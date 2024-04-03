#!/bin/bash
set -euo pipefail

task=$1
cuda_device_id=$2
lang=$3

base_train_args=" --task $task --model_type xlmr --patience 5 --train_batch_size 500 --do_train --do_eval --ratio 1.0 --device_name cuda:${cuda_device_id} --validation loss --lang ${lang}"

for cosda_rate in 0 100 ; do
    # train no-sieving model
    model_dir="model_eval_${task}_no_sieving_cosda_rate_${cosda_rate}"
    if [[ ! -d $model_dir ]] ; then
        train_args="$base_train_args --model_dir $model_dir"
        echo "train $model_dir"
        python3 main.py $train_args --percent_broken 0 \
            --cosda_rate ${cosda_rate}

        rm model_eval_${task}_no_sieving_cosda_rate_${cosda_rate}/pytorch_model.bin
    else
    echo "skip training for $model_dir"
    fi
    
    for fscore_beta in 1 0.5 2 0.1 0.25 3 4 5 ; do
        for sieving_coef in 0.80 ; do
            for do_average_sieving in "" "--do_average_sieving" ; do
                for cosda_as_tp in "" "--cosda_as_tp" ; do
                    # train 1st pass sieving model
                    [[ $cosda_rate -eq 0 ]] && [[ $cosda_as_tp != "" ]] && continue
                    model_dir="model_eval_${task}_cosda_rate_${cosda_rate}_sieving_coef_${sieving_coef}${do_average_sieving/--/_}${cosda_as_tp/--/_}_fscore_beta_${fscore_beta}_eval_sieving"
                    if [[ ! -d $model_dir ]] ; then
                        train_args="$base_train_args --do_sieving --dont_sieve_english --fscore_beta $fscore_beta --percent_broken 100 --cosda_rate ${cosda_rate} $do_average_sieving $cosda_as_tp "
                        echo "train $model_dir"
                        python3 main.py \
                            --model_dir $model_dir \
                            --eval_sieving \
                            --sieving_coef ${sieving_coef} \
                            $train_args
                        rm $model_dir/pytorch_model.bin
                    else
                        echo "skip training for $model_dir"
                    fi

                    # train 2nd pass sieving model
                    if [[ ! -f $model_dir/logs.log ]] ; then
                        echo "DOES NOT EXIST: $model_dir/logs.log"
                        continue
                    fi
                    if [[ ! -d ${model_dir/model_eval/model_eval_optimal_sc} ]] ; then
                        optimal_sc=$(grep -Po 'Eval sieving mode, not doing sieving. Best sieving f.*' $model_dir/logs.log | tail -n 1 | grep -oP '[0-9.]+$')
                        [ -z $optimal_sc ] && continue
                        optimal_sc=$(python -c "print($optimal_sc - 0.01)")

                        train_args="$base_train_args --do_sieving --dont_sieve_english --fscore_beta $fscore_beta --percent_broken 100 --cosda_rate ${cosda_rate} $do_average_sieving $cosda_as_tp"
                        echo "train ${model_dir/model_eval/model_eval_optimal_sc}"

                        python3 main.py \
                            --model_dir ${model_dir/model_eval/model_eval_optimal_sc} \
                            --sieving_coef ${optimal_sc} \
                            $train_args
                        rm "${model_dir/model_eval/model_eval_optimal_sc}"/pytorch_model.bin
                    else
                        echo "skip training for ${model_dir/model_eval/model_eval_optimal_sc}"
                    fi
                done
            done
        done
    done
done
