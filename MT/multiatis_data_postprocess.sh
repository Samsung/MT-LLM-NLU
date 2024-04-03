#!/usr/bin/env bash

DATASET_NAME=$1
SCRIPT_DIR=$(realpath "$(dirname "$0")")

for dataset in ${DATASET_NAME}; do
    cd ${dataset}
    for dataset_name in "train" "dev" "test" "testl1"; do
        dataset=${dataset_name}.tsv
        cd ${dataset_name}
        cut -f1 ${dataset} > domain
        cut -f2 ${dataset} > label
        cut -f3 ${dataset} > seq.in
        cut -f4 ${dataset} > seq.out
        if [[ ${dataset_name} == "train" ]]; then
            cut -f5 ${dataset} > sentence_id
            cut -f6 ${dataset} > language
        fi
        cd ..
    done
    #rm -r testl2/
    #cp -r testl1/ testl2
    cd ..
done

cp -r "${DATASET_NAME}" "${SCRIPT_DIR}/../JointBERT/data"
rm -rf "nlu_input_data/"