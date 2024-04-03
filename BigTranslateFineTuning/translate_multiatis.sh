#/bin/bash

for LANG in  "ES" "DE" "HI" "TR" "ZH" "FR" "JA" "PT"
do
    cat "multiATIS/raw/ldc/train_EN.tsv" | python feedback_loop_translation.py \
        --model-name "BigTranslateSlotTranslator" \
        --output-path "output_datasets/en_to_${LANG,,}" \
        --target-lang "${LANG,,}" \
        --num-variants 5 \
        --chunk-size 100 \

done
