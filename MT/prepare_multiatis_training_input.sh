#!/usr/bin bash
atis_test_file=${1:-"multiATIS/raw/ldc/test_EN.tsv"}
directory=${2:-"output_datasets/multiatis_translated_to_fr_top5__iva_mt_wslot-m2m100_418M-en-fr"}
ENG_TRAIN_DATA=False
MASSIVE=False
multiatis=True
standardize_multiintents=False
OUTPUT_PATH=${3:-"multiatis_train_enfr_test_fr_top5__iva_mt_wslot-m2m100_418M-en-fr"}
SOURCE_LANG="en-GB"
TARGET_LANG=${4:-"fr-FR"}
MAX_HYP=${5:-5}
remove_multiintents=${6:-True}
VALIDATION_DATASET_RATIO=0.95
mkdir -p \
    $OUTPUT_PATH/test \
    $OUTPUT_PATH/dev \
    $OUTPUT_PATH/train \
    $OUTPUT_PATH/testl1 \
    $OUTPUT_PATH/testl2

cat $atis_test_file > test_data.tsv

python prepare_multiatis_jointnlu_traindata.py \
    $directory \
    $ENG_TRAIN_DATA \
    $MASSIVE \
    $multiatis \
    $remove_multiintents \
    $standardize_multiintents \
    $OUTPUT_PATH \
    $MAX_HYP \
    $SOURCE_LANG \
    $TARGET_LANG \
    $VALIDATION_DATASET_RATIO

bash multiatis_data_postprocess.sh $OUTPUT_PATH
