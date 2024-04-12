# Large Language Models for Expansion of Spoken Language Understanding Systems to New Languages

## Abstract
Spoken Language Understanding (SLU) models are a core component of voice assistants (VA), such as Alexa, Bixby, and Google Assistant. In this paper, we introduce a pipeline designed to extend SLU systems to new languages, utilizing Large Language Models (LLMs) that we fine-tune for machine translation of slot-annotated SLU training data. Our approach improved on the MultiATIS++ benchmark, a primary multi-language SLU dataset, in the cloud scenario using an mBERT model. Specifically, we saw an improvement in the Overall Accuracy metric: from 53% to 62.18%, compared to the existing state-of-the-art method, Fine and Coarse-grained Multi-Task Learning Framework (FC-MTLF). In the on-device scenario (tiny and not pretrained SLU), our method improved the Overall Accuracy from 5.31% to 22.06% over the baseline Global-Local Contrastive Learning Framework (GL-CLeF) method. Contrary to both FC-MTLF and GL-CLeF, our LLM-based machine translation does not require changes in the production architecture of SLU. Additionally, our pipeline is slot-type independent: it does not require any slot definitions or examples.

#### HuggingFace
Fine tuned version of BigTranslate for slot translation on multiAtis++ languages can be found on HuggingFace:

https://huggingface.co/Samsung/BigTranslateSlotTranslator

### How to run
#### BigTranslate slot-translation finetuning
Enter the `BigTranslateFineTuning/` dir
1. Prepare MASSIVE dataset

`
python prepare_massive_data.py --output-dir massive`

2. Run BigTranslate LoRA finetuning

`
bash deepspeed_bigtranslate_train.sh
`

#### Translate multiATIS++ dataset
To get the dataset go to https://github.com/amazon-research/multiatis and then put the ``train_EN.tsv`` file into ``BigTranslateFineTuning/multiATIS`` folder.

Model will be automatically downloaded from huggingface hub. You can also specify path to the model in `translate_mulitatis.sh` script (--model-name argument)

`
bash translate_multiatis.sh
`

#### Training input data preparation
```bash
cd MT/
python prepare_multiatis_jointnlu_traindata.py \
    $TRANSLATED_DATA_DIR \
    False \
    False \
    True \
    True \
    False \
    $OUTPUT_TRAINING_DATA_DIR \
    5 \
    $SOURCE_LANG \
    $TARGET_LANG \
    0.95
```
#### Training input data postprocessing
```bash
bash multiatis_data_postprocess.sh $OUTPUT_TRAINING_DATA_DIR
```

#### Run training and evaluation
```bash
cd JointBERT
mkdir eval_results fails
python3 main.py \
    --task $OUTPUT_TRAINING_DATA_DIR \
    --model_type ${MODEL_TYPE} \
    --patience 5 \
    --train_batch_size 500 \
    --do_train \
    --do_eval \
    --ratio 1.0 \
    --device_name cuda:0 \
    --validation loss \
    --lang $TARGET_LANG \
    --model_dir $MODEL_OUTPUT_DIR \
    --percent_broken 0 \
    --num_train_epochs 50
```
where:
- MODEL_TYPE - `multibert`, `xlmr_scratch` or any other model type defined in [utils.py](./JointBERT/utils.py)
