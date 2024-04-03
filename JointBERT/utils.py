import logging
import os
import random

import numpy as np
import pandas as pd
import torch
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from sklearn.metrics import confusion_matrix
from transformers import BertConfig, AlbertConfig, XLMRobertaConfig
from transformers import BertTokenizer, AlbertTokenizer, XLMRobertaTokenizer

from model import JointBERT, JointAlbert, DistillJointBERT, DistillJointLSTM, JointLSTM, \
    DistillScratchBERT, JointXLMR

MODEL_CLASSES = {
    'bert': (BertConfig, JointBERT, BertTokenizer),
    'distillbert': (BertConfig, DistillJointBERT, BertTokenizer),
    'distillberttoxtreme256': (BertConfig, DistillJointBERT, BertTokenizer),
    'distilltinybert': (BertConfig, DistillJointBERT, BertTokenizer),
    'jointlstm': (BertConfig, JointLSTM, BertTokenizer),
    'distilljointlstm': (BertConfig, DistillJointLSTM, BertTokenizer),
    'multibert': (BertConfig, JointBERT, BertTokenizer),
    'xlmr': (XLMRobertaConfig, JointXLMR, XLMRobertaTokenizer),
    'scratch_xlmr': (XLMRobertaConfig, JointXLMR, XLMRobertaTokenizer),
    'albert': (AlbertConfig, JointAlbert, AlbertTokenizer),
    'tinybert': (BertConfig, JointBERT, BertTokenizer),
    'smallbert': (BertConfig, JointBERT, BertTokenizer),
    'xtremedistil': (BertConfig, JointBERT, BertTokenizer),
    'xtremedistil256': (BertConfig, JointBERT, BertTokenizer),
    'distillscratch': (BertConfig, DistillScratchBERT, BertTokenizer),
    'scratch': (BertConfig, JointBERT, BertTokenizer),
    'scratch_multi': (BertConfig, JointBERT, BertTokenizer),
    'spainbert':(BertConfig, JointBERT, BertTokenizer),
    'spaintinybert':(BertConfig, JointBERT, BertTokenizer),
    'multilingual_mini': (BertConfig, JointBERT, BertTokenizer),
    'multilingual_small': (BertConfig, JointBERT, BertTokenizer),
    'spaintinybertner':(BertConfig, JointBERT, BertTokenizer),
    'distillspaintinybert': (BertConfig, DistillJointBERT, BertTokenizer),
    'distillspaintinybertner': (BertConfig, DistillJointBERT, BertTokenizer)
}

MODEL_PATH_MAP = {
    'bert': 'bert-base-uncased',
    'multibert': 'bert-base-multilingual-uncased',
    'xlmr': 'xlm-roberta-base',
    'scratch_xlmr': 'xlm-roberta-base',
    'distillbert': 'prajjwal1/bert-tiny',
    'jointlstm': 'bert-base-uncased',
    'distilljointlstm': 'bert-base-uncased',
    'distillberttoxtreme256': 'microsoft/xtremedistil-l6-h256-uncased',
    'distilltinybert': 'prajjwal1/bert-tiny',
    'albert': 'albert-xxlarge-v1',
    'tinybert': 'prajjwal1/bert-tiny',
    'smallbert': 'prajjwal1/bert-small',
    'xtremedistil' : 'microsoft/xtremedistil-l6-h384-uncased',
    'xtremedistil256' : 'microsoft/xtremedistil-l6-h256-uncased',
    'distillscratch': 'prajjwal1/bert-tiny',
    'scratch': 'prajjwal1/bert-tiny',
    'scratch_multi': 'bert-base-multilingual-uncased',
    'spainbert': 'dccuchile/bert-base-spanish-wwm-uncased',
    'spaintinybert': 'mrm8488/spanish-TinyBERT-betito',
    'multilingual_mini': 'dbmdz/bert-mini-historic-multilingual-cased',
    'multilingual_small': 'dbmdz/bert-small-historic-multilingual-cased',
    'spaintinybertner': 'mrm8488/TinyBERT-spanish-uncased-finetuned-ner'
}

def get_intent_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.intent_label_file), 'r', encoding='utf-8')]


def get_slot_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.slot_label_file), 'r', encoding='utf-8')]

def get_domain_labels(args):
    return [label.strip() for label in open(os.path.join(args.data_dir, args.task, args.domain_label_file), 'r', encoding='utf-8')]

def load_tokenizer(args, mode=None):
    if hasattr(args, 'model_type'):
        return MODEL_CLASSES[args.model_type][2].from_pretrained(args.model_name_or_path)
    elif mode == 'teacher':
        return MODEL_CLASSES[args.teacher_model_type][2].from_pretrained(args.teacher_model_name_or_path)
    elif mode == 'student':
        return MODEL_CLASSES[args.student_model_type][2].from_pretrained(args.student_model_name_or_path)


def init_logger(logdir):
    if not os.path.exists(logdir):
        os.makedirs(logdir, exist_ok=True)
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s -   %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO,
                        handlers=[logging.FileHandler(os.path.join(logdir, "logs.log")),
                                    logging.StreamHandler()]
                        )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # torch.backends.cudnn.deterministic=True
    # torch.use_deterministic_algorithms(True)
    os.environ['PYTHONHASHSEED'] = str(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed) # if you are using multi-GPU.
    # torch.backends.cudnn.benchmark = True

    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(intent_preds, intent_labels, slot_preds, slot_labels, domain_preds, domain_labels, no_domains):
    assert len(intent_preds) == len(intent_labels) == len(slot_preds) == len(slot_labels)
    results = {}
    intent_result = get_intent_acc(intent_preds, intent_labels)
    domain_result = get_domain_acc(domain_preds, domain_labels) if not no_domains else None
    slot_result = get_slot_metrics(slot_preds, slot_labels)
    sementic_result = get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels)

    results.update(intent_result)
    if not no_domains:
        results.update(domain_result)
    results.update(slot_result)
    results.update(sementic_result)

    return results

def compute_metrics_per_label(intent_preds, intent_labels, intent_label_lst,
                              slot_preds, slot_labels, slot_label_lst,
                              domain_preds, domain_labels, domain_label_lst, testset, no_domains=False):
    intent_corr = np.zeros(len(intent_label_lst))
    intent_total = np.zeros(len(intent_label_lst))

    for pred, true in zip(intent_preds, intent_labels):
        if pred == true:
            intent_corr[true] += 1

        intent_total[true] += 1

    if not no_domains:
        domain_corr = np.zeros(len(domain_label_lst))
        domain_total = np.zeros(len(domain_label_lst))

        for pred, true in zip(domain_preds, domain_labels):
            if pred == true:
                domain_corr[true] += 1

            domain_total[true] += 1

    slot_tp = np.zeros(len(slot_label_lst))
    slot_fp = np.zeros(len(slot_label_lst))
    slot_fn = np.zeros(len(slot_label_lst))

    for slots_pred, slots_true in zip(slot_preds, slot_labels):
        for pred, true in zip(slots_pred, slots_true):
            pred_idx = slot_label_lst.index(pred)
            true_idx = slot_label_lst.index(true)
            if pred == true:
                slot_tp[true_idx] += 1
            else:
                slot_fp[pred_idx] += 1
                slot_fn[true_idx] += 1


    if not no_domains:
        domain_results = []
        for i, domain in enumerate(domain_label_lst):
            acc = 0 if domain_total[i] == 0 else domain_corr[i]/domain_total[i]
            domain_results.append((domain, acc, domain_total[i]))

        df_domain = pd.DataFrame(domain_results, columns=['Domain', 'Accuracy', 'Count'])

    intent_results = []

    for i, intent in enumerate(intent_label_lst):
        acc = 0 if intent_total[i] == 0 else intent_corr[i]/intent_total[i]
        intent_results.append((intent, acc, intent_total[i]))

    df_intent = pd.DataFrame(intent_results, columns=['Intent', 'Accuracy', 'Count'])

    slot_dict= classification_report(slot_labels, slot_preds, output_dict=True)

    slot_results = []
    for slot, metrics in slot_dict.items():
        slot_results.append((slot, metrics['precision'], metrics['recall'], metrics['f1-score'], metrics['support']))
    df_slot = pd.DataFrame(slot_results, columns=['Slot', 'Precision', 'Recall', 'F1 Score', 'Support'])

    if not no_domains:
        df_domain.to_csv('./eval_results/domain_res_{}.tsv'.format(testset), sep='\t', index=None)
    df_intent.to_csv('./eval_results/intent_res_{}.tsv'.format(testset), sep='\t', index=None)
    df_slot.to_csv('./eval_results/slot_res_{}.tsv'.format(testset), sep='\t', index=None)

def compute_confusion_matrices(domain_labels, domain_preds, domain_label_lst,
                               intent_labels, intent_preds, intent_label_lst,
                               slot_labels, slot_preds, slot_label_lst,
                               logger, no_domains=False):

    def get_n_biggest_indices(arr, n):
        indices = np.argpartition(arr.ravel(), -n)[-n:]
        indices = indices[np.argsort(-arr.ravel()[indices])]
        return np.unravel_index(indices, arr.shape)

    def remove_correct_predictions(arr):
        for i in range(arr.shape[0]):
            arr[i][i] = 0

        return arr

    def get_most_confused_labels(confusion_matrix, label_list, n):
        confusion_matrix = remove_correct_predictions(confusion_matrix)
        biggest_ix = get_n_biggest_indices(confusion_matrix, n)
        for y, x in zip(biggest_ix[0], biggest_ix[1]):
            logger.info(f"{label_list[y]} recognized as {label_list[x]} {confusion_matrix[y][x]} times")

    flat_slot_labels = [slot for sentence in slot_labels for slot in sentence]
    flat_slot_preds = [slot for sentence in slot_preds for slot in sentence]

    if not no_domains:
        domain_confusion_matrix = confusion_matrix(domain_labels, domain_preds, labels=[i for i in range(len(domain_label_lst))])
    intent_confusion_matrix = confusion_matrix(intent_labels, intent_preds, labels=[i for i in range(len(intent_label_lst))])
    slot_confusion_matrix = confusion_matrix(flat_slot_labels, flat_slot_preds, labels=[i for i in range(len(slot_label_lst))])

    if not no_domains:
        logger.info("**DOMAIN**")
        get_most_confused_labels(domain_confusion_matrix, domain_label_lst, 2)

    logger.info("**INTENT**")
    get_most_confused_labels(intent_confusion_matrix, intent_label_lst, 5)
    logger.info("**SLOT**")
    get_most_confused_labels(slot_confusion_matrix, slot_label_lst, 30)



def get_slot_metrics(preds, labels):
    assert len(preds) == len(labels)
    return {
        "slot_precision": precision_score(labels, preds),
        "slot_recall": recall_score(labels, preds),
        "slot_f1": f1_score(labels, preds)
    }


def get_intent_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "intent_acc": acc
    }

def get_domain_acc(preds, labels):
    acc = (preds == labels).mean()
    return {
        "domain_acc": acc
    }



def read_prediction_text(args):
    return [text.strip() for text in open(os.path.join(args.pred_dir, args.pred_input_file), 'r', encoding='utf-8')]


def get_sentence_frame_acc(intent_preds, intent_labels, slot_preds, slot_labels):
    """For the cases that intent and all the slots are correct (in one sentence)"""
    # Get the intent comparison result
    intent_result = (intent_preds == intent_labels)

    # Get the slot comparision result
    slot_result = []
    for preds, labels in zip(slot_preds, slot_labels):
        assert len(preds) == len(labels)
        one_sent_result = True
        for p, l in zip(preds, labels):
            if p != l:
                one_sent_result = False
                break
        slot_result.append(one_sent_result)
    slot_result = np.array(slot_result)

    sementic_acc = np.multiply(intent_result, slot_result).mean()
    return {
        "sementic_frame_acc": sementic_acc
    }
