from torch.utils.data import Dataset
from copy import deepcopy
from tqdm import tqdm

import torch.nn.functional as F

import random
import torch
import json
import os

from .const import BIGTRANSLATE_LANG_TABLE, LANG_TABLE
from .utils import padded_stack


class TranslationDataset(Dataset):
    def __init__(self, tokenizer, language_pair_dir, language_pairs, mode="train", pretokenize=True):
        
        self.translation_pairs = []
        self.tokenizer = tokenizer
        self.pretokenize = pretokenize
        
        lang_to_en_pairs = [pair.split("-") for pair in language_pairs]
        lang_to_en_pairs = [pair if pair[1] == "en" else reversed(pair) for pair in lang_to_en_pairs]
        lang_to_en_pairs = list(set([f"{lang1}-{lang2}" for lang1, lang2 in lang_to_en_pairs]))
        print(lang_to_en_pairs)
        
        for language_pair in lang_to_en_pairs:
            lang1, lang2 = language_pair.split("-")
            data_path = os.path.join(language_pair_dir, language_pair.replace("-", ""), f"{mode}.{language_pair}.json")
            print(data_path)
            
            with open(data_path, "r") as f:
                translation_pairs = [json.loads(jline)["translation"] for jline in f.read().splitlines()]
                
                for translation_pair in tqdm(translation_pairs, desc=f"Loading {language_pair} {mode} data"):
                    sent1, sent2 = translation_pair[lang1], translation_pair[lang2]
                    
                    if f"{lang1}-{lang2}" in language_pairs:
                        self.translation_pairs.append({"src_lang": lang1, "tgt_lang": lang2, "src_sentence": sent1, "tgt_sentence": sent2})
                    
                    if f"{lang2}-{lang1}" in language_pairs:
                        self.translation_pairs.append({"src_lang": lang2, "tgt_lang": lang1, "src_sentence": sent2, "tgt_sentence": sent1})
        
        random.shuffle(self.translation_pairs)

        if pretokenize:
            self.inputs = []
            
            for record in tqdm(self.translation_pairs, desc=f"Tokenizing {mode} data..."):
                src_lang, tgt_lang, src_sentence, tgt_sentence = record["src_lang"], record["tgt_lang"], record["src_sentence"], record["tgt_sentence"]
                model_inputs = self.get_model_inputs_and_labels(self.get_prompt(src_lang, tgt_lang, src_sentence), tgt_sentence)
                
                self.inputs.append(model_inputs)
            
    
    def get_model_inputs_and_labels(self, prompt, label):
        
        model_inputs = self.tokenizer(prompt + label + self.tokenizer.eos_token, return_tensors="pt", padding="longest", max_length=512, truncation=True, add_special_tokens=False)
        prompt_ids = self.tokenizer(prompt, padding="longest", max_length=512, truncation=True, add_special_tokens=False)["input_ids"]

        model_inputs["labels"] = deepcopy(model_inputs["input_ids"])
        model_inputs["labels"][:, :len(prompt_ids)] = -100
        
        return model_inputs
        
    def get_prompt(self, src_lang, tgt_lang, src_sentence):
        translate_instruct = f"请将以下{BIGTRANSLATE_LANG_TABLE[src_lang]}句子翻译成{BIGTRANSLATE_LANG_TABLE[tgt_lang]}：{src_sentence}"
        return (
            "以下是一个描述任务的指令，请写一个完成该指令的适当回复。\n\n"
            f"### 指令:\n{translate_instruct}\n\n### 回复:")
        
    def __len__(self):
        return len(self.translation_pairs)

    def __getitem__(self, idx):
        if self.pretokenize:
            return self.inputs[idx]
        
        record = self.translation_pairs[idx]
        src_lang, tgt_lang, src_sentence, tgt_sentence = record["src_lang"], record["tgt_lang"], record["src_sentence"], record["tgt_sentence"]
        
        prompt = self.get_prompt(src_lang, tgt_lang, src_sentence)
        model_inputs = self.get_model_inputs_and_labels(prompt, tgt_sentence)
        
        return model_inputs