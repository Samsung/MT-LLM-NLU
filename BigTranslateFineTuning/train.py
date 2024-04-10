from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, LlamaTokenizer, AutoModelForSeq2SeqLM
from peft import LoraConfig, get_peft_model

from transformers.trainer_callback import TrainerCallback
from src.callback import SavePeftModelCallback

from transformers.trainer import Trainer
from src.data import TranslationDataset
from src.utils import collate_fn

from torch.utils.data import DataLoader
from functools import partial
from pprint import pprint

import pandas as pd
import numpy as np

import argparse
import random
import torch
import yaml
import os

from datetime import datetime
from tqdm import tqdm
tqdm.pandas()

def parse_cfg():
    parser = argparse.ArgumentParser(description='')

    parser.add_argument("-C", "--config", help="config path")
    args = parser.parse_args()
    
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)
        
    return cfg


def seed_everything(seed=3407):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    

    
if __name__ == "__main__":
    seed_everything()
    timestamp = datetime.today().strftime("%d-%m-%Y-%H-%M")
    cfg = parse_cfg()
    
    pprint(cfg)
    
    model = AutoModelForCausalLM.from_pretrained(cfg["model"]["name"], torch_dtype=torch.bfloat16)
        
    lora_config = LoraConfig(
        r=cfg["LORA"]["rank"],
        lora_alpha=cfg["LORA"]["rank"] * 2,
        target_modules=cfg["LORA"]["target_modules"],
        lora_dropout=cfg["LORA"]["dropout"],
        bias=cfg["LORA"]["bias"],
        task_type=cfg["LORA"]["task_type"]
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()
    
    tokenizer = LlamaTokenizer.from_pretrained(cfg["model"]["name"])
        
    tokenizer.pad_token = tokenizer.eos_token

    train_dataset = TranslationDataset(tokenizer, cfg["data"]["DATA_PATH"], cfg["data"]["train_language_pairs"], mode="train", pretokenize=False)
    valid_dataset = TranslationDataset(tokenizer, cfg["data"]["DATA_PATH"], cfg["data"]["valid_language_pairs"], mode="valid", pretokenize=False)

    collate_fn = partial(collate_fn, tokenizer=tokenizer)
    
    training_args = TrainingArguments(**cfg["training"])

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        tokenizer=tokenizer,
        data_collator=collate_fn,
        callbacks=[SavePeftModelCallback],
    )
    
    train_result = trainer.train()
    
    model = model.merge_and_unload()
    tokenizer.save_pretrained(cfg["model"]["output_dir"])
    model.save_pretrained(cfg["model"]["output_dir"])
    
