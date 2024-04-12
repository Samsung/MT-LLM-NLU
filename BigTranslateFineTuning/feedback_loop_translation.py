#! pip install transformers sentencepiece datasets
from src.slot_translation_utils import annotate_with_XML, remove_incorrect_tags, decode_translation
from vllm import LLM, SamplingParams
from collections import Counter
from peft import PeftModel
from tqdm import tqdm

import argparse
import torch
import sys
import re
import json
import os

from src.const import BIGTRANSLATE_LANG_TABLE, LANG_TABLE
from time import sleep


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", required=False, default="")
    parser.add_argument("--output-path", required=True)
    parser.add_argument("--target-lang", required=True)
    parser.add_argument("--chunk-size", required=True, type=int)
    parser.add_argument("--num-variants", required=True, type=int)
    parser.add_argument("--tensor-parallel-size", required=True, type=int)
    args = parser.parse_args()
    
    return args


def translation_feedback_loop(translations, input_text_variants, prompts, n=10):
    for _ in range(10):
        
        repeat_generation = []
        repeat_generation_ids = []
    
        for i, (in_text, translation, prompt) in enumerate(zip(input_text_variants, translations, prompts)):
            if Counter(re.findall(r"<[a-z]>", in_text)) != Counter(re.findall(r"<[a-z]>", translation)):
                repeat_generation.append(prompt)
                repeat_generation_ids.append(i)

        if not repeat_generation:
            break
            
        repeated_translations = model.generate(repeat_generation, sampling_params=sampling_params)
        repeated_translations  = [output.outputs[0].text for output in repeated_translations]
        
        for repeat_id, repeat_translation in zip(repeat_generation_ids, repeated_translations):
            translations[repeat_id] = repeat_translation
    
    return translations


def gen_multiple_translations(input_texts, lang, num_variants=5):
    
    prompts = []
    for input_text in input_texts:
        translate_instruct = f"请将以下{BIGTRANSLATE_LANG_TABLE['en']}句子翻译成{BIGTRANSLATE_LANG_TABLE[lang]}：{input_text}"
        prompt = (
            "以下是一个描述任务的指令，请写一个完成该指令的适当回复。\n\n"
            f"### 指令:\n{translate_instruct}\n\n### 回复:"
        )
            
        prompt = re.sub(r"(<[a-z]>) (.*?) <[a-z]>", r"\1\2\1", prompt)
        prompts.extend([prompt] * num_variants)
    
    outputs = model.generate(prompts, sampling_params=sampling_params)
    
    translations = [output.outputs[0].text for output in outputs]
    
    input_text_variants = [input_text for input_text in input_texts for _ in range(5)]
    translations = translation_feedback_loop(translations, input_text_variants, prompts)

    if lang == "tr":
        translations = [re.sub(r"'", " ' ", translation) for translation in translations]
        
    translations = [translations[i: i+num_variants] for i in range(0, len(translations), num_variants)]
    
    return translations


def get_sentence_and_bio(splitted_line):
    return splitted_line[1].lower(), splitted_line[2].lower()

def get_domain_and_intent(splitted_line):
    return splitted_line[3], splitted_line[3]


if __name__ == "__main__":
    args = parse_args()
    
    os.makedirs(args.output_path, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = LLM(args.model_name, tensor_parallel_size=args.tensor_parallel_size)
    sampling_params = SamplingParams(temperature=0.4, top_p=0.95, max_tokens=100, stop=["\n"])
    
    input_lines = sys.stdin.read().splitlines()[1:]
    chunked_lines = [input_lines[i: i+args.chunk_size] for i in range(0, len(input_lines), args.chunk_size)]
    save_id = 0
    
    for chunk_idx, lines in tqdm(enumerate(chunked_lines), total=len(chunked_lines), desc=f"Translating en to {args.target_lang}..."):
            
        splitted_lines = [line.strip().split('\t') for line in lines]
        
        sentence_BIOs_list = [get_sentence_and_bio(splitted_line) for splitted_line in splitted_lines]
        input_jsons = [annotate_with_XML(sentence, BIO) for sentence, BIO in sentence_BIOs_list]

        annotated_sentences = [input_json['annotated_sentence'] for input_json in input_jsons]
        translated_sentences = gen_multiple_translations(annotated_sentences, args.target_lang, args.num_variants)
        
        for i , (batch, input_json, splitted_line, line) in enumerate(zip(translated_sentences, input_jsons, splitted_lines, lines)):
            for hyp_idx, translated_sentence in enumerate(batch):
                tokens, BIO = decode_translation(translated_sentence, input_json)
                domain, intent = get_domain_and_intent(splitted_line)

                output_json = {
                    "input_line": line,
                    "translated_sentence": translated_sentence,
                    "hypothesis_number": hyp_idx,
                    "tokens": tokens,
                    "BIO": BIO,
                    "domain": domain,
                    "intent": intent,
                    "intermediate_json": input_json
                }

                with open(f'{args.output_path}/sentence_{save_id}.json', 'w') as f:
                    json.dump(output_json, f, indent=4, ensure_ascii=False)
                
                save_id += 1
        
