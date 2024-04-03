from tqdm import tqdm

from datasets import load_dataset
import numpy as np
import argparse
import string
import json
import re
import os

_INTENTS = ['datetime_query', 'iot_hue_lightchange', 'transport_ticket', 'takeaway_query', 'qa_stock',
            'general_greet', 'recommendation_events', 'music_dislikeness', 'iot_wemo_off', 'cooking_recipe',
            'qa_currency', 'transport_traffic', 'general_quirky', 'weather_query', 'audio_volume_up',
            'email_addcontact', 'takeaway_order', 'email_querycontact', 'iot_hue_lightup',
            'recommendation_locations', 'play_audiobook', 'lists_createoradd', 'news_query',
            'alarm_query', 'iot_wemo_on', 'general_joke', 'qa_definition', 'social_query',
            'music_settings', 'audio_volume_other', 'calendar_remove', 'iot_hue_lightdim',
            'calendar_query', 'email_sendemail', 'iot_cleaning', 'audio_volume_down',
            'play_radio', 'cooking_query', 'datetime_convert', 'qa_maths', 'iot_hue_lightoff',
            'iot_hue_lighton', 'transport_query', 'music_likeness', 'email_query', 'play_music',
            'audio_volume_mute', 'social_post', 'alarm_set', 'qa_factoid', 'calendar_set',
            'play_game', 'alarm_remove', 'lists_remove', 'transport_taxi', 'recommendation_movies',
            'iot_coffee', 'music_query', 'play_podcasts', 'lists_query']


def parse_args():
    argument_parser = argparse.ArgumentParser()
    argument_parser.add_argument("--output-dir", "-o", required=True)

    return argument_parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    src_lang = "en-US"
    src_abb = "en"
    adjust_tr = True
    include_transport = True

    src_dataset = load_dataset("AmazonScience/massive", src_lang)
    
    for trg_lang in ["tr-TR", "zh-CN", "ja-JP", "de-DE", "es-ES", "fr-FR", "hi-IN", "pt-PT", ]:

        trg_abb = trg_lang.split("-")[0]

        trg_dataset = load_dataset("AmazonScience/massive", trg_lang)

        for (src_split_name, src_split_data), (trg_split_name, trg_split_data) in zip(src_dataset.items(), trg_dataset.items()):

            alma_inputs = []

            if src_split_name == "validation":
                src_split_name = "valid"

            for src_row, trg_row in tqdm(zip(src_split_data, trg_split_data), total=len(src_split_data)):

                intent_id = src_row["intent"]
                if not include_transport and "transport" in _INTENTS[intent_id]:
                    continue

                if (("audio_" in _INTENTS[intent_id]) or ("cooking_" in _INTENTS[intent_id])) and src_split_name == "train":
                    continue

                if (("audio_" not in _INTENTS[intent_id]) and ("cooking_" not in _INTENTS[intent_id])) and src_split_name == "valid":
                    continue

                src_text = src_row["annot_utt"] 
                trg_text = trg_row["annot_utt"]

                all_src_fields = re.findall(r"\[(.*?)\]", src_row["annot_utt"])
                if all_src_fields:
                    all_src_fields = [field.split(":")[0].strip() for field in all_src_fields]
                    all_src_fields = sorted(set(all_src_fields), key=all_src_fields.index)
                    src_tag = {tag: field for tag, field in zip(string.ascii_lowercase, all_src_fields)}

                src_text = src_row["annot_utt"]
                trg_text = trg_row["annot_utt"]

                for tag, field in zip(string.ascii_lowercase, all_src_fields):
                    if trg_lang == "tr-TR" and adjust_tr and "'" in trg_text:
                        trg_text = re.sub(r"(.*?)(')([a-z]*)(\])", r"\1\4 \2 \3", trg_text)

                    src_text = re.sub(fr"\[ ?{field} ?: ? (.*?)\]", fr"<{tag}> \1 <{tag}>", src_text)
                    trg_text = re.sub(fr"\[ ?{field} ?: ? (.*?)\]", fr"<{tag}> \1 <{tag}>", trg_text)

                    trg_text = re.sub(r" +", " ", trg_text)
                    src_text = re.sub(r" +", " ", src_text)

                alma_inputs.append({"translation": {src_abb: src_text, trg_abb: trg_text}})

            out_dir = f"{args.output_dir}/{trg_abb}{src_abb}"

            os.makedirs(out_dir, exist_ok=True)
            with open(f"{out_dir}/{src_split_name}.{trg_abb}-{src_abb}.json", "w", encoding="utf-8") as f:
                out = '\n'.join(json.dumps(i, ensure_ascii=False) for i in alma_inputs)
                f.write(out)
