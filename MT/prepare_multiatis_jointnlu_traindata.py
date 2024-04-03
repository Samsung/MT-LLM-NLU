import json
import os
import random
import shutil
import sys

DOMAIN_LIST = set()
INTENT_LIST = set()
BIO_TAGS_LIST = set()
TEST_DOMAIN_LIST = set()
TEST_INTENT_LIST = set()
TEST_BIO_LIST = set()

directory = sys.argv[1]  # "output_datasets/leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es_refactor_translate"
ENG_TRAIN_DATA = sys.argv[2] == "True"  # False
MASSIVE = sys.argv[3] == "True"  # False
multiatis = sys.argv[4] == "True"  # False
remove_multiintents = sys.argv[5] == "True"  # False
standardize_multiintents = sys.argv[6] == "True"  # False
OUTPUT_PATH = sys.argv[7]  # "leyzer_train_enes_test_es_top1_____leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es_refactor_translate3"
MAX_HYP = int(sys.argv[8])  # 1
SOURCE_LANG = sys.argv[9]  # "en-GB"
TARGET_LANG = sys.argv[10]  # "es-ES"
VALIDATION_DATASET_RATIO = float(sys.argv[11])  # 0.95
random.seed(1234567890)

def clean_BIO(BIO):
    if isinstance(BIO, str):
        BIO = BIO.split()
    return [token.split(":")[0] for token in BIO]


def BILO_to_BIO(BILO, test=False):
    if isinstance(BILO, str):
        BILO = BILO.split()

    BIO_tags = []
    for token in BILO:
        if token == "o":
            BIO_tags.append("o")
        else:
            flag, slot_name = token.split("-", 1)
            flag = "b" if flag == "u" else "i" if flag == "l" else flag
            BIO = f"{flag}-{slot_name}"
            BIO_tags.append(BIO)
            if test:
                TEST_BIO_LIST.add(slot_name)
            else:
                BIO_TAGS_LIST.add(slot_name)
    return BIO_tags


def standardize_multiatis_multiintent(intent):
    intent = intent.lower()
    wrong_map = {
        'atis_flight#atis_airfare' : 'atis_airfare#atis_flight',
        'atis_flight_no#atis_airline' : 'atis_airline#atis_flight_no'
    }
    if intent in wrong_map:
        intent = wrong_map[intent]


    return intent

def split_intent_GLCLEF(intent):
        wrong_map = {
            'airfare': 'atis_airfare',
            'airline': 'atis_airline',
            'flight': 'atis_flight',
            'flight_no': 'atis_flight_no'
        }
        if " " in intent:
            results = intent.strip().split()
        elif "#" in intent:
            results = intent.strip().split('#')
        else:
            if intent in wrong_map:
                results = wrong_map[intent]
            else:
                return intent
        for i in range(len(results)):
            if results[i] in wrong_map:
                results[i] = wrong_map[results[i]]

        return results[0]


def is_out_of_scope(domain, intent, source_signal, target_signal):
    if domain not in TEST_DOMAIN_LIST:
        return True

    if intent not in TEST_INTENT_LIST:
        return True

    for BIO_tag in source_signal + target_signal:
        if BIO_tag == 'o':
            continue
        tag_name = BIO_tag[2:]
        if tag_name not in TEST_BIO_LIST:
            return True
    return False

def process_test_line(line, multiatis, remove_multiintents, standardize_multiintents):
    splitted = [x.lower().strip() for x in line.split('\t')]

    if multiatis:
        if TARGET_LANG in ["es-ES", "fr-FR"]:
            splitted[1] = splitted[1].replace(" ", "")
            
        if remove_multiintents:
            domain = split_intent_GLCLEF(splitted[3])
            intent = split_intent_GLCLEF(splitted[3])
        elif standardize_multiintents:
            domain = standardize_multiatis_multiintent(splitted[3])
            intent = standardize_multiatis_multiintent(splitted[3])
        else:
            domain = splitted[3]
            intent = splitted[3]
            
        sentence = splitted[1].lower().split()        
        BIO = splitted[2].lower().split()
    else:
        raise
    
    return domain, intent, sentence, BIO


with open(f"{OUTPUT_PATH}/test/test.tsv", 'w') as test_file:
    with open(f"test_data.tsv") as text_file:
        # skip header
        text_file.readline()
        for line in text_file:
            if not line:
                continue

            domain, intent, words, slot_tokens = process_test_line(
                line,
                multiatis,
                remove_multiintents,
                standardize_multiintents
            )

            TEST_DOMAIN_LIST.add(domain)
            TEST_INTENT_LIST.add(intent)

            if len(words) != len(slot_tokens):
                print("====FAULTY TC=====")
                print(words)
                print(slot_tokens)
                print("====FAULTY TC=====")
                #continue

            slot_tokens = BILO_to_BIO(slot_tokens, test=True)

            if len(words) != len(slot_tokens):
                print("====FAULTY TC=====")
                print(words)
                print(slot_tokens)
                print("====FAULTY TC=====")
                #continue

            sentence = " ".join(words)
            slot_notation = " ".join(slot_tokens)

            line = "\t".join([domain, intent, sentence, slot_notation])
            test_file.write(line + "\n")


data = []
sentence_id = 0
for filename in [f for f in os.listdir(directory) if f.endswith(".json")]:
     path_to_file = os.path.join(directory, filename)
     with open(path_to_file, encoding="utf-8") as json_data:
        d = json.load(json_data)
        if 'ERROR' in d:
            print(d)
            continue
        
        domain = d['domain'].lower()
        intent = d['intent'].lower()
        if multiatis:
            if remove_multiintents:
                domain = split_intent_GLCLEF(domain)
                intent = split_intent_GLCLEF(intent)
            elif standardize_multiintents:
                domain = standardize_multiatis_multiintent(domain)
                intent = standardize_multiatis_multiintent(intent)

        hypothesis_number = int(d['hypothesis_number'])
        if hypothesis_number >= MAX_HYP:
            continue

        DOMAIN_LIST.add(domain)
        INTENT_LIST.add(intent)

        item = {'domain': domain,
                'intent': intent,
                'source_sentence': d['intermediate_json']['input_sentence'].lower(),  
                'source_BIO': " ".join(
                    BILO_to_BIO(clean_BIO(d['intermediate_json']['input_BIO'].lower()))).lower()
                }
        if ENG_TRAIN_DATA:
            item['target_sentence'] = item['source_sentence'].lower() 
            item['target_BIO'] = item['source_BIO'].lower()
        else:
            assert len(d['BIO']) == len(d['tokens'])
            assert len(clean_BIO(d['BIO'])) == len(d['tokens'])
            item["target_sentence"] = ' '.join(d["tokens"])
            item['target_BIO'] = " ".join(clean_BIO(d['BIO'])).lower()
            if len(item['target_BIO'].split()) != len(item['target_sentence'].split()):
                print("============Inconsistent annotation==========")
                print(d['BIO'])
                print(d['tokens'])
                print(clean_BIO(d['BIO']))
                print(item['target_BIO'])
                print(item['target_BIO'].split())
                print(item['target_sentence'])
                print(item['target_sentence'].split())
                print("============Inconsistent annotation==========")

        if is_out_of_scope(
                domain,
                intent,
                item["source_BIO"].split(),
                item["target_BIO"].split()):
            continue
        
        item["atis_sentence_id"] = int(d["input_line"].split("\t")[0])
        item["sentence_id"] = sentence_id
        data.append(item)
        sentence_id += 1

print("=======DOMAINS present in TEST but not in TRAIN: =======")
print(TEST_DOMAIN_LIST - DOMAIN_LIST)
print("==========INTENTS present in TEST but not in TRAIN: ========")
print(TEST_INTENT_LIST - INTENT_LIST)
print("==========SLOTS present in TEST but not in TRAIN: ========")
print(TEST_BIO_LIST - BIO_TAGS_LIST)


with open(f"{OUTPUT_PATH}/slot_label.txt", 'w') as text_file:
    text_file.write("PAD" + "\n")
    text_file.write("UNK" + "\n")
    text_file.write("o" + "\n")
    for iob_name in sorted(TEST_BIO_LIST):
        text_file.write(f"b-{iob_name}\n")
        text_file.write(f"i-{iob_name}\n")

with open(f"{OUTPUT_PATH}/intent_label.txt", 'w') as text_file:
    text_file.write("UNK" + "\n")
    for intent_name in sorted(TEST_INTENT_LIST):
        text_file.write(intent_name + "\n")

with open(f"{OUTPUT_PATH}/domain_label.txt", 'w') as text_file:
    text_file.write("UNK" + "\n")
    for domain_name in sorted(TEST_DOMAIN_LIST):
        text_file.write(domain_name + "\n")

random.shuffle(data)
    
data_uniq = []
unique_source_lines = set()

for item in data:
    domain = item['domain']
    intent = item['intent']
    source_sentence = item['source_sentence']
    source_BIO = item['source_BIO']
    
    source_line = "".join([domain, intent, source_sentence, source_BIO])
    
    if source_line not in unique_source_lines:
        unique_source_lines.add(source_line)
        data_uniq.append(item)
        
n_of_unique_source_sentences = len(unique_source_lines)

ALL_DATA_SIZE = len(data)
DEVSET_SIZE = 0
TRAINSET_SIZE = 0
split_index = int(round(VALIDATION_DATASET_RATIO * len(data_uniq)))
dev_uniq_lines = set()


devset_sentence_ids = []
lang_abb = TARGET_LANG.split("-")[0].upper()

with open(f"multiATIS/raw/ldc/train_{lang_abb}.tsv") as original_atis_trainset:
    original_atis_trainset = original_atis_trainset.read().splitlines()

    if TARGET_LANG in ["hi-IN", "tr-TR"]:
        dev_idxs = random.sample(list(range(len(original_atis_trainset))), len(data_uniq) - split_index)
        print(dev_idxs)

with open(f"{OUTPUT_PATH}/dev/dev.tsv", 'w') as dev_file:
    for idx, item in enumerate(data_uniq): #here we use unique source sentences
        sentence_id = str(item['sentence_id'])
        domain = item['domain']
        intent = item['intent']
        source_sentence = item['source_sentence']
        source_BIO = item['source_BIO']

        atis_sentence_id = item["atis_sentence_id"]
        if (idx > split_index and TARGET_LANG not in ["hi-IN", "tr-TR"]) or (TARGET_LANG in ["hi-IN", "tr-TR"] and atis_sentence_id in dev_idxs):
            line = "\t".join([domain, intent, source_sentence, source_BIO, sentence_id, SOURCE_LANG])
            line_from_original_train = original_atis_trainset[atis_sentence_id]
            id_, utterance, slot_labels, intent = line_from_original_train.split("\t")
            
            line_from_original_train = "\t".join([intent, intent, utterance.lower(), slot_labels.lower(), sentence_id, SOURCE_LANG])
            
            dev_file.write(line + "\n")
            dev_file.write(line_from_original_train + "\n")
            
            DEVSET_SIZE += 1
            
            # source_line = "".join([domain, intent, source_sentence, source_BIO])
            # dev_uniq_lines.add(source_line)
            devset_sentence_ids.append(atis_sentence_id)
            
assert len(devset_sentence_ids) == DEVSET_SIZE
# assert len(dev_uniq_lines) == DEVSET_SIZE

all_train_sentences = []
n_notuniq_english_devsent = 0
with open(f"{OUTPUT_PATH}/train/train.tsv", 'w') as train_file:
    for idx, item in enumerate(data): #here we use source sentences multiplied by 5 + 5 hypothesis returned by MT
        sentence_id = str(item['sentence_id'])
        domain = item['domain']
        intent = item['intent']
        source_sentence = item['source_sentence']
        source_BIO = item['source_BIO']
        target_sentence = item['target_sentence']
        target_BIO = item['target_BIO']
        
        atis_sentence_id = item["atis_sentence_id"]
        
        if atis_sentence_id in devset_sentence_ids:
            continue

        line = "\t".join([domain, intent, target_sentence, target_BIO, sentence_id, TARGET_LANG])
        if target_sentence not in all_train_sentences:
            all_train_sentences.append(target_sentence)
            train_file.write(line + "\n")
            TRAINSET_SIZE += 1
        
        # source_line = "".join([domain, intent, source_sentence, source_BIO])
        # if source_sentence not in all_train_sentences:
            line = "\t".join([domain, intent, source_sentence, source_BIO, sentence_id, SOURCE_LANG])
            all_train_sentences.append(source_sentence)
            train_file.write(line + "\n")
            TRAINSET_SIZE += 1
        
        # train_file.write(line + "\n")
        # TRAINSET_SIZE += 1
    
    
print(f"All data items: {ALL_DATA_SIZE}")
print(f"Number of unique source sentences: {n_of_unique_source_sentences}")
print(f"Devset split index: {split_index}")
print(f"Devset size: {DEVSET_SIZE}")
print(f"Number of not unique english sentences that are in devset: {n_notuniq_english_devsent}")
print(f"Trainset size: {TRAINSET_SIZE}")

#for backward compatibility
shutil.copyfile(f"{OUTPUT_PATH}/dev/dev.tsv", f"{OUTPUT_PATH}/testl1/testl1.tsv")
shutil.copyfile(f"{OUTPUT_PATH}/dev/dev.tsv", f"{OUTPUT_PATH}/testl2/testl2.tsv")

train_sentences = set()

with open(f"{OUTPUT_PATH}/train/train.tsv") as train_file:
    with open(f"{OUTPUT_PATH}/dev/dev.tsv") as dev_file:
        for line_train in train_file:
            train_splitted = line_train.lower().strip().split("\t")
            train_intent = train_splitted[1]
            train_sentence = train_splitted[2]
            train_lang = train_splitted[5]
            train_info = "".join([train_intent, train_sentence, train_lang])
            train_sentences.add(train_info)
        for line_dev in dev_file:
            dev_splitted = line_dev.lower().strip().split("\t")
            dev_intent = dev_splitted[1]
            dev_sentence = dev_splitted[2]
            dev_lang = dev_splitted[5]
            dev_info = "".join([dev_intent, dev_sentence, dev_lang])
            if dev_info in train_sentences:
                print(dev_info)
                print("This dev sentences is in train...")
                sys.exit(0)
