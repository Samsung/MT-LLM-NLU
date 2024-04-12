import copy
import json
import logging
import os
import random
import sys

import torch
from torch.utils.data import TensorDataset

from utils import get_intent_labels, get_slot_labels, get_domain_labels

logger = logging.getLogger(__name__)


class InputExample(object):
    """
    A single training/test example for simple sequence classification.

    Args:
        guid: Unique id for the example.
        words: list. The words of the sequence.
        intent_label: (Optional) string. The intent label of the example.
        slot_labels: (Optional) list. The slot labels of the example.
    """

    def __init__(self, guid, words, intent_label=None, domain_label=None, slot_labels=None, intent_name=None, language_label_id=None, language_label=None, sentence_id=None, artificially_broken_sentence_id=None, uniq_hash_id=None):
        self.guid = guid
        self.words = words
        self.intent_label = intent_label
        self.domain_label = domain_label
        self.slot_labels = slot_labels
        self.intent_name = intent_name
        self.language_label_id = language_label_id
        self.language_label = language_label
        self.sentence_id = sentence_id
        self.artificially_broken_sentence_id = artificially_broken_sentence_id
        self.uniq_hash_id = uniq_hash_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, intent_label_id, domain_label_id, slot_labels_ids, language_label_id, sentence_id, artificially_broken_sentence_id, uniq_hash_id):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.intent_label_id = intent_label_id
        self.domain_label_id = domain_label_id
        self.slot_labels_ids = slot_labels_ids
        self.language_labels_ids = language_label_id
        self.sentence_ids = sentence_id
        self.artificially_broken_sentence_id = artificially_broken_sentence_id
        self.uniq_hash_id = uniq_hash_id

    def __repr__(self):
        return str(self.to_json_string())

    def to_dict(self):
        """Serializes this instance to a Python dictionary."""
        output = copy.deepcopy(self.__dict__)
        return output

    def to_json_string(self):
        """Serializes this instance to a JSON string."""
        return json.dumps(self.to_dict(), indent=2, sort_keys=True) + "\n"


class JointProcessor(object):
    """Processor for the JointBERT data set """

    def __init__(self, args):
        self.args = args
        self.language_labels = [None, "pl-PL", "en-GB", "es-ES", "fr-FR", "de-DE"]
        self.artificially_broken_labels = ["not_broken", "broken", "unknown"]
        self.intent_labels = get_intent_labels(args)
        self.slot_labels = get_slot_labels(args)
        self.domain_labels = get_domain_labels(args) if not args.no_domains else None

        self.input_text_file = 'seq.in'
        self.intent_label_file = 'label'
        self.slot_labels_file = 'seq.out'
        self.domain_label_file = 'domain'
        self.language_label_file = 'language'
        self.sentence_id = 'sentence_id'
        self.uniq_sentence_hash = 'unique_sentence_hash'
        self.hashes_mapping = {}
        self.max_idx = 0
        self.hashes = set()
        self.target_lang = self.args.lang.lower().split("-")[0]
        if self.args.lang not in self.language_labels:
            raise Exception(self.args.lang + " not in available language labels: " + str(self.language_labels))

    @classmethod
    def _read_file(cls, input_file, quotechar=None):
        """Reads a tab separated value file."""
        if not os.path.isfile(input_file):
            return None 
        with open(input_file, "r", encoding="utf-8") as f:
            lines = []
            for line in f:
                lines.append(line.strip())

            return lines
    
    def data_postprocess(cls, words, slots):
        words = " ".join(words)
        slots = slots.split()
        if "mode" in words.lower() or "tryb" in words.lower() or "panel" in words.lower():
            words = words.split()
            if len(words) != len(slots):
                print("=====INCONSISTENT ANNOTATION=======")
                print(words)
                print(slots)
                print("=====INCONSISTENT ANNOTATION=======")

            import copy
            new_slot_labels = copy.deepcopy(slots)
            print("=======ANNOTATION CHANGE=========")
            for idx, word in enumerate(words):
                if word.lower() == "panel":
                    print(4444)
                    if slots[idx] == "o" and slots[idx-1] == "b-bixby.settingsapp.settingsname":
                        new_slot_labels[idx] = "i-bixby.settingsapp.settingsname"
                        print(words)
                        print(slots)
                        print(new_slot_labels)
                        break
                    try:
                        slots[idx+1]
                    except:
                        break
                    if slots[idx] == "o" and slots[idx+1] == "b-bixby.settingsapp.settingsname":
                        new_slot_labels[idx] = "b-bixby.settingsapp.settingsname"
                        new_slot_labels[idx+1] = "i-bixby.settingsapp.settingsname"
                        print(words)
                        print(slots)
                        print(new_slot_labels)
                        break
                    else:
                        print("===========NOT CHANGED===============")
                        print(words)
                        print(slots)
                        print(new_slot_labels)
                        #sys.exit(0)
                        break

                if word.lower() == "mode" or word.lower() == "tryb":
                    if slots[idx].startswith("i-"):
                        new_slot_labels[idx] = "o"
                        print(words)
                        print(slots)
                        print(new_slot_labels)
                        break
                    elif slots[idx].startswith("b-"):
                        new_slot_labels[idx] = "o"
                        if slots[idx + 1] == "o":
                            print(words)
                            print(slots)
                            print(new_slot_labels)
                            break
                        else:
                            new_slot_labels[idx + 1] = "b-" + slots[idx + 1].split("-")[1]
                        print(words)
                        print(slots)
                        print(new_slot_labels)
                        break
                    else:
                        print("===========NOT CHANGED===============")
                        print(words)
                        print(slots)
                        print(new_slot_labels)
                        #sys.exit(0)
                        break
            print("=========ANNOTATION CHANGE=============")
            return " ".join(new_slot_labels)
        else:
            return " ".join(slots)
    
    def broke_example(self, text, slot, intent):

        intent_source = copy.deepcopy(intent)
        text_source = copy.deepcopy(text)
        slot_source = copy.deepcopy(slot)
        slot = slot.split(" ")
        
        slot_present = False
        if any([bio != "o" for bio in slot]):
            slot_present = True

        
        if slot_present:
            rand = random.randrange(100)
            
            if rand < 30:
                #change some slot to "o"
                trial = 0
                while True:
                    randomIndex = random.choice(range(len(slot)))
                    if slot[randomIndex] == "o":
                        trial += 1
                        if trial > len(slot) * 10:
                            break
                        else:
                            continue
                    else:
                        slot[randomIndex] = "o"
                        break

            #change some second slot to "o" with 30% P
            if rand < 60: 
                trial = 0
                while True:
                    randomIndex = random.choice(range(len(slot)))
                    if slot[randomIndex] == "o":
                        trial += 1
                        if trial > len(slot) * 10:
                            break
                        else:
                            continue
                    else:
                        slot[randomIndex] = "o"
                        break
            
            # play (rihanna)[artist] on spotify => (play rihanna)[artist] on spotify
            if rand >= 60:
                trial = 0
                while True:
                    randomIndex = random.choice(range(len(slot)))
                    if slot[randomIndex].startswith("b-"):
                        if randomIndex != 0:
                            slot[randomIndex - 1] = slot[randomIndex]
                            slot_name = "i-" + slot[randomIndex].split("-")[1]
                            if slot_name in self.slot_labels:
                                slot[randomIndex] = "i-" + slot[randomIndex].split("-")[1]
                            break
                            
                        if randomIndex == 0:
                            if slot[-1] == slot[randomIndex]:
                                slot[-1] = "o"
                            else:
                                slot[-1] = slot[randomIndex]
                                break
                    else:
                        trial += 1
                        if trial > len(slot) * 20:
                            raise Exception("During broking of training data, cannot find b- slot in following sentence annotation: " + str(slot))

        else:
            while True:
                intent = self.intent_labels[random.choice(range(len(self.intent_labels)))]
                if intent != intent_source:
                    break
                    
                    
                    
            

        slot = " ".join(slot)

#         print(text)
#         print(text_source)
#         print(slot_source)
#         print(slot)
#         print(intent_source)
#         print(intent)
        if text == text_source and slot == slot_source and intent == intent_source:
            logger.error(f"Utterance not broken: {text_source}, {slot_source}, {intent_source}")

        return text, slot, intent

    def _create_example(self, set_type, i, text, intent, domain, slot, language_label, sentence_id, uniq_hash, artificially_broken):
        
        guid = "%s-%s" % (set_type, i)
        # 1. input_text
        words = text.split()  # Some are spaced twice

        if artificially_broken == "broken":
            words, slot, intent = self.broke_example(words, slot, intent)
            
        
        
        # 2. intent
        intent_label = self.intent_labels.index(intent) if intent in self.intent_labels else self.intent_labels.index("UNK")
        if intent_label == self.intent_labels.index("UNK"):
            print(f'Unknown intent:{intent}')

        
        
        if domain:
            domain_label = self.domain_labels.index(domain) if domain in self.domain_labels else self.domain_labels.index("UNK")
        else:
            domain_label = 0
        # 3. slot

        if intent.lower() == "bixby.settingsapp.enablesettings-none-none" or intent.lower() == "bixby.settingsapp.disablesettings-none-none":
            slot = self.data_postprocess(words, slot)


        slot_labels = []
        for s in slot.split():
            slot_labels.append(self.slot_labels.index(s) if s in self.slot_labels else self.slot_labels.index("UNK"))
            if s not in self.slot_labels:
                print(f'Unknown slot:{s}')


        if len(words) != len(slot_labels):
            print("=============INCONSISTENT ANNOTATION=================")
            print(words)
            print(slot_labels)
            print("=============INCONSISTENT ANNOTATION=================")

        #4. language
        
        language_label_id = self.language_labels.index(language_label)

        #5. sentence_id
        sentence_id = sentence_id
        
        #6. artificially broken sentence
        artificially_broken_sentence_id = self.artificially_broken_labels.index(artificially_broken)
        
        #7
        if set_type == "train" and self.args.hash:
            if self.max_idx + 1 in self.hashes_mapping and self.max_idx - 1 not in self.hashes_mapping or uniq_hash in self.hashes:
                print("456 check code")
                sys.exit(0)
            else:
                self.hashes_mapping[self.max_idx] = uniq_hash
                self.hashes.add(uniq_hash)
                uniq_hash_id = self.max_idx
                self.max_idx += 1
        else:
            uniq_hash_id = 0
            self.hashes_mapping[0] = 0




        return InputExample(guid=guid, words=words, intent_label=intent_label, domain_label=domain_label, slot_labels=slot_labels,    intent_name=intent, language_label_id=language_label_id, language_label=language_label, sentence_id=sentence_id, artificially_broken_sentence_id=artificially_broken_sentence_id, uniq_hash_id=uniq_hash_id)

    
    
    def _create_examples(self, texts, intents, domains, slots, set_type, language_labels, sentence_ids, uniq_sentence_hashes):
        """Creates examples for the training and dev sets."""
        if not domains:
            domains = [None for _ in range(len(texts))]
        if not language_labels:
            language_labels = [None for _ in range(len(texts))]
        if not sentence_ids:
            sentence_ids = [0 for _ in range(len(texts))]
        if not uniq_sentence_hashes:
            uniq_sentence_hashes = ["0"] * len(texts)
        
        examples = []
        source_traindata_size = len(texts)
        n_broken = 0

        for i, (text, intent, domain, slot, language_label, sentence_id, uniq_sentence_hash) in enumerate(zip(texts, intents, domains, slots, language_labels, sentence_ids, uniq_sentence_hashes)):

            assert language_label in self.language_labels
            
            rand = random.randrange(100)
            if self.args.do_sieving == True and rand < self.args.percent_broken and set_type == "train" and language_label == "en-GB":
                n_broken += 1
                example = self._create_example(set_type, i, text, intent, domain, slot, language_label, sentence_id, uniq_sentence_hash, artificially_broken="not_broken")
                broke_example = self._create_example(set_type, i, text, intent, domain, slot, language_label, sentence_id, uniq_sentence_hash + "1234", artificially_broken="broken")
                examples.append(broke_example)
                examples.append(example)
            elif language_label == "en-GB" and self.args.percent_broken > 0:
                raise Exception("If sieving is turned off, you should not broke english training sentences.")
            else:
                example = self._create_example(set_type, i, text, intent, domain, slot, language_label, sentence_id, uniq_sentence_hash, artificially_broken="unknown")
                examples.append(example)
        
        
        if set_type == "train":
            logger.info("================")
            logger.info("Traindata size (without broken sentences): " + str(source_traindata_size))
            logger.info("Traindata size (including broken): " + str(len(examples)))
            logger.info("Broken sentences in trainset: " + str(n_broken))
            logger.info("==============================")

            assert len(examples) == source_traindata_size + n_broken
            if not self.args.do_sieving:
                assert n_broken == 0
                
        random.shuffle(examples)
        
        return examples

    def get_examples(self, mode, pretrain=False):
        """
        Args:
            mode: train, dev, testl1, testl2
        """
        task = self.args.task if not pretrain else self.args.pretrain_task
        data_path = os.path.join(self.args.data_dir, task, mode)
        logger.info("LOOKING AT {}".format(data_path))
        domains = self._read_file(os.path.join(data_path, self.domain_label_file)) if not self.args.no_domains else None
        return self._create_examples(texts=self._read_file(os.path.join(data_path, self.input_text_file)),
                                     intents=self._read_file(os.path.join(data_path, self.intent_label_file)),
                                     domains=domains,
                                     slots=self._read_file(os.path.join(data_path, self.slot_labels_file)),
                                     set_type=mode,
                                     language_labels=self._read_file(os.path.join(data_path, self.language_label_file)),
                                     sentence_ids=self._read_file(os.path.join(data_path, self.sentence_id)),
                                     uniq_sentence_hashes=self._read_file(os.path.join(data_path, self.uniq_sentence_hash))
                                    )


processors = {
    "en_kd": JointProcessor,
    "en_exp_kd": JointProcessor,
    "en_uniq_kd": JointProcessor,
    "es_kd": JointProcessor,
    "leyzer": JointProcessor,
    "leyzer_en": JointProcessor,
    "leyzer_es": JointProcessor,
    "leyzer_pl": JointProcessor,
    "leyzer-big": JointProcessor,
    "leyzer-big2": JointProcessor,
    "leyzer-mal": JointProcessor,
    "leyzer-snips": JointProcessor,
    "leyzer_pl_faulted": JointProcessor,
    "pl_kd": JointProcessor,
    "2_massive_pl-PL_translated": JointProcessor,
    "massive_en-US": JointProcessor,
    "prod_pl": JointProcessor,
    "train_en_test_pl" : JointProcessor,
    "train_enpl_test_pl" : JointProcessor,
    "train_pl_test_pl" : JointProcessor,
    "train_pl_large_test_pl" : JointProcessor,
    "train_enpl_large_test_pl" : JointProcessor,
    "train_en_large_test_pl" : JointProcessor,
    "train_enpl_valid_en_test_pl" : JointProcessor,
    "train_enpl_large_valid_en_test_pl" : JointProcessor,
    "sieved_01_slot_enpl_test_pl" : JointProcessor,
    "sieved_large_003_slot_enpl_test_pl": JointProcessor,
    "train_enpl_test_pl_sed": JointProcessor,
    "broda_xlmr_largepl_sieving6": JointProcessor,
    "xlmr_largepl_sieving3": JointProcessor,
    "xlmr_largepl_sieving4" : JointProcessor,
    "sieving_quality_train_enpl_test_pl" : JointProcessor,
    "contrastive_sieving_test" : JointProcessor,
    "massive_train_enpl_test_pl": JointProcessor,
    "massive_train_enpl_test_pl3": JointProcessor,
    "massive_train_en_test_pl": JointProcessor,
    "multiatis_train_ende_test_de" : JointProcessor,
    "multiatis_train_ende_test_de_top5" : JointProcessor,
    "multiatis_train_enes_test_es_top5" : JointProcessor,
    "multiatis_train_enfr_test_fr": JointProcessor,
    "multiatis_train_enpt_test_pt": JointProcessor,
    "multiatis_train_enes_test_es": JointProcessor,
    "multiatis_train_enes_test_es_standardized" : JointProcessor,
    "multiatis_train_enfr_test_fr_removedmulti" : JointProcessor,
    "multiatis_train_enfr_test_fr_removedmulti2" : JointProcessor,
    "multiatis_train_ende_test_de_removedmulti" : JointProcessor,
    "multiatis_train_enpt_test_pt_removedmulti" : JointProcessor,
    "leyzer_train_enpl_test_pl" : JointProcessor,
    "leyzer_train_en_test_pl" :  JointProcessor,
    "leyzer_train_enpl_test_pl_top1" : JointProcessor,
    "leyzer_train_en_test_pl_top1" : JointProcessor,
    "leyzer_pure_pl_test_pl" : JointProcessor,
    "leyzer_train_enes_test_pl___top5_iva_mt_wslot-m2m100_418M-en-pl-massive_filtered" : JointProcessor,
    "leyzer_train_enes_test_es_top5_____iva_mt_wslot-m2m100_418M-en-es" : JointProcessor,
    "leyzer_train_enes_test_es_top5_true_____iva_mt_wslot-m2m100_418M-en-es" : JointProcessor,
    "leyzer_train_en_test_es_top5_true_____iva_mt_wslot-m2m100_418M-en-es" : JointProcessor,
    "leyzer_train_enes_test_es_top5_true_____iva_mt_wslot-m2m100_418M-en-es-massive_unfiltered" : JointProcessor,
    "leyzer_train_enpl_test_pl___top5_iva_mt_wslot-m2m100_418M-en-pl-massive_filtered" : JointProcessor,
    "leyzer_train_enpl_test_pl_top5_____iva_mt_wslot-m2m100_418M-en-pl-noleyzer" : JointProcessor,
    "multiatis_train_enfr_test_fr_top5__iva_mt_wslot-m2m100_418M-en-fr" : JointProcessor,
    "multiatis_train_ende_test_de__iva_mt_wslot-m2m100_418M-en-de" : JointProcessor,
    "multiatis_train_ende_test_de_newvalid__iva_mt_wslot-m2m100_418M-en-de" : JointProcessor,
    "leyzer_train_enes_test_es_top5_true_newvalid_____leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es" : JointProcessor,
    "leyzer_train_enes_test_es_top5_true_newvalid_____leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es_refactor_translate" : JointProcessor,
    "leyzer_train_enes_test_es_top5_true_newvalid_____leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es_refactor_translate2" : JointProcessor,
    "leyzer_train_enes_test_es_top5_true_newvalid_____leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es_refactor_translate3" : JointProcessor,
    "contrastive_sieving_test_prod" : JointProcessor,
    "dnlu_train_enpl_test_pl" : JointProcessor,
    "leyzer_train_enes_test_es_top1_____leyzer_translated_to_es_top5_iva_mt_wslot-m2m100_418M-en-es_refactor_translate3" : JointProcessor,
    "leyzer_es_top5_20230531" : JointProcessor,
    "leyzer_es_20230531" : JointProcessor,
    "output_123" : JointProcessor,
}



def convert_examples_to_features(examples, max_seq_len, tokenizer,
                                 pad_token_label_id=-100,
                                 cls_token_segment_id=0,
                                 pad_token_segment_id=0,
                                 sequence_a_segment_id=0,
                                 mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    features = []
    for (ex_index, example) in enumerate(examples):
        if ex_index % 5000 == 0:
            logger.info("Writing example %d of %d" % (ex_index, len(examples)))


        # Tokenize word by word (for NER)
        tokens = []
        slot_labels_ids = []
        for word, slot_label in zip(example.words, example.slot_labels):
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            slot_labels_ids.extend([int(slot_label)] + [pad_token_label_id] * (len(word_tokens) - 1))

        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > max_seq_len - special_tokens_count:
            tokens = tokens[:(max_seq_len - special_tokens_count)]
            slot_labels_ids = slot_labels_ids[:(max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        slot_labels_ids += [pad_token_label_id]
        token_type_ids = [sequence_a_segment_id] * len(tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        slot_labels_ids = [pad_token_label_id] + slot_labels_ids
        token_type_ids = [cls_token_segment_id] + token_type_ids

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real
        # tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = max_seq_len - len(input_ids)
        input_ids = input_ids + ([pad_token_id] * padding_length)
        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_labels_ids = slot_labels_ids + ([pad_token_label_id] * padding_length)

        assert len(input_ids) == max_seq_len, "Error with input length {} vs {}".format(len(input_ids), max_seq_len)
        assert len(attention_mask) == max_seq_len, "Error with attention mask length {} vs {}".format(len(attention_mask), max_seq_len)
        assert len(token_type_ids) == max_seq_len, "Error with token type length {} vs {}".format(len(token_type_ids), max_seq_len)
        assert len(slot_labels_ids) == max_seq_len, "Error with slot labels length {} vs {}".format(len(slot_labels_ids), max_seq_len)

        intent_label_id = int(example.intent_label)
        domain_label_id = int(example.domain_label)
        language_label_id = int(example.language_label_id)
        sentence_id = int(example.sentence_id)
        artificially_broken_sentence_id = int(example.artificially_broken_sentence_id)
        uniq_hash_id = int(example.uniq_hash_id)

        


        if ex_index < 5:
            logger.info("*** Example ***")
            logger.info("guid: %s" % example.guid)
            logger.info("tokens: %s" % " ".join([str(x) for x in tokens]))
            logger.info("input_ids: %s" % " ".join([str(x) for x in input_ids]))
            logger.info("attention_mask: %s" % " ".join([str(x) for x in attention_mask]))
            logger.info("token_type_ids: %s" % " ".join([str(x) for x in token_type_ids]))
            logger.info("intent_label: %s (id = %d)" % (example.intent_label, intent_label_id))
            logger.info("language_label: %s (id = %d)" % (example.language_label, language_label_id))
            logger.info("sentence_id: %s" % (example.sentence_id))
            logger.info("slot_labels: %s" % " ".join([str(x) for x in slot_labels_ids]))
            logger.info("artificially broken sentence: %s", artificially_broken_sentence_id)
            logger.info("uniq_hash_id: %s", uniq_hash_id)

        features.append(
            InputFeatures(input_ids=input_ids,
                          attention_mask=attention_mask,
                          token_type_ids=token_type_ids,
                          intent_label_id=intent_label_id,
                          domain_label_id=domain_label_id,
                          slot_labels_ids=slot_labels_ids,
                          language_label_id=language_label_id,
                          sentence_id=sentence_id,
                          artificially_broken_sentence_id=artificially_broken_sentence_id,
                          uniq_hash_id=uniq_hash_id
                          ))
        


    return features


def load_and_cache_examples(args, tokenizer, mode, teacher=False, pretrain=False, ratio=1.0):
    task = args.task if not pretrain else args.pretrain_task
    processor = processors.get(task, JoinProcessor)(args)

    # Load data features from cache or dataset file
    if hasattr(args, 'model_name_or_path'):
        model_name = args.model_name_or_path
    else:
        if teacher:
            model_name = args.teacher_model_name_or_path
        else:
            model_name = args.student_model_name_or_path

    cached_features_file = os.path.join(
        args.data_dir,
        'cached_{}_{}_{}_{}'.format(
            mode,
            task,
            list(filter(None, model_name.split("/"))).pop(),
            args.max_seq_len
        )
    )

    if False:
    #if os.path.exists(cached_features_file):
        logger.info("Loading features from cached file %s", cached_features_file)
        features = torch.load(cached_features_file)
    else:
        # Load data features from dataset file
        logger.info("Creating features from dataset file at %s", args.data_dir)
        if mode in ["train", "dev", "test", "testl1", "testl2"]:
            examples = processor.get_examples(mode, pretrain)
        else:
            raise Exception("For mode, Only train, dev, test is available")

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        pad_token_label_id = args.ignore_index
        features = convert_examples_to_features(examples, args.max_seq_len, tokenizer,
                                                pad_token_label_id=pad_token_label_id)
        logger.info("Saving features into cached file %s", cached_features_file)
        torch.save(features, cached_features_file)

    dataset_size = int(len(features) * ratio)
    features = features[:dataset_size]

    #if mode == "train":
    # features = [ele for ele in features for _ in range(40)]

    all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)
    all_intent_label_ids = torch.tensor([f.intent_label_id for f in features], dtype=torch.long)
    all_domain_label_ids = torch.tensor([f.domain_label_id for f in features], dtype=torch.long)
    all_slot_labels_ids = torch.tensor([f.slot_labels_ids for f in features], dtype=torch.long)
    all_language_labels_ids = torch.tensor([f.language_labels_ids for f in features], dtype=torch.long)
    all_sentence_ids = torch.tensor([f.sentence_ids for f in features], dtype=torch.long)
    all_broken_ids = torch.tensor([f.artificially_broken_sentence_id for f in features], dtype=torch.long)
    all_hashes_ids = torch.tensor([f.uniq_hash_id for f in features], dtype=torch.long)


    dataset = TensorDataset(all_input_ids, all_attention_mask,
                            all_token_type_ids, all_intent_label_ids, all_domain_label_ids, all_slot_labels_ids,
                            all_language_labels_ids, all_sentence_ids, all_broken_ids, all_hashes_ids)

    return dataset, processor
