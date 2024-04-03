import logging
import os
import sys
from operator import itemgetter
from statistics import mean

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torch.utils.data import TensorDataset
from tqdm import tqdm, trange
from transformers import AdamW, get_linear_schedule_with_warmup

from display_model_parameters import display_parameters
from utils import MODEL_CLASSES, compute_metrics, compute_metrics_per_label, get_intent_labels, \
    get_slot_labels, get_domain_labels, compute_confusion_matrices

logger = logging.getLogger(__name__)

class Trainer(object):
    #@profile
    def __init__(self, args, train_dataset=None, dev_dataset=None, test_dataset=None, test_dataset_l1=None, test_dataset_l2=None, tokenizer=None, data_processor=None):
        self.args = args
        self.train_dataset = train_dataset
        self.train_dataloader = None
        self.dev_dataset = dev_dataset
        self.test_dataset = test_dataset
        self.test_dataset_l1 = test_dataset_l1
        self.test_dataset_l2 = test_dataset_l2
        
        self.args.eval_batch_size = self.args.train_batch_size

        self.intent_label_lst = get_intent_labels(args)
        self.slot_label_lst = get_slot_labels(args)
        self.domain_label_lst = get_domain_labels(args) if not args.no_domains  else None
        self.sieving_stop = False
        
        self.data_processor = data_processor
        self.language_labels = self.data_processor.language_labels
        self.artificially_broken_labels = self.data_processor.artificially_broken_labels

        self.hashes_mapping = self.data_processor.hashes_mapping

        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index
        self.sieving = args.do_sieving
        self.contrastive_learning = args.do_contrastive
        


        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task=args.task)

        if 'scratch' in args.model_type:
            self.config.hidden_size = 64
            self.config.num_attention_heads = 2
            self.config.num_hidden_layers = 2
            self.model = self.model_class(config=self.config,
                                        args=args,
                                        intent_label_lst=self.intent_label_lst,
                                        slot_label_lst=self.slot_label_lst,
                                        domain_label_lst=self.domain_label_lst)
        else:
            self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                        config=self.config,
                                                        args=args,
                                                        intent_label_lst=self.intent_label_lst,
                                                        slot_label_lst=self.slot_label_lst,
                                                        domain_label_lst=self.domain_label_lst)


        self.tokenizer = tokenizer

        # GPU or CPU
        if args.device_name:
            self.device = args.device_name
        else:
            self.device = "cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        #self.device = "cpu"
        self.model.to(self.device)


        display_parameters(logger, self.model, max_level=2)
        self.train_loss = []
        self.val_loss = []
        self.best_sieving_fscore = 0

    def prepare_train_for_contrastive(self):
        dict_ = {}
        for example in self.train_dataset:
            domain = example[3].int().item()
            intent = example[4].int().item()
            language = example[6].int().item()
            sentence_id = example[7].int().item()
            broken_id = example[8].int().item()
            if sentence_id in dict_:
                assert domain == dict_[sentence_id][0][3].int().item()
                assert intent == dict_[sentence_id][0][4].int().item()
                dict_[sentence_id].append(example)
            else:
                dict_[sentence_id] = []
                dict_[sentence_id].append(example)

        train_dataset = []
        tensordatasets = []
        for sentence_id in dict_:
            if len(dict_[sentence_id]) == 3:
                example_A = dict_[sentence_id][0]
                example_A_2 = dict_[sentence_id][1]
                example_A_3 = dict_[sentence_id][2]
                tensordatasets.append(example_A)
                tensordatasets.append(example_A_2)
                tensordatasets.append(example_A_3)
            elif len(dict_[sentence_id]) == 2:
                example_A = dict_[sentence_id][0]
                example_A_2 = dict_[sentence_id][1]
                tensordatasets.append(example_A)
                tensordatasets.append(example_A_2)
            elif len(dict_[sentence_id]) == 1: #after sieving, pair sentences could be removed
                example_A = dict_[sentence_id][0]
                tensordatasets.append(example_A)
            else:
                print("======ERROR 12345=======")
                sys.exit(0)


        train_dataset = torch.utils.data.ConcatDataset([tensordatasets])
        train_dataloader = DataLoader(train_dataset, batch_size=self.args.train_batch_size)


        return (train_dataloader, train_dataset)
    
    
    
    def train(self):
        def seed_worker(worker_id):
            worker_seed = torch.initial_seed() % 2**32
            numpy.random.seed(worker_seed)
            random.seed(worker_seed)

        if self.contrastive_learning == True:
            result = self.prepare_train_for_contrastive()
            train_dataloader = result[0]
            self.train_dataset = result[1]
        else:
            if self.sieving == True:
                train_sampler = SequentialSampler(self.train_dataset)
                train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
            else:
                g = torch.Generator()
                g.manual_seed(0)
                train_sampler = RandomSampler(self.train_dataset)
                train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size, num_workers=0,
                                                worker_init_fn=seed_worker, generator=g)


        self.train_dataloader = train_dataloader


        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)

        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        worse_epochs = 0
        epoch = 0

        best_validation = 0
        if self.args.validation == 'loss':
            best_validation = 900
        val_res = {}

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")

        for _ in train_iterator:
            if self.sieving==True and self.contrastive_learning == False:
                train_sampler = SequentialSampler(self.train_dataset)
                train_dataloader = DataLoader(self.train_dataset, batch_size=self.args.train_batch_size)
                self.train_dataloader = train_dataloader
            if self.contrastive_learning == True:
                result = self.prepare_train_for_contrastive()
                self.train_dataloader = result[0]
                self.train_dataset = result[1]

            improved = False
            epoch += 1
            epoch_iterator = tqdm(self.train_dataloader, desc="Iteration")
            sieving_input = []
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'domain_label_ids': batch[4],
                          'slot_labels_ids': batch[5]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]
                if self.contrastive_learning == True:
                    inputs['language_id'] = batch[6]
                    inputs['sentence_id'] = batch[7]
                

                if self.contrastive_learning == True:
                    outputs = self.model(**inputs, contrastive=True)
                    loss = outputs[0]
                else:
                    outputs = self.model(**inputs)
                    loss = outputs[0]


                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break



            #train_res = self.evaluate("train")
            train_res = {}
            train_res['loss'] = 0 #to speedup training when trainset is large
            val_res = self.evaluate("dev")
            testl1_res = self.evaluate("testl1")
            test_res = self.evaluate("test")

            test_accuracy = round(test_res["sementic_frame_acc"], 2)

            current_validation = val_res[self.args.validation]

            if self.sieving == True and self.sieving_stop == False:
                logger.info("=====Sieving forward pass=====")
                self.model.zero_grad()
                with torch.no_grad():
                    for step, batch in enumerate(epoch_iterator):
                        batch = tuple(t.to(self.device) for t in batch) 
                        inputs = {'input_ids': batch[0],
                                    'attention_mask': batch[1],
                                    'intent_label_ids': batch[3],
                                    'domain_label_ids': batch[4],
                                    'slot_labels_ids': batch[5]}
                        inputs['token_type_ids'] = batch[2]
                        if self.contrastive_learning == True:
                            inputs['language_id'] = batch[6]
                            inputs['sentence_id'] = batch[7]
                            outputs_sieving = self.model(**inputs, sieving=True, contrastive=True)
                        else:
                            outputs_sieving = self.model(**inputs, sieving=True)
                        sieving_losses = outputs_sieving[1]
                        sieving_input.append([batch[0].detach(), sieving_losses])
                self.model.zero_grad()
                print("=====Finished Sieving forward pass=====")
                
                self.sieve(sieving_input, epoch=_, test_accuracy=test_accuracy)


            self.train_loss.append(train_res['loss'])
            self.val_loss.append(val_res['loss'])


            if self.args.validation == 'loss' and current_validation < best_validation:
                
                best_validation = current_validation
                logger.info("Saving model because eval loss decreased to %s", current_validation)
                logger.info("Saving model from epoch %d", epoch)
                self.save_model()
                worse_epochs = 0
            elif self.args.validation != 'loss' and best_validation < current_validation:
                best_validation = current_validation
                logger.info(f"Saving model because {self.args.validation} increased to {current_validation}")
                logger.info("Saving model from epoch %d", epoch)
                self.save_model()
                worse_epochs = 0
                sys.exit(0)
            else:
                worse_epochs += 1

            if worse_epochs == self.args.patience or 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def sieve(self, sieving_input, epoch=0, test_accuracy=0):

        data = []
        global_idx = 0
        column_names = ["removed", "sentence_id", "utterance", "domain_loss", "intent_loss", "slot_loss", "contrastive_loss", "max_loss", "average_loss", "type_of_error", "true_domain", "pred_domain", "true_intent", "predicted_intent", "true_slots", "predicted_slots", "true_slots_merged", "predicted_slots_merged", "broken", "lang", "hash", "hash_id"]
            
        for batched_inputids, batched_sieving_losses in sieving_input:
            batch_utterances = [self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)) for ids in batched_inputids]
            domain_losses = batched_sieving_losses[0]
            intent_losses = batched_sieving_losses[1]
            slot_losses = batched_sieving_losses[2]
            contrastive_losses = batched_sieving_losses[3]
            for idx in range(len(batch_utterances)):
                utterance = batch_utterances[idx]
                domain_loss = float(domain_losses[idx].float())
                intent_loss = float(intent_losses[idx].float())
                slot_loss = float(slot_losses[idx].float())
                contrastive_loss = float(contrastive_losses[idx])
                max_loss = max([domain_loss, intent_loss, slot_loss, contrastive_loss])
                average_loss = mean([slot_loss, intent_loss])
               

                data.append([global_idx, utterance, domain_loss, intent_loss, slot_loss, contrastive_loss, max_loss, average_loss])

                tensors_tuple = self.train_dataset[global_idx]
                input_ids = tensors_tuple[0]
                utterance_2 = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True))


                assert utterance == utterance_2
                global_idx+=1


        #to lepiej przeniesc do linijki z train_res = self.evaluate("train"), zeby nie duplikowac
        utterances, domain_preds, out_domain_label_ids, intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list = self.evaluate("train", sieving=True)


        assert len(data) == len(self.train_dataset) == len(utterances) == global_idx

        data_updated = []
        hashes = set()
        for item in data:
            type_of_error = []
            item_globalidx = item[0]
            item_utterance = item[1]
            eval_utterance = utterances[item_globalidx]
            assert item_utterance==eval_utterance
            tensors_tuple = self.train_dataset[item_globalidx]
            input_ids = tensors_tuple[0]
            broken_id = int(tensors_tuple[8].cpu().detach().numpy())
            hash_id = int(tensors_tuple[9].cpu().detach().numpy())
            lang_id = int(tensors_tuple[6].cpu().detach().numpy())
            lang = self.language_labels[lang_id].lower()
            broken = self.artificially_broken_labels[broken_id]
            hash_ = self.hashes_mapping[hash_id]

            if hash_ in hashes and self.args.hash:
                raise Exception("Hash " + str(hash_) + " is not unique in training dataset, something went wrong")
            else:
                hashes.add(hash_)
                
            utterance_2 = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True))
            assert item_utterance == utterance_2
            predicted_domain = self.domain_label_lst[domain_preds[item_globalidx]]
            true_domain = self.domain_label_lst[out_domain_label_ids[item_globalidx]]
            predicted_intent = self.intent_label_lst[intent_preds[item_globalidx]]
            true_intent = self.intent_label_lst[out_intent_label_ids[item_globalidx]]
            predicted_slots = []
            true_slots = []
            for slots_pred, slots_true in zip(slot_preds_list[item_globalidx], out_slot_label_list[item_globalidx]):
                predicted_slots.append(slots_pred)
                true_slots.append(slots_true)
            true_slots_merged = " ".join(true_slots)
            predicted_slots_merged = " ".join(predicted_slots)
            if predicted_slots != true_slots:
                type_of_error.append("slots error")
            if predicted_domain != true_domain:
                type_of_error.append("domain error")
            if true_intent != predicted_intent:
                type_of_error.append("intent error")
            all_columns = item + [type_of_error, true_domain, predicted_domain, true_intent, predicted_intent, true_slots, predicted_slots, true_slots_merged, predicted_slots_merged, broken, lang, hash_, hash_id]
            data_updated.append(all_columns)
            

        assert len(data) == len(data_updated)
        data_notsorted = data_updated
        
        #column_names = ["sentence_id", "utterance", "domain_loss", "intent_loss", "slot_loss", "contrastive_loss", "max_loss", "average_loss", "type_of_error", "true_domain", "pred_domain", "true_intent", "predicted_intent", "true_slots", "predicted_slots", "true_slots_merged", "predicted_slots_merged", "broken", "lang", "hash"]
        if self.args.do_average_sieving:
            sieving_loss_type = 7 #average_loss == average(slot_loss, intent_loss)
        else:
            sieving_loss_type = 4 #slot_loss


        #print(data_notsorted)
        data_sorted = sorted(data_notsorted, key=itemgetter(sieving_loss_type))
        assert data_sorted != data_notsorted

        logger.info("=======Top 5 sentences with lowest loss=========")
        for example in data_sorted[:5]:
            logger.info(example)
        logger.info("=========Top 5 sentences with highest loss============")
        for example in data_sorted[-5:]:
            logger.info(example)


        
        max_loss = 0
        max_idx = 0
        min_loss = 100000000
        min_idx = 0
        for idx, item in enumerate(data_sorted):
            loss = item[sieving_loss_type]
            if loss >= max_loss:
                max_loss = loss
                max_idx = idx
            if loss < min_loss:
                min_loss = loss
                min_idx = idx
        

        assert max_idx == len(data_sorted) - 1
        assert min_idx == 0
        logger.info("Max loss value: " + str(max_loss))
        logger.info("Min loss value: " + str(min_loss))
        interval = (max_loss - min_loss) / 1000
        loss_thresholds = [min_loss]
        loss = min_loss
        for i in range(1000):
            loss += interval
            loss_thresholds.append(loss)
        logger.info("Top 5 lowest loss thresholds: " + str(loss_thresholds[:5]))
        logger.info("Top 5 highest loss thresholds: " + str(loss_thresholds[-5:]))

        best_acc = 0
        best_global_idx_to_remove = set()
        sieved_sentences = 0
        best_false_positives = 0
        best_true_positives = 0
        for loss_threshold in loss_thresholds:
            true_positives = 0
            false_negatives = 0
            false_positives  = 0
            true_negatives = 0
            sieved_unknown_not_en = 0
            not_sieved_unknown_not_en = 0
            sieved_unknown_en = 0
            not_sieved_unknown_en = 0
            global_idx_to_remove = set()

            for idx, example in enumerate(data_sorted):
                sieved = False
                item_global_idx = example[0]
                lang = example[-3]
                broken = example[-4]
                loss_value = example[sieving_loss_type]

                if loss_value > loss_threshold:
                    sieved = True

                    if self.args.dont_sieve_english:
                        if lang != "en-gb" or broken == "broken": #remove only broken english sentences
                            global_idx_to_remove.add(item_global_idx)
                    else:
                        global_idx_to_remove.add(item_global_idx)


                if broken == "broken" and sieved == True:
                    true_positives += 1
                elif broken == "broken" and sieved == False:
                    false_negatives += 1
                elif broken == "not_broken" and lang == "en-gb" and sieved == True:
                    false_positives += 1
                elif broken == "not_broken" and lang == "en-gb" and sieved == False:
                    true_negatives += 1
                elif broken == "unknown" and lang != "en-gb" and sieved == True:
                    sieved_unknown_not_en += 1
                elif broken == "unknown" and lang != "en-gb" and sieved == False:
                    not_sieved_unknown_not_en += 1
                elif broken == "unknown" and lang == "en-gb" and sieved == True:
                    sieved_unknown_en += 1
                elif broken == "unknown" and lang == "en-gb" and sieved == False:
                    not_sieved_unknown_en += 1
                else:
                    raise Exception("Something went wrong with labelling of this training sentence.")

            if self.args.dont_sieve_english:
                assert len(global_idx_to_remove) == sieved_unknown_not_en + true_positives
            else:
                assert len(global_idx_to_remove) ==  sieved_unknown_not_en + true_positives + sieved_unknown_en + false_positives
            assert true_positives + false_negatives + false_positives + true_negatives + sieved_unknown_not_en + not_sieved_unknown_not_en + sieved_unknown_en + not_sieved_unknown_en  == len(data_sorted)


            try:
                precision = (true_positives) / (true_positives + false_positives)
                recall = (true_positives) / (true_positives + false_negatives)

                # https://en.wikipedia.org/wiki/F-score
                beta = self.args.fscore_beta
                fscore = (1 + beta ** 2) * precision * recall / (beta ** 2 * precision + recall)
            except:
                fscore = 0
                recall = 0
                precision = 0

            if fscore > best_acc:
                best_acc = fscore
                best_recall = recall
                best_precision = precision
                best_loss_threshold = loss_threshold
                best_global_idx_to_remove = global_idx_to_remove
                sieved_sentences = len(global_idx_to_remove)
                best_true_positives = true_positives
                best_false_positives = false_positives
                best_false_negatives = false_negatives
                best_true_negatives = true_negatives
                best_sieved_unknown_en = sieved_unknown_en
                best_not_sieved_unknown_en = not_sieved_unknown_en
                best_sieved_unknown_not_en = sieved_unknown_not_en
                best_not_sieved_unknown_not_en = not_sieved_unknown_not_en



        logger.info("=====STATISTIC FOR BEST SIEVING THRESHOLD IN THIS EPOCH===========")
        logger.info(f"f{self.args.fscore_beta}score: {best_acc}")
        logger.info("Sieving threshold: " + str(best_loss_threshold))
        logger.info("How much sentences are sieved: " + str(len(best_global_idx_to_remove)))
        logger.info("Precision: " + str(best_precision))
        logger.info("Recall: " + str(best_recall))
        logger.info("True positives: " + str(best_true_positives))
        logger.info("False negatives: " + str(best_false_negatives))
        logger.info("False positives: " + str(best_false_positives))
        logger.info("True negatives: " + str(best_true_negatives))
        logger.info("[en] Unknown and sieved: " + str(best_sieved_unknown_en))
        logger.info("[en] Unknown and not sieved: " + str(best_not_sieved_unknown_en))
        logger.info("[not en] Unknown and sieved: " + str(best_sieved_unknown_not_en))
        logger.info("[not en] Unknown and not sieved: " + str(best_not_sieved_unknown_not_en))
        logger.info("====Accuracy statistics=====")
        
        
        n_new_broken = 0
        if best_acc < self.args.sieving_coef:
            logger.info("=======================123================================")
            logger.info("Best sieving accuracy is lower than " + str(self.args.sieving_coef) + " not doing sieving: " + str(best_acc))
            best_global_idx_to_remove = set()
            sieved_sentences = 0
            logger.info("=======================123================================")
        else:
            logger.info("=======================456================================")
            logger.info("Best sieving accuracy is higher than " + str(self.args.sieving_coef) + " performing sieving: " + str(best_acc))
            logger.info("Also, stopping sieving and removing all broken training data")
            logger.info("=======================456================================")
            self.sieving_stop = True

            for example in data_sorted:
                item_global_idx = example[0]
                broken = example[-4]
                lang = example[-3]
                if broken == "broken":
                    if item_global_idx not in best_global_idx_to_remove:
                        n_new_broken += 1
                        best_global_idx_to_remove.add(item_global_idx)
                    if lang != "en-gb":
                        raise Exception("Broken sentence is not english, something went wrong because we break only english sentences.")
            if n_new_broken == 0:
                logger.info("All broken sentences have been sieved")
        
        
        if self.args.eval_sieving == True:
            if best_acc > self.best_sieving_fscore:
                self.best_sieving_fscore = best_acc
            
            logger.info("=======================987================================")
            logger.info(f"Eval sieving mode, not doing sieving. Best sieving f{self.args.fscore_beta}score for this epoch is: " + str(best_acc))
            logger.info(f"Eval sieving mode, not doing sieving. Best sieving f{self.args.fscore_beta}score so far: " + str(self.best_sieving_fscore))
            logger.info("=======================987================================")
            best_global_idx_to_remove = set()
            sieved_sentences = 0
            n_new_broken = 0
            self.sieving_stop = False
            
        
        
        input_ids = []
        attention_masks = []
        token_type_ids = []
        intent_label_ids = []
        domain_label_ids = []
        slot_label_ids = []
        language_labels_ids = []
        sentence_ids = []
        broken_ids = []
        hashes_ids = []
        for idx in range(len(data_sorted)):
            tensors_tuple = self.train_dataset[idx]
            
            if idx not in best_global_idx_to_remove:
                    
                input_ids.append(tensors_tuple[0].unsqueeze(0))
                utterance_2 = self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(tensors_tuple[0], skip_special_tokens=True))
                utterance = data_notsorted[idx][1]

                assert utterance == utterance_2
                attention_masks.append(tensors_tuple[1].unsqueeze(0))
                token_type_ids.append(tensors_tuple[2].unsqueeze(0))
                intent_label_ids.append(tensors_tuple[3].view(1))
                domain_label_ids.append(tensors_tuple[4].view(1))
                slot_label_ids.append(tensors_tuple[5].unsqueeze(0))
                language_labels_ids.append(tensors_tuple[6].view(1))
                sentence_ids.append(tensors_tuple[7].view(1))
                broken_ids.append(tensors_tuple[8].view(1))
                hashes_ids.append(tensors_tuple[9].view(1))

        assert len(input_ids) == len(self.train_dataset) - sieved_sentences - n_new_broken

        all_input_ids = torch.cat(input_ids).type(torch.LongTensor)
        all_attention_mask = torch.cat(attention_masks).type(torch.LongTensor)
        all_token_type_ids = torch.cat(token_type_ids).type(torch.LongTensor)
        all_intent_label_ids = torch.cat(intent_label_ids).type(torch.LongTensor)
        all_domain_label_ids = torch.cat(domain_label_ids).type(torch.LongTensor)
        all_slot_labels_ids = torch.cat(slot_label_ids).type(torch.LongTensor)
        all_language_labels_ids = torch.cat(language_labels_ids).type(torch.LongTensor)
        all_sentence_ids = torch.cat(sentence_ids).type(torch.LongTensor)
        all_broken_ids = torch.cat(broken_ids).type(torch.LongTensor)
        all_hashes_ids = torch.cat(hashes_ids).type(torch.LongTensor)

        self.train_dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_intent_label_ids, all_domain_label_ids, all_slot_labels_ids,all_language_labels_ids, all_sentence_ids, all_broken_ids, all_hashes_ids)
        

        output_tsv = os.path.join(self.args.model_dir, "sieving_output")
        if not os.path.exists(output_tsv):
            os.makedirs(output_tsv)
        
        output_file = os.path.join(output_tsv, "epoch_" + str(epoch) + "_testaccuracy_" + str(test_accuracy) + ".tsv")
        try:
            os.remove(output_file)
        except OSError:
            pass

        with open(output_file, 'a') as the_file:
            column_line = "\t".join(column_names)                 
            the_file.write(column_line + "\n")
            for idx, item in enumerate(data_notsorted):
                if idx in best_global_idx_to_remove:
                    remove = "removed"
                else:
                    remove = "not_removed"
                
                if "powiedz christine" in item[1]:
                    print(item)
                    print([str(elem) for elem in item])
                    print("\t".join([remove] + [str(elem) for elem in item]))
                    continue

                column_values = [remove] + [str(elem) for elem in item]
                assert len(column_values) == len(column_names)
                output_line = "\t".join(column_values) + "\n"
                the_file.write(output_line)
        

    
    def evaluate(self, mode, eval_dataloader=None, per_label=False, head_mask=None, sieving=False):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'testl1':
            dataset = self.test_dataset_l1
        elif mode == 'testl2':
            dataset = self.test_dataset_l2
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Unsupported dataset")

        if sieving == True:
            eval_dataloader = self.train_dataloader
        else:
            eval_sampler = SequentialSampler(dataset)
            eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        # Eval!
        logger.info("***** Running evaluation on %s dataset *****", mode)
        logger.info("  Num examples = %d", len(dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        domain_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_domain_label_ids = None
        out_slot_labels_ids = None

        res_dict = {
            'utterances': [],
            'slot_label': [],
            'slot_pred': [],
            'domain_label': [],
            'domain_pred': [],
            'intent_label': [],
            'intent_pred': [],
        }

        self.model.eval()
        print(f"GPU memory allocated {torch.cuda.memory_allocated(0) / 1e6} MB" )
        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'domain_label_ids': batch[4],
                          'slot_labels_ids': batch[5],
                          'head_mask': head_mask}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]

                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, domain_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            res_dict['utterances'].extend([self.tokenizer.convert_tokens_to_string(self.tokenizer.convert_ids_to_tokens(ids, skip_special_tokens=True)) for ids in inputs['input_ids']])

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            if not self.args.no_domains:
                if domain_preds is None:
                    domain_preds = domain_logits.detach().cpu().numpy()
                    out_domain_label_ids = inputs['domain_label_ids'].detach().cpu().numpy()
                else:
                    domain_preds = np.append(domain_preds, domain_logits.detach().cpu().numpy(), axis=0)
                    out_domain_label_ids = np.append(
                        out_domain_label_ids, inputs['domain_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits))
                else:
                    slot_preds = slot_logits.detach().cpu().numpy()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy()
            else:
                if self.args.use_crf:
                    slot_preds = np.append(slot_preds, np.array(self.model.crf.decode(slot_logits)), axis=0)
                else:
                    slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)

                out_slot_labels_ids = np.append(out_slot_labels_ids, inputs["slot_labels_ids"].detach().cpu().numpy(), axis=0)


        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        domain_preds = np.argmax(domain_preds, axis=1) if not self.args.no_domains else None

        # Slot result
        if not self.args.use_crf:
            slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_preds_ids_filtered = [[] for _ in range(out_slot_labels_ids.shape[0])]
        slot_labels_ids_filtered = [[] for _ in range(out_slot_labels_ids.shape[0])]

        for i in range(out_slot_labels_ids.shape[0]):
            for j in range(out_slot_labels_ids.shape[1]):
                if out_slot_labels_ids[i, j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])
                    slot_preds_ids_filtered[i].append(slot_preds[i][j])
                    slot_labels_ids_filtered[i].append(out_slot_labels_ids[i][j])
        
        total_result = compute_metrics(intent_preds, out_intent_label_ids,
                                       slot_preds_list, out_slot_label_list,
                                       domain_preds, out_domain_label_ids, no_domains=self.args.no_domains)
        results.update(total_result)

        if sieving==True:
            return (res_dict['utterances'], domain_preds, out_domain_label_ids, intent_preds, out_intent_label_ids, 
                                    slot_preds_list, out_slot_label_list)

        if per_label:
            compute_metrics_per_label(intent_preds, out_intent_label_ids, self.intent_label_lst,
                                      slot_preds_list, out_slot_label_list, self.slot_label_lst,
                                      domain_preds, out_domain_label_ids, self.domain_label_lst,
                                      mode, self.args.no_domains)

            compute_confusion_matrices(out_domain_label_ids, domain_preds, self.domain_label_lst,
                                       out_intent_label_ids, intent_preds, self.intent_label_lst,
                                       slot_labels_ids_filtered, slot_preds_ids_filtered, self.slot_label_lst,
                                       logger, self.args.no_domains)

        logger.info(f"***** Eval results for {mode} *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        # Save failing utterances to csv
        if per_label:
            res_dict['intent_pred'].extend([self.intent_label_lst[intent] for intent in intent_preds])
            res_dict['domain_pred'].extend([self.domain_label_lst[domain] for domain in domain_preds])
            res_dict['slot_pred'].extend([' '.join([self.slot_label_lst[slot] for slot in slots]) for slots in slot_preds_ids_filtered])

            res_dict['intent_label'].extend([self.intent_label_lst[intent] for intent in out_intent_label_ids])
            res_dict['domain_label'].extend([self.domain_label_lst[domain] for domain in out_domain_label_ids])
            res_dict['slot_label'].extend([' '.join([self.slot_label_lst[slot] for slot in slots]) for slots in slot_labels_ids_filtered])

            def are_not_the_same(pred, label):
                if pred == label:
                    return False
                return True
            df = pd.DataFrame(res_dict)
            df = df[df[['slot_pred', 'slot_label']].apply(lambda x: are_not_the_same(*x), axis=1)
                    | df[['domain_pred', 'domain_label']].apply(lambda x: are_not_the_same(*x), axis=1)
                    |    df[['intent_pred', 'intent_label']].apply(lambda x: are_not_the_same(*x), axis=1)
            
                            ]
            df.to_csv(f'fails/{mode}_res.tsv', sep='\t')

        return results

    def cache_outputs(self, mode):
        if mode == 'test':
            dataset = self.test_dataset
        elif mode == 'dev':
            dataset = self.dev_dataset
        elif mode == 'train':
            dataset = self.train_dataset
        else:
            raise Exception("Unsupported dataset")

        sampler = SequentialSampler(dataset)
        dataloader = DataLoader(dataset, sampler=sampler, batch_size=self.args.eval_batch_size)

        for batch in tqdm(dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                          'attention_mask': batch[1],
                          'intent_label_ids': batch[3],
                          'domain_label_ids': batch[4],
                          'slot_labels_ids': batch[5]}
                if self.args.model_type != 'distilbert':
                    inputs['token_type_ids'] = batch[2]

                outputs = self.model(**inputs)
                tmp_eval_loss, (teacher_intent_logits, teacher_domain_logits, teacher_slot_logits) = outputs[:2]
                teacher_last_hidden_states = outputs[2][-1] #batch_size x seq_length x hidden_dim_size
                teacher_first_hidden_states = outputs[2][0]
                teacher_last_attention = outputs[3][-1]


    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst,
                                                          domain_label_lst=self.domain_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")

    def make_plots(self):
        epochs = len(self.train_loss)
        train, val = plt.plot([i for i in range(1, epochs+1)], self.train_loss, self.val_loss)
        plt.legend([train, val], ['train', 'validaton'])
        plt.savefig(os.path.join(self.args.model_dir, 'loss.png'))
