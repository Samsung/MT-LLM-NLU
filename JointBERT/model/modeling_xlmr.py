import sys

import numpy as np
import torch
import torch.nn as nn
from torchcrf import CRF
from transformers.models.xlm_roberta.modeling_xlm_roberta import XLMRobertaPreTrainedModel, \
    XLMRobertaModel

from .module import IntentClassifier, SlotClassifier, DomainClassifier


class JointXLMR(XLMRobertaPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, domain_label_lst, slot_label_lst):
        super(JointXLMR, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst) if not self.args.no_domains else None
        config.output_hidden_states = True
        config.output_attentions=True

        try:
            if 'scratch' in args.model_type:
                config.hidden_size = args.hidden_size
                config.num_attention_heads = 2
                config.num_hidden_layers = 2
        except:
            pass

        self.roberta = XLMRobertaModel(config=config)  # Load pretrained bert
        self.hidden_dim = config.hidden_size

        #freezing bert:
        # for param in self.roberta.parameters():
        #     param.requires_grad = False


        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.domain_classifier = DomainClassifier(config.hidden_size, self.num_domain_labels, args.dropout_rate) if not self.args.no_domains else None
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)
        
        self.projection_layer = torch.nn.Linear(in_features=self.hidden_dim, out_features=self.hidden_dim, bias=True)


        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, domain_label_ids, slot_labels_ids, 
               sentence_id=None, language_id=None, sieving=False, contrastive=False, head_mask=None):
        
        device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        outputs = self.roberta(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, head_mask = head_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]


        
        total_loss = 0
        contrastive_loss = 0
        sieving_contrastive_loss = [float(0.0)] * len(intent_label_ids)
        if contrastive == True:
            #print("Performing contrastive learning")
            sieving_contrastive_loss = []
            sentence_id = sentence_id.cpu().numpy()
            contrastive_criterion = torch.nn.CosineSimilarity(dim=0)
            
            indices_set = set()
            loss_count = 0
            for idx in sentence_id:

                indices = np.where(sentence_id == idx)[0]

                if len(indices) == 3: #TODO: remove broken sentences and calculate loss for not broken pair of sentences
                    sieving_contrastive_loss.append(float(0.0))
                    continue
                elif len(indices) == 2:
                    idx_1 = indices[0]
                    idx_2 = indices[1]
                    assert domain_label_ids[idx_1] == domain_label_ids[idx_2]
                    assert intent_label_ids[idx_1] == intent_label_ids[idx_2]
                    if language_id[idx_1] != language_id[idx_2]: #TODO: remove broken sentences and calculate loss for not broken pair of sentences
                        sieving_contrastive_loss.append(float(0.0))
                        continue

                    if False:
                        repr_1 = self.projection_layer(pooled_output[idx_1])
                        repr_2 = self.projection_layer(pooled_output[idx_2])
                    else:
                        repr_1 = pooled_output[idx_1]
                        repr_2 = pooled_output[idx_2]
                    
                    pair_loss = contrastive_criterion(repr_1, repr_2)

                    sieving_contrastive_loss.append(pair_loss.detach().cpu().numpy().item())

                    if self.args.turnon_contrastiveloss:
                        contrastive_loss = contrastive_loss + pair_loss

                    loss_count += 1
                elif len(indices) == 1: #in case of sieving, some translated sentences will be removed
                    sieving_contrastive_loss.append(float(0.0))
                    continue
                else:
                    print("=====ERROR543==========")
                    sys.exit(0)

                    
#             for idx in sentence_id:
#                 indices = np.where(sentence_id == idx)[0]
#                 if len(indices) == 2:
#                     idx_1 = indices[0]
#                     idx_2 = indices[1]
#                     assert sieving_contrastive_loss[idx_1] == sieving_contrastive_loss[idx_2]
            
            #print("===========================")
            assert len(sieving_contrastive_loss) == len(sentence_id)
            #print(contrastive_loss)
            #print("Contrastive loss: ", contrastive_loss.detach().cpu().numpy().item())
            contrastive_loss = -torch.div(contrastive_loss, loss_count)
            #print("Average Contrastive loss: ", contrastive_loss.detach().cpu().numpy().item())
            #print(self.args.contrastive_loss_coef * contrastive_loss.detach().cpu().numpy().item())
            #total_loss += self.args.contrastive_loss_coef * contrastive_loss
            #print(total_loss)
            #print("===========================")
        
        
        
        
        
        intent_logits = self.intent_classifier(pooled_output)
        domain_logits = self.domain_classifier(pooled_output) if not self.args.no_domains else None
        slot_logits = self.slot_classifier(sequence_output)

        sieving_intent_loss = None
        sieving_domain_loss = None
        sieving_slot_loss = None
        # 1. Intent Softmax
        if intent_label_ids is not None:
            if self.num_intent_labels == 1:
                intent_loss_fct = nn.MSELoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1), intent_label_ids.view(-1))
            else:
                if sieving == True:
                    sieving_intent_loss_fct = nn.CrossEntropyLoss(reduction='none')
                    sieving_intent_loss = sieving_intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))

                intent_loss_fct = nn.CrossEntropyLoss()
                intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
            total_loss += intent_loss

        if not self.args.no_domains:
            if domain_label_ids is not None:
                if sieving == True:
                    sieving_domain_loss_fct = nn.CrossEntropyLoss(reduction='none')
                    sieving_domain_loss = sieving_domain_loss_fct(domain_logits.view(-1, self.num_domain_labels), domain_label_ids.view(-1))

                domain_loss_fct = nn.CrossEntropyLoss()
                domain_loss = domain_loss_fct(domain_logits.view(-1, self.num_domain_labels), domain_label_ids.view(-1))
                total_loss += domain_loss

        # 2. Slot Softmax
        if slot_labels_ids is not None:
            if self.args.use_crf:
                slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
                slot_loss = -1 * slot_loss  # negative log-likelihood
            else:
                if sieving == True:
                    sieving_slot_loss_fct = nn.CrossEntropyLoss(reduction='none', ignore_index=self.args.ignore_index)

                slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
                # Only keep active parts of the loss
                if attention_mask is not None:
                    active_loss = attention_mask.view(-1) == 1
                    active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
                    active_labels = slot_labels_ids.view(-1)[active_loss]
                    slot_loss = slot_loss_fct(active_logits, active_labels)
                    
                    if sieving == True:
                        sieving_slot_loss_flattened = sieving_slot_loss_fct(active_logits, active_labels)
                        
                        current_len = 0
                        sieving_slot_loss = []
                        for att_mask_for_example in attention_mask:
                            active_att = att_mask_for_example == 1
                            active_len = int(torch.sum(active_att).int())
                            
                            slot_losses_for_example = sieving_slot_loss_flattened[current_len:current_len+active_len]
                            assert slot_losses_for_example.shape[0] == active_len
                            current_len+=active_len

                            if self.args.sieving_slot_mean:
                                average_slot_loss_for_example = torch.mean(slot_losses_for_example).detach()
                            else:
                                average_slot_loss_for_example = torch.max(slot_losses_for_example).detach()

                            
                            sieving_slot_loss.append(average_slot_loss_for_example)

                        assert sieving_slot_loss_flattened.shape[0] == current_len
                        assert len(sieving_slot_loss) == input_ids.shape[0]
                else:
                    slot_loss = slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))
                    if sieving == True: #not implemented
                        print("===========NOT IMPLEMENTED============")
                        sys.exit(0)
                        #sieving_slot_loss = sieving_slot_loss_fct(slot_logits.view(-1, self.num_slot_labels), slot_labels_ids.view(-1))


            total_loss += self.args.slot_loss_coef * slot_loss

        outputs = ((intent_logits, domain_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here

        #print(total_loss.item())
        #sys.exit(0)
        
        if contrastive == True:
            total_loss += self.args.contrastive_loss_coef * contrastive_loss
        
        outputs = (total_loss,) + outputs
        

        if sieving==True:
            assert sieving_domain_loss.shape[0] == sieving_intent_loss.shape[0] == len(sieving_slot_loss)
            return (outputs, ) + ((sieving_domain_loss, sieving_intent_loss, sieving_slot_loss, sieving_contrastive_loss),)
        else:
            return outputs # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits



    def forward_temwis(self, input_ids):

        device = "cuda" if torch.cuda.is_available() and not self.args.no_cuda else "cpu"
        #print(input_ids)
        token_type_ids = torch.zeros(input_ids.size()[0], input_ids.size()[1], dtype=torch.long).to(device)
        input_ids = input_ids.to(device)
        attention_mask = torch.ones(input_ids.size()[0], input_ids.size()[1], dtype=torch.long).to(device)
        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # #sys.exit(0)


        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooled_output = outputs[1]  # [CLS]

        # active_hidden_outputs = torch.zeros(input_ids.size()[0], self.hidden_dim).to(device)
        # for idx, (case, att_mask) in enumerate(zip(sequence_output, attention_mask)):
        #     #case => max_seq_len (50) x hidden_dim
        #     active_outputs = case[att_mask==1] #active states x hidden_dim
        #     active_outputs_summed = torch.sum(active_outputs, dim=0) # hidden_dim
        #     active_hidden_outputs[idx] = active_outputs_summed

        intent_logits = self.intent_classifier(pooled_output)
        domain_logits = self.domain_classifier(pooled_output) if not self.args.no_domains else None
        slot_logits = self.slot_classifier(sequence_output)

        return domain_logits, intent_logits, slot_logits