import torch
import torch.nn as nn
from torchcrf import CRF
from transformers import BertPreTrainedModel

from .module import IntentClassifier, SlotClassifier, DomainClassifier


class JointLSTM(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, domain_label_lst, slot_label_lst):
        super(JointLSTM, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst) if not self.args.no_domains else None


        self.hidden_dim = 256
        self.n_layers = 1
        self.embedding_dim = 64
        self.dictionary_size = 30522
        self.embedding = nn.Embedding(self.dictionary_size, self.embedding_dim, padding_idx=0)

        self.rnn = nn.LSTM(self.embedding.embedding_dim,
                           self.hidden_dim,
                           num_layers=self.n_layers,
                           bidirectional=True,
                           dropout=0.1,
                           batch_first=True)

        self.intent_classifier = IntentClassifier(2 * self.hidden_dim, self.num_intent_labels, args.dropout_rate)
        self.domain_classifier = DomainClassifier(2 * self.hidden_dim, self.num_domain_labels, args.dropout_rate) if not self.args.no_domains else None
        self.slot_classifier = SlotClassifier(2 * self.hidden_dim, self.num_slot_labels, args.dropout_rate) #2 * bo bidirectional

        

        if args.use_crf:
        #if True:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, domain_label_ids, slot_labels_ids):



        embeddings = self.embedding(input_ids)

        output, (h_n, c_n) = self.rnn(embeddings)



        slot_logits = self.slot_classifier(output)

        active_hidden_outputs = torch.zeros(input_ids.size()[0], 2 * self.hidden_dim).to("cuda")
        for idx, (case, att_mask) in enumerate(zip(output, attention_mask)):
            #case => max_seq_len (50) x 2*hidden_dim
            active_outputs = case[att_mask==1] #active states x 2*hidden_dim
            active_outputs_summed = torch.sum(active_outputs, dim=0) # 2*hidden_dim
            active_hidden_outputs[idx] = active_outputs_summed
        
        
        #output = batch size x n_tokens x 2 * hidden_dim
        #output = torch.sum(output, dim=1) # summing with inactive states
        #summed_lstm_output = batch_size x 2 *hidden_dim

        intent_logits = self.intent_classifier(active_hidden_outputs)
        domain_logits = self.domain_classifier(active_hidden_outputs) if not self.args.no_domains else None


        #sys.exit(0)

        total_loss = 0
        # 1. Intent Softmax

        intent_loss_fct = nn.CrossEntropyLoss()
        intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
        total_loss += intent_loss

        if not self.args.no_domains:
            domain_loss_fct = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fct(domain_logits.view(-1, self.num_domain_labels), domain_label_ids.view(-1))
            total_loss += domain_loss


        if self.args.use_crf:
            slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss = -1 * slot_loss  # negative log-likelihood
        else:
            active_loss = attention_mask.view(-1) == 1 
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
            active_labels = slot_labels_ids.view(-1)[active_loss]
            slot_loss = slot_loss_fct(active_logits, active_labels)

        
        total_loss += self.args.slot_loss_coef * slot_loss
        
        outputs = ((intent_logits, domain_logits, slot_logits),)

        outputs = (total_loss,) + outputs

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits
