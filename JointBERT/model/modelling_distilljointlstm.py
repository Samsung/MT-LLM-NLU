import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel, BertConfig

from .module import IntentClassifier, SlotClassifier, DomainClassifier


class DistillJointLSTM(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, domain_label_lst, slot_label_lst, distillation_bool, distillation_alpha, pretrain=False):
        super(DistillJointLSTM, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst) if not self.args.no_domains else None

        self.distillation_switchon = distillation_bool
        self.distillation_alpha = distillation_alpha
        self.pretrain = pretrain

        self.hidden_dim = 256
        self.n_layers = 1
        self.embedding_dim = 32
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


        self.intermediate_projection = self.linear_projection(2 * self.hidden_dim, 768)
        self.embedding_projection = self.linear_projection(self.embedding_dim, 768)
        

        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)


    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, domain_label_ids, slot_labels_ids, 
            teacher_intent_logits, teacher_domain_logits, teacher_slot_logits, teacher_last_hidden_states, teacher_first_hidden_states, teacher_last_attention, head_mask=None):

        
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
        domain_logits = self.domain_classifier(active_hidden_outputs)  if not self.args.no_domains else None


        #self.distillation_switchon = True
        #self.distillation_alpha = 0.7
        temperature = float(self.args.temperature)

        student_loss = 0
        distillation_loss = 0
        intermediate_loss = 0
        # 1. Intent Softmax

        intent_loss_fct = nn.CrossEntropyLoss()
        #output_logits = self.select_logits_with_mask(intent_logits, attention_mask)
        #print(output_logits.cpu().detach().numpy().shape)
        #print(intent_logits.cpu().detach().numpy().shape)

        intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
        student_loss += intent_loss

        if self.distillation_switchon:
            distillation_intent_loss_fct = nn.MSELoss()
            #distillation_intent_loss = distillation_intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), teacher_intent_logits.view(-1, self.num_intent_labels))
            
            distillation_intent_loss = self.kd_ce_loss(intent_logits, teacher_intent_logits, temperature)
            distillation_loss += distillation_intent_loss

        if not self.args.no_domains:
            domain_loss_fct = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fct(domain_logits.view(-1, self.num_domain_labels), domain_label_ids.view(-1))
            student_loss += domain_loss


            if self.distillation_switchon:
                distillation_domain_loss_fct = nn.MSELoss()
                #distillation_domain_loss = distillation_domain_loss_fct(domain_logits.view(-1, self.num_domain_labels), teacher_domain_logits.view(-1, self.num_domain_labels))
                
                distillation_domain_loss = self.kd_ce_loss(domain_logits, teacher_domain_logits, temperature)
                distillation_loss += distillation_domain_loss


        if self.args.use_crf:
            slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss = -1 * slot_loss  # negative log-likelihood
        else:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            
        
            active_loss = attention_mask.view(-1) == 1 #czy to jest w ogole potrzebne, jak jest paddinx_idx ?
            active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
            active_labels = slot_labels_ids.view(-1)[active_loss]
            slot_loss = slot_loss_fct(active_logits, active_labels)


            if self.distillation_switchon:
                active_teacher_logits = teacher_slot_logits.view(-1, self.num_slot_labels)[active_loss]
                distillation_slot_loss_fct = nn.MSELoss()   
                #distillation_slot_loss = distillation_slot_loss_fct(active_logits, active_teacher_logits)

                distillation_slot_loss = self.kd_ce_loss(active_logits, active_teacher_logits, temperature)
                
                distillation_loss += distillation_slot_loss


        student_loss += self.args.slot_loss_coef * slot_loss


        if self.distillation_switchon and self.args.intermediate_loss:
            projection = self.intermediate_projection(output)
            hidd_loss = self.hid_mse_loss(projection, teacher_last_hidden_states, attention_mask)
            if self.pretrain:
                intermediate_loss += hidd_loss


            embedding_projection = self.embedding_projection(embeddings)
            emb_loss = self.hid_mse_loss(embedding_projection, teacher_first_hidden_states, attention_mask)
            if self.pretrain:
                intermediate_loss += emb_loss



        if self.distillation_switchon:
            if self.pretrain:
                total_loss =  intermediate_loss
            else:
                total_loss =  intermediate_loss +  student_loss + distillation_loss
        else:
            total_loss = student_loss
        


        outputs = (total_loss, intent_logits, domain_logits, slot_logits, distillation_loss, student_loss, intermediate_loss)

        return outputs  # (loss), logits, (hidden_states), (attentions) # Logits is a tuple of intent and slot logits

    def hid_mse_loss(self, state_S, state_T, mask=None):
        '''
        * Calculates the mse loss between `state_S` and `state_T`, which are the hidden state of the models.
        * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.
        * If the hidden sizes of student and teacher are different, 'proj' option is required in `inetermediate_matches` to match the dimensions.

        :param torch.Tensor state_S: tensor of shape  (*batch_size*, *length*, *hidden_size*)
        :param torch.Tensor state_T: tensor of shape  (*batch_size*, *length*, *hidden_size*)
        :param torch.Tensor mask:    tensor of shape  (*batch_size*, *length*)
        '''
        if mask is None:
            loss = F.mse_loss(state_S, state_T)
        else:
            mask = mask.to(state_S)
            valid_count = mask.sum() * state_S.size(-1)
            loss = (F.mse_loss(state_S, state_T, reduction='none') * mask.unsqueeze(-1)).sum() / valid_count
        return loss

    def linear_projection(self, dim_in, dim_out):
        model = torch.nn.Linear(in_features=dim_in, out_features=dim_out, bias=True)
        model.weight.data.normal_(mean=0.0, std=2)
        return model

    def kd_ce_loss(self,logits_S, logits_T, temperature=1):
        '''
        Calculate the cross entropy between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        p_T = F.softmax(beta_logits_T, dim=-1)
        loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
        return loss