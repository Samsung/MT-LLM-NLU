import torch
import torch.nn as nn
import torch.nn.functional as F
from torchcrf import CRF
from transformers import BertPreTrainedModel, BertModel, BertConfig

from .module import IntentClassifier, SlotClassifier, DomainClassifier


class DistillScratchBERT(BertPreTrainedModel):
    def __init__(self, config, args, intent_label_lst, domain_label_lst, slot_label_lst, distillation_bool, distillation_alpha, pretrain=False):
        super(DistillScratchBERT, self).__init__(config)
        self.args = args
        self.num_intent_labels = len(intent_label_lst)
        self.num_slot_labels = len(slot_label_lst)
        self.num_domain_labels = len(domain_label_lst) if not self.args.no_domains else None
        config.output_hidden_states = True
        config.output_attentions=True
        config.hidden_size = args.hidden_size
        config.num_attention_heads = 2
        config.num_hidden_layers = 2
        self.hidden_dim = config.hidden_size
        self.bert = BertModel(config=config)

        self.distillation_switchon = distillation_bool
        self.distillation_alpha = distillation_alpha
        self.pretrain = pretrain

        self.intent_classifier = IntentClassifier(config.hidden_size, self.num_intent_labels, args.dropout_rate)
        self.domain_classifier =  DomainClassifier(config.hidden_size, self.num_domain_labels, args.dropout_rate) if not self.args.no_domains else None
        self.slot_classifier = SlotClassifier(config.hidden_size, self.num_slot_labels, args.dropout_rate)


        self.last_intermediate_projection = self.linear_projection(config.hidden_size, 768)
        self.first_intermediate_projection = self.linear_projection(config.hidden_size, 768)


        if args.use_crf:
            self.crf = CRF(num_tags=self.num_slot_labels, batch_first=True)

    def forward_temwis(self, input_ids):


        token_type_ids = torch.zeros(input_ids.size()[0], input_ids.size()[1], dtype=torch.long).to("cuda")
        input_ids = input_ids.to("cuda")
        attention_mask = torch.ones(input_ids.size()[0], input_ids.size()[1], dtype=torch.long).to("cuda")

        # print(input_ids)
        # print(attention_mask)
        # print(token_type_ids)
        # #sys.exit(0)

        outputs = self.bert(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
        sequence_output = outputs[0]
        pooled_output = outputs[1]

        active_hidden_outputs = torch.zeros(input_ids.size()[0], self.hidden_dim).to("cuda")
        for idx, (case, att_mask) in enumerate(zip(sequence_output, attention_mask)):
            #case => max_seq_len (50) x hidden_dim
            active_outputs = case[att_mask==1] #active states x hidden_dim
            active_outputs_summed = torch.mean(active_outputs, dim=0) # hidden_dim
            active_hidden_outputs[idx] = active_outputs_summed


        intent_logits = self.intent_classifier(active_hidden_outputs)
        domain_logits = self.domain_classifier(active_hidden_outputs) if not self.args.no_domains else None
        slot_logits = self.slot_classifier(sequence_output)

        return domain_logits, intent_logits, slot_logits







    def forward(self, input_ids, attention_mask, token_type_ids, intent_label_ids, domain_label_ids, slot_labels_ids,
        teacher_intent_logits, teacher_domain_logits, teacher_slot_logits, teacher_last_hidden_states, teacher_first_hidden_states, teacher_last_attention, head_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask,
                            token_type_ids=token_type_ids, head_mask=head_mask)  # sequence_output, pooled_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        #pooled_output = outputs[1]  # [CLS]


        active_hidden_outputs = torch.zeros(input_ids.size()[0], self.hidden_dim).to("cuda")
        for idx, (case, att_mask) in enumerate(zip(sequence_output, attention_mask)):
            #case => max_seq_len (50) x hidden_dim
            active_outputs = case[att_mask==1] #active states x hidden_dim
            active_outputs_summed = torch.mean(active_outputs, dim=0) # hidden_dim
            active_hidden_outputs[idx] = active_outputs_summed


        #self.distillation_switchon = False
        self.distillation_alpha = 0.5
        temperature = float(self.args.temperature)

        intent_logits = self.intent_classifier(active_hidden_outputs)
        domain_logits = self.domain_classifier(active_hidden_outputs) if not self.args.no_domains else None
        slot_logits = self.slot_classifier(sequence_output)


        student_loss = 0
        distillation_loss = 0
        intermediate_loss = 0

        # 1. Intent Softmax

        intent_loss_fct = nn.CrossEntropyLoss()
        intent_loss = intent_loss_fct(intent_logits.view(-1, self.num_intent_labels), intent_label_ids.view(-1))
        student_loss += intent_loss

        # print(self.distillation_switchon)
        if self.distillation_switchon:
            distillation_intent_loss_fct = nn.MSELoss()

            temp_intent_logits = intent_logits / temperature
            temp_teacher_intent_logits = teacher_intent_logits / temperature

            #distillation_intent_loss = distillation_intent_loss_fct(temp_intent_logits.view(-1, self.num_intent_labels), temp_teacher_intent_logits.view(-1, self.num_intent_labels))

            distillation_intent_loss = self.kd_ce_loss(intent_logits, teacher_intent_logits, temperature)
            distillation_loss += distillation_intent_loss


        if not self.args.no_domains:
            domain_loss_fct = nn.CrossEntropyLoss()
            domain_loss = domain_loss_fct(domain_logits.view(-1, self.num_domain_labels), domain_label_ids.view(-1))
            student_loss += domain_loss

            if self.distillation_switchon:
                distillation_domain_loss_fct = nn.MSELoss()

                temp_domain_logits = domain_logits / temperature
                temp_teacher_domain_logits = teacher_domain_logits / temperature


                #distillation_domain_loss = distillation_domain_loss_fct(temp_domain_logits.view(-1, self.num_domain_labels), temp_teacher_domain_logits.view(-1, self.num_domain_labels))


                distillation_domain_loss = self.kd_ce_loss(domain_logits, teacher_domain_logits, temperature)


                distillation_loss += distillation_domain_loss


        # 2. Slot Softmax
        if self.args.use_crf:
            slot_loss = self.crf(slot_logits, slot_labels_ids, mask=attention_mask.byte(), reduction='mean')
            slot_loss = -1 * slot_loss  # negative log-likelihood
        else:
            slot_loss_fct = nn.CrossEntropyLoss(ignore_index=self.args.ignore_index)
            # Only keep active parts of the loss

            active_loss = attention_mask.view(-1) == 1
            active_logits = slot_logits.view(-1, self.num_slot_labels)[active_loss]
            active_labels = slot_labels_ids.view(-1)[active_loss]
            slot_loss = slot_loss_fct(active_logits, active_labels)

            student_loss += self.args.slot_loss_coef * slot_loss

            if self.distillation_switchon:
                active_teacher_logits = teacher_slot_logits.view(-1, self.num_slot_labels)[active_loss]
                distillation_slot_loss_fct = nn.MSELoss()

                temp_active_logits = active_logits / temperature
                temp_active_teacher_logits = active_teacher_logits / temperature

                #distillation_slot_loss = distillation_slot_loss_fct(temp_active_logits, temp_active_teacher_logits)


                distillation_slot_loss = self.kd_ce_loss(active_logits, active_teacher_logits, temperature)


                distillation_loss += distillation_slot_loss



        if self.distillation_switchon:
            if self.args.intermediate_loss or self.pretrain:
                student_last_hidden_states = outputs[2][-1]
                # intermediate_attention_mask = torch.tensor(np.stack([np.concatenate((np.array([0]), mask[np.where(mask[1:-1]==1)], np.array([0] * (50 - len(mask[np.where(mask == 1)]))))) for mask in np.array(attention_mask.cpu())])).to('cuda')
                #last_projection = self.last_intermediate_projection(student_last_hidden_states)
                # last_hidd_loss = self.hid_mse_loss(teacher_last_hidden_states, student_last_hidden_states, intermediate_attention_mask)
                last_hidd_loss = self.hid_mse_loss(teacher_last_hidden_states, student_last_hidden_states, attention_mask)
                intermediate_loss += last_hidd_loss

                student_first_hidden_states = outputs[2][0]
                first_projection = self.first_intermediate_projection(student_first_hidden_states)
                first_hidd_loss = self.hid_mse_loss(teacher_first_hidden_states, first_projection, attention_mask)
                if self.pretrain:
                    intermediate_loss += first_hidd_loss

                student_last_attention = outputs[3][-1]
                last_attention_loss = self.att_mse_sum_loss(student_last_attention, teacher_last_attention, attention_mask)
                #if self.pretrain:
                #    intermediate_loss += last_attention_loss



        if self.distillation_switchon:
            if self.pretrain:
                total_loss =  intermediate_loss
            else:
                total_loss =  intermediate_loss +  student_loss + distillation_loss
                #total_loss = distillation_loss
        else:
            total_loss = student_loss


        #outputs = ((intent_logits, domain_logits, slot_logits),) + outputs[2:]  # add hidden states and attention if they are here
        outputs = (total_loss, intent_logits, domain_logits, slot_logits, distillation_loss, student_loss, intermediate_loss) + outputs[2:]

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

    def att_mse_sum_loss(self, attention_S, attention_T, mask=None):

        '''
        * Calculates the mse loss between `attention_S` and `attention_T`.
        * If the the shape is (*batch_size*, *num_heads*, *length*, *length*), sums along the `num_heads` dimension and then calcuates the mse loss between the two matrices.
        * If the `inputs_mask` is given, masks the positions where ``input_mask==0``.

        :param torch.Tensor logits_S: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
        :param torch.Tensor logits_T: tensor of shape  (*batch_size*, *num_heads*, *length*, *length*) or (*batch_size*, *length*, *length*)
        :param torch.Tensor mask:     tensor of shape  (*batch_size*, *length*)
        '''
        if len(attention_S.size())==4:
            attention_T = attention_T.sum(dim=1)
            attention_S = attention_S.sum(dim=1)
        if mask is None:
            attention_S_select = torch.where(attention_S <= -1e-3, torch.zeros_like(attention_S), attention_S)
            attention_T_select = torch.where(attention_T <= -1e-3, torch.zeros_like(attention_T), attention_T)
            loss = F.mse_loss(attention_S_select, attention_T_select)
        else:
            mask = mask.to(attention_S)
            valid_count = torch.pow(mask.sum(dim=1), 2).sum()
            loss = (F.mse_loss(attention_S, attention_T, reduction='none') * mask.unsqueeze(-1) * mask.unsqueeze(1)).sum() / valid_count
        return loss

    def kd_mse_loss(self,logits_S, logits_T, temperature=1):
        '''
        Calculate the mse loss between logits_S and logits_T

        :param logits_S: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param logits_T: Tensor of shape (batch_size, length, num_labels) or (batch_size, num_labels)
        :param temperature: A float or a tensor of shape (batch_size, length) or (batch_size,)
        '''
        if isinstance(temperature, torch.Tensor) and temperature.dim() > 0:
            temperature = temperature.unsqueeze(-1)
        beta_logits_T = logits_T / temperature
        beta_logits_S = logits_S / temperature
        loss = F.mse_loss(beta_logits_S, beta_logits_T)
        return loss


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
        #print(p_T)
        #print(F.softmax(beta_logits_S, dim=-1))
        loss = -(p_T * F.log_softmax(beta_logits_S, dim=-1)).sum(dim=-1).mean()
        return loss
