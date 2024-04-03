# import os
# import argparse
# import tensorflow as tf
# import pandas as pd
# import numpy as np
# import sys

# from transformers import BertTokenizer, TFBertModel, TFBertPreTrainedModel, BertConfig
# from tensorflow.keras import Input, layers, Model
# from sklearn.model_selection import train_test_split



# class KerasBERT(TFBertPreTrainedModel):
#     def __init__(self, config, args, intent_label_lst, domain_label_lst, slot_label_lst):
#         super().__init__(config)
#         self.args = args
#         self.num_intent_labels = len(intent_label_lst)
#         self.num_slot_labels = len(slot_label_lst)
#         self.num_domain_labels = len(domain_label_lst)
        
        
#         self.bert = TFBertModel(config=config)
#         self.dropout = tf.keras.layers.Dropout(0.1, name='dropout') 
#         self.domain_classifier = tf.keras.layers.Dense(self.num_domain_labels, name="domain_classifier")
#         self.intent_classifier = tf.keras.layers.Dense(self.num_intent_labels, name="intent_classifier")
#         self.slot_classifier = tf.keras.layers.Dense(self.num_slot_labels, name="slot_classifier")

#     def model(self, max_seq_length):
#         input_word_ids = tf.keras.layers.Input(shape=(max_seq_length,  ), dtype=tf.int32,name="input_word_ids")
#         input_mask = tf.keras.layers.Input(shape=(max_seq_length, ), dtype=tf.int32,name="input_mask")
#         segment_ids = tf.keras.layers.Input(shape=(max_seq_length, ), dtype=tf.int32,name="segment_ids")
#         input_dict = {'input_ids': input_word_ids, 'attention_mask': input_mask, 'token_type_ids': segment_ids}
#         return Model(inputs=input_dict, outputs=self.__call__(input_dict))

#     def __call__(self, inputs, **kwargs):
#         outputs = self.bert(inputs, **kwargs)
#         pooled_output = self.dropout(outputs[1])
#         sequence_output = self.dropout(outputs[0])

#         intent_logits = self.intent_classifier(pooled_output)
#         domain_logits = self.domain_classifier(pooled_output)

#         if 'attention_mask' not in inputs: 
#             #for handling the case when Transformers make sanity test
#             #  File "/usr/local/lib/python3.6/dist-packages/transformers/modeling_tf_utils.py", line 1286, in from_pretrained
#             #model(model.dummy_inputs)  # build the network with dummy inputs 
#             slot_logits = self.slot_classifier(sequence_output)
#             return domain_logits, intent_logits, slot_logits
#         else:
#             attention_mask_1 = tf.cast(inputs['attention_mask'],  sequence_output.dtype)
#             attention_mask_expand = tf.tile(tf.expand_dims(attention_mask_1, axis=-1), multiples=[1,1, tf.shape(sequence_output)[-1]])
#             attention_masked_data = tf.multiply(sequence_output, attention_mask_expand)
            
#             masking_layer = layers.Masking(mask_value=tf.constant(0, dtype=sequence_output.dtype))
#             masked = masking_layer(attention_masked_data)
#             #print(masked._keras_mask)  keras_mask to ostateczna booolean maska, która jest propagowana do kolejnych warstw i funkcji loss
#             #https://www.tensorflow.org/guide/keras/masking_and_padding
#             slot_logits = self.slot_classifier(masked)

#             return domain_logits, intent_logits, slot_logits