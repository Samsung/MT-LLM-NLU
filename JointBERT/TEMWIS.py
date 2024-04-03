# from transformers import AutoModelForSequenceClassification, AutoTokenizer
# model_name = "distilbert-base-uncased-finetuned-sst-2-english"
# model = AutoModelForSequenceClassification.from_pretrained(model_name)
# tokenizer = AutoTokenizer.from_pretrained(model_name)

# # With both the model and tokenizer initialized we are now able to get explanations on an example text.

# from transformers_interpret.transformers_interpret import SequenceClassificationExplainer
# cls_explainer = SequenceClassificationExplainer(
#     model,
#     tokenizer)
# word_attributions = cls_explainer("I love you, I like you")
# print(word_attributions)
from scipy.stats import entropy
from math import log

def measure_entropy(frequency: list, normalized: bool = True):
        """
        Calculates entropy and normalized entropy of list of elements that have specific frequency
        :param frequency: The frequency of the elements.
        :param normalized: Calculate normalized entropy
        :return: entropy or (entropy, normalized entropy)
        """
        entropy, normalized_ent, n = 0, 0, 0
        pk = []
        for word in frequency:
            pk.append(abs(word[1]))
        
        sum_freq = sum(pk)
        for i, x in enumerate(pk):
            p_x = float(pk[i] / sum_freq)
            if p_x > 0:
                n += 1
                entropy += - p_x * log(p_x, 2)
        if normalized:
            if log(n) > 0:
                normalized_ent = entropy / log(n, 2)
            return entropy, normalized_ent
        else:
            return entropy


def evaluate_TEMWIS(teacher_model, student_model, tokenizer, dataset):
    from transformers_interpret.transformers_interpret import DomainClassificationExplainer
    teacher_cls_explainer = DomainClassificationExplainer(teacher_model,tokenizer)
    student_cls_explainer = DomainClassificationExplainer(student_model,tokenizer)
    
    total_student_entropy = 0
    total_teacher_entropy = 0
    
    student_n = 0
    teacher_n = 0
    n = 0
    for example in dataset:
        n += 1
        input_ids = example[0]
        attention_mask = example[1]
        #token_type_ids = example[2]
        intent_label = example[3].item()
        input_ids = input_ids[attention_mask==1]
        reference_tokens = tokenizer.convert_ids_to_tokens(input_ids)
        input_ids = input_ids.unsqueeze(0)
        attention_mask = attention_mask[attention_mask==1].unsqueeze(0)

        #teacher_word_attributions = teacher_cls_explainer(input_ids, attention_mask, reference_tokens)
        student_word_attributions = student_cls_explainer(input_ids, attention_mask, reference_tokens)
        
        #print("========================")
        # if teacher_cls_explainer.predicted_class_name != intent_label:
        # # # #if measure_entropy(teacher_word_attributions)[1] < 0.4:
        #     print("========================")
        # #print("Teacher error")
        #     teacher_n += 1
        #     print(teacher_word_attributions)
        #     print(measure_entropy(teacher_word_attributions)[1])
        #     #print(teacher_cls_explainer.predicted_class_name)
        #     total_teacher_entropy += measure_entropy(teacher_word_attributions)[1]
        #     print(total_teacher_entropy)
        #     print(teacher_n)
        #     print(total_teacher_entropy/teacher_n)
        #     print(teacher_cls_explainer.predicted_class_name)
        #     print("=========================")


        # if student_cls_explainer.predicted_class_name != intent_label:
        # # if measure_entropy(student_word_attributions)[1] > 0.9:
        print("=====================")
        #print("Student error")
        student_n += 1
        print(student_word_attributions)
        print(measure_entropy(student_word_attributions)[1])
        total_student_entropy += measure_entropy(student_word_attributions)[1]
        print(total_student_entropy)
        print(student_n)
        print("Average entropy: ", total_student_entropy/student_n)
        print(student_cls_explainer.predicted_class_name)
        print("=======================")
        #break

        # if n == 50:
        #     break