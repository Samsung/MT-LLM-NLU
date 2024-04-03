import torch
import sys
import numpy as np
from trainer import Trainer






def evaluate_teacher_heads_importance(trainer):

    
    per_head_results = {}

    per_head_results['without_masking'] = trainer.evaluate("test", head_mask = None)['sementic_frame_acc']

    n_layers = trainer.config.num_hidden_layers
    n_heads = trainer.config.num_attention_heads
    
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            a = np.ones((n_layers, n_heads))
            a[layer_idx, head_idx] = 0
            head_name = "turnedoff_head_layer" + str(layer_idx) + "_head" + str(head_idx) + "_" + "teacher"
            head_mask = torch.from_numpy(a).to(trainer.device)
            results = trainer.evaluate("test", head_mask = head_mask)
            per_head_results[head_name] = results['sementic_frame_acc']

    without_mask_results = per_head_results['without_masking']

    heads_importance = {}
    for key, value in per_head_results.items():
        if key != 'without_masking':
            diff = without_mask_results - value
            #diff_normalized = (diff - min_value) / (max_value - min_value) #normalization leads to 0 score for least important head, which might introduce confusion
            heads_importance[key] = diff

    print(per_head_results)
    print(heads_importance)
    max_print = 10
    for key, value in sorted(heads_importance.items(), key=lambda item: item[1], reverse=True):
            max_print += 1
            print(key, value)
            if max_print == 10:
                break
    return None