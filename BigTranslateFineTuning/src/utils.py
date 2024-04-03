import torch.nn.functional as F
import torch

def padded_stack(
    tensors, side="left", mode="constant", value=0):
    
    full_size = max([x.size(-1) for x in tensors])

    def make_padding(pad):
        if side == "left":
            return (pad, 0)
        elif side == "right":
            return (0, pad)
        else:
            raise ValueError(f"side for padding '{side}' is unknown")

    out = torch.stack(
        [
            F.pad(x, make_padding(full_size - x.size(-1)), mode=mode, value=value) if full_size - x.size(-1) > 0 else x
            for x in tensors
        ],
        dim=0,
    )
    return out


def collate_fn(batch, tokenizer):
    input_ids = padded_stack([x["input_ids"] for x in batch], value=tokenizer.pad_token_id)
    attention_mask = padded_stack([x["attention_mask"] for x in batch])
        
    output_dict = {
        "input_ids": input_ids.squeeze(),
        "attention_mask": attention_mask.squeeze(),
    }
    
    if "labels" in batch[0]:
        label_ids = padded_stack([x["labels"] for x in batch], value=-100)
        output_dict["labels"] = label_ids.squeeze()
    
    return output_dict