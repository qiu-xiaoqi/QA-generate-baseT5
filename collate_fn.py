import torch
import numpy as np

def collate_fn(batch, tokenizer, max_source_len, max_target_len):
    batched_data = {
        "input_ids": [],
        "attention_mask": [],
        "decoder_input_ids": [],
        "labels": [],
    }
    for _, (input_seq, output_seq) in enumerate(batch):
        # tokenize输入
        inputs = tokenizer(text=input_seq, truncation=True, max_length=max_source_len, padding=True)
       
        # tokenize输出,并将输出的ids作为inputs的label
        output_ids = tokenizer.encode(text=output_seq, truncation=True, max_length=max_target_len)
        decoder_input_ids = output_ids[:-2] # 去掉eos和[cls]
        decoder_input_ids = decoder_input_ids + [tokenizer.pad_token_id] * (max_target_len - len(decoder_input_ids)) # padding
        
        labels = output_ids[1: -1] # 去掉起始token和[cls]
        labels = labels + [-100] * (max_target_len - len(labels))
        
        batched_data["input_ids"].append(inputs["input_ids"])
        batched_data["attention_mask"].append(inputs["attention_mask"])
        batched_data["decoder_input_ids"].append(decoder_input_ids)
        batched_data["labels"].append(labels)
        
    for k, v in batched_data.items():
        batched_data[k] = torch.tensor(np.array(v))
    return batched_data



