import torch
import numpy as np
from transformers import AutoTokenizer

def collate_fn(batch, tokenizer, max_source_len, max_target_len):
    batched_data = {
        "input_ids": [],
        "attention_mask": [],
        "decoder_input_ids": [],
        "labels": [],
    }
    for _, (input_seq, output_seq) in enumerate(batch):
        # tokenize输入
        inputs = tokenizer(text=input_seq, truncation=True, max_length=max_source_len, padding='max_length')
       
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
        batched_data[k] = torch.tensor(np.array(v), dtype=torch.long)
    return batched_data

if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("./uer/t5-base-chinese-cluecorpussmall")

    test_batch = [
        ("This is a test input sequence 1.", "This is the corresponding output 1."),
        ("Another test input sequence 2.", "Another corresponding output 2."),
        ("Short input.", "Short output."),
        ("A very very long input sequence that should be truncated.", "A very very long output sequence that should be truncated.")
    ]
    
    max_source_len = 128
    max_target_len = 64

    batched_data = collate_fn(test_batch, tokenizer, max_source_len, max_target_len)
    
    for key, value in batched_data.items():
        print(f"{key}: {value.shape}")
