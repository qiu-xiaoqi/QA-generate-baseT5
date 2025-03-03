from torch.utils.data import Dataset
import json

class QADataset(Dataset):
    def __init__(self, data_path, tokenizer):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = f"问题：{item['question']}{self.tokenizer.sep_token}原文：{item['context']}"
        output_seq = f"答案：{item['answer']}{self.tokenizer.eos_token}"
        return input_seq, output_seq


if __name__ == "__main__":
    from transformers import AutoTokenizer, T5ForConditionalGeneration, get_scheduler
    pretrain_path = "./uer/roberta-base-finetuned-dianping-chinese"
    tokenizer = AutoTokenizer.from_pretrained(pretrain_path)
    data_path = './DuReaderQG/train.json'
    dataset = QADataset(data_path, tokenizer)
    print(dataset[0])

