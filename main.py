import os
import time
import torch
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_scheduler
from torch.utils.data import DataLoader
from dataset import QADataset
from collate_fn import collate_fn
from tools import evaluate_model
from trainer import Trainer

def main():
    # 设置设备
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # 模型和分词器路径
    pretrained_model = "./uer/t5-base-chinese-cluecorpussmall"
    max_source_seq_len = 256
    max_target_seq_len = 32
    batch_size = 16
    num_train_epochs = 20
    logging_steps = 10
    learning_rate = 5e-5
    save_dir = "./checkpoints"

    # 加载模型和分词器
    model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model)

    # 设置 eos_token 和 bos_token
    tokenizer.eos_token = tokenizer.sep_token
    tokenizer.bos_token = tokenizer.cls_token

    # 初始化 Trainer
    trainer = Trainer(
        model=model,
        tokenizer=tokenizer,
        train_path="./DuReaderQG/train.json",
        test_path="./DuReaderQG/dev.json",
        max_source_len=max_source_seq_len,
        max_target_len=max_target_seq_len,
        batch_size=batch_size,
        lr=learning_rate,
        num_train_epochs=num_train_epochs,
        save_path=save_dir,
        logging_steps=logging_steps
    )

    # 开始训练
    trainer.train()

if __name__ == "__main__":
    main()