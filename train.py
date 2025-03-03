import os
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, T5ForConditionalGeneration, get_scheduler
from torch.utils.data import Dataset, DataLoader
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import json
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
pretrained_model = "./uer/t5-base-chinese-cluecorpussmall"
max_source_seq_len = 256
max_target_seq_len = 32
batch_size = 16
num_train_epochs = 20
valid_steps = 200
logging_steps = 10
learning_rate = 5e-5
save_dir = "./checkpoints"

model = T5ForConditionalGeneration.from_pretrained(pretrained_model)
tokenizer = AutoTokenizer.from_pretrained(pretrained_model)


# 自定义数据集类
class QADataset(Dataset):
    def __init__(self, data_path):
        self.data = []
        with open(data_path, "r", encoding='utf-8') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]
        input_seq = (
            f"问题：{item['question']}{tokenizer.sep_token}原文：{item['context']}"
        )
        output_seq = f"答案：{item['answer']}{tokenizer.eos_token}"
        return input_seq, output_seq


def collate_fn(batch):
    batched_data = {
        "input_ids": [],
        "attention_mask": [],
        "decoder_input_ids": [],  # decoder 输入，包含了起始的提示 token，不包含 eos_token
        "labels": [],  # decoder 标签，用于计算损失，不包含起始的提示 token，包含了 eos_token
    }
    for _, (input_seq, output_seq) in enumerate(batch):
        output_ids = tokenizer.encode(
            text=output_seq, truncation=True, max_length=max_target_seq_len
        )

        decoder_input_ids = output_ids[
            :-2
        ]  # 去掉 eos_token 和 tokenizer 自己加上的 [CLS]

        decoder_input_ids = decoder_input_ids + [tokenizer.pad_token_id] * (
            max_target_seq_len - len(decoder_input_ids)
        )  # 补 padding
        lables = output_ids[1:-1]  # 去掉起始的提示 token 和 tokenizer 自己加上的 [CLS]

        # 用 -100 用于在计算 loss 时忽略，因为该 T5 模型的实现中 loss 的 ignore_token 设置为了 -100
        # `loss_fct = CrossEntropyLoss(ignore_index=-100)`
        lables = lables + [-100] * (max_target_seq_len - len(lables))

        # input 不需要特殊处理，直接 tokenize 即可
        inputs = tokenizer(
            text=input_seq,
            truncation=True,
            max_length=max_source_seq_len,
            padding="max_length",
        )
        batched_data["input_ids"].append(inputs["input_ids"])
        batched_data["attention_mask"].append(inputs["attention_mask"])
        batched_data["decoder_input_ids"].append(decoder_input_ids)
        batched_data["labels"].append(lables)

    for k, v in batched_data.items():
        batched_data[k] = torch.tensor(np.array(v), dtype=torch.long)
    return batched_data


# 绘制收敛曲线
def plot_metrics(value, name):
    plt.figure()
    plt.plot(value)
    plt.xlabel("Batch")
    plt.ylabel(f"{name}")
    plt.title(f"{name}")
    plt.savefig(f"./log/{name}.png")


def evaluate_model(data_loader):
    model.eval()
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    smoothie = SmoothingFunction().method4
    print("Evaluation")
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device)
            )  # (batch, seq_len)

            label_tokens = batch["labels"].cpu().numpy()
            decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            label_tokens = np.where(
                batch["labels"] != -100, label_tokens, tokenizer.pad_token_id
            )
            decoded_labels = tokenizer.batch_decode(
                label_tokens, skip_special_tokens=True
            )
            for pred, ref in zip(decoded_preds, decoded_labels):
                bleu_1_scores.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        weights=(1, 0, 0, 0),
                        smoothing_function=smoothie,
                    )
                )
                bleu_2_scores.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        weights=(0.5, 0.5, 0, 0),
                        smoothing_function=smoothie,
                    )
                )
                bleu_3_scores.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        weights=(0.33, 0.33, 0.33, 0),
                        smoothing_function=smoothie,
                    )
                )
                bleu_4_scores.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        weights=(0.25, 0.25, 0.25, 0.25),
                        smoothing_function=smoothie,
                    )
                )
    model.train()
    return [
        sum(bleu_1_scores) / len(bleu_1_scores),
        sum(bleu_2_scores) / len(bleu_2_scores),
        sum(bleu_3_scores) / len(bleu_3_scores),
        sum(bleu_4_scores) / len(bleu_4_scores),
    ]


def train():
    # 设置 eos_token 为 sep_token, 这个 tokenizer 默认是没有 eos_token 的
    tokenizer.eos_token = tokenizer.sep_token
    # 设置 bos_token 为 cls_token，这个 tokenizer 默认是没有 bos_token 的， 而 cls_token 是有的且在当前任务中没有用
    tokenizer.bos_token = tokenizer.cls_token

    train_dataset = QADataset("./DuReaderQG/train.json")
    eval_dataset = QADataset("./DuReaderQG/dev.json")
    print("train dataset size: ", len(train_dataset))
    print("eval dataset size: ", len(eval_dataset))

    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    lr_scheduler = get_scheduler(
        "linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_train_epochs * len(train_dataloader),
    )
    model.to(device)

    loss_list = []
    bleu1_list = []
    bleu2_list = []
    bleu3_list = []
    bleu4_list = []
    tic_train = time.time()
    global_step, best_bleu4 = 0, 0
    print("Start Training")
    for epoch in range(num_train_epochs):
        for batch in train_dataloader:
            outputs = model(
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
                decoder_input_ids=batch["decoder_input_ids"].to(device),
                labels=batch["labels"].to(device),
            )
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            loss_list.append(float(loss.cpu().detach()))

            global_step += 1
            if global_step % logging_steps == 0:
                time_diff = time.time() - tic_train
                loss_avg = sum(loss_list) / len(loss_list)
                print(
                    "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                    % (global_step, epoch, loss_avg, logging_steps / time_diff)
                )
                tic_train = time.time()

            if global_step % valid_steps == 0:
                cur_save_dir = os.path.join(save_dir, "model_%d" % global_step)
                if not os.path.exists(cur_save_dir):
                    os.makedirs(cur_save_dir)
                model.save_pretrained(os.path.join(cur_save_dir))
                tokenizer.save_pretrained(os.path.join(cur_save_dir))

                bleu1, bleu2, bleu3, bleu4 = evaluate_model(eval_dataloader)
                print(
                    "Evaluation bleu1: %.5f, bleu2: %.5f, bleu3: %.5f, bleu4: %.5f"
                    % (bleu1, bleu2, bleu3, bleu4)
                )
                bleu1_list.append(bleu1)
                bleu2_list.append(bleu2)
                bleu3_list.append(bleu3)
                bleu4_list.append(bleu4)

                plot_metrics(loss_list, "loss")
                plot_metrics(bleu1_list, "bleu1")
                plot_metrics(bleu2_list, "bleu2")
                plot_metrics(bleu3_list, "bleu3")
                plot_metrics(bleu4_list, "bleu4")

                print("Evaluation bleu4: %.5f" % (bleu4))
                if bleu4 > best_bleu4:
                    print(
                        f"best BLEU-4 performence has been updated: {best_bleu4:.5f} --> {bleu4:.5f}"
                    )
                    best_bleu4 = bleu4
                    cur_save_dir = os.path.join(save_dir, "model_best")
                    if not os.path.exists(cur_save_dir):
                        os.makedirs(cur_save_dir)
                    model.save_pretrained(os.path.join(cur_save_dir))
                    tokenizer.save_pretrained(os.path.join(cur_save_dir))
                tic_train = time.time()


if __name__ == "__main__":
    train()
