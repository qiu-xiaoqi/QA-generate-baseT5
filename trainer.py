import torch
from transformers import get_scheduler
from dataset import QADataset
import time
from torch.utils.data import DataLoader
from tools import evaluate_model
import matplotlib.pyplot as plt
from collate_fn import collate_fn

class Trainer:
    def __init__(self, model, tokenizer, train_path, test_path, max_source_len, max_target_len, batch_size, lr, num_train_epochs, save_path, logging_steps):
        self.model = model
        self.tokenizer = tokenizer
        self.train_path = train_path
        self.test_path = test_path
        self.max_source_len = max_source_len
        self.max_target_len = max_target_len
        self.batch_size = batch_size
        self.lr = lr
        self.num_train_epochs = num_train_epochs
        self.save_path = save_path
        self.logging_steps = logging_steps
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.lr_scheduler = get_scheduler(
            "linear",
            optimizer=self.optimizer,
            num_warmup_steps=0,
            num_training_steps=self.num_train_epochs * len(self.get_train_loader()),
        )
        self.loss_list = []
        self.global_step = 0
        self.best_bleu4 = 0
        self.tic_train = time.time()

    def get_train_loader(self):
        train_dataset = QADataset(data_path=self.train_path, tokenizer=self.tokenizer)
        return DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, self.tokenizer, self.max_source_len, self.max_target_len))

    def get_test_loader(self):
        test_dataset = QADataset(data_path=self.test_path, tokenizer=self.tokenizer)
        return DataLoader(test_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=lambda batch: collate_fn(batch, self.tokenizer, self.max_source_len, self.max_target_len))

    def train(self):
        for epoch in range(self.num_train_epochs):
            for batch in self.get_train_loader():
                outputs = self.model(
                    input_ids=batch["input_ids"].to(self.device),
                    attention_mask=batch["attention_mask"].to(self.device),
                    decoder_input_ids=batch["decoder_input_ids"].to(self.device),
                    labels=batch["labels"].to(self.device),
                )
                loss = outputs.loss
                loss.backward()
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()
                self.loss_list.append(float(loss.cpu().detach()))

                self.global_step += 1
                if self.global_step % self.logging_steps == 0:
                    time_diff = time.time() - self.tic_train
                    loss_avg = sum(self.loss_list) / len(self.loss_list)
                    print(
                        "global step %d, epoch: %d, loss: %.5f, speed: %.2f step/s"
                        % (self.global_step, epoch, loss_avg, self.logging_steps / time_diff)
                    )
                    self.tic_train = time.time()

            # 在每个epoch结束后评估模型
            bleu_scores = evaluate_model(self.get_test_loader(), self.model, self.tokenizer, self.device)
            print(f"Epoch {epoch}, BLEU-1: {bleu_scores[0]}, BLEU-2: {bleu_scores[1]}, BLEU-3: {bleu_scores[2]}, BLEU-4: {bleu_scores[3]}")
            if bleu_scores[3] > self.best_bleu4:
                self.best_bleu4 = bleu_scores[3]
                torch.save(self.model.state_dict(), f"{self.save_path}/best_model.pth")
        
            # 训练结束后绘制曲线
            self.plot_metrics()
    def plot_metrics(self):
        # 绘制损失曲线
        plt.figure()
        plt.plot(self.loss_list)
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss")
        plt.savefig(f"{self.save_path}/loss.png")

        # 绘制BLEU分数曲线
        plt.figure()
        plt.plot(self.bleu1_list, label="BLEU-1")
        plt.plot(self.bleu2_list, label="BLEU-2")
        plt.plot(self.bleu3_list, label="BLEU-3")
        plt.plot(self.bleu4_list, label="BLEU-4")
        plt.xlabel("Epoch")
        plt.ylabel("BLEU Score")
        plt.title("BLEU Scores")
        plt.legend()
        plt.savefig(f"{self.save_path}/bleu_scores.png")