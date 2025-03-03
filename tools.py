import matplotlib.pyplot as plt
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import torch

def plot_metrics(value, name):
    plt.figure()
    plt.plot(value)
    plt.xlabel("Batch")
    plt.ylabel(f"{name}")
    plt.title(f"{name}")
    plt.savefig(f"log/{name}.png")

def evaluate_model(data_loader, model, tokenizer, device):
    model.eval()
    bleu1 = []
    bleu2 = []
    bleu3 = []
    bleu4 = []
    smoothie = SmoothingFunction().method4
    print("Evaluation")
    with torch.no_grad():
        for _, batch in enumerate(data_loader):
            outputs = model.generate(
                input_ids=batch["input_ids"].to(device)
            )
            label_tokens = batch["labels"].cpu().numpy()
            decode_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            label_tokens = np.where(
                batch["labels"] != -100, label_tokens, tokenizer.pad_token_id
            )
            decode_labels = tokenizer.batch_decode(
                label_tokens, skip_special_tokens=True
            )
            for pred, ref in zip(decode_preds, decode_labels):
                bleu1.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        smoothing_function=smoothie,
                        weights=(1, 0, 0, 0),
                    )
                )
                bleu2.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        smoothing_function=smoothie,
                        weights=(0.5, 0.5, 0, 0),
                    )
                )
                bleu3.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        smoothing_function=smoothie,
                        weights=(0.33, 0.33, 0.33, 0),
                    )
                )
                bleu4.append(
                    sentence_bleu(
                        [ref.split()],
                        pred.split(),
                        smoothing_function=smoothie,
                        weights=(0.25, 0.25, 0.25, 0.25),
                    )
                )
    model.train()  
    return [
        sum(bleu1) / len(bleu1),
        sum(bleu2) / len(bleu2),
        sum(bleu3) / len(bleu3),  
        sum(bleu4) / len(bleu4),
    ]