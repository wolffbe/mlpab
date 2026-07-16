"""Fine-tuning job — runs directly in HopsFS mounted directory where data files already exist."""
import os
import json
import numpy as np

cwd = os.getcwd()
print(f"Working directory: {cwd}")
print(f"Files available: {os.listdir('.')}")

# ---- Fine-tuning logic (from finetune_model.py, unmodified logic) ----
RANK = 4
LEARNING_RATE = 2.0
ITERATIONS = 400
ADAPTER_SCALE = 0.5


def bigram_counts(text, vocab):
    index = {ch: i for i, ch in enumerate(vocab)}
    idx = np.array([index[ch] for ch in text], dtype=int)
    counts = np.zeros((len(vocab), len(vocab)), dtype=float)
    np.add.at(counts, (idx[:-1], idx[1:]), 1.0)
    return counts


def cross_entropy(logits, counts):
    logp = logits - logits.max(axis=1, keepdims=True)
    logp = logp - np.log(np.exp(logp).sum(axis=1, keepdims=True))
    return -(counts * logp).sum() / counts.sum()


ckpt = np.load("base_model.npz")
base_logits = ckpt["logits"].astype(float)
vocab = [str(ch) for ch in ckpt["vocab"]]
ft_counts = bigram_counts(open("finetune.txt").read(), vocab)
eval_counts = bigram_counts(open("eval.txt").read(), vocab)

n_vocab = base_logits.shape[0]
A = np.zeros((n_vocab, RANK), dtype=float)
for i in range(n_vocab):
    A[i, i % RANK] = ADAPTER_SCALE
B = np.zeros((RANK, n_vocab), dtype=float)
n = ft_counts.sum()
row_totals = ft_counts.sum(axis=1, keepdims=True)
for iteration in range(ITERATIONS):
    logits = base_logits + A @ B
    e = np.exp(logits - logits.max(axis=1, keepdims=True))
    p = e / e.sum(axis=1, keepdims=True)
    g = (row_totals * p - ft_counts) / n
    A, B = A - LEARNING_RATE * (g @ B.T), B - LEARNING_RATE * (A.T @ g)
    if (iteration + 1) % 100 == 0:
        print(f"Iteration {iteration + 1}/{ITERATIONS}")

np.savez("finetuned_model.npz", logits=base_logits + A @ B,
         adapter_a=A, adapter_b=B)
metrics = {
    "eval_loss": round(float(cross_entropy(base_logits + A @ B, eval_counts)), 4),
    "base_eval_loss": round(float(cross_entropy(base_logits, eval_counts)), 4),
}
json.dump(metrics, open("metrics.json", "w"), indent=2)

print(f"Fine-tuning complete!")
print(f"METRICS: {json.dumps(metrics)}")
print(f"Output files: {[f for f in os.listdir('.') if f.endswith(('.npz', '.json'))]}")
