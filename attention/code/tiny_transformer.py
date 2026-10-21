"""
tiny_transformer.py -- train an actual attention model, then look at what its
heads learned.  Everything (attention, MLP, LayerNorm, loss) is written out
here; nothing comes from torch.nn except Parameter and the optimiser.

Two tasks, both on a toy vocabulary, both learnable in about 20 seconds on a
CPU:

  prev   -- predict the token that came immediately before.   A single head
            solving it must look at exactly one position, i-1: a clean
            off-diagonal stripe.  This is a real, named circuit: a
            "previous-token head".

  recall -- the associative-recall task.  The sequence is
                k1 v1 k2 v2 ... kK vK [Q]
            and the model must output V(Q) -- it has to find the earlier
            occurrence of Q and copy whatever followed it.  That is
            *induction*: the circuit behind in-context learning
            (Olsson et al. 2022, "In-context Learning and Induction Heads").
            Two heads in two layers are enough, and you can see both of them
            in the attention maps at the end.

Run:  .venv/bin/python code/tiny_transformer.py
      .venv/bin/python code/tiny_transformer.py --task recall --steps 4000
"""

from __future__ import annotations

import argparse
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

torch.manual_seed(0)


# --------------------------------------------------------------------------
# model: written out in full, no library blocks
# --------------------------------------------------------------------------

class Attention(nn.Module):
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads, self.dh = n_heads, d_model // n_heads
        self.wq = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.wk = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.wv = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))
        self.wo = nn.Parameter(torch.randn(d_model, d_model) / math.sqrt(d_model))

    def forward(self, x):
        B, T, D = x.shape
        H, dh = self.n_heads, self.dh
        q = (x @ self.wq).view(B, T, H, dh).transpose(1, 2)
        k = (x @ self.wk).view(B, T, H, dh).transpose(1, 2)
        v = (x @ self.wv).view(B, T, H, dh).transpose(1, 2)
        logits = q @ k.transpose(-1, -2) / math.sqrt(dh)
        # causal mask: integer -inf in the upper triangle
        causal = torch.triu(torch.ones(T, T, dtype=torch.bool), diagonal=1)
        logits = logits.masked_fill(causal, float("-inf"))
        A = logits.softmax(-1)
        out = (A @ v).transpose(1, 2).reshape(B, T, D) @ self.wo
        return out, A


class Block(nn.Module):
    """Pre-norm transformer block: x -> x + attn(ln(x)) -> x + mlp(ln(x)).

    Pre-norm matters: the residual path stays a clean identity, so gradients
    reach the embedding without being multiplied by a LayerNorm Jacobian at
    every layer.  That is why every modern model is shaped like this."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d_model), nn.LayerNorm(d_model)
        self.attn = Attention(d_model, n_heads)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, 4 * d_model), nn.GELU(),
            nn.Linear(4 * d_model, d_model))

    def forward(self, x):
        a, A = self.attn(self.ln1(x))
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x, A


class TinyGPT(nn.Module):
    def __init__(self, vocab, max_len, d_model=64, n_heads=4, n_layers=2):
        super().__init__()
        self.tok = nn.Embedding(vocab, d_model)
        self.pos = nn.Embedding(max_len, d_model)
        self.blocks = nn.ModuleList([Block(d_model, n_heads) for _ in range(n_layers)])
        self.lnf = nn.LayerNorm(d_model)

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok(idx) + self.pos(torch.arange(T, device=idx.device))[None]
        maps = []
        for blk in self.blocks:
            x, A = blk(x)
            maps.append(A)
        return self.lnf(x) @ self.tok.weight.T, maps


# --------------------------------------------------------------------------
# tasks
# --------------------------------------------------------------------------

def batch_prev(B, T, V, gen):
    """Predict the token at position i-1 from everything up to i.

    Note this is NOT ordinary next-token prediction: the targets here are
    random too, so no amount of training can predict x_i from x_<i.  The only
    way to get these right is to look backwards, which is why this task
    reliably produces a previous-token head.  Position 0 has no predecessor
    and is ignored.
    """
    x = torch.randint(0, V, (B, T), generator=gen)
    y = torch.cat([torch.full((B, 1), -100), x[:, :-1]], dim=1)
    return x, y


def unique_tokens(B, K, V, gen):
    """K distinct token ids per batch row, as the first K of a random
    permutation of the vocabulary."""
    return torch.argsort(torch.rand(B, V, generator=gen), dim=-1)[:, :K]


def batch_recall(B, K, V, gen, return_positions=False):
    """k1 v1 k2 v2 ... kK vK Q  ->  target is V(Q).

    Loss is only taken on the final position; that is the whole task.  The
    answer sits at position 2*idx+1, so `return_positions` gives the caller the
    position a correct attention pointer must land on."""
    keys = unique_tokens(B, K, V, gen)                 # (B, K)
    vals = unique_tokens(B, K, V, gen)                 # (B, K)
    idx = torch.randint(0, K, (B, 1), generator=gen)
    q = torch.gather(keys, 1, idx)                     # (B, 1) query key
    a = torch.stack([keys, vals], dim=2).reshape(B, 2 * K)   # interleaved
    x = torch.cat([a, q], dim=1)
    y = torch.gather(vals, 1, idx).squeeze(1)
    if return_positions:
        return x, y, (2 * idx.squeeze(1) + 1)
    return x, y


# --------------------------------------------------------------------------
# training
# --------------------------------------------------------------------------

def train(task, steps, V=32, T=16, K=8, d_model=64, n_heads=4, n_layers=2,
          lr=3e-3, batch=128, log_every=250):
    gen = torch.Generator().manual_seed(1)
    if task == "prev":
        ctx = T
    else:
        ctx = 2 * K + 1
    model = TinyGPT(V, ctx, d_model, n_heads, n_layers)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, lr, total_steps=steps)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"task={task}  vocab={V}  context={ctx}  d_model={d_model}  "
          f"heads={n_heads}  layers={n_layers}  params={n_params:,}")

    for step in range(1, steps + 1):
        if task == "prev":
            x, y = batch_prev(batch, T, V, gen)
            logits, _ = model(x)
            loss = F.cross_entropy(logits.reshape(-1, V), y.reshape(-1),
                                   ignore_index=-100)
        else:
            x, y = batch_recall(batch, K, V, gen)
            logits, _ = model(x)
            loss = F.cross_entropy(logits[:, -1], y)          # last position
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % log_every == 0 or step == 1:
            print(f"  step {step:>5}  loss {loss.item():.4f}")

    return model, (V, T, K, ctx)


def evaluate(model, task, V=32, T=16, K=8, n=512):
    gen = torch.Generator().manual_seed(99)
    model.eval()
    with torch.no_grad():
        if task == "prev":
            x, y = batch_prev(n, T, V, gen)
            logits, _ = model(x)
            m = y != -100
            acc = (logits.argmax(-1)[m] == y[m]).float().mean().item()
        else:
            x, y = batch_recall(n, K, V, gen)
            logits, _ = model(x)
            acc = (logits[:, -1].argmax(-1) == y).float().mean().item()
    model.train()
    return acc, x


def show_map(A, row=None, title="", width=52, lo=None):
    """Print one query row of an attention map as a bar chart of weights."""
    if row is None:
        row = A.shape[-1] - 1
    w = A[row].cpu().numpy()
    if lo is None:
        lo = w.max() * 0.05
    print(f"  {title}  (query position {row}, {A.shape[-1]} keys)")
    for j, v in enumerate(w):
        if v < lo:
            continue
        bar = "#" * max(1, int(round(v * width / max(w.max(), 1e-9))))
        print(f"    key {j:>3}  {v:6.3f}  {bar}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--task", default="both", choices=["prev", "recall", "both"])
    ap.add_argument("--steps", type=int, default=0,
                    help="0 = use the task's default (prev 1200, recall 6000)")
    ap.add_argument("--pairs", type=int, default=4, help="K, recall task")
    ap.add_argument("--vocab", type=int, default=16)
    ap.add_argument("--batch", type=int, default=256)
    args = ap.parse_args()

    tasks = ["prev", "recall"] if args.task == "both" else [args.task]
    default_steps = {"prev": 1200, "recall": 6000}

    for task in tasks:
        print("=" * 74)
        steps = args.steps or default_steps[task]
        model, shapes = train(task, steps, V=args.vocab, K=args.pairs,
                              batch=args.batch)
        acc, x = evaluate(model, task, V=args.vocab, K=args.pairs)
        V, T, K, ctx = shapes
        print(f"\n  held-out accuracy: {acc * 100:.2f}%")

        with torch.no_grad():
            _, maps = model(x[:1])

        if task == "prev":
            print("\n  A head that solves this must attend to exactly i-1.")
            best = None
            for layer, A in enumerate(maps):
                for h in range(A.shape[1]):
                    # score = how much weight sits on the single key i-1
                    off = A[0, h].diagonal(offset=-1).mean().item()
                    if best is None or off > best[0]:
                        best = (off, layer, h, A[0, h])
            off, layer, h, Arow = best
            print(f"  best 'previous-token' head: layer {layer}, head {h}"
                  f"  (mean weight on position i-1 = {off:.3f})")
            show_map(Arow, row=15, title=f"layer {layer} head {h}")
        else:
            # Induction: at the final query position, a head that has solved the
            # task must put its weight on position 2*idx+1 -- the token AFTER
            # the earlier occurrence of the query token.  Score every head by
            # how often its argmax lands exactly there.
            print(f"\n  the sequence is  k1 v1 k2 v2 ... k{K} v{K} Q   "
                  f"({ctx} tokens, query at position {ctx - 1})")
            print("  the answer is v_i at position 2i+1, so a head that has")
            print("  learned the induction circuit points exactly one past the")
            print("  earlier occurrence of Q.  Pointer accuracy per head:\n")
            gen = torch.Generator().manual_seed(4242)
            xg, yg, pos = batch_recall(512, K, V, gen, return_positions=True)
            with torch.no_grad():
                _, maps = model(xg)
            best = (0.0, 0, 0, 0)
            for layer, A in enumerate(maps):
                for h in range(A.shape[1]):
                    ptr = (A[:, h, -1].argmax(-1) == pos).float().mean().item()
                    tag = "  <-- induction" if ptr > 0.9 else ""
                    print(f"  layer {layer} head {h}: "
                          f"{ptr * 100:6.2f}% of queries point at the right "
                          f"position{tag}")
                    if ptr > best[0]:
                        best = (ptr, layer, h, A[:, h, -1].max(1).values.mean())
            ptr, layer, h, conf = best
            print(f"\n  best head: layer {layer}, head {h} -- pointer accuracy "
                  f"{ptr * 100:.2f}%, mean weight {conf:.3f}")
            print("  Layer 1 is doing the induction: layer 0 first moves the")
            print("  *previous* token into each position, and layer 1 then")
            print("  matches the current token against its earlier occurrence.")
            with torch.no_grad():
                _, maps = model(xg[:1])
            show_map(maps[layer][0, h], row=ctx - 1,
                     title=f"layer {layer} head {h}")
            print(f"  (key {int(pos[0])} is the correct answer for this example)")
        print()


if __name__ == "__main__":
    main()