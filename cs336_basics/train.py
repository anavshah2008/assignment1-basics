from __future__ import annotations
import argparse
import math
import os
import time
import numpy as np
import torch
from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, clip_gradients, cross_entropy, get_lr_cosine_schedule


def load_memmap_dataset(path: str, dtype: np.dtype = np.uint16) -> np.ndarray:
    if path.endswith(".npy"):
        return np.load(path, mmap_mode="r")
    return np.memmap(path, dtype=dtype, mode="r")


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    val_data: np.ndarray,
    batch_size: int,
    context_length: int,
    eval_iters: int,
    device: str,
) -> float:
    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(val_data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    return float(sum(losses) / len(losses))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10000.0)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--max_iters", type=int, default=20000)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--eval_interval", type=int, default=250)
    parser.add_argument("--eval_iters", type=int, default=25)
    parser.add_argument("--save_interval", type=int, default=1000)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--resume_from", type=str, default=None)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    args = parser.parse_args()

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    train_data = load_memmap_dataset(args.train_data)
    val_data = load_memmap_dataset(args.val_data)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=args.device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume_from is not None and os.path.exists(args.resume_from):
        start_iter = load_checkpoint(args.resume_from, model, optimizer)

    start_time = time.time()
    model.train()

    for it in range(start_iter, args.max_iters):
        lr = get_lr_cosine_schedule(
            it=it,
            max_learning_rate=args.max_lr,
            min_learning_rate=args.min_lr,
            warmup_iters=args.warmup_iters,
            cosine_cycle_iters=args.max_iters,
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits, y)
        loss.backward()
        if args.grad_clip > 0.0:
            clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            val_loss = evaluate(
                model,
                val_data,
                args.batch_size,
                args.context_length,
                args.eval_iters,
                args.device,
            )
            elapsed = time.time() - start_time
            val_ppl = math.exp(min(val_loss, 20.0))
            print(
                f"step={it} time={elapsed:.2f}s lr={lr:.6e} "
                f"train_loss={loss.item():.4f} val_loss={val_loss:.4f} val_ppl={val_ppl:.2f}"
            )

        if (it + 1) % args.save_interval == 0 or it == args.max_iters - 1:
            ckpt_path = os.path.join(args.checkpoint_dir, f"ckpt_step_{it + 1}.pt")
            save_checkpoint(model, optimizer, it + 1, ckpt_path)


if __name__ == "__main__":
    main()
