from __future__ import annotations
import math
from collections.abc import Callable, Iterable
import torch


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_logits
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = shifted.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    loss = log_sum_exp - target_logits
    return torch.mean(loss)


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> None:
    params_with_grad = [p for p in parameters if p.grad is not None]
    if not params_with_grad:
        return
    total_norm_sq = sum(torch.sum(p.grad.data ** 2) for p in params_with_grad)
    total_norm = torch.sqrt(total_norm_sq).item()
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for p in params_with_grad:
            p.grad.data.mul_(scale)


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    if it > cosine_cycle_iters:
        return min_learning_rate
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameters: {betas}")
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable | None = None) -> torch.Tensor | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                if len(state) == 0:
                    state["t"] = 0
                    state["m"] = torch.zeros_like(p.data)
                    state["v"] = torch.zeros_like(p.data)

                state["t"] += 1
                t = state["t"]
                m = state["m"]
                v = state["v"]

                if weight_decay != 0.0:
                    p.data.mul_(1.0 - lr * weight_decay)

                m.mul_(beta1).add_(grad, alpha=1.0 - beta1)
                v.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

                alpha_t = lr * math.sqrt(1.0 - beta2 ** t) / (1.0 - beta1 ** t)
                p.data.addcdiv_(m, torch.sqrt(v) + eps, value=-alpha_t)

        return loss
