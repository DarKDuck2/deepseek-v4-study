"""
Demo 03: Muon Optimizer

本 Demo 展示 Muon 优化器的核心机制：
1. Newton-Schulz 迭代近似正交化
2. 参数分组：2D 矩阵用 Muon，其他用 AdamW
3. 与 AdamW 在简单合成任务上的收敛对比
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def newton_schulz(G, steps=10):
    """
    Newton-Schulz 迭代：将梯度矩阵 G 近似正交化

    使用 V4 中的两阶段策略：
    - 前 8 步使用快速收敛系数
    - 后 2 步使用稳定化系数

    简化版实现，基于原始 Muon 论文思路。
    """
    # 归一化
    G_norm = G.norm()
    if G_norm < 1e-8:
        return G
    X = G / G_norm

    # 两阶段 Newton-Schulz
    for i in range(steps):
        if i < 8:
            # 快速收敛阶段 (激进系数)
            a, b = 3.0, -1.5
        else:
            # 稳定化阶段 (保守系数)
            a, b = 1.5, -0.5

        X = a * X + b * X @ X.T @ X

    # 恢复尺度
    return X * G_norm


class MuonOptimizer(torch.optim.Optimizer):
    """
    简化版 Muon 优化器

    核心思想：
    - 对 2D 矩阵参数（如 Linear.weight），用 Newton-Schulz 正交化梯度后更新
    - 对其他参数（bias, norm, embedding），回退到 AdamW
    """
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0.01, ns_steps=10):
        defaults = dict(lr=lr, betas=betas, eps=eps,
                       weight_decay=weight_decay, ns_steps=ns_steps)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            wd = group['weight_decay']
            ns_steps = group['ns_steps']

            for p in group['params']:
                if p.grad is None:
                    continue

                grad = p.grad.data
                state = self.state[p]

                # 初始化状态
                if len(state) == 0:
                    state['step'] = 0
                    if len(p.shape) == 2:
                        # Muon 参数：动量
                        state['momentum'] = torch.zeros_like(p)
                    else:
                        # AdamW 参数：一阶、二阶动量
                        state['exp_avg'] = torch.zeros_like(p)
                        state['exp_avg_sq'] = torch.zeros_like(p)

                state['step'] += 1

                # 参数分组策略
                if len(p.shape) == 2:
                    # ===== Muon 路径：2D 矩阵参数 =====
                    # 更新动量
                    m = state['momentum']
                    m.mul_(beta1).add_(grad, alpha=1 - beta1)

                    # Newton-Schulz 正交化
                    update = newton_schulz(m, steps=ns_steps)

                    # 带权重衰减的更新
                    p.data.mul_(1 - lr * wd).add_(update, alpha=-lr)

                else:
                    # ===== AdamW 路径：其他参数 =====
                    exp_avg = state['exp_avg']
                    exp_avg_sq = state['exp_avg_sq']

                    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                    bias_correction1 = 1 - beta1 ** state['step']
                    bias_correction2 = 1 - beta2 ** state['step']

                    step_size = lr / bias_correction1
                    denom = (exp_avg_sq.sqrt() / (bias_correction2 ** 0.5)).add_(eps)

                    # AdamW: 权重衰减直接作用在参数上
                    p.data.mul_(1 - lr * wd)
                    p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def create_synthetic_task(dim=128, rank=16):
    """创建一个合成任务：低秩矩阵恢复"""
    # 目标：学习一个低秩线性映射
    U = torch.randn(dim, rank)
    V = torch.randn(rank, dim)
    W_target = U @ V  # 低秩目标矩阵

    def loss_fn(model, batch_size=64):
        x = torch.randn(batch_size, dim)
        y_target = x @ W_target.T
        y_pred = model(x)
        return nn.functional.mse_loss(y_pred, y_target)

    return loss_fn


def train_model(model, optimizer, loss_fn, steps=500, log_every=50):
    """训练并返回 loss 历史"""
    losses = []
    for step in range(steps):
        optimizer.zero_grad()
        loss = loss_fn(model)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if step % log_every == 0:
            print(f"  Step {step:4d}: loss = {loss.item():.6f}")

    return losses


def main():
    print("=" * 60)
    print("Demo 03: Muon 优化器")
    print("=" * 60)

    dim = 128
    torch.manual_seed(42)

    # 创建合成任务
    loss_fn = create_synthetic_task(dim=dim, rank=16)

    # 定义模型结构（包含 2D Linear 和 1D Bias/Norm）
    class TinyModel(nn.Module):
        def __init__(self, dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(dim, dim * 2),
                nn.LayerNorm(dim * 2),
                nn.GELU(),
                nn.Linear(dim * 2, dim),
                nn.LayerNorm(dim),
            )

        def forward(self, x):
            return self.net(x)

    print(f"\n合成任务: 低秩矩阵恢复 (dim={dim})")
    print(f"模型包含 Linear(2D) + LayerNorm(1D) + GELU(无参数)")

    # 实验 1: AdamW 训练
    print("\n--- Experiment 1: AdamW ---")
    model_adam = TinyModel(dim)
    opt_adam = torch.optim.AdamW(model_adam.parameters(), lr=3e-4, weight_decay=0.01)
    losses_adam = train_model(model_adam, opt_adam, loss_fn, steps=300, log_every=60)

    # 实验 2: Muon 训练（2D 矩阵用 Muon，1D 用 AdamW）
    print("\n--- Experiment 2: Muon (2D params) + AdamW (others) ---")
    model_muon = TinyModel(dim)

    # 手动分组参数
    muon_params = []
    adamw_params = []
    for name, p in model_muon.named_parameters():
        if len(p.shape) == 2:
            muon_params.append(p)
            print(f"  Muon param: {name}, shape={p.shape}")
        else:
            adamw_params.append(p)
            print(f"  AdamW param: {name}, shape={p.shape}")

    opt_muon = MuonOptimizer([
        {'params': muon_params, 'lr': 3e-4},
        {'params': adamw_params, 'lr': 3e-4}
    ], weight_decay=0.01, ns_steps=10)

    losses_muon = train_model(model_muon, opt_muon, loss_fn, steps=300, log_every=60)

    # 实验 3: 纯 AdamW 但更高学习率
    print("\n--- Experiment 3: AdamW (lr=1e-3, higher) ---")
    model_adam_high = TinyModel(dim)
    opt_adam_high = torch.optim.AdamW(model_adam_high.parameters(), lr=1e-3, weight_decay=0.01)
    losses_adam_high = train_model(model_adam_high, opt_adam_high, loss_fn, steps=300, log_every=60)

    # 结果对比
    print("\n--- 最终 Loss 对比 ---")
    print(f"  AdamW (lr=3e-4):    {losses_adam[-1]:.6f}")
    print(f"  Muon + AdamW:       {losses_muon[-1]:.6f}")
    print(f"  AdamW (lr=1e-3):    {losses_adam_high[-1]:.6f}")

    # 可视化
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(losses_adam, label='AdamW (lr=3e-4)', linewidth=2)
    plt.plot(losses_muon, label='Muon + AdamW', linewidth=2)
    plt.plot(losses_adam_high, label='AdamW (lr=1e-3)', linewidth=2)
    plt.xlabel('Step')
    plt.ylabel('MSE Loss')
    plt.title('Muon vs AdamW Convergence')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('figures/demo_muon_convergence.png', dpi=150)
    print("\n图表已保存到 figures/demo_muon_convergence.png")

    # 展示 Newton-Schulz 效果
    print("\n--- Newton-Schulz 正交化效果 ---")
    G = torch.randn(32, 32)
    G_orth = newton_schulz(G, steps=10)

    # 检查正交化程度: G_orth @ G_orth.T 应接近单位矩阵
    I_approx = G_orth @ G_orth.T
    identity = torch.eye(32)
    orth_error = torch.norm(I_approx - identity).item()

    print(f"  原始梯度谱范数: {torch.linalg.matrix_norm(G, 2).item():.4f}")
    print(f"  正交化后谱范数: {torch.linalg.matrix_norm(G_orth, 2).item():.4f}")
    print(f"  G@G^T 与 I 的误差: {orth_error:.4f} (越小越正交)")

    print("\n" + "=" * 60)
    print("核心思想：")
    print("1. Newton-Schulz 迭代将梯度矩阵近似正交化")
    print("2. 对 2D 权重矩阵使用结构感知的矩阵级更新")
    print("3. 对 Embedding/Norm/Bias 等回退到 AdamW")
    print("4. 参数更新规则本身也可以是模型效率的来源")
    print("=" * 60)


if __name__ == "__main__":
    main()
