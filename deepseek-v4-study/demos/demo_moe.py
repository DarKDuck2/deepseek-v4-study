"""
Demo 06: MoE 架构改进

本 Demo 展示：
1. 标准 Top-K MoE 路由
2. Hash-routed MoE（基于 token ID 哈希直接映射专家）
3. Sqrt(Softplus()) 激活函数
4. 序列级负载均衡损失
5. 专家负载分布可视化
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def sqrt_softplus(x):
    """Sqrt(Softplus()) 激活函数
    V4 中替代 Sigmoid 用于计算专家亲和度
    """
    return torch.sqrt(F.softplus(x))


class StandardTopKMoE(nn.Module):
    """
    标准 Top-K MoE

    每个 token 通过门控网络计算对所有专家的亲和度，
    选择 top-k 个专家处理。
    """
    def __init__(self, d_model=64, num_experts=8, top_k=2, d_ff=128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 门控网络
        self.gate = nn.Linear(d_model, num_experts)

        # 专家们
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        """
        x: (batch, seq_len, d_model)
        """
        B, T, D = x.shape

        # 计算门控 logits
        gate_logits = self.gate(x)  # (B, T, num_experts)

        # Top-K 选择
        topk_vals, topk_idx = torch.topk(gate_logits, self.top_k, dim=-1)

        # 归一化权重（softmax）
        topk_weights = F.softmax(topk_vals, dim=-1)

        # 每个 expert 分别处理
        output = torch.zeros_like(x)
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            expert_ids = topk_idx[:, :, k]  # (B, T)
            weights = topk_weights[:, :, k]  # (B, T)

            for i in range(self.num_experts):
                mask = (expert_ids == i)  # (B, T)
                if mask.any():
                    # 取对应 token 过对应专家
                    expert_input = x * mask.unsqueeze(-1)
                    expert_output = self.experts[i](expert_input)
                    output += expert_output * weights.unsqueeze(-1) * mask.unsqueeze(-1)
                    expert_counts[i] += mask.sum().item()

        return output, gate_logits, topk_idx, expert_counts


class HashRoutedMoE(nn.Module):
    """
    Hash-routed MoE

    简化版：基于 token ID 哈希直接映射到专家，
    无需门控网络，适合浅层使用。
    """
    def __init__(self, d_model=64, num_experts=8, top_k=2, d_ff=128):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        # 专家们
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d_model, d_ff),
                nn.GELU(),
                nn.Linear(d_ff, d_model)
            ) for _ in range(num_experts)
        ])

    def forward(self, x, token_ids=None):
        """
        x: (batch, seq_len, d_model)
        token_ids: 可选，用于哈希（简化版直接用位置索引）
        """
        B, T, D = x.shape

        # 简化哈希：基于位置和 batch 索引
        if token_ids is None:
            token_ids = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)

        # 为每个位置生成 top_k 个哈希专家 ID
        expert_ids_list = []
        for offset in range(self.top_k):
            # 简单哈希: (batch_idx * seq_len + token_idx + offset) % num_experts
            flat_ids = (torch.arange(B, device=x.device).unsqueeze(1) * T +
                       token_ids + offset) % self.num_experts
            expert_ids_list.append(flat_ids)

        expert_ids = torch.stack(expert_ids_list, dim=-1)  # (B, T, top_k)

        # 均匀权重（哈希路由天然均匀）
        weights = torch.ones_like(expert_ids, dtype=torch.float32) / self.top_k

        # 执行
        output = torch.zeros_like(x)
        expert_counts = torch.zeros(self.num_experts, device=x.device)

        for k in range(self.top_k):
            cur_expert_ids = expert_ids[:, :, k]
            for i in range(self.num_experts):
                mask = (cur_expert_ids == i)
                if mask.any():
                    expert_input = x * mask.unsqueeze(-1)
                    expert_output = self.experts[i](expert_input)
                    output += expert_output * (weights[:, :, k] / self.top_k).unsqueeze(-1) * mask.unsqueeze(-1)
                    expert_counts[i] += mask.sum().item()

        return output, expert_ids, expert_counts


def sequence_level_balance_loss(gate_logits, topk_idx, top_k):
    """
    序列级负载均衡损失

    目标：让单个序列内的专家使用也保持均衡
    避免某些序列过度依赖少数专家
    """
    B, T, E = gate_logits.shape

    # 计算每个序列内每个专家被选中的频率
    seq_expert_counts = torch.zeros(B, E, device=gate_logits.device)

    for b in range(B):
        for e in range(E):
            seq_expert_counts[b, e] = (topk_idx[b] == e).sum().item()

    # 归一化频率
    seq_freq = seq_expert_counts / (T * top_k)

    # 熵正则：鼓励均匀分布（最大化熵 = 均匀分布）
    entropy = -(seq_freq * torch.log(seq_freq + 1e-8)).sum(dim=-1).mean()

    # 均衡损失：方差（越小越均衡）
    freq_variance = seq_freq.var(dim=-1).mean()

    return freq_variance, entropy


def compute_load_distribution(expert_counts, num_experts):
    """计算专家负载分布"""
    total = sum(expert_counts.values()) if isinstance(expert_counts, dict) else expert_counts.sum().item()
    if isinstance(expert_counts, dict):
        distribution = {k: v / total for k, v in expert_counts.items()}
    else:
        distribution = expert_counts / total
    return distribution


def main():
    print("=" * 60)
    print("Demo 06: MoE 架构改进")
    print("=" * 60)

    d_model = 64
    d_ff = 128
    num_experts = 8
    top_k = 2
    batch_size = 4
    seq_len = 32

    torch.manual_seed(42)

    # 创建输入
    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n输入: batch={batch_size}, seq_len={seq_len}, d_model={d_model}")
    print(f"专家数: {num_experts}, Top-K: {top_k}")

    # ===== 实验 1: 标准 Top-K MoE =====
    print("\n--- 实验 1: 标准 Top-K MoE ---")
    moe = StandardTopKMoE(d_model, num_experts, top_k, d_ff)

    with torch.no_grad():
        output, gate_logits, topk_idx, expert_counts = moe(x)

    print(f"输出 shape: {output.shape}")
    print(f"门控 logits 范围: [{gate_logits.min():.3f}, {gate_logits.max():.3f}]")

    # 专家负载分布
    total_tokens = batch_size * seq_len * top_k
    print("\n专家负载分布 (Top-K MoE):")
    print(f"{'Expert':>8s} | {'Count':>8s} | {'Percentage':>12s} | {'理想值':>10s}")
    print("-" * 50)
    ideal = total_tokens / num_experts
    for i in range(num_experts):
        count = expert_counts[i].item()
        pct = count / total_tokens * 100
        ideal_pct = 100 / num_experts
        print(f"  E{i:2d}     | {count:8d} | {pct:11.1f}% | {ideal_pct:9.1f}%")

    # 序列级均衡损失
    balance_var, entropy = sequence_level_balance_loss(gate_logits, topk_idx, top_k)
    print(f"\n序列级均衡指标:")
    print(f"  专家频率方差: {balance_var:.4f} (越小越均衡)")
    print(f"  平均熵: {entropy:.4f} (越大越均衡)")

    # ===== 实验 2: Sqrt(Softplus()) vs Sigmoid =====
    print("\n--- 实验 2: Sqrt(Softplus()) vs Sigmoid ---")

    x_test = torch.linspace(-5, 5, 100)

    sig_out = torch.sigmoid(x_test)
    sp_out = sqrt_softplus(x_test)

    print(f"  Sigmoid(5.0)    = {sig_out[-1].item():.4f}")
    print(f"  Sqrt(Softplus)(5.0) = {sp_out[-1].item():.4f}")
    print(f"  Sigmoid(-5.0)   = {sig_out[0].item():.6f}")
    print(f"  Sqrt(Softplus)(-5.0) = {sp_out[0].item():.6f}")
    print("  (Sqrt(Softplus) 在极端值处更受抑制)")

    # ===== 实验 3: Hash-routed MoE =====
    print("\n--- 实验 3: Hash-routed MoE ---")
    hash_moe = HashRoutedMoE(d_model, num_experts, top_k, d_ff)

    with torch.no_grad():
        output_hash, expert_ids, hash_counts = hash_moe(x)

    print(f"Hash 路由输出 shape: {output_hash.shape}")

    print("\n专家负载分布 (Hash-routed MoE):")
    print(f"{'Expert':>8s} | {'Count':>8s} | {'Percentage':>12s}")
    print("-" * 40)
    for i in range(num_experts):
        count = hash_counts[i].item()
        pct = count / total_tokens * 100
        print(f"  E{i:2d}     | {count:8d} | {pct:11.1f}%")
    print("  (Hash 路由天然均匀，理想状态下各专家负载相等)")

    # ===== 实验 4: V4 中的激活函数改进 =====
    print("\n--- 实验 4: Sqrt(Softplus()) 激活函数特性 ---")

    # 在 MoE 门控中使用
    gate_test = torch.randn(10, num_experts)

    gate_sigmoid = torch.sigmoid(gate_test)
    gate_sqrt_sp = sqrt_softplus(gate_test)

    # 对比
    print(f"  Sigmoid 亲和度范围: [{gate_sigmoid.min():.4f}, {gate_sigmoid.max():.4f}]")
    print(f"  SqrtSP 亲和度范围:  [{gate_sqrt_sp.min():.4f}, {gate_sqrt_sp.max():.4f}]")
    print(f"  SqrtSP 的值域更宽，更能区分强/弱专家")

    # ===== 可视化 =====
    os.makedirs('figures', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # 专家负载分布
    ax1 = axes[0]
    experts = [f'E{i}' for i in range(num_experts)]
    topk_loads = [expert_counts[i].item() / total_tokens * 100 for i in range(num_experts)]
    hash_loads = [hash_counts[i].item() / total_tokens * 100 for i in range(num_experts)]

    x_pos = torch.arange(num_experts).numpy()
    width = 0.35

    ax1.bar(x_pos - width/2, topk_loads, width, label='Top-K MoE', alpha=0.8)
    ax1.bar(x_pos + width/2, hash_loads, width, label='Hash-routed', alpha=0.8)
    ax1.axhline(y=100/num_experts, color='r', linestyle='--', label='Ideal (12.5%)')
    ax1.set_xlabel('Expert')
    ax1.set_ylabel('Load (%)')
    ax1.set_title('Expert Load Distribution')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # 激活函数对比
    ax2 = axes[1]
    x_plot = torch.linspace(-5, 5, 100).numpy()
    ax2.plot(x_plot, torch.sigmoid(torch.tensor(x_plot)).numpy(), label='Sigmoid', linewidth=2)
    ax2.plot(x_plot, sqrt_softplus(torch.tensor(x_plot)).numpy(), label='Sqrt(Softplus)', linewidth=2)
    ax2.set_xlabel('Input')
    ax2.set_ylabel('Output')
    ax2.set_title('Activation Function Comparison')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/demo_moe_comparison.png', dpi=150)
    print("\n图表已保存到 figures/demo_moe_comparison.png")

    # ===== V4 改进总结 =====
    print("\n" + "=" * 60)
    print("DeepSeek V4 MoE 改进总结:")
    print("1. Hash-routed MoE 替代浅层 Dense FFN (O(1) 路由，无门控)")
    print("2. Sqrt(Softplus()) 替代 Sigmoid (更稳定，值域更宽)")
    print("3. 序列级负载均衡损失 (防止单序列内专家过度集中)")
    print("4. 移除节点限制 (路由更自由)")
    print("5. 细粒度专家并行 + 通信重叠 (隐藏 all-to-all 延迟)")
    print("=" * 60)


if __name__ == "__main__":
    main()
