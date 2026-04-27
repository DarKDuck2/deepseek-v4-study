"""
Demo 01: 混合注意力 — CSA (Compressed Sparse Attention) 与 HCA (Heavily Compressed Attention)

本 Demo 展示 DeepSeek V4 如何通过三种不同精度的记忆路径来处理长上下文：
- 滑动窗口：保留最近 128 tokens 的原始 KV
- CSA: 4:1 压缩 KV，通过 Lightning Indexer 选 top-k
- HCA: 128:1 重压缩 KV，做稠密全局注意力
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class SlidingWindowAttention(nn.Module):
    """滑动窗口注意力：保留最近 window_size 个 tokens 的原始 KV"""
    def __init__(self, d_model=64, num_heads=4, window_size=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.window_size = window_size

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, x):
        """x: (batch, seq_len, d_model)"""
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 只保留最近 window_size 个位置
        if T > self.window_size:
            K_win = K[:, :, -self.window_size:, :]
            V_win = V[:, :, -self.window_size:, :]
        else:
            K_win, V_win = K, V

        scores = torch.matmul(Q, K_win.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_win)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class CSAAttention(nn.Module):
    """Compressed Sparse Attention: 4:1 压缩 + top-k 稀疏选择"""
    def __init__(self, d_model=64, num_heads=4, compress_ratio=4, top_k=16):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.compress_ratio = compress_ratio
        self.top_k = top_k

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 压缩投影：把 compress_ratio 个 token 压成 1 个
        self.compress_proj = nn.Linear(d_model * compress_ratio, d_model)

    def compress_kv(self, K, V):
        """对 KV 做 4:1 压缩
        K, V: (batch, num_heads, seq_len, head_dim)
        返回: (batch, num_heads, compressed_len, head_dim)
        """
        B, H, T, D = K.shape
        # 只处理能被 compress_ratio 整除的部分
        valid_len = (T // self.compress_ratio) * self.compress_ratio
        K_valid = K[:, :, :valid_len, :]
        V_valid = V[:, :, :valid_len, :]

        # reshape: (B, H, T//r, r, D) -> (B, H, T//r, r*D)
        K_blocks = K_valid.view(B, H, -1, self.compress_ratio, D)
        V_blocks = V_valid.view(B, H, -1, self.compress_ratio, D)

        # 简单压缩：把 block 内所有 head_dim 拼接后投影
        K_flat = K_blocks.reshape(B, H, -1, self.compress_ratio * D)
        V_flat = V_blocks.reshape(B, H, -1, self.compress_ratio * D)

        K_comp = self.compress_proj(K_flat.transpose(2, 3)).transpose(2, 3)
        V_comp = self.compress_proj(V_flat.transpose(2, 3)).transpose(2, 3)

        return K_comp, V_comp

    def lightning_indexer(self, Q, K_comp):
        """Lightning Indexer: 用余弦相似度选 top-k 压缩块"""
        # Q: (B, H, T, D), K_comp: (B, H, T_comp, D)
        Q_norm = F.normalize(Q, dim=-1)
        K_norm = F.normalize(K_comp, dim=-1)

        # 相似度: (B, H, T, T_comp)
        sim = torch.matmul(Q_norm, K_norm.transpose(-2, -1))

        # 每个 query 选 top-k 个压缩块
        topk_sim, topk_idx = torch.topk(sim, min(self.top_k, sim.size(-1)), dim=-1)
        return topk_idx

    def forward(self, x):
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 1. 压缩 KV
        K_comp, V_comp = self.compress_kv(K, V)
        T_comp = K_comp.size(2)

        # 2. Lightning Indexer 选 top-k
        topk_idx = self.lightning_indexer(Q, K_comp)  # (B, H, T, top_k)

        # 3. 只拿 top-k 压缩块做注意力
        B_idx = torch.arange(B, device=x.device).view(B, 1, 1, 1)
        H_idx = torch.arange(self.num_heads, device=x.device).view(1, self.num_heads, 1, 1)

        # gather: (B, H, T, top_k, D)
        K_selected = K_comp[B_idx, H_idx, topk_idx, :]
        V_selected = V_comp[B_idx, H_idx, topk_idx, :]

        # 注意力: (B, H, T, D) x (B, H, T, top_k, D) -> (B, H, T, top_k)
        scores = torch.matmul(Q.unsqueeze(3), K_selected.transpose(-2, -1)).squeeze(3)
        scores = scores / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)

        # (B, H, T, top_k) x (B, H, top_k, D) -> (B, H, T, D)
        out = torch.matmul(attn.unsqueeze(3), V_selected).squeeze(3)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class HCAAttention(nn.Module):
    """Heavily Compressed Attention: 128:1 重压缩 + 稠密注意力"""
    def __init__(self, d_model=64, num_heads=4, compress_ratio=128):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.compress_ratio = compress_ratio

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        self.compress_proj = nn.Linear(d_model * compress_ratio, d_model)

    def compress_kv(self, K, V):
        """128:1 压缩"""
        B, H, T, D = K.shape
        valid_len = (T // self.compress_ratio) * self.compress_ratio
        K_valid = K[:, :, :valid_len, :]
        V_valid = V[:, :, :valid_len, :]

        K_blocks = K_valid.view(B, H, -1, self.compress_ratio, D)
        V_blocks = V_valid.view(B, H, -1, self.compress_ratio, D)

        K_flat = K_blocks.reshape(B, H, -1, self.compress_ratio * D)
        V_flat = V_blocks.reshape(B, H, -1, self.compress_ratio * D)

        K_comp = self.compress_proj(K_flat.transpose(2, 3)).transpose(2, 3)
        V_comp = self.compress_proj(V_flat.transpose(2, 3)).transpose(2, 3)

        return K_comp, V_comp

    def forward(self, x):
        B, T, D = x.shape

        Q = self.W_q(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(x).view(B, T, self.num_heads, self.head_dim).transpose(1, 2)

        # 128:1 压缩
        K_comp, V_comp = self.compress_kv(K, V)

        # 在压缩空间做稠密注意力
        scores = torch.matmul(Q, K_comp.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V_comp)

        out = out.transpose(1, 2).contiguous().view(B, T, D)
        return self.W_o(out)


class HybridAttention(nn.Module):
    """混合注意力：融合滑动窗口 + CSA + HCA"""
    def __init__(self, d_model=64, num_heads=4, window_size=128,
                 csa_compress=4, csa_topk=16, hca_compress=128):
        super().__init__()
        self.sw_attn = SlidingWindowAttention(d_model, num_heads, window_size)
        self.csa_attn = CSAAttention(d_model, num_heads, csa_compress, csa_topk)
        self.hca_attn = HCAAttention(d_model, num_heads, hca_compress)

        # 融合三个路径的输出
        self.fusion = nn.Linear(d_model * 3, d_model)

    def forward(self, x):
        out_sw = self.sw_attn(x)
        out_csa = self.csa_attn(x)
        out_hca = self.hca_attn(x)

        concat = torch.cat([out_sw, out_csa, out_hca], dim=-1)
        return self.fusion(concat)


def compute_kv_cache_size(seq_len, d_model, num_heads, compress_ratios):
    """计算不同注意力机制的 KV Cache 大小（以 float 数量计）"""
    head_dim = d_model // num_heads

    # 标准全注意力
    full_attn = 2 * seq_len * d_model

    # CSA: 4:1 压缩 + top-k
    csa_len = seq_len // compress_ratios['csa']
    csa_cache = 2 * csa_len * d_model  # 压缩后的 K 和 V

    # HCA: 128:1 压缩
    hca_len = seq_len // compress_ratios['hca']
    hca_cache = 2 * hca_len * d_model

    # 滑动窗口
    sw_cache = 2 * min(128, seq_len) * d_model

    return {
        'full_attention': full_attn,
        'csa_only': csa_cache,
        'hca_only': hca_cache,
        'sliding_window': sw_cache,
        'hybrid_csa+sw': csa_cache + sw_cache,
        'hybrid_all': csa_cache + hca_cache + sw_cache,
    }


def main():
    print("=" * 60)
    print("Demo 01: 混合注意力 CSA + HCA")
    print("=" * 60)

    # 模拟一个 1M tokens 的场景（为了 demo 速度，用更小的规模）
    seq_len = 4096  # 教学演示用，实际 V4 支持 1M
    d_model = 64
    num_heads = 4
    batch_size = 2

    x = torch.randn(batch_size, seq_len, d_model)

    print(f"\n输入 shape: {x.shape}")
    print(f"序列长度: {seq_len}, 维度: {d_model}, 头数: {num_heads}")

    # 混合注意力
    hybrid = HybridAttention(d_model, num_heads, window_size=128,
                            csa_compress=4, csa_topk=16, hca_compress=128)

    with torch.no_grad():
        out = hybrid(x)

    print(f"\n混合注意力输出 shape: {out.shape}")

    # KV Cache 大小对比
    print("\n--- KV Cache 大小对比 ---")
    sizes = compute_kv_cache_size(seq_len, d_model, num_heads,
                                  {'csa': 4, 'hca': 128})

    for name, size in sizes.items():
        ratio = size / sizes['full_attention'] * 100
        print(f"  {name:25s}: {size:8d} floats ({ratio:5.1f}% of full)")

    # 实际 V4 的比例（相对 V3.2）
    print("\n--- DeepSeek V4 官方数据（1M 上下文，相对 V3.2）---")
    print("  V4-Pro  KV Cache: ~10% of V3.2")
    print("  V4-Flash KV Cache: ~7% of V3.2")
    print("  对应单 token FLOPs: V4-Pro ~27%, V4-Flash ~10%")

    # 展示各子模块的独立输出
    print("\n--- 各注意力子模块输出示例 ---")
    sw = hybrid.sw_attn(x)
    csa = hybrid.csa_attn(x)
    hca = hybrid.hca_attn(x)
    print(f"  滑动窗口输出: {sw.shape}, 均值={sw.mean().item():.4f}")
    print(f"  CSA 输出:     {csa.shape}, 均值={csa.mean().item():.4f}")
    print(f"  HCA 输出:     {hca.shape}, 均值={hca.mean().item():.4f}")

    print("\n" + "=" * 60)
    print("核心思想：不是让所有远程历史都原样参与注意力，")
    print("而是对不同时距的信息采用不同压缩率和访问策略。")
    print("=" * 60)


if __name__ == "__main__":
    main()
