"""
Demo 07: Engram 条件记忆

本 Demo 展示（简化版实现）：
1. 多粒度 N-Gram 提取 (2-gram, 3-gram)
2. 多头哈希映射（减少碰撞）
3. 简化记忆库（embedding table）
4. 门控融合机制
5. 模拟异步预取

注意：这是教学简化版，展示 Engram 的核心设计思路。
官方论文中使用的是更复杂的哈希函数和记忆库实现。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class NGramExtractor:
    """
    多粒度 N-Gram 提取器

    从 token 序列中提取不同长度的连续片段（n-grams）
    """
    def __init__(self, max_n=3):
        self.max_n = max_n

    def extract(self, tokens):
        """
        从 token 序列中提取所有 n-grams

        Args:
            tokens: (seq_len,) token ID 序列

        Returns:
            ngrams: dict, key=n, value=list of n-tuple token IDs
        """
        ngrams = {n: [] for n in range(2, self.max_n + 1)}

        for n in range(2, self.max_n + 1):
            for i in range(len(tokens) - n + 1):
                ngram = tuple(tokens[i:i + n].tolist())
                ngrams[n].append(ngram)

        return ngrams


class MultiHeadHasher:
    """
    多头哈希映射

    使用多个独立的哈希函数减少碰撞，
    模拟 Engram 中的"conditional memory lookup"
    """
    def __init__(self, num_heads=4, hash_range=1024, seed=42):
        self.num_heads = num_heads
        self.hash_range = hash_range

        # 不同的哈希种子
        torch.manual_seed(seed)
        self.seeds = torch.randint(0, 10000, (num_heads,))

    def hash_ngram(self, ngram_tuple):
        """
        用多个哈希函数映射一个 n-gram

        Args:
            ngram_tuple: tuple of token IDs

        Returns:
            indices: (num_heads,) 哈希到的索引
        """
        ngram_tensor = torch.tensor(list(ngram_tuple), dtype=torch.long)

        indices = []
        for seed in self.seeds:
            # 简化的哈希：基于内容的混合
            hash_val = (ngram_tensor.float() * (seed + 1)).sum()
            idx = int(hash_val.item()) % self.hash_range
            indices.append(idx)

        return torch.tensor(indices, dtype=torch.long)


class MemoryBank:
    """
    简化记忆库

    存储预训练好的 N-Gram embeddings
    模拟 Engram 中"静态知识"的存储
    """
    def __init__(self, hash_range=1024, embed_dim=32, num_heads=4):
        self.hash_range = hash_range
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 记忆库：每个 head 的每个哈希槽一个嵌入
        # shape: (num_heads, hash_range, embed_dim)
        self.embeddings = nn.Parameter(
            torch.randn(num_heads, hash_range, embed_dim) * 0.02
        )

    def lookup(self, indices):
        """
        查表

        Args:
            indices: (num_heads,) 哈希索引

        Returns:
            embeddings: (embed_dim,) 平均后的嵌入
        """
        # gather 获取对应嵌入
        embs = self.embeddings[:, indices, :]  # (num_heads, num_heads, embed_dim)
        # 对多头结果取平均
        emb = embs.mean(dim=0)  # (embed_dim,)
        return emb

    def prefetch(self, indices_list):
        """
        模拟异步预取：提前把多个索引的嵌入加载出来
        这是 Engram 的关键优势之一
        """
        with torch.no_grad():
            embeddings = []
            for indices in indices_list:
                emb = self.lookup(indices)
                embeddings.append(emb)
            return torch.stack(embeddings)


class EngramModule(nn.Module):
    """
    Engram 条件记忆模块（简化版）

    工作流程：
    1. 提取多粒度 N-Grams
    2. 多头哈希映射
    3. 查记忆库获取嵌入
    4. 门控融合到隐藏状态
    """
    def __init__(self, vocab_size=10000, d_model=64, embed_dim=32,
                 num_heads=4, hash_range=1024, max_n=3):
        super().__init__()
        self.d_model = d_model
        self.embed_dim = embed_dim
        self.num_heads = num_heads

        # 组件
        self.ngram_extractor = NGramExtractor(max_n=max_n)
        self.hasher = MultiHeadHasher(num_heads, hash_range)
        self.memory_bank = MemoryBank(hash_range, embed_dim, num_heads)

        # 投影层
        self.to_hidden = nn.Linear(embed_dim, d_model)

        # 门控机制
        self.gate_net = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, token_ids, hidden_state):
        """
        Args:
            token_ids: (batch, seq_len) token ID 序列
            hidden_state: (batch, seq_len, d_model) 当前隐藏状态

        Returns:
            enhanced_hidden: (batch, seq_len, d_model) 融合记忆后的隐藏状态
            memory_info: dict 记忆融合的调试信息
        """
        B, T = token_ids.shape

        # 对每个 batch 和位置提取 n-grams 并查表
        memory_outputs = []
        gate_values = []

        for b in range(B):
            seq_tokens = token_ids[b]
            seq_hidden = hidden_state[b]

            # 提取所有 n-grams
            ngrams = self.ngram_extractor.extract(seq_tokens)

            # 对每种 n-gram 查记忆库
            seq_memory = torch.zeros(T, self.embed_dim, device=token_ids.device)

            for n, ngram_list in ngrams.items():
                if not ngram_list:
                    continue

                for pos, ngram in enumerate(ngram_list):
                    if pos >= T:
                        break
                    # 哈希映射
                    indices = self.hasher.hash_ngram(ngram)
                    # 查表
                    emb = self.memory_bank.lookup(indices)
                    seq_memory[pos] += emb

            # 投影到 d_model
            memory_proj = self.to_hidden(seq_memory)

            # 门控融合
            gate = self.gate_net(seq_hidden + memory_proj)  # (T, 1)

            # 融合
            enhanced = seq_hidden + gate * memory_proj
            memory_outputs.append(enhanced)
            gate_values.append(gate)

        enhanced_hidden = torch.stack(memory_outputs, dim=0)

        memory_info = {
            'gate_mean': torch.stack(gate_values).mean().item(),
            'memory_norm': memory_proj.norm().item(),
        }

        return enhanced_hidden, memory_info


def simulate_async_prefetch():
    """
    模拟 Engram 的异步预取特性

    由于查表只依赖 token IDs（不依赖激活），
    可以提前异步加载，无需等待计算完成
    """
    print("\n--- 异步预取模拟 ---")

    memory_bank = MemoryBank(hash_range=1024, embed_dim=32, num_heads=4)

    # 模拟一个长序列的 token IDs
    token_ids = torch.randint(0, 10000, (100,))

    extractor = NGramExtractor(max_n=3)
    hasher = MultiHeadHasher(num_heads=4, hash_range=1024)

    # 提取所有 n-gram 的索引
    all_indices = []
    ngrams = extractor.extract(token_ids)

    for n in range(2, 4):
        for ngram in ngrams[n][:10]:  # 取前 10 个演示
            indices = hasher.hash_ngram(ngram)
            all_indices.append(indices)

    print(f"  序列长度: {len(token_ids)}")
    print(f"  提取的 n-gram 数: {len(all_indices)}")

    # 模拟预取（实际应该在 GPU/CPU 之间异步传输）
    prefetched = memory_bank.prefetch(all_indices[:20])
    print(f"  预取 embeddings shape: {prefetched.shape}")
    print(f"  预取耗时（模拟）: < 1ms (vs 计算可能需要 10+ ms)")

    # 对比：如果是计算式（MoE forward）
    print("\n  异步预取 vs 同步计算:")
    print("  - 预取: O(1) 哈希查表，不依赖激活，可提前执行")
    print("  - MoE forward: O(d) 矩阵乘，必须等待前向完成")
    print("  - 关键: Engram 查表可以在上一 token 的 forward 时就开始")


def main():
    print("=" * 60)
    print("Demo 07: Engram 条件记忆")
    print("=" * 60)

    vocab_size = 10000
    d_model = 64
    embed_dim = 32
    num_heads = 4
    hash_range = 1024
    batch_size = 2
    seq_len = 16

    torch.manual_seed(42)

    # 创建模块
    engram = EngramModule(vocab_size, d_model, embed_dim, num_heads, hash_range)

    # 创建输入
    token_ids = torch.randint(0, vocab_size, (batch_size, seq_len))
    hidden_state = torch.randn(batch_size, seq_len, d_model)

    print(f"\n输入: batch={batch_size}, seq_len={seq_len}, vocab={vocab_size}")
    print(f"模型: d_model={d_model}, embed_dim={embed_dim}, num_heads={num_heads}")

    # ===== N-Gram 提取演示 =====
    print("\n--- N-Gram 提取示例 ---")
    extractor = NGramExtractor(max_n=3)
    sample_tokens = token_ids[0, :8]
    ngrams = extractor.extract(sample_tokens)

    print(f"Token 序列前 8 个: {sample_tokens.tolist()}")
    print(f"2-grams: {ngrams[2][:5]}")
    print(f"3-grams: {ngrams[3][:5]}")

    # ===== 多头哈希演示 =====
    print("\n--- 多头哈希映射 ---")
    hasher = MultiHeadHasher(num_heads=4, hash_range=1024)
    sample_ngram = ngrams[3][0] if ngrams[3] else ngrams[2][0]
    indices = hasher.hash_ngram(sample_ngram)

    print(f"示例 n-gram: {sample_ngram}")
    print(f"多头哈希索引 (4 heads): {indices.tolist()}")
    print("  (不同 head 用不同哈希函数，减少碰撞)")

    # ===== Engram 前向传播 =====
    print("\n--- Engram 前向传播 ---")

    with torch.no_grad():
        enhanced_hidden, memory_info = engram(token_ids, hidden_state)

    print(f"原始隐藏状态 shape: {hidden_state.shape}")
    print(f"融合后隐藏状态 shape: {enhanced_hidden.shape}")
    print(f"门控均值: {memory_info['gate_mean']:.4f}")
    print(f"记忆嵌入范数: {memory_info['memory_norm']:.4f}")

    # ===== 门控效果 =====
    print("\n--- 门控机制效果 ---")
    # 门控值接近 0 表示忽略记忆，接近 1 表示信任记忆
    print("门控机制的作用:")
    print("  - gate ≈ 0: 当前 token 依赖动态推理，不依赖记忆")
    print("  - gate ≈ 1: 记忆提供关键信息，直接使用")
    print(f"当前门控均值: {memory_info['gate_mean']:.4f}")

    # ===== 稀疏分配定律 =====
    print("\n--- 稀疏分配定律（来自 Engram 论文）---")
    print("研究发现在总稀疏参数固定时:")
    print("  - 20-25% 分配给记忆 (Engram)")
    print("  - 75-80% 分配给计算 (MoE)")
    print("  - 这是一个 U 型曲线的最优平衡点")

    # ===== 异步预取模拟 =====
    simulate_async_prefetch()

    # ===== Offload 能力 =====
    print("\n--- 记忆库 Offload 能力 ---")
    total_memory_params = num_heads * hash_range * embed_dim
    print(f"记忆库参数量: {total_memory_params:,} ({total_memory_params/1e6:.2f}M)")
    print(f"如果是 100B 参数的记忆库:")
    print(f"  - 可完全放在 CPU/SSD 上")
    print(f"  - GPU 推理开销增加 < 3%")
    print(f"  - 打破了 GPU 显存必须容纳所有参数的约束")

    print("\n" + "=" * 60)
    print("核心思想：")
    print("1. 静态知识用 O(1) 查表代替动态计算")
    print("2. 多粒度 N-Gram 捕获不同尺度的局部模式")
    print("3. 多头哈希减少碰撞，异步预取掩盖延迟")
    print("4. 门控决定何时信任记忆，何时依赖推理")
    print("5. 记忆库可 offload，突破 GPU 显存墙")
    print("=" * 60)


if __name__ == "__main__":
    main()
