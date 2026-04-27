"""
Demo 02: mHC (Manifold-Constrained Hyper-Connections)

本 Demo 展示三种残差连接方式的前向传播稳定性对比：
1. 普通残差连接 (Residual)
2. 无约束超连接 (Unconstrained HC)
3. mHC: Sinkhorn-Knopp 投影到双随机矩阵流形

通过追踪多层传播后的信号范数，直观展示 mHC 的稳定性优势。
"""

import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os


def sinkhorn_projection(M, iterations=20):
    """
    Sinkhorn-Knopp 算法：将矩阵投影到双随机矩阵流形
    M: (n, n) 输入矩阵
    返回: (n, n) 双随机矩阵（每行每列和都为 1）
    """
    M = torch.exp(M)  # 保证正性
    for _ in range(iterations):
        M = M / M.sum(dim=1, keepdim=True)  # 行归一化
        M = M / M.sum(dim=0, keepdim=True)  # 列归一化
    return M


class ResidualBlock(nn.Module):
    """普通残差连接: x_{l+1} = x_l + F(x_l)"""
    def __init__(self, dim):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim)
        )

    def forward(self, x):
        return x + self.ffn(x)


class HCBlock(nn.Module):
    """
    无约束超连接 (Hyper-Connections):
    将单流扩展为 n=4 个流，用可学习矩阵混合，但无约束。
    """
    def __init__(self, dim, n_streams=4):
        super().__init__()
        self.n_streams = n_streams
        self.dim_per_stream = dim // n_streams

        # n 个并行的 FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim_per_stream, self.dim_per_stream),
                nn.GELU(),
                nn.Linear(self.dim_per_stream, self.dim_per_stream)
            ) for _ in range(n_streams)
        ])

        # 无约束的混合矩阵
        self.mix = nn.Parameter(torch.randn(n_streams, n_streams))

    def forward(self, x):
        B, D = x.shape
        # 将输入拆分为 n 个流
        streams = x.view(B, self.n_streams, self.dim_per_stream)

        # 每个流过 FFN
        streams_out = torch.stack([self.ffns[i](streams[:, i, :])
                                    for i in range(self.n_streams)], dim=1)

        # 用无约束矩阵混合
        mixed = torch.einsum('ns,bsd->bnd', self.mix, streams_out)

        return mixed.view(B, D)


class mHCBlock(nn.Module):
    """
    mHC (Manifold-Constrained Hyper-Connections):
    混合矩阵通过 Sinkhorn-Knopp 投影到双随机矩阵流形。
    """
    def __init__(self, dim, n_streams=4, sinkhorn_iter=20):
        super().__init__()
        self.n_streams = n_streams
        self.dim_per_stream = dim // n_streams
        self.sinkhorn_iter = sinkhorn_iter

        # n 个并行的 FFN
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(self.dim_per_stream, self.dim_per_stream),
                nn.GELU(),
                nn.Linear(self.dim_per_stream, self.dim_per_stream)
            ) for _ in range(n_streams)
        ])

        # 原始可学习参数，每次前向时投影到双随机矩阵
        self.mix_raw = nn.Parameter(torch.randn(n_streams, n_streams))

    def get_mix_matrix(self):
        """通过 Sinkhorn-Knopp 投影得到双随机矩阵"""
        return sinkhorn_projection(self.mix_raw, self.sinkhorn_iter)

    def forward(self, x):
        B, D = x.shape
        # 拆分为 n 个流
        streams = x.view(B, self.n_streams, self.dim_per_stream)

        # 每个流过 FFN
        streams_out = torch.stack([self.ffns[i](streams[:, i, :])
                                    for i in range(self.n_streams)], dim=1)

        # 用投影后的双随机矩阵混合
        mix = self.get_mix_matrix()
        mixed = torch.einsum('ns,bsd->bnd', mix, streams_out)

        return mixed.view(B, D)


def build_stack(block_cls, dim, num_layers, **kwargs):
    """构建指定层数的模块堆叠"""
    return nn.Sequential(*[block_cls(dim, **kwargs) for _ in range(num_layers)])


def trace_signal_amplification(model, x, num_layers):
    """
    逐层追踪信号范数，返回每层输出的 L2 范数
    """
    norms = [torch.norm(x).item()]

    for layer in model:
        x = layer(x)
        norms.append(torch.norm(x).item())

    return norms


def compute_spectral_norms(model):
    """计算各层混合矩阵的谱范数"""
    spectral_norms = []
    for layer in model:
        if hasattr(layer, 'mix'):
            # HC: 直接使用 mix
            M = layer.mix.detach()
            sn = torch.linalg.matrix_norm(M, 2).item()
        elif hasattr(layer, 'get_mix_matrix'):
            # mHC: 使用投影后的矩阵
            M = layer.get_mix_matrix().detach()
            sn = torch.linalg.matrix_norm(M, 2).item()
        else:
            # Residual: 恒等映射的谱范数为 1
            sn = 1.0
        spectral_norms.append(sn)
    return spectral_norms


def main():
    print("=" * 60)
    print("Demo 02: mHC 流形约束超连接")
    print("=" * 60)

    dim = 64
    num_layers = 20
    batch_size = 4

    # 初始输入
    x = torch.randn(batch_size, dim)

    print(f"\n参数: dim={dim}, layers={num_layers}, batch={batch_size}")

    # 构建三种连接方式的深层网络
    residual_net = build_stack(ResidualBlock, dim, num_layers)
    hc_net = build_stack(HCBlock, dim, num_layers, n_streams=4)
    mhc_net = build_stack(mHCBlock, dim, num_layers, n_streams=4, sinkhorn_iter=20)

    # 追踪信号放大
    with torch.no_grad():
        res_norms = trace_signal_amplification(residual_net, x.clone(), num_layers)
        hc_norms = trace_signal_amplification(hc_net, x.clone(), num_layers)
        mhc_norms = trace_signal_amplification(mhc_net, x.clone(), num_layers)

    # 打印关键结果
    print("\n--- 信号范数随层数变化 ---")
    print(f"{'Layer':>6s} | {'Residual':>10s} | {'HC (unconstrained)':>20s} | {'mHC':>10s}")
    print("-" * 60)
    for i in range(0, num_layers + 1, 5):
        print(f"{i:6d} | {res_norms[i]:10.2f} | {hc_norms[i]:20.2f} | {mhc_norms[i]:10.2f}")

    print(f"\n最终层信号放大倍数:")
    print(f"  Residual: {res_norms[-1]/res_norms[0]:.2f}x")
    print(f"  HC:       {hc_norms[-1]/hc_norms[0]:.2f}x  (容易信号爆炸)")
    print(f"  mHC:      {mhc_norms[-1]/mhc_norms[0]:.2f}x  (受控)")

    # 谱范数对比
    print("\n--- 混合矩阵谱范数对比 ---")
    hc_sn = compute_spectral_norms(hc_net)
    mhc_sn = compute_spectral_norms(mhc_net)

    print(f"HC 谱范数范围:   [{min(hc_sn):.3f}, {max(hc_sn):.3f}], 均值={sum(hc_sn)/len(hc_sn):.3f}")
    print(f"mHC 谱范数范围:  [{min(mhc_sn):.3f}, {max(mhc_sn):.3f}], 均值={sum(mhc_sn)/len(mhc_sn):.3f}")
    print("(mHC 的谱范数被约束在 <= 1.0 附近)")

    # 可视化
    os.makedirs('figures', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(res_norms, 'o-', label='Residual', linewidth=2, markersize=4)
    plt.plot(hc_norms, 's-', label='HC (unconstrained)', linewidth=2, markersize=4)
    plt.plot(mhc_norms, '^-', label='mHC (Sinkhorn)', linewidth=2, markersize=4)
    plt.xlabel('Layer')
    plt.ylabel('Signal Norm (L2)')
    plt.title('Signal Amplification: Residual vs HC vs mHC')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.yscale('log')
    plt.tight_layout()
    plt.savefig('figures/demo_mhc_signal_amplification.png', dpi=150)
    print("\n图表已保存到 figures/demo_mhc_signal_amplification.png")

    print("\n" + "=" * 60)
    print("核心思想：")
    print("1. 双随机矩阵的谱范数 <= 1，防止信号层层放大")
    print("2. Sinkhorn-Knopp 是可微投影，可用于训练")
    print("3. mHC 用数学约束替代经验调参，保证深层稳定性")
    print("=" * 60)


if __name__ == "__main__":
    main()
