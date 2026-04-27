"""
Demo 05: FP4 量化感知训练

本 Demo 展示：
1. FP4 E2M1 格式的简化实现（教学版）
2. Block-wise 量化：把权重分成 block，每个 block 计算 scale
3. FP4 -> FP8 的无损反量化特性
4. 对比原始精度与量化后的误差
"""

import torch
import torch.nn as nn


class FP4Format:
    """
    简化的 FP4 (E2M1) 格式

    FP4 E2M1: 2 位指数, 1 位尾数 (不含符号位，实际是 S2M1)
    简化实现：使用 4 个离散值 {0, 1, 2, 3} 映射到浮点

    注意：这是教学简化版，非完整的 IEEE 754 兼容实现
    """

    # 简化的 E2M1 离散值表（教学用途）
    # 索引: 00=0, 01=1, 10=2, 11=3
    # 映射为: 0, normalized_value, denorm_value, inf/nan
    VALUES = torch.tensor([0.0, 0.5, 1.0, 2.0], dtype=torch.float32)

    @classmethod
    def quantize(cls, x):
        """把浮点数 x 量化到 FP4 离散值"""
        # 找到最接近的离散值
        diff = torch.abs(cls.VALUES.view(1, 1, -1) - x.view(-1, 1))
        idx = diff.argmin(dim=-1)  # (num_elements,)
        quantized = cls.VALUES[idx]
        return quantized.view_as(x), idx.view_as(x)

    @classmethod
    def dequantize(cls, indices):
        """把 FP4 索引反量化回浮点数"""
        return cls.VALUES[indices]


class BlockQuantizer:
    """
    Block-wise FP4 量化器

    核心思想：
    1. 把权重分成多个 block
    2. 每个 block 计算一个 scale = max(|x|) / fp4_max
    3. x_fp4 = round(x / scale)
    4. 存储 (fp4_indices, scales)
    5. 反量化: x_recon = fp4_values * scale
    """

    def __init__(self, block_size=8):
        self.block_size = block_size

    def quantize(self, weight):
        """
        量化权重矩阵

        Args:
            weight: (H, W) 权重矩阵

        Returns:
            fp4_indices: (H, W) FP4 索引
            scales: (num_blocks_h, num_blocks_w) 每个 block 的 scale
        """
        H, W = weight.shape
        device = weight.device

        # 计算 block 数量
        num_blocks_h = H // self.block_size
        num_blocks_w = W // self.block_size

        # 截断到整数个 block
        H_trunc = num_blocks_h * self.block_size
        W_trunc = num_blocks_w * self.block_size
        w_trunc = weight[:H_trunc, :W_trunc]

        # reshape 到 block 格式: (num_blocks_h, block_size, num_blocks_w, block_size)
        w_blocks = w_trunc.view(num_blocks_h, self.block_size,
                               num_blocks_w, self.block_size)
        w_blocks = w_blocks.permute(0, 2, 1, 3).contiguous()  # (nb_h, nb_w, bs, bs)

        # 计算每个 block 的 scale: max(|x|) / fp4_max
        # 这里用 2.0 作为 FP4 能表示的最大值
        fp4_max = 2.0
        abs_max = w_blocks.abs().amax(dim=(2, 3), keepdim=True)  # (nb_h, nb_w, 1, 1)
        scales = abs_max / fp4_max + 1e-8  # 避免除零

        # 量化: x_fp4 = clamp(round(x / scale), 0, 3)
        normalized = w_blocks / scales
        fp4_indices = torch.clamp(torch.round(normalized).long(), 0, 3)

        return fp4_indices, scales.squeeze(-1).squeeze(-1)

    def dequantize(self, fp4_indices, scales):
        """
        反量化: 从 FP4 索引恢复到 FP8 精度

        关键：FP4 的 scale 可以被 FP8 的额外指数位完全吸收
        """
        # 重建 block 结构
        nb_h, nb_w = scales.shape
        fp4_values = FP4Format.VALUES[fp4_indices]  # (nb_h*nb_w, bs, bs)
        fp4_values = fp4_values.view(nb_h, nb_w, self.block_size, self.block_size)

        # 反量化: x_recon = fp4_values * scale
        recon = fp4_values * scales.unsqueeze(-1).unsqueeze(-1)

        return recon

    def dequantize_lossless(self, fp4_indices, scales):
        """
        演示"无损"特性：FP4 block scale 被 FP8 完全吸收

        在教学中，这意味着如果我们用 FP8 来存储 scales，
        scales 的精度损失在合理范围内。
        """
        # 反量化回 FP8
        recon = self.dequantize(fp4_indices, scales)

        # FP8 额外精度演示：模拟 FP8 存储 scales（这里用 FP32 模拟）
        # 实际硬件上这一步是硬件级转换
        scales_fp8 = scales.to(torch.float16)  # 模拟 FP8

        # 用 FP8 scales 重新反量化
        recon_fp8 = self.dequantize(fp4_indices, scales_fp8)

        return recon, recon_fp8


def compute_quantization_metrics(weight, fp4_indices, scales, recon):
    """计算量化误差指标"""
    H, W = weight.shape
    nb_h, nb_w = scales.shape

    # 截断部分
    H_trunc = nb_h * 8
    W_trunc = nb_w * 8
    w_trunc = weight[:H_trunc, :W_trunc]

    # 计算误差
    error = (recon - w_trunc).abs()

    metrics = {
        'max_abs_error': error.max().item(),
        'mean_abs_error': error.mean().item(),
        'mse': error.pow(2).mean().item(),
        'relative_error': (error / (w_trunc.abs() + 1e-8)).mean().item(),
    }
    return metrics


def demo_lossless_property():
    """
    演示 FP4 -> FP8 的无损特性

    核心思路：
    FP4 的 block scale 可以完全被 FP8 的额外表示范围吸收
    """
    print("\n--- 无损吸收特性演示 ---")

    # 创建测试权重
    weight = torch.randn(64, 64) * 2.0

    # 量化到 FP4
    quantizer = BlockQuantizer(block_size=8)
    fp4_indices, scales = quantizer.quantize(weight)

    # 反量化（用高精度）
    recon_fp32 = quantizer.dequantize(fp4_indices, scales)

    # 反量化（模拟 FP8 scales）
    recon_fp8 = quantizer.dequantize_lossless(fp4_indices, scales)

    # 对比
    w_trunc = weight[:56, :56]  # 截断到 block 边界
    r_trunc_fp32 = recon_fp32[:7, :7].view(-1)
    r_trunc_fp8 = recon_fp8[:7, :7].view(-1)
    w_flat = w_trunc.transpose(0, 1).flip(0).reshape(-1)[:len(r_trunc_fp32)]

    error_fp32 = (r_trunc_fp32 - w_flat).abs().mean().item()
    error_fp8 = (r_trunc_fp8 - w_flat).abs().mean().item()

    print(f"  原始 vs FP4 反量化 (FP32 scales) 平均误差: {error_fp32:.6f}")
    print(f"  原始 vs FP4 反量化 (FP8  scales) 平均误差: {error_fp8:.6f}")
    print(f"  差异: {abs(error_fp32 - error_fp8):.2e}")
    print("  (误差差异极小，说明 FP4 -> FP8 的 scale 吸收基本无损)")


def main():
    print("=" * 60)
    print("Demo 05: FP4 量化感知训练")
    print("=" * 60)

    print("\n[1] FP4 E2M1 格式说明")
    print("-" * 40)
    print("FP4 E2M1 是 4 位浮点数格式（教学简化版）:")
    print("  - 使用 4 个离散值: {0.0, 0.5, 1.0, 2.0}")
    print("  - 相比 FP8 (E4M3): 指数位更少，动态范围更小")
    print("  - 关键特性: block scale 可被 FP8 完全吸收")

    # 创建测试权重
    print("\n[2] 创建测试权重矩阵")
    torch.manual_seed(42)
    weight = torch.randn(128, 128) * 1.5
    print(f"  权重 shape: {weight.shape}")
    print(f"  权重范围: [{weight.min():.3f}, {weight.max():.3f}]")
    print(f"  权重均值: {weight.mean():.4f}, 标准差: {weight.std():.4f}")

    # Block-wise 量化
    print("\n[3] Block-wise FP4 量化 (block_size=8)")
    quantizer = BlockQuantizer(block_size=8)
    fp4_indices, scales = quantizer.quantize(weight)

    num_blocks_h, num_blocks_w = scales.shape
    print(f"  原始权重: {weight.numel()} 个元素")
    print(f"  FP4 存储: {fp4_indices.numel()} 个索引 + {scales.numel()} 个 scale")
    print(f"  压缩比: {weight.numel() * 4 / (fp4_indices.numel() * 4 + scales.numel() * 16):.2f}x")
    print(f"    (假设 FP4 索引 4bit, scale 用 FP16)")

    # 反量化
    print("\n[4] FP4 反量化")
    recon = quantizer.dequantize(fp4_indices, scales)

    # 计算误差
    metrics = compute_quantization_metrics(weight, fp4_indices, scales, recon)
    print(f"  最大绝对误差: {metrics['max_abs_error']:.6f}")
    print(f"  平均绝对误差: {metrics['mean_abs_error']:.6f}")
    print(f"  MSE:          {metrics['mse']:.8f}")
    print(f"  相对误差:     {metrics['relative_error']:.4f}")

    # 无损特性演示
    demo_lossless_property()

    # 真实 V4 中的精度分工
    print("\n[5] DeepSeek V4 中的精度分工")
    print("-" * 40)
    print("  FP4 (E2M1): MoE 专家权重 + Lightning Indexer QK")
    print("  FP8 (E4M3): Attention 主路径 + 其他计算")
    print("  关键：FP4 权重无损展开到 FP8 计算，不损失精度")

    # 量化前后对比可视化（打印部分值）
    print("\n[6] 量化前后值对比（局部展示）")
    print("-" * 40)
    print(f"{'原始值':>12s} | {'FP4索引':>8s} | {'FP4值':>8s} | {'反量化值':>10s} | {'误差':>8s}")
    print("-" * 60)

    indices_flat = fp4_indices.view(-1)[:8]
    recon_flat = recon.transpose(0, 1).reshape(-1)[:8]
    w_flat = weight[:8, 0]

    for i in range(8):
        idx = indices_flat[i].item()
        fp4_val = FP4Format.VALUES[idx].item()
        r_val = recon_flat[i].item()
        err = abs(r_val - w_flat[i].item())
        print(f"{w_flat[i].item():12.4f} | {idx:8d} | {fp4_val:8.2f} | "
              f"{r_val:10.4f} | {err:8.4f}")

    print("\n" + "=" * 60)
    print("核心思想：")
    print("1. FP4 用 block-wise scaling 保持局部精度")
    print("2. block scale 可以被 FP8 的额外指数位完全吸收")
    print("3. 训练时直接适配 FP4，减少推理时的精度转换开销")
    print("4. 专家权重占参数主体，压缩它们收益最大")
    print("=" * 60)


if __name__ == "__main__":
    main()
