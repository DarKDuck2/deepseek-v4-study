# DeepSeek V4 技术模块代码 Demo

本目录包含 8 个独立的 Python Demo，分别对应 DeepSeek V4 技术学习手册中的 8 个技术模块。每个 Demo 都是教学性质的最小可运行示例，展示该技术的核心思想，而非完整复现生产级模型。

## 环境要求

- Python >= 3.8
- PyTorch >= 2.0
- NumPy >= 1.24
- Matplotlib >= 3.7

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行方式

每个 Demo 都是自包含的，可以直接运行：

```bash
python demo_csa_hca.py       # 01. 混合注意力：CSA 与 HCA
python demo_mhc.py           # 02. mHC：流形约束超连接
python demo_muon.py          # 03. Muon：面向隐藏层矩阵参数的优化器
python demo_post_training.py # 04. 两阶段后训练
python demo_fp4.py           # 05. FP4 量化感知训练
python demo_moe.py           # 06. MoE 架构改进
python demo_engram.py        # 07. Engram 条件记忆
python demo_dsec.py          # 08. DSec：弹性计算沙盒
```

## Demo 说明

| 文件 | 技术模块 | 核心展示内容 |
|------|----------|-------------|
| `demo_csa_hca.py` | 混合注意力 | CSA 4:1 压缩 + top-k 稀疏选择、HCA 128:1 重压缩、滑动窗口、KV Cache 大小对比 |
| `demo_mhc.py` | 流形约束超连接 | Sinkhorn-Knopp 投影、多流残差混合、对比普通残差/无约束 HC/mHC 的信号稳定性 |
| `demo_muon.py` | Muon 优化器 | Newton-Schulz 迭代、矩阵正交化更新、参数分组策略、与 AdamW 的收敛对比 |
| `demo_post_training.py` | 两阶段后训练 | 领域专家分别训练、On-Policy Distillation、KL 散度蒸馏损失 |
| `demo_fp4.py` | FP4 量化 | E2M1 简化表示、block-wise 量化/反量化、"无损"特性验证 |
| `demo_moe.py` | MoE 架构 | Top-K 路由、Hash-routed MoE、序列级负载均衡损失、Sqrt(Softplus) 激活 |
| `demo_engram.py` | Engram 条件记忆 | N-Gram 哈希查表、门控融合、异步预取模拟 |
| `demo_dsec.py` | DSec 沙盒 | 四种后端沙盒、调度器、轨迹日志、Fast-forward 断点续训模拟 |

## 注意事项

1. 所有 Demo 均为**教学简化版**，在真实生产环境中需要更复杂的工程和优化。
2. `demo_fp4.py` 中的 FP4 是简化教学实现，非完整的 IEEE 754 兼容版本。
3. `demo_dsec.py` 是纯 Python 模拟，展示 DSec 的设计理念，非真实的 Rust/3FS 实现。
4. 部分 Demo 运行时会生成 matplotlib 图表并保存到 `figures/` 目录下。
