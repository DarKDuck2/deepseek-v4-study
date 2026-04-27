"""
Demo 04: 两阶段后训练 — 专家培养与 On-Policy Distillation

本 Demo 展示：
1. 阶段一：分别训练代码、数学、通用三个领域专家
2. 阶段二：On-Policy Distillation，把专家能力蒸馏到统一学生模型
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    """领域专家模型：一个小 MLP"""
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def get_distribution(self, x, temperature=1.0):
        """返回输出分布（用于蒸馏）"""
        logits = self.forward(x)
        return F.softmax(logits / temperature, dim=-1)


class Student(nn.Module):
    """学生模型：和专家结构相同，蒸馏后能力应接近专家集合"""
    def __init__(self, input_dim=64, hidden_dim=128, num_classes=10):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x):
        return self.net(x)

    def generate_trajectory(self, x):
        """模拟 On-Policy 生成：在自身输出上采样下一步输入
        简化版：直接返回当前输出作为蒸馏目标
        """
        with torch.no_grad():
            return self.forward(x)


def create_domain_data(domain, num_samples=200, input_dim=64, num_classes=10):
    """为不同领域生成合成数据（每个领域有不同的数据分布）"""
    torch.manual_seed(42 + hash(domain) % 100)

    # 每个领域的特征分布不同
    if domain == 'code':
        # 代码数据：偏向高频模式
        centers = torch.randn(3, input_dim) * 2
        labels = torch.randint(0, 3, (num_samples,))
    elif domain == 'math':
        # 数学数据：更结构化
        centers = torch.randn(5, input_dim)
        labels = torch.randint(0, 5, (num_samples,))
    else:  # general
        # 通用数据：均匀分布
        centers = torch.randn(num_classes, input_dim)
        labels = torch.randint(0, num_classes, (num_samples,))

    # 以 centers 为中心生成数据
    x = centers[labels] + torch.randn(num_samples, input_dim) * 0.5

    return x, labels


def train_expert(expert, x, labels, domain, steps=100):
    """阶段一：训练单个领域专家"""
    optimizer = torch.optim.AdamW(expert.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for step in range(steps):
        optimizer.zero_grad()
        logits = expert(x)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

    # 最终准确率
    with torch.no_grad():
        preds = expert(x).argmax(dim=-1)
        acc = (preds == labels).float().mean().item()

    print(f"  [{domain}] 最终准确率: {acc:.2%}")
    return expert


def on_policy_distillation(student, experts, x_data, labels_data, domains, steps=100):
    """
    阶段二：On-Policy Distillation

    关键点：
    - 学生模型先在当前分布上生成轨迹
    - 然后学习多个教师专家的输出分布
    - KL 散度衡量分布差异
    """
    optimizer = torch.optim.AdamW(student.parameters(), lr=5e-4)
    criterion = nn.CrossEntropyLoss()

    # 领域权重（可学习或固定）
    domain_weights = {'code': 0.35, 'math': 0.35, 'general': 0.30}

    for step in range(steps):
        optimizer.zero_grad()
        total_loss = 0.0

        for domain in domains:
            # 获取该领域的数据
            domain_idx = domains.index(domain)
            x = x_data[domain]
            labels = labels_data[domain]
            weight = domain_weights[domain]

            # On-Policy: 学生生成自己的轨迹分布
            student_dist = student.get_distribution(x, temperature=1.0)

            # 教师分布
            teacher_dist = experts[domain].get_distribution(x, temperature=2.0)

            # KL 散度蒸馏损失
            kl_loss = F.kl_div(
                student_dist.log(),
                teacher_dist,
                reduction='batchmean'
            )

            # 原始任务损失（加权）
            task_loss = criterion(student(x), labels)

            # 组合损失
            loss = kl_loss + 0.5 * task_loss
            total_loss += weight * loss

        total_loss.backward()
        optimizer.step()

        if step % 20 == 0:
            print(f"  [Distillation] Step {step:3d}: combined_loss = {total_loss.item():.4f}")

    return student


def evaluate(model, x_data, labels_data, domains):
    """在所有领域上评估模型"""
    results = {}
    with torch.no_grad():
        for domain in domains:
            preds = model(x_data[domain]).argmax(dim=-1)
            acc = (preds == labels_data[domain]).float().mean().item()
            results[domain] = acc
    return results


def main():
    print("=" * 60)
    print("Demo 04: 两阶段后训练 — 专家培养与 On-Policy Distillation")
    print("=" * 60)

    input_dim = 64
    hidden_dim = 128
    num_classes = 10
    domains = ['code', 'math', 'general']

    # ===== 阶段零：准备数据 =====
    print("\n--- 阶段零: 生成各领域数据 ---")
    x_data = {}
    labels_data = {}
    for domain in domains:
        x_data[domain], labels_data[domain] = create_domain_data(domain)
        print(f"  {domain}: {x_data[domain].shape}, labels: {labels_data[domain].shape}")

    # ===== 阶段一：分别训练领域专家 =====
    print("\n--- 阶段一: 分别训练领域专家 ---")
    experts = {}
    for domain in domains:
        print(f"\n训练 {domain} 专家...")
        expert = Expert(input_dim, hidden_dim, num_classes)
        experts[domain] = train_expert(expert, x_data[domain],
                                       labels_data[domain], domain)

    # 评估各专家在其专长领域的表现
    print("\n专家在各领域的表现（对角线应为最高）:")
    print(f"{'Expert':>10s} | {'code':>8s} | {'math':>8s} | {'general':>8s}")
    print("-" * 45)
    for expert_domain in domains:
        results = evaluate(experts[expert_domain], x_data, labels_data, domains)
        print(f"{expert_domain:>10s} | {results['code']:8.2%} | "
              f"{results['math']:8.2%} | {results['general']:8.2%}")

    # ===== 阶段二：On-Policy Distillation =====
    print("\n--- 阶段二: On-Policy Distillation ---")
    print("将三个专家的能力蒸馏到统一学生模型...")

    student = Student(input_dim, hidden_dim, num_classes)

    # 蒸馏前的学生表现
    print("\n蒸馏前学生表现:")
    before_results = evaluate(student, x_data, labels_data, domains)
    for d, acc in before_results.items():
        print(f"  {d}: {acc:.2%}")

    # 执行蒸馏
    student = on_policy_distillation(student, experts, x_data, labels_data, domains)

    # 蒸馏后的学生表现
    print("\n蒸馏后学生表现:")
    after_results = evaluate(student, x_data, labels_data, domains)
    for d, acc in after_results.items():
        print(f"  {d}: {acc:.2%}")

    # 对比
    print("\n--- 蒸馏前后对比 ---")
    print(f"{'领域':>10s} | {'蒸馏前':>10s} | {'蒸馏后':>10s} | {'提升':>10s}")
    print("-" * 50)
    for d in domains:
        improvement = after_results[d] - before_results[d]
        print(f"{d:>10s} | {before_results[d]:10.2%} | "
              f"{after_results[d]:10.2%} | {improvement:+10.2%}")

    print("\n" + "=" * 60)
    print("核心思想：")
    print("1. 专家阶段允许每种能力在更合适的目标下单独变强")
    print("2. On-Policy 让学生在自己真实的分布上学习教师分布")
    print("3. KL 散度蒸馏的不只是答案，还有行为模式和推理方式")
    print("4. 最终统一模型继承了所有专家的上限能力")
    print("=" * 60)


if __name__ == "__main__":
    main()
