"""
Demo 08: DSec 弹性计算沙盒

本 Demo 展示（纯 Python 模拟版）：
1. 沙盒类：支持函数/容器/MicroVM/VM 四种后端
2. 调度器：管理多个并发沙盒
3. 轨迹日志：记录 command-output
4. Fast-forward 重放：断点续训
5. 模拟任务抢占与恢复

注意：这是教学简化版，DSec 真实实现用 Rust + 3FS，
本 Demo 纯 Python 模拟其核心设计理念。
"""

import time
import random
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional
from enum import Enum
from collections import defaultdict


class BackendType(Enum):
    """DSec 支持的四种执行后端"""
    FUNCTION = "function"       # 进程级，最快
    CONTAINER = "container"    # 内核命名空间隔离
    MICROVM = "microvm"       # 硬件虚拟化
    VM = "vm"                  # 完整系统虚拟化


@dataclass
class Command:
    """沙盒中执行的一条命令"""
    id: int
    cmd: str
    timestamp: float = field(default_factory=time.time)


@dataclass
class CommandResult:
    """命令执行结果"""
    command_id: int
    stdout: str
    stderr: str
    exit_code: int
    duration_ms: float


@dataclass
class TrajectoryLog:
    """
    全局轨迹日志

    记录每个沙盒的命令执行历史，用于断点续训
    """
    def __init__(self):
        self.logs: Dict[str, List[tuple]] = defaultdict(list)
        # 格式: {sandbox_id: [(command, result), ...]}

    def record(self, sandbox_id: str, command: Command, result: CommandResult):
        """记录一条命令及其结果"""
        self.logs[sandbox_id].append((command, result))

    def get_log(self, sandbox_id: str) -> List[tuple]:
        """获取某个沙盒的完整日志"""
        return self.logs.get(sandbox_id, [])

    def get_length(self, sandbox_id: str) -> int:
        """获取某个沙盒已执行的命令数"""
        return len(self.logs.get(sandbox_id, []))


class Sandbox:
    """
    沙盒实例

    支持四种后端类型，模拟 DSec 的统一抽象
    """
    def __init__(self, sandbox_id: str, backend: BackendType):
        self.sandbox_id = sandbox_id
        self.backend = backend
        self.state: Dict[str, Any] = {}
        self.is_active = True
        self.exec_count = 0

        # 后端启动时间（模拟）
        self.startup_time_ms = {
            BackendType.FUNCTION: 1,
            BackendType.CONTAINER: 50,
            BackendType.MICROVM: 200,
            BackendType.VM: 500,
        }

    def start(self):
        """启动沙盒（模拟毫秒级冷启动）"""
        startup = self.startup_time_ms[self.backend]
        time.sleep(startup / 1000)  # 实际是毫秒级
        self.is_active = True
        return f"[{self.backend.value}] Sandbox {self.sandbox_id} started"

    def execute(self, command: Command) -> CommandResult:
        """
        在沙盒中执行命令

        返回模拟的执行结果
        """
        if not self.is_active:
            raise RuntimeError(f"Sandbox {self.sandbox_id} is not active")

        self.exec_count += 1

        # 模拟执行（根据后端类型）
        if self.backend == BackendType.FUNCTION:
            # 函数调用：纯计算
            stdout = f"func_result_{command.id}"
            duration = random.uniform(0.1, 0.5)
        elif self.backend == BackendType.CONTAINER:
            # 容器：启动子进程
            stdout = f"container_result_{command.id}"
            duration = random.uniform(1, 5)
        elif self.backend == BackendType.MICROVM:
            # MicroVM：更重
            stdout = f"microvm_result_{command.id}"
            duration = random.uniform(5, 15)
        else:
            # 完整 VM：最重
            stdout = f"vm_result_{command.id}"
            duration = random.uniform(15, 50)

        # 模拟执行时间
        time.sleep(duration / 1000)

        # 更新状态
        self.state[f"step_{self.exec_count}"] = stdout

        return CommandResult(
            command_id=command.id,
            stdout=stdout,
            stderr="",
            exit_code=0,
            duration_ms=duration * 1000
        )

    def stop(self):
        """停止沙盒"""
        self.is_active = False
        return f"Sandbox {self.sandbox_id} stopped"

    def get_state_snapshot(self) -> Dict[str, Any]:
        """获取状态快照（用于恢复）"""
        return {
            "sandbox_id": self.sandbox_id,
            "backend": self.backend.value,
            "state": self.state.copy(),
            "exec_count": self.exec_count,
            "is_active": self.is_active,
        }


class Scheduler:
    """
    DSec 调度器

    管理多个并发沙盒，支持任务抢占与恢复
    """
    def __init__(self, trajectory_log: TrajectoryLog):
        self.sandboxes: Dict[str, Sandbox] = {}
        self.trajectory_log = trajectory_log
        self.next_sandbox_id = 0

    def create_sandbox(self, backend: BackendType = BackendType.CONTAINER) -> Sandbox:
        """创建一个新沙盒"""
        sandbox_id = f"sbox_{self.next_sandbox_id}"
        self.next_sandbox_id += 1

        sandbox = Sandbox(sandbox_id, backend)
        sandbox.start()
        self.sandboxes[sandbox_id] = sandbox

        return sandbox

    def run_task(self, sandbox: Sandbox, num_steps: int) -> List[CommandResult]:
        """
        在沙盒上运行一个任务

        返回执行结果列表
        """
        results = []
        for i in range(num_steps):
            cmd = Command(id=i, cmd=f"step_{i}")
            result = sandbox.execute(cmd)
            self.trajectory_log.record(sandbox.sandbox_id, cmd, result)
            results.append(result)

        return results

    def preempt_task(self, sandbox: Sandbox):
        """
        模拟任务被抢占

        关键：沙盒资源不释放，只是标记
        """
        # 记录抢占点
        state = sandbox.get_state_snapshot()
        print(f"  [Preempt] Task {sandbox.sandbox_id} preempted at step {state['exec_count']}")
        return state

    def resume_task(self, sandbox: Sandbox, preemption_state: Dict, from_step: int = None):
        """
        恢复被抢占的任务

        关键：Fast-forward 重放，从日志恢复已执行命令的结果
        而不是重新实际执行
        """
        exec_count = preemption_state['exec_count']

        # 模拟 Fast-forward：直接重放日志
        log = self.trajectory_log.get_log(sandbox.sandbox_id)
        replayed = 0

        for cmd, result in log:
            if replayed >= exec_count:
                break
            # 这里只是模拟重放，实际会用缓存的结果
            replayed += 1

        print(f"  [Resume] Fast-forward replayed {replayed} cached results")

        # 恢复状态
        sandbox.state = preemption_state['state'].copy()
        sandbox.exec_count = preemption_state['exec_count']
        sandbox.is_active = True

        return f"Task {sandbox.sandbox_id} resumed from step {exec_count}"

    def destroy_sandbox(self, sandbox: Sandbox):
        """销毁沙盒"""
        sandbox.stop()
        del self.sandboxes[sandbox.sandbox_id]


def demo_backend_comparison():
    """演示四种后端的启动时间和执行特性"""
    print("\n--- 四种执行后端对比 ---")

    backends = [
        BackendType.FUNCTION,
        BackendType.CONTAINER,
        BackendType.MICROVM,
        BackendType.VM,
    ]

    print(f"{'后端':>12s} | {'启动时间':>10s} | {'隔离级别':>12s} | {'适用场景':>20s}")
    print("-" * 65)

    for backend in backends:
        sb = Sandbox(f"demo_{backend.value}", backend)

        start = time.time()
        sb.start()
        startup_ms = (time.time() - start) * 1000

        if backend == BackendType.FUNCTION:
            scene = "简单工具调用"
        elif backend == BackendType.CONTAINER:
            scene = "代码执行、文件操作"
        elif backend == BackendType.MICROVM:
            scene = "不可信代码隔离"
        else:
            scene = "完整 OS 环境"

        isolation = {
            BackendType.FUNCTION: "进程级",
            BackendType.CONTAINER: "内核命名空间",
            BackendType.MICROVM: "硬件虚拟化",
            BackendType.VM: "完整系统",
        }[backend]

        print(f"{backend.value:>12s} | {startup_ms:>10.1f}ms | {isolation:>12s} | {scene:>20s}")


def demo_unified_api():
    """演示统一 Python SDK：一个参数切换后端"""
    print("\n--- 统一 API 演示 ---")

    backends = [
        BackendType.FUNCTION,
        BackendType.CONTAINER,
    ]

    for backend in backends:
        # 模拟 DSec 的统一 API
        sandbox = Sandbox(f"api_demo_{backend.value}", backend)
        sandbox.start()

        cmd = Command(id=0, cmd="print('hello')")
        result = sandbox.execute(cmd)

        print(f"backend='{backend.value}': exit_code={result.exit_code}, "
              f"stdout='{result.stdout}', duration={result.duration_ms:.1f}ms")

        sandbox.stop()


def demo_trajectory_recovery():
    """演示轨迹日志与断点续训"""
    print("\n--- 轨迹日志与断点续训 ---")

    # 创建调度器和日志
    trajectory_log = TrajectoryLog()
    scheduler = Scheduler(trajectory_log)

    # 创建沙盒并运行几步
    sandbox = scheduler.create_sandbox(BackendType.CONTAINER)
    print(f"[Init] Created sandbox: {sandbox.sandbox_id}")

    # 运行 5 步
    print("\n--- Phase 1: 正常运行 5 步 ---")
    for i in range(5):
        cmd = Command(id=i, cmd=f"work_step_{i}")
        result = sandbox.execute(cmd)
        trajectory_log.record(sandbox.sandbox_id, cmd, result)
        print(f"  Step {i}: {result.stdout}, duration={result.duration_ms:.1f}ms")

    print(f"\n日志长度: {trajectory_log.get_length(sandbox.sandbox_id)}")

    # 模拟抢占
    print("\n--- Phase 2: 任务被抢占 ---")
    preemption_state = scheduler.preempt_task(sandbox)

    # 创建新沙盒（或恢复旧沙盒）
    print("[Recovery] Creating new sandbox for recovery...")
    new_sandbox = Sandbox(f"recovered_{sandbox.sandbox_id}", sandbox.backend)
    # 模拟状态恢复
    new_sandbox.state = preemption_state['state']
    new_sandbox.exec_count = preemption_state['exec_count']

    # Fast-forward 重放
    print("\n--- Phase 3: Fast-forward 恢复 ---")
    recovered = scheduler.resume_task(new_sandbox, preemption_state)
    print(recovered)

    # 继续执行（从第 6 步开始）
    print("\n--- Phase 4: 继续执行（不重复已完成步骤）---")
    for i in range(5, 10):
        cmd = Command(id=i, cmd=f"work_step_{i}")
        result = new_sandbox.execute(cmd)
        trajectory_log.record(new_sandbox.sandbox_id, cmd, result)
        print(f"  Step {i}: {result.stdout}")

    print(f"\n最终日志长度: {trajectory_log.get_length(new_sandbox.sandbox_id)}")
    print("  (如果重新执行，这里应该是 10；断点续训只执行了 5 步新命令)")


def demo_large_scale_concurrency():
    """演示大规模并发调度"""
    print("\n--- 大规模并发调度演示（模拟）---")

    trajectory_log = TrajectoryLog()
    scheduler = Scheduler(trajectory_log)

    num_sandboxes = 20  # 模拟 20 个（实际 DSec 可达数十万）

    print(f"创建 {num_sandboxes} 个并发沙盒...")

    start = time.time()

    # 批量创建
    sandboxes = []
    for i in range(num_sandboxes):
        backend = random.choice([
            BackendType.FUNCTION,
            BackendType.CONTAINER,
            BackendType.MICROVM,
        ])
        sb = scheduler.create_sandbox(backend)
        sandboxes.append(sb)

    create_time = (time.time() - start) * 1000
    print(f"创建耗时: {create_time:.1f}ms (平均 {create_time/num_sandboxes:.2f}ms/个)")

    # 批量运行
    print(f"\n每个沙盒运行 3 步...")
    for sb in sandboxes:
        scheduler.run_task(sb, num_steps=3)

    total_time = (time.time() - start) * 1000
    print(f"总耗时: {total_time:.1f}ms")

    # 清理
    for sb in sandboxes:
        scheduler.destroy_sandbox(sb)

    print(f"\nDSec 实际能力:")
    print(f"  - 单集群可调度: 数十万个并发沙盒")
    print(f"  - 毫秒级冷启动 (层级按需加载)")
    print(f"  - 毫秒级调度延迟")


def main():
    print("=" * 60)
    print("Demo 08: DSec 弹性计算沙盒")
    print("=" * 60)

    print("\n[说明] 本 Demo 是 Python 模拟版，")
    print("真实 DSec 用 Rust + 3FS 实现，性能远超此模拟。")

    # 四种后端对比
    demo_backend_comparison()

    # 统一 API
    demo_unified_api()

    # 轨迹日志与断点续训
    demo_trajectory_recovery()

    # 大规模并发
    demo_large_scale_concurrency()

    # DSec 的核心价值
    print("\n" + "=" * 60)
    print("DSec 的核心设计价值：")
    print("1. 统一抽象：Python SDK + 四种异构后端，一行切换")
    print("2. 弹性规模：单集群数十万并发，层级按需加载")
    print("3. 容错恢复：全局轨迹日志 + Fast-forward 重放")
    print("4. 非幂等保护：避免重复执行导致的副作用")
    print("5. 工具即代码：Agent 训练的基础设施资产")
    print("=" * 60)


if __name__ == "__main__":
    main()
