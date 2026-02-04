import numpy as np
import matplotlib.pyplot as plt
from meta_env import MetaExecutionEnv

def test_env_logic():
    print("🚀 初始化 MetaExecutionEnv...")
    env = MetaExecutionEnv(n_assets=5, episode_length=50, cost_rate=0.0005)
    
    obs, _ = env.reset(seed=42)
    print(f"✅ 环境重置成功. Obs Dim: {obs.shape}")
    print(f"🔮 Ground Truth 计算完成. Shape: {env.ground_truth_w.shape}")
    
    # 存储记录
    static_benchmark_wealth = [1.0]
    ground_truth_wealth = [1.0]
    
    # --- 1. 运行 Static Benchmark (模拟一个固定的传统策略) ---
    # 假设 lambda=5.0 (对应 action[0] approx -0.5), gamma=0.005 (对应 action[1] approx -0.9)
    static_action = np.array([-0.5, -0.9, 0.0]) 
    
    print("\n🏃 开始运行 Static Benchmark (Lambda=5.0, Gamma=0.005)...")
    terminated = False
    
    rewards = []
    regrets = []
    
    while not terminated:
        obs, reward, terminated, truncated, info = env.step(static_action)
        rewards.append(reward)
        regrets.append(info['regret'])
        
        # 简单的复利计算
        static_benchmark_wealth.append(static_benchmark_wealth[-1] * (1 + info['net_return']))
        
        # 计算 Ground Truth 的财富曲线 (用于对比)
        ground_truth_wealth.append(ground_truth_wealth[-1] * (1 + info['gt_net_return']))

    print("✅ 运行结束.")
    
    # --- 2. 结果可视化 ---
    plt.figure(figsize=(12, 6))
    
    # 子图 1: 净值曲线
    plt.subplot(2, 1, 1)
    plt.plot(ground_truth_wealth, label='Risk-Adjusted Ground Truth (Ceiling)', linestyle='--', color='red')
    plt.plot(static_benchmark_wealth, label='Static Benchmark (Baseline)', color='blue')
    plt.title('Wealth Curve: Static Benchmark vs. God Mode')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 子图 2: 遗憾值 (Regret)
    plt.subplot(2, 1, 2)
    plt.plot(np.cumsum(regrets), label='Cumulative Regret', color='orange')
    plt.title('Cumulative Regret (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
    
    print(f"\n📊 最终统计:")
    print(f"Ground Truth Final Wealth: {ground_truth_wealth[-1]:.4f}")
    print(f"Static Bench Final Wealth: {static_benchmark_wealth[-1]:.4f}")
    print(f"Efficiency: {static_benchmark_wealth[-1] / ground_truth_wealth[-1] * 100:.2f}%")
    print("如果 RL 有效，其曲线应位于蓝线和红线之间。")

if __name__ == "__main__":
    test_env_logic()