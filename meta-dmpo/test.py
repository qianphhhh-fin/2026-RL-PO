import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_meta import MetaSyntheticEnv
from dmpo_model import MetaDMPOActionWrapper

def run_test():
    # 1. 准备测试环境
    # 注意：测试时不要 Shuffle，要按顺序
    raw_env = MetaSyntheticEnv(n_assets=5, n_steps=1000, lookback=30)
    env = MetaDMPOActionWrapper(raw_env)
    env = DummyVecEnv([lambda: env])
    
    # 加载归一化参数 (非常重要，否则 Agent 看不懂数据)
    env = VecNormalize.load("meta_vec_normalize.pkl", env)
    env.training = False # 测试模式，不更新均值方差
    env.norm_reward = False
    
    # 加载模型
    model = PPO.load("meta_dmpo_agent")
    
    print("📊 开始测试运行...")
    obs = env.reset()
    
    # 记录数据
    portfolio_values = [1.0]
    alpha_weights_history = []
    
    # 拿到底层环境引用用于 Benchmark
    base_env = env.envs[0].env.unwrapped
    market_returns = base_env.returns # 真实市场收益
    
    done = False
    step_idx = 0
    
    while not done:
        # 预测动作
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # 记录净值
        ret = infos[0].get('terminal_observation', {}) # SB3 的 quirk，忽略
        # 我们手动算累计收益更准
        # 这里用一种简单近似：利用记录的 info
        
        # 实际上 VecEnv 的 step 返回的 reward 是归一化过的，不能直接用算净值
        # 最好是我们重新跑一遍逻辑，或者在 Wrapper 里记录真实收益
        # 这里为了简单，我们假设 Wrapper 没改 Reward，直接用 env.render 或 hack
        
        # 正确做法：直接复现逻辑或让 Wrapper 返回真实收益
        # 这里我们假定 base_env.current_step 已经推进了
        # 拿到上一步的收益
        
        # Hack: 从 Info 中提取 Alpha 权重
        alpha_weights = infos[0]['alpha_weights']
        alpha_weights_history.append(alpha_weights)
        
        # 简单计算 Benchmark 收益 (1/N Alpha)
        # ... (略)
        
        step_idx += 1
        if step_idx >= 900: # 稍微提前结束避免越界
            break
            
    # --- 可视化分析 ---
    alpha_weights_history = np.array(alpha_weights_history)
    
    plt.figure(figsize=(12, 8))
    
    # 图1: Alpha 权重分配热力图 (The Money Shot!)
    plt.subplot(2, 1, 1)
    plt.title("Meta-Agent Decision: Which Alpha to Trust?")
    # Alpha 0: Oracle, Alpha 1: Inverse, Alpha 2: Noise, Alpha 3: Regime
    labels = ["Oracle", "Inverse", "Noise", "Regime"]
    
    plt.stackplot(range(len(alpha_weights_history)), alpha_weights_history.T, labels=labels, alpha=0.8)
    plt.legend(loc='upper left')
    plt.ylabel("Weight Allocation")
    plt.xlabel("Time Step")
    
    # 图2: 市场状态 (Regime)
    # 我们把真实的 Regime 画出来，看看 Agent 有没有在 Bear 时切换策略
    regimes = base_env.regimes[:len(alpha_weights_history)]
    plt.subplot(2, 1, 2)
    plt.title("True Market Regime (0: Bull, 1: Bear)")
    plt.plot(regimes, color='black', drawstyle='steps-post', lw=1)
    plt.fill_between(range(len(regimes)), 0, regimes, color='gray', alpha=0.3)
    plt.xlabel("Time Step")
    
    plt.tight_layout()
    plt.savefig("meta_dmpo_analysis.png")
    print("📈 分析图已保存为 meta_dmpo_analysis.png")
    print("  -> 观察图1：如果 'Oracle' (蓝色) 占据主导，说明 Agent 学会了。")
    print("  -> 观察图2：如果在 'Bear' 阴影区，Agent 增加了 'Inverse' 或 'Regime' 的权重，说明它学会了择时。")

if __name__ == "__main__":
    run_test()