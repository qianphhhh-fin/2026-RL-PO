import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import SAC  # <--- 修改 1: 导入 SAC
# from sbx import SAC  # 如果你用的是 sbx 版本的 Stable Baselines3
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env_meta import MetaSyntheticEnv
from dmpo_model import MetaDMPOActionWrapper

def run_test():
    # 1. 准备测试环境
    # 务必保证这里的参数与训练时完全一致
    raw_env = MetaSyntheticEnv(n_assets=5, n_steps=1000, lookback=30)
    env = MetaDMPOActionWrapper(raw_env)
    env = DummyVecEnv([lambda: env])
    
    # 2. 加载归一化参数 (非常重要！)
    # <--- 修改 2: 加载 SAC 对应的归一化文件
    # 如果你训练时用了 save("meta_vec_normalize_sac.pkl")，这里就要对应
    try:
        env = VecNormalize.load("meta_vec_normalize_sac.pkl", env)
    except FileNotFoundError:
        print("⚠️ 警告: 找不到归一化参数文件，正在尝试使用无归一化环境（可能会导致效果极差）...")
        # 如果找不到文件，就用原始环境（仅用于 Debug，实际效果通常不好）
    
    env.training = False # 测试模式，冻结均值方差更新
    env.norm_reward = False
    
    # 3. 加载 SAC 模型
    # <--- 修改 3: 加载 SAC 模型文件
    model = SAC.load("meta_dmpo_sac")
    
    print("📊 开始 SAC 模型测试...")
    obs = env.reset()
    
    # 记录数据
    alpha_weights_history = []
    
    # 拿到底层环境引用用于画图
    base_env = env.envs[0].env.unwrapped
    
    done = False
    step_idx = 0
    
    while not done:
        # 预测动作
        # deterministic=True 会输出均值（Mean），去除随机性，适合测试
        action, _ = model.predict(obs, deterministic=True)
        
        obs, rewards, dones, infos = env.step(action)
        
        # 记录 Alpha 权重
        # 这里的 alpha_weights 已经是 Wrapper 经过 Scaling + Softmax 后的结果
        if 'alpha_weights' in infos[0]:
            alpha_weights = infos[0]['alpha_weights']
            alpha_weights_history.append(alpha_weights)
        else:
            # 兜底：如果 info 没传出来，可能是 Wrapper 没写好，打印个空占位
            alpha_weights_history.append(np.zeros(4))
        
        step_idx += 1
        # 防止无限循环（虽然 env 有 n_steps 限制）
        if dones[0]: 
            break
            
    # --- 可视化分析 ---
    alpha_weights_history = np.array(alpha_weights_history)
    
    plt.figure(figsize=(12, 10))
    
    # 图1: Alpha 权重分配热力图
    plt.subplot(3, 1, 1)
    plt.title("Meta-Agent (SAC) Decision: Trust Distribution")
    # 假设顺序: 0:Oracle, 1:Inverse, 2:Noise, 3:Regime
    labels = ["Oracle", "Inverse", "Noise", "Regime"]
    
    # 使用 stackplot 堆叠图查看占比
    plt.stackplot(range(len(alpha_weights_history)), alpha_weights_history.T, labels=labels, alpha=0.8)
    plt.legend(loc='upper left')
    plt.ylabel("Weight Allocation")
    plt.xlabel("Time Step")
    
    # 图2: 重点展示 Oracle 权重 (单线图)
    plt.subplot(3, 1, 2)
    plt.title("Oracle Weight Trajectory (Did it learn to trust?)")
    plt.plot(alpha_weights_history[:, 0], color='blue', label='Oracle Weight', linewidth=2)
    plt.axhline(y=0.9, color='r', linestyle='--', alpha=0.3, label='Target (>0.9)')
    plt.ylabel("Oracle Weight")
    plt.legend()
    
    # 图3: 市场状态 (Regime)
    # 看看在不同市场状态下，Agent 行为是否有变化
    if hasattr(base_env, 'regimes'):
        regimes = base_env.regimes[:len(alpha_weights_history)]
        plt.subplot(3, 1, 3)
        plt.title("True Market Regime (0: Bull, 1: Bear)")
        plt.plot(regimes, color='black', drawstyle='steps-post', lw=1)
        plt.fill_between(range(len(regimes)), 0, regimes, color='gray', alpha=0.3)
        plt.xlabel("Time Step")
    
    plt.tight_layout()
    plt.savefig("meta_dmpo_sac_analysis.png")
    print("📈 分析图已保存为 meta_dmpo_sac_analysis.png")
    
    # 简单的控制台统计
    avg_oracle_weight = np.mean(alpha_weights_history[:, 0])
    print(f"🏆 测试集平均 Oracle 权重: {avg_oracle_weight:.4f}")
    if avg_oracle_weight > 0.8:
        print("✅ 成功！Agent 已经学会了重仓 Oracle。")
    else:
        print("⚠️ 还有提升空间，检查是否测试集数据分布与训练集差异过大。")

if __name__ == "__main__":
    run_test()