import gymnasium as gym
import numpy as np
import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

# 导入自定义模块
from env import PortfolioEnv
from dmpo_model import PortfolioExtractor, DMPOActionWrapper

# 创建日志目录
log_dir = "./dmpo_tensorboard/"
os.makedirs(log_dir, exist_ok=True)

def make_env():
    # 1. 基础环境
    env = PortfolioEnv(n_assets=10, lookback=30, max_turnover=0.10)
    # 2. 包装 QP 约束 (Wrapper)
    env = DMPOActionWrapper(env, max_turnover=0.10)
    # 3. 监控器 (记录未归一化的真实 Reward)
    env = Monitor(env, log_dir)
    return env

# 1. 使用 DummyVecEnv 包装 (PPO 标准用法)
env = DummyVecEnv([make_env])

# 2. ⚡️ 关键修复：输入与奖励归一化 ⚡️
# VecNormalize 会自动计算运行均值和方差，把 Obs 和 Reward 缩放到标准正态分布
# clip_obs=10, clip_reward=10 防止异常值破坏模型
env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)

print("🚀 开始训练 DMPO 模型 (Improved)...")

# 3. 初始化 PPO (增加 entropy_coef 鼓励探索)
model = PPO(
    "MultiInputPolicy",
    env,
    policy_kwargs={
        "features_extractor_class": PortfolioExtractor,
        "features_extractor_kwargs": {"features_dim": 64},
        "net_arch": dict(pi=[64, 64], vf=[64, 64]),
        # 使用 Tanh 激活函数通常在连续控制中更稳定
        "activation_fn": th.nn.Tanh 
    },
    learning_rate=3e-4,
    n_steps=2048,
    batch_size=64,
    ent_coef=0.01, # 增加熵正则化，防止过早收敛到局部最优(比如一直持有现金)
    verbose=1,
    tensorboard_log=log_dir
)

try:
    # 增加训练步数：50k 可能太少，建议 100k+
    model.learn(total_timesteps=100000)
    
    # 保存模型时，必须同时也保存 VecNormalize 的统计数据！
    # 否则测试时无法正确归一化输入
    model.save("./model/dmpo_agent_fixed")
    env.save("./model/vec_normalize.pkl")
    print("✅ 模型与归一化参数已保存。")
except Exception as e:
    print(f"❌ 训练中断: {e}")
    import traceback
    traceback.print_exc()