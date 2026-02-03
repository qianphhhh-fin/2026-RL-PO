import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback

# 导入我们定义的 Env (假设你把上面的 Env 代码存为了 env_meta.py)
from env_meta import MetaSyntheticEnv 
from dmpo_model import MetaDMPOActionWrapper

def main():
    log_dir = "logs_meta_dmpo"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 创建并包装环境
    def make_env():
        env = MetaSyntheticEnv(n_assets=5, n_steps=2000, lookback=30)
        env = MetaDMPOActionWrapper(env) # 加上我们的 Meta 逻辑
        env = Monitor(env, log_dir)
        return env
    
    # 向量化环境
    env = DummyVecEnv([make_env])
    
    # 归一化是必须的，因为金融数据的 Reward 尺度很小
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. 定义模型 (使用简单的 MLP)
    # policy="MlpPolicy" 会自动构建几个全连接层
    model = PPO(
        "MlpPolicy", 
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01, # 稍微加点熵正则化，防止过早收敛到单一 Alpha
        tensorboard_log=log_dir
    )
    
    # 3. 训练
    print("🚀 开始训练 Meta-DMPO Agent...")
    model.learn(total_timesteps=100000) # 跑 100k 步试水
    
    # 4. 保存
    model.save("meta_dmpo_agent")
    env.save("meta_vec_normalize.pkl") # 必须保存归一化参数！
    print("✅ 模型已保存")

if __name__ == "__main__":
    main()