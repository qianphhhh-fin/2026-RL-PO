import os
import torch as th
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor

from env import PortfolioEnv
from dmpo_model import PortfolioTransformerExtractor, DMPOActionWrapper

def main():
    log_dir = "./dmpo_logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 组装环境
    def make_env():
        env = PortfolioEnv(n_assets=10, lookback=30, max_turnover=0.10)
        env = DMPOActionWrapper(env, max_turnover=0.10)
        return Monitor(env, log_dir)
    
    env = DummyVecEnv([make_env])
    
    # 2. 归一化 (关键!)
    # 金融数据通常很小 (1e-3)，必须归一化 Obs 和 Reward
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10., clip_reward=10.)
    
    # 3. 初始化 PPO
    model = PPO(
        "MultiInputPolicy",
        env,
        policy_kwargs={
            "features_extractor_class": PortfolioTransformerExtractor,
            "features_extractor_kwargs": {"features_dim": 64},
            "net_arch": dict(pi=[64, 64], vf=[64, 64]),
            "activation_fn": th.nn.Tanh 
        },
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01, # 鼓励探索
        verbose=1,
        tensorboard_log=log_dir
    )
    
    print("🚀 开始训练 Transformer-DMPO (Simulated FF5 Data)...")
    try:
        # 建议训练至少 100k 步
        model.learn(total_timesteps=100000)
        
        model.save("dmpo_transformer_agent")
        env.save("vec_normalize.pkl")
        print("✅ 模型保存成功")
    except Exception as e:
        print(f"❌ 训练失败: {e}")

if __name__ == "__main__":
    main()