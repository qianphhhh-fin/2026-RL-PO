import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3 import PPO  # 回归标准 PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from meta_env import MetaExecutionEnv

# 奖励放大依然需要，但因为移除了 Penalty，现在 Reward 应该是正负都有，量级在 1e-3 左右
# 放大 1000 倍，让它在 [-1, 1] 左右波动
class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=1000.0):
        super().__init__(env)
        self.scale = scale
    def reward(self, reward):
        return reward * self.scale

class EfficiencyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    def _on_step(self) -> bool:
        if self.n_calls % 1000 == 0:
            infos = self.locals['infos']
            effs = []
            regrets = []
            lambdas = []
            for info in infos:
                if 'gt_net_return' in info and 'net_return' in info:
                    if abs(info['gt_net_return']) > 1e-5:
                        effs.append(info['net_return'] / info['gt_net_return'])
                    regrets.append(info['regret'])
                    lambdas.append(info['risk_aversion'])
            if effs: self.logger.record("custom/efficiency", np.mean(effs))
            if regrets: self.logger.record("custom/regret", np.mean(regrets))
            if lambdas: self.logger.record("params/risk_aversion", np.mean(lambdas))
        return True

def make_env(rank, seed=0):
    def _init():
        env = MetaExecutionEnv(n_assets=5, episode_length=50, cost_rate=0.0005)
        env = RewardScaleWrapper(env, scale=1000.0) 
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    N_ENVS = 16  # 尽量多开核，cvxpy 比较慢
    TOTAL_TIMESTEPS = 500_000 
    LOG_DIR = "./logs/meta_ppo_v3/"
    MODEL_PATH = "meta_execution_ppo_v3"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    set_random_seed(42)

    print(f"🚀 启动 {N_ENVS} 个并行环境 (PPO + LogSpace + PureReward)...")

    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    # 依然使用 VecNormalize
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    env = VecMonitor(env, LOG_DIR)

    # --- PPO 配置 ---
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        ent_coef=0.01,  # PPO 的 entropy 更好控。0.01 足够维持探索
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]), # 独立网络
    )

    print("🏃 开始训练...")
    callbacks = CallbackList([EfficiencyCallback(), CheckpointCallback(save_freq=50000, save_path=LOG_DIR)])
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks, progress_bar=True,log_interval=10)

    print(f"✅ 保存模型: {MODEL_PATH}")
    model.save(MODEL_PATH)
    env.save(f"{MODEL_PATH}_vecnorm.pkl")
    env.close()