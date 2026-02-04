import gymnasium as gym
import numpy as np
import os
import torch
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor, VecNormalize
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.utils import set_random_seed
from sbx import SAC
from meta_env import MetaExecutionEnv

# --- 1. 奖励缩放 Wrapper ---
class RewardScaleWrapper(gym.RewardWrapper):
    def __init__(self, env, scale=100.0):
        super().__init__(env)
        self.scale = scale
        
    def reward(self, reward):
        return reward * self.scale

# --- 2. 效率监控 Callback ---
class EfficiencyCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)

    def _on_step(self) -> bool:
        # 降低频率到 1000 step 记录一次，减少开销
        if self.n_calls % 1000 == 0:
            infos = self.locals['infos']
            effs = []
            regrets = []
            lambdas = []
            gammas = []
            
            for info in infos:
                if 'gt_net_return' in info and 'net_return' in info:
                    # 避免除以极小值
                    if abs(info['gt_net_return']) > 1e-5:
                        effs.append(info['net_return'] / info['gt_net_return'])
                    
                    regrets.append(info['regret'])
                    lambdas.append(info['risk_aversion'])
                    gammas.append(info['trade_penalty'])

            if effs:
                self.logger.record("custom/efficiency_vs_god", np.mean(effs))
            if regrets:
                self.logger.record("custom/regret_mean", np.mean(regrets))
            if lambdas:
                self.logger.record("params/risk_aversion", np.mean(lambdas))
            if gammas:
                self.logger.record("params/trade_penalty", np.mean(gammas))
        return True

def make_env(rank, seed=0):
    def _init():
        env = MetaExecutionEnv(n_assets=5, episode_length=50, cost_rate=0.0005)
        # 核心修改：放大奖励
        env = RewardScaleWrapper(env, scale=100.0) 
        env.reset(seed=seed + rank)
        return env
    return _init

if __name__ == "__main__":
    # 配置参数
    N_ENVS = 8
    TOTAL_TIMESTEPS = 300_000 
    
    # 保持原来的 log 目录不变
    LOG_DIR = "./logs/meta_sac_sbx/"
    # 保持原来的模型名字不变
    MODEL_PATH = "meta_execution_sac_sbx"
    
    os.makedirs(LOG_DIR, exist_ok=True)
    set_random_seed(42)

    print(f"🚀 启动 {N_ENVS} 个并行环境 (Reward Scaled x100)...")

    # --- 1. 创建环境 ---
    env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    
    # --- 2. 核心修改：自动归一化 Observation ---
    # norm_reward=False 是因为我们已经手动 Scale 了
    # clip_obs=10.0 防止异常值干扰网络
    env = VecNormalize(env, norm_obs=True, norm_reward=False, clip_obs=10.)
    
    # 监控器
    env = VecMonitor(env, LOG_DIR)

    # --- 3. 初始化模型 ---
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100_000,
        batch_size=256,
        # 核心修改：固定 Entropy 为 0.05，强制它探索，不准躺平
        ent_coef='auto', 
        gamma=0.99,
        tau=0.005,
        tensorboard_log=LOG_DIR,
        policy_kwargs=dict(net_arch=[256, 256]),
    )

    print("🏃 开始训练 (Fixed Entropy + Normalized Obs)...")
    
    # 组合 Callbacks (效率监控 + 自动保存Checkpoint防止意外)
    eff_callback = EfficiencyCallback()
    checkpoint_callback = CheckpointCallback(save_freq=50000, save_path=LOG_DIR, name_prefix='ckpt')
    
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=CallbackList([eff_callback, checkpoint_callback]), progress_bar=True)

    print(f"✅ 训练完成，保存模型至 {MODEL_PATH}...")
    model.save(MODEL_PATH)
    
    # 重要：必须保存 VecNormalize 的统计数据 (均值和方差)，否则测试时模型就是瞎子
    env.save(f"{MODEL_PATH}_vecnorm.pkl")
    
    env.close()