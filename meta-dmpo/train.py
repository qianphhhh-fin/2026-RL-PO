# train.py
import os
import numpy as np
# from stable_baselines3 import PPO,SAC
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.monitor import Monitor
# 引入 BaseCallback 用于自定义记录
from stable_baselines3.common.callbacks import BaseCallback 

from env_meta import MetaSyntheticEnv 
from dmpo_model import MetaDMPOActionWrapper

# --- 新增：自定义 Callback 类 ---
class OracleWeightLogger(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
    
    def _on_step(self) -> bool:
        # self.locals['infos'] 包含了当前步所有环境的 info 列表
        infos = self.locals['infos']
        
        # 遍历所有环境（通常 DummyVecEnv 只有一个环境）
        for info in infos:
            if 'alpha_weights' in info:
                weights = info['alpha_weights']
                # 记录 Oracle (Index 0) 的权重
                # "custom/oracle_weight" 会出现在 TensorBoard 的 custom 标签下
                self.logger.record("custom/oracle_weight", weights[0])
                
                # 如果你想看反指策略 (Index 1) 的权重也可以加上：
                self.logger.record("custom/inverse_weight", weights[1])
                
                # 如果你想看 Noise (Index 2)
                self.logger.record("custom/noise_weight", weights[2])

                # 如果你想看 Regime (Index 3)
                self.logger.record("custom/regime_weight", weights[3])

        return True

def main():
    log_dir = "logs_meta_dmpo" # 改个名字区分
    os.makedirs(log_dir, exist_ok=True)
    
    # 1. 环境设置 (保持不变)
    def make_env():
        # 建议配合之前的修改：风险中性 Solver + 增加 MSE 特征
        env = MetaSyntheticEnv(n_assets=5, n_steps=2000, lookback=30)
        env = MetaDMPOActionWrapper(env)
        env = Monitor(env, log_dir)
        return env
    
    env = DummyVecEnv([make_env])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)
    
    # 2. 定义 SAC 模型
    # SAC 不需要 n_steps (它是 Off-policy)，而是用 buffer_size
    model = SAC(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=3e-4,
        buffer_size=100000, # 经验回放池大小
        batch_size=256,     # SAC 通常用大一点的 Batch
        ent_coef='auto',    # 关键！让它自动调整探索力度
        policy_kwargs=dict(net_arch=[64,64]), 
        train_freq=4,       # 每个 step 都训练
        gradient_steps=4,   # 每次更新一步
        tensorboard_log=log_dir,
        device='cpu'  # 如果有 GPU 可以改成 'cuda'
    )
    
    print("🚀 开始训练 Meta-DMPO (SAC 版)...")
    
    # 因为 SAC 训练更慢（每个 step 都反向传播），同样的 total_timesteps 会比 PPO 慢
    # 但它的收敛通常需要更少的 steps
    model.learn(total_timesteps=200000, callback=OracleWeightLogger(),log_interval=1) 
    
    model.save("meta_dmpo_sac")
    env.save("meta_vec_normalize_sac.pkl")

if __name__ == "__main__":
    main()