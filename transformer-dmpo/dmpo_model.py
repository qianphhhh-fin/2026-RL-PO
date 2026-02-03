import torch as th
import torch.nn as nn
import numpy as np
import cvxpy as cp
import gymnasium as gym
import math

from cvxpylayers.torch import CvxpyLayer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# --- 1. 位置编码 ---
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = th.zeros(max_len, d_model)
        position = th.arange(0, max_len, dtype=th.float).unsqueeze(1)
        div_term = th.exp(th.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = th.sin(position * div_term)
        pe[:, 1::2] = th.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]

# --- 2. Transformer 特征提取器 ---
class PortfolioTransformerExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        n_assets = observation_space['stocks'].shape[1]
        n_factors = observation_space['factors'].shape[1]
        lookback = observation_space['stocks'].shape[0]
        
        # 输入维度 = 个股数 + 因子数 (拼接处理)
        input_dim = n_assets + n_factors 
        
        d_model = 64
        nhead = 4
        num_layers = 2
        
        self.input_proj = nn.Linear(input_dim, d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_len=lookback)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=128, 
            batch_first=True, dropout=0.1
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.weight_enc = nn.Linear(n_assets, 16)
        self.fusion = nn.Linear(d_model + 16, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        # 拼接个股和因子: (Batch, T, N) + (Batch, T, 5) -> (Batch, T, N+5)
        stocks = observations['stocks']
        factors = observations['factors']
        weights = observations['weights']
        
        x = th.cat([stocks, factors], dim=2)
        
        # Transformer 流程
        x = self.input_proj(x)
        x = self.pos_encoder(x)
        x = self.transformer(x)
        
        # 取最后一个时间步 (Global Context)
        context = x[:, -1, :] # (Batch, 64)
        
        # 融合当前持仓
        w_feat = self.relu(self.weight_enc(weights))
        
        return self.relu(self.fusion(th.cat([context, w_feat], dim=1)))

# --- 3. 可微 QP 求解器 ---
class QPSolver:
    def __init__(self, n_assets, max_turnover=0.10):
        self.n_assets = n_assets
        
        # CVXPY 变量
        self.w = cp.Variable(n_assets)
        self.mu_hat = cp.Parameter(n_assets)      
        self.w_prev = cp.Parameter(n_assets)      
        
        # 目标: Max mu^T w - 0.5 * w^T w
        objective = cp.Maximize(self.mu_hat @ self.w - 0.5 * cp.sum_squares(self.w))
        
        constraints = [
            cp.sum(self.w) == 1,
            self.w >= 0,
            cp.norm(self.w - self.w_prev, 1) <= max_turnover
        ]
        
        self.problem = cp.Problem(objective, constraints)
        
    def solve(self, pred_mu, prev_weights):
        self.mu_hat.value = pred_mu
        self.w_prev.value = prev_weights
        
        try:
            # 提高精度以通过硬约束测试
            self.problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6)
            
            if self.problem.status not in ["optimal", "optimal_inaccurate"]:
                return prev_weights 
            
            # 数值净化
            val = self.w.value
            val = np.maximum(val, 0)
            val /= np.sum(val)
            return val
        except:
            return prev_weights

# --- 4. 动作包装器 (连接 Policy Output 和 QP) ---
class DMPOActionWrapper(gym.ActionWrapper):
    def __init__(self, env, max_turnover=0.10):
        super().__init__(env)
        self.solver = QPSolver(env.n_assets, max_turnover)
        self.last_weights = np.ones(env.n_assets) / env.n_assets

    def action(self, action):
        # Action 这里是 PPO 输出的 mu (View)
        action = np.array(action).flatten()
        
        # 求解 QP
        weights = self.solver.solve(action, self.last_weights)
        return weights

    def step(self, action):
        obs, reward, done, truncated, info = super().step(action)
        # 必须从 obs 更新 last_weights，因为 step 内部可能有漂移
        if 'weights' in obs:
            self.last_weights = obs['weights']
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if 'weights' in obs:
            self.last_weights = obs['weights']
        else:
            self.last_weights = np.ones(self.env.n_assets) / self.env.n_assets
        return obs, info