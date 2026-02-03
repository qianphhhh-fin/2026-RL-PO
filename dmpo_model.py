import torch as th
import torch.nn as nn
import cvxpy as cp
import numpy as np
import gymnasium as gym
from cvxpylayers.torch import CvxpyLayer
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.policies import ActorCriticPolicy

# --- 1. QP 求解器封装 (核心修复部分) ---
class QPSolver:
    def __init__(self, n_assets, max_turnover=0.10):
        self.n_assets = n_assets
        
        # Define CVXPY Variables
        self.w = cp.Variable(n_assets)
        
        # Define CVXPY Parameters (保存为 self 属性，方便后续赋值)
        self.mu_hat = cp.Parameter(n_assets)      
        self.w_prev = cp.Parameter(n_assets)      
        
        # 目标：最大化预测收益 - 风险惩罚
        # 这是一个标准的二次规划 (QP)
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
            # 1. 提高精度到 1e-6 (原来是默认或1e-4)
            self.problem.solve(solver=cp.OSQP, warm_start=True, eps_abs=1e-6, eps_rel=1e-6)
            
            if self.problem.status not in ["optimal", "optimal_inaccurate"]:
                return prev_weights 
            
            w_val = self.w.value
            
            # 2. 数值净化 (兜底逻辑)
            # 有时候 solver 会输出 -1e-10 这种负数，导致后续计算出错
            w_val = np.maximum(w_val, 0) 
            w_val = w_val / np.sum(w_val) # 重新归一化确保和为1
            
            return w_val
        except Exception as e:
            return prev_weights

# --- 2. Gym Wrapper (将 QP 层应用到环境交互中) ---
class DMPOActionWrapper(gym.ActionWrapper):
    """
    核心机制实现：
    Agent 输出 -> Unconstrained Mu ([-inf, inf])
    Wrapper -> 截获 Mu -> 调用 QP Solver -> Constrained Weights
    Env 接收 -> Weights
    """
    def __init__(self, env, max_turnover=0.10):
        super().__init__(env)
        self.solver = QPSolver(env.n_assets, max_turnover)
        # 初始化上一期权重
        self.last_obs_weights = np.ones(env.n_assets) / env.n_assets

    def action(self, action):
        # action 这里是 PPO 输出的 pred_mu (带有噪声)
        # 确保输入是 Numpy 数组且维度正确
        action = np.array(action).flatten()
        
        # 1. 获取上一期权重
        current_prev_weights = self.last_obs_weights
        
        # 2. 调用 QP
        valid_weights = self.solver.solve(action, current_prev_weights)
        
        # 3. 防止 NaN 传递给 Env
        if valid_weights is None or np.isnan(valid_weights).any():
            return current_prev_weights
            
        return valid_weights

    def step(self, action):
        # 覆写 step 以便更新 last_obs_weights
        # ActionWrapper 会自动调用上面的 .action() 方法处理 action
        obs, reward, done, truncated, info = super().step(action)
        
        # 从 obs 更新权重，供下一步 QP 使用
        # 假设 Env 返回的 obs['weights'] 是准确的
        if 'weights' in obs:
            self.last_obs_weights = obs['weights']
            
        return obs, reward, done, truncated, info

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        if 'weights' in obs:
            self.last_obs_weights = obs['weights']
        else:
             self.last_obs_weights = np.ones(self.env.n_assets) / self.env.n_assets
        return obs, info

# --- 3. 神经网络策略 (LSTM 特征提取) ---
class PortfolioExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space, features_dim=64):
        super().__init__(observation_space, features_dim)
        
        # 自动获取资产数量
        n_assets = observation_space['market'].shape[1]
        
        self.lstm = nn.LSTM(input_size=n_assets, hidden_size=64, batch_first=True)
        self.weight_encoder = nn.Linear(n_assets, 16)
        self.fusion = nn.Linear(64 + 16, features_dim)
        self.relu = nn.ReLU()

    def forward(self, observations):
        market = observations['market']   
        weights = observations['weights'] 
        
        lstm_out, _ = self.lstm(market)
        market_feat = lstm_out[:, -1, :]  
        
        weight_feat = self.relu(self.weight_encoder(weights))
        
        combined = th.cat([market_feat, weight_feat], dim=1)
        return self.relu(self.fusion(combined))