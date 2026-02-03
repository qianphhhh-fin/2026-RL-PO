import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class FF5MarketGenerator:
    """
    基于 Fama-French 5因子模型的市场生成器
    包含: 时变Beta (Random Walk) + 牛熊转换 (Regime Switching)
    """
    def __init__(self, n_assets=10, n_steps=3000):
        self.n_assets = n_assets
        self.n_steps = n_steps
        self.n_factors = 5 # Mkt-RF, SMB, HML, RMW, CMA
        
    def generate(self):
        dt = 1/252
        
        # --- 1. 生成因子收益 (Regime Switching) ---
        # Regime 0: 牛市 (高收益, 低波动, 因子动量正)
        mu_bull = np.array([0.08, 0.03, 0.02, 0.02, 0.01]) * dt
        sigma_bull = np.array([0.15, 0.10, 0.10, 0.08, 0.08]) * np.sqrt(dt)
        
        # Regime 1: 熊市 (负收益, 高波动, 因子相关性飙升)
        mu_bear = np.array([-0.15, -0.05, 0.01, 0.01, 0.01]) * dt # 价值因子在熊市可能略好
        sigma_bear = np.array([0.30, 0.20, 0.20, 0.15, 0.15]) * np.sqrt(dt)
        
        factors = np.zeros((self.n_steps, self.n_factors))
        regimes = np.zeros(self.n_steps)
        current_regime = 0 
        
        for t in range(self.n_steps):
            # 转移概率
            prob_switch = 0.01 if current_regime == 0 else 0.05
            if np.random.rand() < prob_switch:
                current_regime = 1 - current_regime
            
            regimes[t] = current_regime
            mu = mu_bull if current_regime == 0 else mu_bear
            sigma = sigma_bull if current_regime == 0 else sigma_bear
            
            # 因子之间加一点相关性
            cov = np.diag(sigma**2)
            if current_regime == 1: # 熊市因子相关性增加
                cov += 0.5 * np.outer(sigma, sigma)
                
            factors[t] = np.random.multivariate_normal(mu, cov)
            
        # --- 2. 生成时变 Beta (Random Walk) ---
        # 初始 Beta: 围绕 1.0 (Mkt) 和 0.0 (其他) 波动
        betas = np.zeros((self.n_steps, self.n_assets, self.n_factors))
        
        # 初始化
        betas[0, :, 0] = np.random.normal(1.0, 0.2, self.n_assets) # Market Beta
        betas[0, :, 1:] = np.random.normal(0.0, 0.3, (self.n_assets, self.n_factors-1)) # Style Betas
        
        # 随机游走: beta_t = beta_{t-1} + noise
        beta_drift_std = 0.01 # 每天变动一点点
        for t in range(1, self.n_steps):
            betas[t] = betas[t-1] + np.random.normal(0, beta_drift_std, (self.n_assets, self.n_factors))
            
        # --- 3. 生成个股收益 ---
        # R_i = Beta * F + Idiosyncratic_Noise
        stock_returns = np.zeros((self.n_steps, self.n_assets))
        resid_std = 0.15 * np.sqrt(dt) # 特质波动率
        
        for t in range(self.n_steps):
            # 因子驱动部分
            systematic = np.sum(betas[t] * factors[t].reshape(1, 5), axis=1)
            # 残差部分
            residual = np.random.normal(0, resid_std, self.n_assets)
            stock_returns[t] = systematic + residual
            
        prices = 100 * np.cumprod(1 + stock_returns, axis=0)
        
        return prices, stock_returns, factors, regimes

class PortfolioEnv(gym.Env):
    def __init__(self, n_assets=10, lookback=30, max_turnover=0.10):
        super().__init__()
        self.n_assets = n_assets
        self.n_factors = 5
        self.lookback = lookback
        self.max_turnover = max_turnover
        
        self.generator = FF5MarketGenerator(n_assets=n_assets)
        # 初始化数据
        self._regenerate_data()
        
        # Action: 目标权重 (归一化前) -> 由 Wrapper 处理
        self.action_space = spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        
        # Observation: 
        # 1. Stocks: (Lookback, N)
        # 2. Factors: (Lookback, 5) -> 显式给因子，降低学习难度
        # 3. Weights: (N,)
        self.observation_space = spaces.Dict({
            "stocks": spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, n_assets), dtype=np.float32),
            "factors": spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, self.n_factors), dtype=np.float32),
            "weights": spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        })

    def _regenerate_data(self):
        self.prices, self.returns, self.factors, self.regimes = self.generator.generate()
        self.n_steps = len(self.prices)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        # 训练时每次 Reset 都在平行宇宙里重生
        self._regenerate_data()
        
        self.current_step = self.lookback
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs(), {}

    def _get_obs(self):
        # 提取窗口数据
        t = self.current_step
        stock_window = self.returns[t-self.lookback:t]
        factor_window = self.factors[t-self.lookback:t]
        
        # 数值保护
        stock_window = np.nan_to_num(stock_window, nan=0.0)
        factor_window = np.nan_to_num(factor_window, nan=0.0)
        
        return {
            "stocks": stock_window.astype(np.float32),
            "factors": factor_window.astype(np.float32),
            "weights": self.current_weights.astype(np.float32)
        }

    def step(self, action):
        # Action 是经过 Wrapper 处理后的合规权重
        # 1. 归一化 & 保护
        if np.isnan(action).any(): action = self.current_weights
        target_weights = np.abs(action) / (np.sum(np.abs(action)) + 1e-8)
        
        # 2. 计算真实换手率
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        
        # 3. 计算收益 (扣除万分之五交易成本)
        cost = turnover * 0.0005
        raw_ret = np.sum(target_weights * self.returns[self.current_step])
        portfolio_ret = raw_ret - cost
        
        # 4. 更新权重 (价格漂移)
        self.current_weights = target_weights * (1 + self.returns[self.current_step])
        self.current_weights /= (np.sum(self.current_weights) + 1e-8)
        
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        # 5. Reward: 简单收益率 (依赖 VecNormalize 缩放)
        reward = portfolio_ret 
        
        info = {
            "return": portfolio_ret,
            "turnover": turnover,
            "regime": self.regimes[self.current_step]
        }
        
        return self._get_obs(), reward, done, False, info