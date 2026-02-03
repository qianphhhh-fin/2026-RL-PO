import numpy as np
import gymnasium as gym
from gymnasium import spaces

class MarketGenerator:
    """
    生成模拟市场数据：包含牛市(Normal)和熊市(Stress)的切换
    """
    def __init__(self, n_assets=10, n_steps=3000):
        self.n_assets = n_assets
        self.n_steps = n_steps
    
    def generate(self):
        # 1. 定义两个Regime的参数
        mu_bull = np.array([0.0005] * self.n_assets) 
        sigma_bull = np.array([0.01] * self.n_assets)
        corr_bull = 0.2
        
        mu_bear = np.array([-0.001] * self.n_assets)
        sigma_bear = np.array([0.04] * self.n_assets)
        corr_bear = 0.8
        
        def get_cov(sigma, corr):
            cov = np.outer(sigma, sigma) * corr
            np.fill_diagonal(cov, sigma**2)
            return cov

        cov_bull = get_cov(sigma_bull, corr_bull)
        cov_bear = get_cov(sigma_bear, corr_bear)
        
        # 2. 生成数据
        returns = np.zeros((self.n_steps, self.n_assets))
        regimes = np.zeros(self.n_steps)
        
        current_regime = 0 
        
        for t in range(self.n_steps):
            if current_regime == 0:
                if np.random.rand() > 0.99: current_regime = 1
            else:
                if np.random.rand() > 0.95: current_regime = 0
            
            regimes[t] = current_regime
            cov = cov_bull if current_regime == 0 else cov_bear
            mu = mu_bull if current_regime == 0 else mu_bear
            
            returns[t] = np.random.multivariate_normal(mu, cov)
            
        prices = 100 * np.cumprod(1 + returns, axis=0)
        return prices, returns, regimes

class PortfolioEnv(gym.Env):
    """
    Gym环境：接收合法的权重 (Weights) 并计算收益
    """
    def __init__(self, n_assets=10, lookback=30, max_turnover=0.10):
        super().__init__()
        self.n_assets = n_assets
        self.lookback = lookback
        self.max_turnover = max_turnover
        
        self.generator = MarketGenerator(n_assets=n_assets)
        self.prices, self.returns, self.regimes = self.generator.generate()
        self.n_steps = len(self.prices)
        
        # Action: 这里的 Action 应该是已经经过 QP 约束的权重
        self.action_space = spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        
        self.observation_space = spaces.Dict({
            "market": spaces.Box(low=-np.inf, high=np.inf, shape=(lookback, n_assets), dtype=np.float32),
            "weights": spaces.Box(low=0, high=1, shape=(n_assets,), dtype=np.float32)
        })
        
        self.current_step = 0
        self.current_weights = np.ones(n_assets) / n_assets

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.prices, self.returns, self.regimes = self.generator.generate()
        self.current_step = self.lookback
        self.current_weights = np.ones(self.n_assets) / self.n_assets
        return self._get_obs(), {}

    def _get_obs(self):
        market_data = self.returns[self.current_step - self.lookback : self.current_step]
        # 保护：防止 NaN 传入网络
        market_data = np.nan_to_num(market_data, nan=0.0)
        
        return {
            "market": market_data.astype(np.float32),
            "weights": self.current_weights.astype(np.float32)
        }

    def step(self, action):
        # 0. NaN 检查 (至关重要)
        if np.isnan(action).any():
            print("⚠️ Env received NaN action! Using previous weights.")
            action = self.current_weights

        # 1. 归一化保护
        # 如果 Action 已经在 Wrapper 里处理过，这里理论上不需要，但为了鲁棒性加上
        raw_sum = np.sum(np.abs(action))
        
        if raw_sum < 1e-6: # 防止除以零
            target_weights = self.current_weights
        else:
            target_weights = np.abs(action) / raw_sum
        
        # 2. 计算实际换手率
        turnover = np.sum(np.abs(target_weights - self.current_weights))
        
        # 3. 计算收益
        transaction_cost = turnover * 0.0005
        current_return = self.returns[self.current_step]
        portfolio_return = np.sum(target_weights * current_return) - transaction_cost
        
        # 4. 更新状态
        self.current_weights = target_weights * (1 + current_return) 
        
        # 再次防止除零 (如果资产全部归零)
        weight_sum = np.sum(self.current_weights)
        if weight_sum < 1e-6:
            self.current_weights = np.ones(self.n_assets) / self.n_assets
        else:
            self.current_weights /= weight_sum
            
        self.current_step += 1
        done = self.current_step >= self.n_steps - 1
        
        reward = portfolio_return * 100 
        
        info = {
            "return": portfolio_return,
            "turnover": turnover,
            "regime": self.regimes[self.current_step]
        }
        
        return self._get_obs(), reward, done, False, info