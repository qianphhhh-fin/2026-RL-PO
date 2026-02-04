import numpy as np
import pandas as pd
import gymnasium as gym
from gymnasium import spaces

class MetaSyntheticEnv(gym.Env):
    """
    Meta Portfolio Environment
    
    核心逻辑：
    生成真实市场回报 (True Returns) 的同时，生成 K 个虚拟的 Alpha 信号。
    RL Agent 的任务是根据市场状态，动态决定信任哪些信号，从而组合出最优的 weights。
    
    信号设定 (K=4):
    0. Oracle+:  真实收益 + 低噪声 (神级因子)
    1. Oracle-:  真实收益 * -1 + 噪声 (反指因子, 也可以利用)
    2. Noise:    纯高斯噪声 (干扰项)
    3. Regime:   牛市时准，熊市时乱猜 (模拟特定风格策略)
    """
    def __init__(self, n_assets=5, n_steps=1000, lookback=30):
        super().__init__()
        self.n_assets = n_assets
        self.n_alphas = 4  # 定义4个虚拟策略
        self.n_steps = n_steps
        self.lookback = lookback
        
        # 动作空间：RL 输出的是对 4 个 Alpha 的 "置信度权重"
        # 注意：这只是给 CvxpyLayer 的参数，不是最终仓位
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_alphas,), dtype=np.float32)
        
        # 观察空间：
        # 1. 市场特征 (Vol, Returns)
        # 2. 各个 Alpha 过去一段时间的预测值 vs 真实值 (让 RL 学会谁准)
# 3. [新增] 过去一段时间各 Alpha 的预测误差 (MSE) -> shape (n_alphas,)
        obs_dim = (self.n_assets * self.lookback) + \
                  (self.n_alphas * self.n_assets) + \
                  self.n_alphas  # <--- 新增这行，给每个 Alpha 一个评分位
        
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()

    def _generate_market_data(self):
        """生成底层资产的真实收益 (Fama-French 风格)"""
        # 简单模拟：随机游走 + Regime Switching
        returns = np.zeros((self.n_steps + 100, self.n_assets))
        regimes = np.zeros(self.n_steps + 100) # 0: Bull, 1: Bear
        
        current_price = np.ones(self.n_assets)
        current_regime = 0 
        
        for t in range(self.n_steps + 100):
            # 状态转移
            if current_regime == 0 and np.random.rand() < 0.02: current_regime = 1
            elif current_regime == 1 and np.random.rand() < 0.05: current_regime = 0
            regimes[t] = current_regime
            
            if current_regime == 0: # Bull
                mu, sigma = 0.0005, 0.01
            else: # Bear
                mu, sigma = -0.0005, 0.02
                
            r = np.random.normal(mu, sigma, self.n_assets)
            # 添加一点相关性
            market_mode = np.random.normal(0, 0.005)
            r += market_mode
            returns[t] = r
            
        return returns, regimes

    def _generate_signals(self, returns, regimes):
        """基于真实收益，生成 K 个虚拟 Alpha 信号"""
        T, N = returns.shape
        # signals shape: (T, K, N) -> 每个时间步，K个策略对N个资产的预测收益
        signals = np.zeros((T, self.n_alphas, N))
        
        for t in range(T):
            true_next_ret = returns[t] # 假设 t 时刻预测 t+1 (简化，实际要有 shift)
            
            # Alpha 0: Oracle (高相关)
            signals[t, 0, :] = true_next_ret + np.random.normal(0, 0.00001, N)
            
            # Alpha 1: Inverse (反指)
            signals[t, 1, :] = -1 * true_next_ret + np.random.normal(0, 0.005, N)
            
            # Alpha 2: Noise (纯噪声)
            signals[t, 2, :] = np.random.normal(0, 0.1, N)
            
            # Alpha 3: Regime Sensitive (牛市准，熊市瞎)
            if regimes[t] == 0: # Bull
                signals[t, 3, :] = true_next_ret + np.random.normal(0, 0.008, N)
            else:
                signals[t, 3, :] = np.random.normal(0, 0.03, N) # 瞎猜
                
        return signals

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.lookback
        
        # 1. 生成新一轮数据
        self.returns, self.regimes = self._generate_market_data()
        self.signals = self._generate_signals(self.returns, self.regimes)
        
        # 2. 初始化持仓
        self.weights = np.ones(self.n_assets) / self.n_assets
        
        return self._get_obs(), {}

    def _get_obs(self):
        # 1. 市场历史特征
        # returns shape: (Total_Steps, N_Assets)
        # 切片范围: [current_step - lookback : current_step]
        window_start = self.current_step - self.lookback
        window_end = self.current_step
        
        # 容错处理：如果刚开始，用 0 填充
        if window_start < 0:
            mkt_history = np.zeros((self.lookback, self.n_assets))
            # 简单处理：把能取的取出来，剩下的补0 (或者直接由 Reset 保证 current_step >= lookback)
            # 这里假设 Reset 已经设置 current_step = lookback，所以通常不会越界
        else:
            mkt_history = self.returns[window_start:window_end]
            
        mkt_feat = mkt_history.flatten()
        
        # 2. 当期 Alpha 信号
        alpha_feat = self.signals[self.current_step].flatten()

        # --- 修改开始: 计算成绩单 (Past Performance) ---
        # 我们要看过去 Lookback 天，谁预测得准
        # 历史真实收益: mkt_history (L, N)
        # 历史 Alpha 预测: self.signals[window_start:window_end] -> (L, K, N)
        
        # 为了广播相减，扩展 mkt_history 维度: (L, N) -> (L, 1, N)
        true_vals = mkt_history[:, np.newaxis, :]
        pred_vals = self.signals[window_start:window_end]
        
        # 计算 MSE: (Pred - True)^2 -> Mean over Time(L) and Assets(N)
        # 结果 shape: (K,) 即每个 Alpha 一个分值，越小越好
        mse_scores = np.mean(np.square(pred_vals - true_vals), axis=(0, 2))
        
        # 可选：为了让神经网络好理解，可以取负对数或者倒数，或者直接归一化
        # 这里直接传原始 MSE，由 VecNormalize 处理
        perf_feat = mse_scores.astype(np.float32)
        
        return np.concatenate([mkt_feat, alpha_feat, perf_feat])

    def step(self, action):
        """
        修改后的 Step:
        接收的 action 已经是 Wrapper 算好的【资产权重】 (N_assets,)
        """
        # 1. 接收资产权重
        self.weights = action 
        
        # 2. 计算组合收益 (简单的一期收益)
        # returns[t] 是 t 到 t+1 的收益
        true_ret = self.returns[self.current_step]
        portfolio_ret = np.sum(self.weights * true_ret)
        
        # 3. 推进时间
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        # 4. 构造 Next Obs
        next_obs = self._get_obs() if not done else np.zeros_like(self.observation_space.sample())
        
        return next_obs, portfolio_ret, done, False, {}