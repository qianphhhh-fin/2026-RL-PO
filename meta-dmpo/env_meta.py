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
        # 为了 MVP 简单化，我们先只给 Alpha 的当期预测值 + 历史收益特征
        obs_dim = (self.n_assets * self.lookback) + (self.n_alphas * self.n_assets)
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
            signals[t, 0, :] = true_next_ret + np.random.normal(0, 0.005, N)
            
            # Alpha 1: Inverse (反指)
            signals[t, 1, :] = -1 * true_next_ret + np.random.normal(0, 0.005, N)
            
            # Alpha 2: Noise (纯噪声)
            signals[t, 2, :] = np.random.normal(0, 0.02, N)
            
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
        # 简单 Observation: 
        # 1. 过去 Lookback 天的真实收益 (T, N) -> Flatten
        mkt_feat = self.returns[self.current_step-self.lookback:self.current_step].flatten()
        
        # 2. 当期的 Alpha 信号预测值 (K, N) -> Flatten
        #    RL 需要看到这些信号，结合 mkt_feat 来决定信谁
        alpha_feat = self.signals[self.current_step].flatten()
        
        return np.concatenate([mkt_feat, alpha_feat]).astype(np.float32)

    def step(self, action):
        """
        Action: (K,) 策略权重向量 (例如 [0.8, 0, 0.2, 0])
        这里的 Action 不是最终资产权重，而是对 Signal 的加权
        """
        # 1. 归一化 Action (Softmax or L1 normalize)
        alpha_weights = np.exp(action) / np.sum(np.exp(action))
        
        # 2. 信号融合 (Meta-Aggregation)
        # (K,) @ (K, N) -> (N,)
        current_signals = self.signals[self.current_step] # (K, N)
        blended_prediction = alpha_weights @ current_signals # (N,) 融合后的预测收益率
        
        # --- 这里通常连接 CvxpyLayer ---
        # 但为了Env能独立运行测试，我们这里做一个简单的模拟优化
        # 假设我们直接根据融合预测做 Long-Only
        # (在实际训练中，这一步是在 Network Forward 里做的，Env 只接收最终资产权重)
        # 为了兼容目前的 Step 接口，我们假设 Step 接收的是 'Alpha Weights'
        # 然后我们在 Env 内部算 PnL (这其实是简化版，正规版应该由 Agent 输出 Asset Weights)
        
        # 真正的 Meta-RL 流程：
        # Agent(Obs) -> Alpha_Weights -> CvxpyLayer(Alpha_Signals) -> Asset_Weights -> Env.step(Asset_Weights)
        
        # 为了方便 MVP，我们这里暂时假设 env.step 接收的是 Alpha_Weights，
        # 并在内部简单模拟一个 "无约束优化" (即直接按预测值排序买入)
        # *注意*：你在写 train.py 时，应该把 CvxpyLayer 放在 Policy Network 里，
        # 让 Env 接收最终 Asset Weights。但为了测试 DGP，我们先这样写。
        
        # 临时逻辑：根据融合信号直接生成仓位
        target_weights = np.maximum(blended_prediction, 0) # Long only
        if np.sum(target_weights) > 1e-6:
            target_weights /= np.sum(target_weights)
        else:
            target_weights = np.ones(self.n_assets) / self.n_assets
            
        # 3. 计算收益
        true_ret = self.returns[self.current_step]
        portfolio_ret = np.sum(target_weights * true_ret)
        
        self.current_step += 1
        done = self.current_step >= self.n_steps
        
        return self._get_obs(), portfolio_ret, done, False, {"alpha_weights": alpha_weights}