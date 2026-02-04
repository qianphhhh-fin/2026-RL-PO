import numpy as np

class DataGenerator:
    """
    生成具有 Regime Switching (三状态马尔可夫) 特性的市场数据。
    State 0: Bull (低波, 低相关, 高收益)
    State 1: Choppy (中波, 中相关, 零收益)
    State 2: Crash (高波, 高相关, 负收益)
    """
    def __init__(self, n_assets=5, seed=None):
        self.n_assets = n_assets
        self.rng = np.random.default_rng(seed)
        
        # 参数设定
        self.means = {0: 0.0005, 1: 0.0000, 2: -0.0020}
        self.vols =  {0: 0.0080, 1: 0.0150, 2: 0.0300}
        self.corrs = {0: 0.2000, 1: 0.5000, 2: 0.9000}
        
        # 状态转移矩阵 (Sticky Regime)
        self.trans_mat = np.array([
            [0.95, 0.04, 0.01],
            [0.10, 0.85, 0.05],
            [0.10, 0.00, 0.90], 
        ])
        
    def generate_episode(self, n_steps):
        """一次性生成一个完整的 Episode 数据"""
        returns = np.zeros((n_steps, self.n_assets))
        sigmas = np.zeros((n_steps, self.n_assets, self.n_assets))
        regimes = np.zeros(n_steps, dtype=int)
        
        current_regime = 0
        
        for t in range(n_steps):
            # 1. 状态转移
            current_regime = self.rng.choice(3, p=self.trans_mat[current_regime])
            regimes[t] = current_regime
            
            # 2. 构建协方差矩阵 Sigma = D * R * D
            vol_base = self.vols[current_regime]
            corr_base = self.corrs[current_regime]
            
            # 随机扰动个股波动率
            asset_vols = vol_base * (1 + self.rng.normal(0, 0.1, self.n_assets))
            D = np.diag(asset_vols)
            
            # 构建相关系数矩阵
            R = np.full((self.n_assets, self.n_assets), corr_base)
            np.fill_diagonal(R, 1.0)
            
            Sigma = D @ R @ D
            sigmas[t] = Sigma
            
            # 3. 生成收益率
            mu_base = self.means[current_regime]
            mu_vec = mu_base + self.rng.normal(0, 0.0001, self.n_assets) # 个股漂移
            returns[t] = self.rng.multivariate_normal(mu_vec, Sigma)
            
        return returns, sigmas, regimes

class AlphaModel:
    """
    模拟 4 种有缺陷的 Alpha 策略
    """
    def __init__(self, n_assets, rng=None):
        self.n_assets = n_assets
        self.rng = rng if rng else np.random.default_rng()
        
    def generate_signals(self, returns, regimes, sigmas):
        """
        基于未来收益(Oracle)或历史价格(Mom/MR)生成预测信号 mu_pred
        returns: (T, N)
        """
        T, N = returns.shape
        n_alphas = 4
        signals = np.zeros((T, n_alphas, N))
        
        # 模拟价格路径 (用于计算技术指标)
        prices = np.zeros((T + 100, N))
        prices[0] = 100.0
        # 预热数据
        pre_returns = self.rng.normal(0, 0.01, (99, N))
        for t in range(99):
            prices[t+1] = prices[t] * (1 + pre_returns[t])
            
        # 开始生成
        for t in range(T):
            # 更新当期价格
            prices[t+100] = prices[t+99] * (1 + returns[t])
            
            curr_vol = np.sqrt(np.diag(sigmas[t]))
            curr_regime = regimes[t]
            
            # --- Alpha 0: Oracle (基本面) ---
            # 崩盘时 IC 极低
            ic = 0.15 if curr_regime != 2 else 0.01
            noise = self.rng.normal(0, curr_vol * (1-ic) * 2.0)
            signals[t, 0] = returns[t] * ic + noise
            
            # --- Alpha 1: Momentum (动量) ---
            # 过去 20 天收益
            past_20 = prices[t+99] / prices[t+99-20] - 1
            z_score = (past_20 - np.mean(past_20)) / (np.std(past_20) + 1e-6)
            # 在震荡市(1)失效(负IC)，在牛市(0)和崩盘(2)有效
            mom_ic = 0.05 if curr_regime != 1 else -0.02
            signals[t, 1] = z_score * curr_vol * mom_ic
            
            # --- Alpha 2: Mean Reversion (反转) ---
            # 过去 5 天收益负号
            past_5 = prices[t+99] / prices[t+99-5] - 1
            z_score_mr = -(past_5 - np.mean(past_5)) / (np.std(past_5) + 1e-6)
            # 在震荡市(1)有效，趋势市失效
            mr_ic = 0.05 if curr_regime == 1 else -0.02
            signals[t, 2] = z_score_mr * curr_vol * mr_ic
            
            # --- Alpha 3: Noise ---
            signals[t, 3] = self.rng.normal(0, curr_vol * 0.05)
            
        return signals