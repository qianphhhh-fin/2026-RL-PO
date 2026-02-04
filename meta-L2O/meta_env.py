import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from market_dynamics import DataGenerator, AlphaModel

class MetaExecutionEnv(gym.Env):
    """
    v2.0 改动：
    1. Action 映射改为 Log-Space (对数空间)，覆盖数量级差异。
    2. Reward 移除人工风险惩罚，改为纯净收益 (Net Return)，让市场波动自然惩罚风险。
    """
    def __init__(self, n_assets=5, episode_length=100, cost_rate=0.0005):
        super().__init__()
        self.n_assets = n_assets
        self.episode_length = episode_length
        self.cost_rate = cost_rate
        
        self.data_gen = DataGenerator(n_assets)
        self.alpha_model = AlphaModel(n_assets)
        
        # Action: [log_lambda, log_gamma, kappa]
        # 范围 [-1, 1]
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # Obs: [Signals, Vols, Holdings, Macro]
        obs_dim = (4 * n_assets) + n_assets + n_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        self.reset()
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.real_returns, self.sigmas, regimes = self.data_gen.generate_episode(self.episode_length)
        self.signals = self.alpha_model.generate_signals(self.real_returns, regimes, self.sigmas)
        
        # 计算 Ground Truth (Risk-Adjusted) 仅作为参考标尺，不参与 Reward
        self.ground_truth_w = self._calculate_risk_adjusted_ground_truth(
            self.real_returns, self.sigmas, risk_aversion=5.0
        )
        
        self.t = 0
        self.prev_w = np.ones(self.n_assets) / self.n_assets
        self.cumulative_return = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # --- 1. Log-Space Action Mapping (关键修改) ---
        # 原始动作范围 [-1, 1]
        
        # Lambda (风险厌恶): 映射到 [0.01, 100]
        # action[0]= -1 -> exp(-4.6) ≈ 0.01 (极度激进)
        # action[0]=  1 -> exp(4.6)  ≈ 100  (极度保守)
        risk_aversion = np.exp(action[0] * 4.6)
        
        # Gamma (交易惩罚): 映射到 [0.0001, 1.0]
        # action[1] 控制换手意愿
        trade_penalty = np.exp(action[1] * 4.6 - 4.6) # range approx [0.0001, 1.0]
        
        # Kappa (信号置信度): [0, 2] 保持线性
        alpha_scale = action[2] + 1.0 
        
        # --- 2. Solver ---
        raw_mu = np.mean(self.signals[self.t], axis=0) 
        blended_mu = raw_mu * alpha_scale
        current_sigma = self.sigmas[self.t]
        
        target_w = self._solve_cvxpy(blended_mu, current_sigma, risk_aversion, trade_penalty, self.prev_w)
        
        # --- 3. Reward (关键修改: 纯利模式) ---
        r_t = self.real_returns[self.t]
        turnover = np.sum(np.abs(target_w - self.prev_w))
        cost = turnover * self.cost_rate
        
        gross_ret = np.dot(target_w, r_t)
        net_ret = gross_ret - cost
        
        # 移除人工定义的风险惩罚 (- lambda * risk)
        # 理由：如果 Agent 风险太大，遇到 Crash Regime 自然会巨亏，不需要我们替它操心。
        # 我们希望 Agent 追求的是扣费后的“真金白银”。
        reward = net_ret 
        
        # --- 4. Info & Update ---
        self.prev_w = target_w
        self.t += 1
        
        # 计算 Regret
        t_idx = min(self.t, self.episode_length - 1)
        gt_w = self.ground_truth_w[t_idx-1]
        gt_turnover = np.sum(np.abs(gt_w - (self.ground_truth_w[t_idx-2] if t_idx > 1 else np.ones(self.n_assets)/self.n_assets)))
        gt_net_ret = np.dot(gt_w, r_t) - gt_turnover * self.cost_rate
        
        truncated = False
        terminated = (self.t >= self.episode_length)
        
        info = {
            "risk_aversion": risk_aversion,
            "trade_penalty": trade_penalty,
            "net_return": net_ret,
            "gt_net_return": gt_net_ret,
            "regret": gt_net_ret - net_ret
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        t_idx = min(self.t, self.episode_length - 1)
        signals_flat = self.signals[t_idx].flatten()
        vols = np.sqrt(np.diag(self.sigmas[t_idx]))
        holdings = self.prev_w
        macro = np.array([np.mean(vols), 0.5]) 
        return np.concatenate([signals_flat, vols, holdings, macro]).astype(np.float32)

    def _solve_cvxpy(self, mu, sigma, risk_aversion, trade_penalty, w_prev):
        w = cp.Variable(self.n_assets)
        # 目标函数：收益 - 风险 - 成本
        # 即使 Reward 移除了风险项，Solver 内部必须保留风险项，否则就是线性规划(全仓一只票)
        obj = w @ mu - (risk_aversion / 2) * cp.quad_form(w, sigma) - trade_penalty * cp.norm(w - w_prev, 1)
        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0])
        try:
            prob.solve(solver=cp.ECOS, verbose=False) # ECOS 依然是最稳的
            if w.value is None: return w_prev
            return w.value
        except:
            return w_prev

    def _calculate_risk_adjusted_ground_truth(self, returns, sigmas, risk_aversion):
        # GT 计算保持不变，作为参考系
        T, N = returns.shape
        W = cp.Variable((T, N))
        w_prev = np.ones(N) / N
        objs = []
        for t in range(T):
            ret = W[t] @ returns[t]
            cost = cp.norm(W[t] - w_prev, 1) * self.cost_rate
            risk = (risk_aversion / 2) * cp.quad_form(W[t], sigmas[t])
            objs.append(ret - cost - risk)
            w_prev = W[t]
        prob = cp.Problem(cp.Maximize(cp.sum(objs)), [cp.sum(W, axis=1)==1, W>=0])
        try:
            prob.solve(solver=cp.ECOS)
        except:
            prob.solve(solver=cp.SCS)
        return W.value if W.value is not None else np.ones((T, N))/N