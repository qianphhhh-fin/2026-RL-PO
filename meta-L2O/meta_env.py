import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cvxpy as cp
from market_dynamics import DataGenerator, AlphaModel

class MetaExecutionEnv(gym.Env):
    """
    RL 调参器环境:
    Action: [lambda_risk, gamma_trade, kappa_conf] -> Solver -> Weights
    Reward: Realized Return - Realized Cost - Risk Penalty
    """
    def __init__(self, n_assets=5, episode_length=100, cost_rate=0.0005):
        super().__init__()
        self.n_assets = n_assets
        self.episode_length = episode_length
        self.cost_rate = cost_rate
        
        # 核心组件
        self.data_gen = DataGenerator(n_assets)
        self.alpha_model = AlphaModel(n_assets)
        
        # 动作空间: [lambda, gamma, kappa] (连续值 -1 到 1，内部映射)
        self.action_space = spaces.Box(low=-1, high=1, shape=(3,), dtype=np.float32)
        
        # 观察空间: 
        # [Alpha预测(N*4), 波动率(N), 历史持仓(N), 宏观特征(2)] -> 简化版
        # 这里为了演示，我们flatten所有特征
        obs_dim = (4 * n_assets) + n_assets + n_assets + 2
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        
        # 缓存数据
        self.real_returns = None
        self.sigmas = None
        self.signals = None
        self.ground_truth_returns = None
        
        # 状态变量
        self.t = 0
        self.prev_w = None
        self.cumulative_return = 0
        
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # 1. 生成整个 Episode 的数据
        self.real_returns, self.sigmas, regimes = self.data_gen.generate_episode(self.episode_length)
        self.signals = self.alpha_model.generate_signals(self.real_returns, regimes, self.sigmas)
        
        # 2. 计算 Risk-Adjusted Ground Truth (上帝视角)
        # 这是一个耗时操作，但在训练中可以接受，或者可以缓存
        self.ground_truth_w = self._calculate_risk_adjusted_ground_truth(
            self.real_returns, self.sigmas, risk_aversion=5.0
        )
        
        # 3. 初始化状态
        self.t = 0
        self.prev_w = np.ones(self.n_assets) / self.n_assets # 初始均仓
        self.cumulative_return = 0
        
        return self._get_obs(), {}
    
    def step(self, action):
        # 1. Action 映射 (Meta-Parameters)
        # lambda: 0.1 ~ 20.0
        risk_aversion = 0.1 + 19.9 * (0.5 * (action[0] + 1)) 
        # gamma: 0.0 ~ 0.10 (0 ~ 1000 bps)
        trade_penalty = 0.10 * (0.5 * (action[1] + 1))
        # kappa: 0.0 ~ 2.0 (Alpha 置信度缩放)
        alpha_scale = 2.0 * (0.5 * (action[2] + 1))
        
        # 2. 准备 Solver 输入
        # 简单平均混合 Alpha (RL 可以学着输出不同 Alpha 的权重，这里简化为统一缩放)
        # 真实的实现里，RL Action 应该包含对每个 Alpha 的权重
        raw_mu = np.mean(self.signals[self.t], axis=0) 
        blended_mu = raw_mu * alpha_scale
        current_sigma = self.sigmas[self.t]
        
        # 3. 调用 Inner Solver (Myopic Convex Optimization)
        target_w = self._solve_cvxpy(blended_mu, current_sigma, risk_aversion, trade_penalty, self.prev_w)
        
        # 4. 计算真实环境反馈 (Realized Reward)
        r_t = self.real_returns[self.t]
        
        # 真实交易成本 (固定费率)
        turnover = np.sum(np.abs(target_w - self.prev_w))
        cost = turnover * self.cost_rate
        
        # 组合收益
        gross_ret = np.dot(target_w, r_t)
        net_ret = gross_ret - cost
        
        # Reward 设计: 追求高夏普，惩罚大回撤
        # 使用 Risk-Adjusted Return 作为 Reward
        # 这里的 5.0 是一个基准风险厌恶，引导 RL 逼近 Ground Truth 的行为
        reward = net_ret - (5.0 / 2) * np.dot(target_w.T, np.dot(current_sigma, target_w))
        
        # 5. 更新状态
        self.prev_w = target_w
        self.cumulative_return += net_ret
        self.t += 1
        
        # 6. 计算 Regret (可选，用于 Info)
        # Ground Truth 在这一步的理论收益
        gt_w = self.ground_truth_w[self.t-1]
        gt_turnover = np.sum(np.abs(gt_w - (self.ground_truth_w[self.t-2] if self.t > 1 else np.ones(self.n_assets)/self.n_assets)))
        gt_net_ret = np.dot(gt_w, r_t) - gt_turnover * self.cost_rate
        regret = gt_net_ret - net_ret
        
        truncated = False
        terminated = (self.t >= self.episode_length)
        
        info = {
            "risk_aversion": risk_aversion,
            "trade_penalty": trade_penalty,
            "turnover": turnover,
            "net_return": net_ret,
            "gt_net_return": gt_net_ret,
            "regret": regret
        }
        
        return self._get_obs(), reward, terminated, truncated, info

    def _get_obs(self):
        # --- 修复开始: 索引安全锁 ---
        # 如果 self.t 已经增加到了 50 (episode结束)，我们强制读取索引 49 的数据
        # 这样能保证 terminated=True 时也能返回一个合法的 observation
        t_idx = min(self.t, self.episode_length - 1)
        # --- 修复结束 ---

        # 1. Alpha 信号 (N*4)
        signals_flat = self.signals[t_idx].flatten()
        # 2. 波动率 (N)
        vols = np.sqrt(np.diag(self.sigmas[t_idx]))
        # 3. 当前持仓 (N)
        holdings = self.prev_w
        # 4. 简单宏观特征 (例如平均波动率)
        mean_vol = np.mean(vols)
        # 占位符
        macro = np.array([mean_vol, 0.5]) 
        
        return np.concatenate([signals_flat, vols, holdings, macro]).astype(np.float32)

    def _solve_cvxpy(self, mu, sigma, risk_aversion, trade_penalty, w_prev):
        """Myopic Solver (内层优化器)"""
        w = cp.Variable(self.n_assets)
        
        # 目标: mu^T w - lambda * w^T Sigma w - gamma * ||w - w_prev||_1
        # 注意: cvxpy 的 quad_form 是凸的，前面要减号
        # 为了速度，可以开启 parameter 编译，这里为清晰直接重构
        
        obj = w @ mu \
              - (risk_aversion / 2) * cp.quad_form(w, sigma) \
              - trade_penalty * cp.norm(w - w_prev, 1)
              
        prob = cp.Problem(cp.Maximize(obj), [cp.sum(w) == 1, w >= 0])
        
        try:
            prob.solve(solver=cp.ECOS, verbose=False) # 或 OSQP
            if w.value is None:
                return w_prev # Solver 失败，保持不动
            return w.value
        except:
            return w_prev

    def _calculate_risk_adjusted_ground_truth(self, returns, sigmas, risk_aversion):
        """
        计算全局最优路径 (上帝视角)
        Max sum( r_t*w_t - cost*|dw| - lambda/2 * risk )
        """
        T, N = returns.shape
        W = cp.Variable((T, N))
        w_prev_init = np.ones(N) / N
        
        objs = []
        w_prev = w_prev_init
        
        for t in range(T):
            ret = W[t] @ returns[t]
            cost = cp.norm(W[t] - w_prev, 1) * self.cost_rate
            risk = (risk_aversion / 2) * cp.quad_form(W[t], sigmas[t])
            objs.append(ret - cost - risk)
            w_prev = W[t]
            
        prob = cp.Problem(cp.Maximize(cp.sum(objs)), [cp.sum(W, axis=1)==1, W>=0])
        
        # 求解大问题，可能需要几秒
        try:
            prob.solve(solver=cp.ECOS) # ECOS 擅长 SOCP
        except:
            prob.solve(solver=cp.SCS)
            
        if W.value is None:
            # Fallback
            return np.ones((T, N)) / N
            
        return W.value