import numpy as np
import cvxpy as cp
import gymnasium as gym

# 注意：在 PPO (Policy Gradient) 框架下，我们不需要 CvxpyLayer 的反向传播梯度，
# 因为 PPO 将环境视为黑盒。只有在使用 DDPG/SAC 等算法做 End-to-End 优化时才需要 cvxpylayers。
# 为了保持 Demo 简单且稳定，我们在 Wrapper 里直接用 cvxpy 求解。

class MetaQPSolver:
    """
    负责将 RL 输出的 'Alpha 混合权重' 转换为最终的 '资产持仓权重'
    """
    def __init__(self, n_assets, risk_aversion=1.0):
        self.n_assets = n_assets
        self.risk_aversion = risk_aversion

    def solve(self, blended_mu, covariance_matrix):
        """
        求解标准的马科维茨优化:
        Max w.T * mu - gamma * w.T * Sigma * w
        s.t. sum(w) = 1, w >= 0
        """
        w = cp.Variable(self.n_assets)
        
        # 目标函数
        risk = cp.quad_form(w, covariance_matrix)
        ret = w @ blended_mu
        objective = cp.Maximize(ret - self.risk_aversion * risk)
        
        # 约束条件
        constraints = [
            cp.sum(w) == 1,
            w >= 0
        ]
        
        prob = cp.Problem(objective, constraints)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-5, eps_rel=1e-5)
            if w.value is None:
                return np.ones(self.n_assets) / self.n_assets
            
            # 数值清洗
            res = w.value
            res[res < 1e-4] = 0.0
            res /= np.sum(res)
            return res
        except:
            # 求解失败时的兜底策略 (等权重)
            return np.ones(self.n_assets) / self.n_assets

class MetaDMPOActionWrapper(gym.ActionWrapper):
    """
    Meta-RL 动作包装器
    
    Flow:
    1. RL Agent 输出 Logits (针对 K 个 Alpha 的打分)
    2. Wrapper 做 Softmax -> 得到 Alpha 权重 (lambda)
    3. Wrapper 从 Env 获取当前的 Alpha 预测值 (Signals)
    4. 融合信号: mu = sum(lambda_i * Signal_i)
    5. 调用 QP Solver -> 得到 Asset 权重 (w)
    6. 将 w 传给 Env 执行
    """
    def __init__(self, env):
        super().__init__(env)
        # 我们的 Solver
        self.solver = MetaQPSolver(env.n_assets)
        # 记录上一次的混合权重用于观察
        self.last_alpha_weights = np.zeros(env.n_alphas)

    def action(self, action):
        # 1. Softmax 处理 RL 输出的 Logits
        # 加上数值稳定性处理
        exp_x = np.exp(action - np.max(action))
        alpha_weights = exp_x / exp_x.sum()
        self.last_alpha_weights = alpha_weights
        
        # 2. 获取环境中的信号 (这是 Hack，但在 Wrapper 中很常见)
        # 我们需要访问 env.unwrapped 来拿到当前的信号矩阵
        # 信号矩阵 shape: (K, N_assets)
        # 注意：这里要拿 current_step 的信号
        env_core = self.env.unwrapped
        
        # 边界检查
        if env_core.current_step >= len(env_core.signals):
             return np.ones(env_core.n_assets) / env_core.n_assets
             
        current_signals = env_core.signals[env_core.current_step] 
        
        # 3. 信号融合 (加权平均)
        # blended_mu: (N_assets,)
        blended_mu = alpha_weights @ current_signals
        
        # 4. 估计简单的协方差矩阵 (为了 MVP，这里用近期历史估计)
        # 在真实场景中，这应该由专门的 Risk Model 提供
        lookback = 30
        start = max(0, env_core.current_step - lookback)
        end = env_core.current_step
        
        if end - start < 10:
            # 数据不足时使用单位阵
            cov_matrix = np.eye(env_core.n_assets) * 0.01
        else:
            recent_returns = env_core.returns[start:end]
            cov_matrix = np.cov(recent_returns.T) + np.eye(env_core.n_assets) * 1e-6
            
        # 5. 求解优化问题
        asset_weights = self.solver.solve(blended_mu, cov_matrix)
        
        return asset_weights

    def step(self, action):
        # Step 会调用上面的 action() 转换动作
        obs, reward, done, truncated, info = self.env.step(action)
        
        # 将 Alpha 权重记录到 Info 里，方便我们 Debug
        info['alpha_weights'] = self.last_alpha_weights
        return obs, reward, done, truncated, info