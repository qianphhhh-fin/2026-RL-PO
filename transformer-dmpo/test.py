import numpy as np
import matplotlib.pyplot as plt
import cvxpy as cp
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

from env import PortfolioEnv
from dmpo_model import DMPOActionWrapper

# --- Benchmark: 带约束的均值方差 ---
def run_constrained_mv(returns, lookback=30, max_turnover=0.10):
    T, N = returns.shape
    weights = np.ones(N) / N
    equity = [1.0]
    
    for t in range(lookback, T):
        # 估计参数
        window = returns[t-lookback:t]
        mu = np.mean(window, axis=0)
        Sigma = np.cov(window.T)
        
        # 求解带约束 QP
        w = cp.Variable(N)
        w_prev = cp.Parameter(N); w_prev.value = weights
        
        # 目标: Max Return - Risk
        obj = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, Sigma))
        cons = [
            cp.sum(w) == 1, 
            w >= 0,
            cp.norm(w - w_prev, 1) <= max_turnover # 公平对比!
        ]
        
        prob = cp.Problem(obj, cons)
        try:
            prob.solve(solver=cp.OSQP, eps_abs=1e-4, eps_rel=1e-4)
            if w.value is not None:
                weights = np.maximum(w.value, 0)
                weights /= np.sum(weights)
        except:
            pass
        
        # 计算净值
        r = np.sum(weights * returns[t]) - max_turnover * 0.0005 # 估算成本
        equity.append(equity[-1] * (1 + r))
        
    return np.array(equity)

def main():
    # 1. 创建测试环境 (保持随机种子固定以复现)
    # 注意: 这里不需要重新生成数据，我们reset后获取数据
    raw_env = PortfolioEnv(n_assets=10, lookback=30, max_turnover=0.10)
    
    # 2. 包装环境 (用于模型预测)
    def make_test_env():
        e = PortfolioEnv(n_assets=10, lookback=30, max_turnover=0.10)
        e = DMPOActionWrapper(e, max_turnover=0.10)
        # 强制同步数据，确保和 raw_env 使用同一套"平行宇宙"
        e.env.generator = raw_env.generator
        e.env.prices = raw_env.prices
        e.env.returns = raw_env.returns
        e.env.factors = raw_env.factors
        e.env.regimes = raw_env.regimes
        return e

    env = DummyVecEnv([make_test_env])
    
    # 3. 加载归一化参数 (Training=False)
    try:
        env = VecNormalize.load("vec_normalize.pkl", env)
        env.training = False
        env.norm_reward = False
    except:
        print("⚠️ 未找到归一化参数，结果可能不准")

    # 4. 加载模型
    model = PPO.load("dmpo_transformer_agent", env=env)
    
    # --- 运行回测 ---
    print("📊 运行 DMPO 回测...")
    obs = env.reset()
    dmpo_equity = [1.0]
    dmpo_violations = 0
    
    # 获取真实数据用于 Benchmark
    returns_data = raw_env.returns
    
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, _, dones, infos = env.step(action)
        
        info = infos[0]
        dmpo_equity.append(dmpo_equity[-1] * (1 + info['return']))
        
        if info['turnover'] > 0.10 + 1e-4:
            dmpo_violations += 1
            
        if dones[0]: break
            
    # --- 运行 Benchmark ---
    print("📊 运行 Constrained Mean-Variance 回测...")
    mv_equity = run_constrained_mv(returns_data)
    
    # 对齐长度
    min_len = min(len(dmpo_equity), len(mv_equity))
    dmpo_equity = dmpo_equity[:min_len]
    mv_equity = mv_equity[:min_len]
    
    # --- 结果 ---
    print("\n" + "="*40)
    print(f"DMPO Return: {(dmpo_equity[-1]-1):.2%}")
    print(f"MV Return:   {(mv_equity[-1]-1):.2%}")
    print(f"Constraint Violations: {dmpo_violations}")
    print("="*40)
    
    plt.plot(dmpo_equity, label='DMPO (Transformer+QP)')
    plt.plot(mv_equity, label='Mean-Variance (Constrained)', linestyle='--')
    plt.legend()
    plt.title("Backtest: DMPO vs Constrained MV")
    plt.grid(True)
    plt.savefig("final_result.png")
    print("✅ 结果图已保存")

if __name__ == "__main__":
    main()