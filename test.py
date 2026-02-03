import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cvxpy as cp
from stable_baselines3 import PPO
from env import PortfolioEnv
from dmpo_model import DMPOActionWrapper
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


# --- 辅助函数：计算金融指标 ---
def calculate_metrics(returns):
    """计算累计收益, 夏普, 最大回撤"""
    cum_ret = np.cumprod(1 + returns)
    total_ret = cum_ret[-1] - 1
    
    # 年化夏普 (假设252个交易日)
    mean = np.mean(returns)
    std = np.std(returns)
    if std == 0:
        sharpe = 0
    else:
        sharpe = (mean / std) * np.sqrt(252)
        
    # 最大回撤
    running_max = np.maximum.accumulate(cum_ret)
    drawdown = (cum_ret - running_max) / running_max
    max_dd = np.min(drawdown)
    
    return total_ret, sharpe, max_dd, cum_ret

# --- 基准策略 1: 均值方差 (Mean-Variance) ---
def run_mean_variance(returns_history, lookback=30, max_turnover=0.10): # 增加参数
    """
    修正后的 MV 策略：增加了硬换手率约束，实现公平对比
    """
    n_steps, n_assets = returns_history.shape
    portfolio_returns = []
    
    # 初始权重
    weights = np.ones(n_assets) / n_assets
    
    for t in range(lookback, n_steps):
        window = returns_history[t-lookback:t]
        Sigma = np.cov(window.T)
        mu = np.mean(window, axis=0)
        
        # --- 增加约束 ---
        w = cp.Variable(n_assets)
        w_prev = cp.Parameter(n_assets) # 上一期权重参数
        w_prev.value = weights
        
        gamma = 0.5
        obj = cp.Maximize(mu @ w - gamma * cp.quad_form(w, Sigma))
        
        # 这里的约束必须和 DMPO 完全一致！
        cons = [
            cp.sum(w) == 1, 
            w >= 0,
            cp.norm(w - w_prev, 1) <= max_turnover # <--- 加上这一行！
        ]
        
        prob = cp.Problem(obj, cons)
        
        try:
            # 同样提高求解精度
            prob.solve(solver=cp.OSQP, eps_abs=1e-6, eps_rel=1e-6)
            if w.value is not None:
                # 简单的数值清洗
                new_w = np.maximum(w.value, 0)
                new_w /= np.sum(new_w)
                weights = new_w
        except:
            pass # 求解失败则不动
            
        # 计算收益 (含交易成本，为了公平)
        # 注意：这里我们简单计算，实际应该和 env 逻辑一致
        r = np.sum(weights * returns_history[t])
        # 如果你想算得更细，可以扣除 costs，但作为 baseline 纯收益对比也可以
        portfolio_returns.append(r)
        
    return np.array(portfolio_returns)


def main():
    print("📊 初始化测试环境...")
    
    # 1. 创建基础环境
    raw_env = PortfolioEnv(n_assets=10, lookback=30, max_turnover=0.10)
    # 保存数据用于 Benchmark
    obs, _ = raw_env.reset()
    market_returns_data = raw_env.returns 
    
    # 2. 重新构建与训练时一致的 Wrapper 栈
    def make_test_env():
        e = PortfolioEnv(n_assets=10, lookback=30, max_turnover=0.10)
        e = DMPOActionWrapper(e, max_turnover=0.10)
        # 注入相同数据
        e.env.returns = market_returns_data 
        e.env.prices = raw_env.prices
        e.env.regimes = raw_env.regimes
        e.env.n_steps = len(market_returns_data)
        return e

    env = DummyVecEnv([make_test_env])
    
    # 3. ⚡️ 加载 VecNormalize 统计数据 ⚡️
    try:
        env = VecNormalize.load("./model/vec_normalize.pkl", env)
        env.training = False # 测试模式：不更新均值方差
        env.norm_reward = False # 测试模式：我们需要真实的 Reward 来评估
        print("✅ 成功加载归一化参数")
    except:
        print("❌ 未找到 vec_normalize.pkl，结果将不可靠！")

    try:
        model = PPO.load("./model/dmpo_agent_fixed", env=env)
        print("✅ 成功加载 DMPO 模型")
    except:
        print("❌ 未找到模型文件")
        return

    # --- 运行 DMPO ---
    print("🤖 正在运行 DMPO 策略...")
    dmpo_returns = []
    dmpo_violation_count = 0
    total_steps = 0
    
    obs = env.reset() # VecEnv 返回的 obs 已经是归一化过的
    
    # VecEnv 的 step 循环略有不同
    for _ in range(len(market_returns_data) - 32): # 减去 lookback
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        
        # VecEnv 返回的 infos 是一个列表
        info = infos[0]
        
        dmpo_returns.append(info['return'])
        
        # 提高容忍度到 1e-4，因为 solver 精度是 1e-6，Python 浮点累加可能有误差
        if info['turnover'] > 0.10 + 1e-4:
            dmpo_violation_count += 1
        
        total_steps += 1
        if dones[0]: break

    # --- 4. 运行基准策略 ---
    print("📉 正在运行 Benchmark (1/N 等权)...")
    # 1/N 策略收益 = 每日所有资产收益的平均值
    # 注意要对齐时间轴：DMPO 是从第30天(lookback)开始交易的
    bench_equal_returns = np.mean(market_returns_data[30:], axis=1)
    
    print("📉 正在运行 Benchmark (Mean-Variance)...")
    bench_mv_returns = run_mean_variance(market_returns_data)
    # MV 可能会少几天数据，截断对齐
    min_len = min(len(dmpo_returns), len(bench_equal_returns), len(bench_mv_returns))
    
    dmpo_returns = np.array(dmpo_returns[:min_len])
    bench_equal_returns = bench_equal_returns[:min_len]
    bench_mv_returns = bench_mv_returns[:min_len]

    # --- 5. 计算指标与展示 ---
    metrics_dmpo = calculate_metrics(dmpo_returns)
    metrics_equal = calculate_metrics(bench_equal_returns)
    metrics_mv = calculate_metrics(bench_mv_returns)
    
    print("\n" + "="*60)
    print(f"{'Metric':<15} | {'DMPO (Ours)':<15} | {'1/N (Benchmark)':<15} | {'Mean-Var':<15}")
    print("-" * 60)
    print(f"{'Total Return':<15} | {metrics_dmpo[0]:>14.2%} | {metrics_equal[0]:>14.2%} | {metrics_mv[0]:>14.2%}")
    print(f"{'Sharpe Ratio':<15} | {metrics_dmpo[1]:>14.2f} | {metrics_equal[1]:>14.2f} | {metrics_mv[1]:>14.2f}")
    print(f"{'Max Drawdown':<15} | {metrics_dmpo[2]:>14.2%} | {metrics_equal[2]:>14.2%} | {metrics_mv[2]:>14.2%}")
    print("-" * 60)
    
    # 硬约束统计 (A/B 格式)
    violation_rate = dmpo_violation_count / total_steps
    print(f"Hard Constraint Violations (Turnover > 10%):")
    print(f"👉 {dmpo_violation_count}/{total_steps} ({violation_rate:.2%})")
    
    if dmpo_violation_count == 0:
        print("✅ Validated: The differentiable layer successfully enforced strict constraints.")
    else:
        print("⚠️ Warning: Some constraints were violated.")

    # --- 6. 绘图 ---
    plt.figure(figsize=(10, 6))
    plt.plot(metrics_dmpo[3], label='DMPO (RL + QP)', linewidth=2, color='red')
    plt.plot(metrics_equal[3], label='1/N Benchmark', linestyle='--', color='gray')
    plt.plot(metrics_mv[3], label='Mean-Variance', linestyle=':', color='blue')
    
    plt.title('Cumulative Returns Comparison')
    plt.xlabel('Trading Days')
    plt.ylabel('Portfolio Value')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('./graph/backtest_result.png')
    print("\n📊 净值曲线已保存为 './graph/backtest_result.png'")
    plt.show()

if __name__ == "__main__":
    main()