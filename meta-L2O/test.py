import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from sbx import SAC
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from meta_env import MetaExecutionEnv

def run_strategy(env, model=None, static_action=None, label="Strategy"):
    """
    运行策略回测
    注意：这里的 env 已经是被 VecNormalize 包裹过的
    """
    # 强制固定种子，确保对比公平
    # VecEnv 的 reset 不需要 seed 参数，它在内部管理
    obs = env.reset()
    
    # 这里的 env 是 VecEnv，所以 step 返回的是数组，我们需要取第一个元素
    terminated = False
    truncated = False
    
    wealth = [1.0]
    actions = []
    regrets = []
    
    # 我们需要手动访问内部的原始环境来获取 info 中的 net_return (未归一化的真实值)
    # 因为 VecNormalize 可能会修改 reward，虽然我们设了 norm_reward=False，但为了保险起见，
    # 我们直接从 info 里读原始数据
    
    while not (terminated or truncated):
        if model:
            # deterministic=True 意味着测试时使用确定性策略（不加噪声）
            action, _ = model.predict(obs, deterministic=True)
        else:
            # 静态策略需要扩展维度以适配 VecEnv: (3,) -> (1, 3)
            action = np.array([static_action])
            
        obs, rewards, dones, infos = env.step(action)
        
        info = infos[0] # 取第一个环境的 info
        terminated = dones[0]
        truncated = info.get("TimeLimit.truncated", False)
        
        # 记录真实净值变化 (使用 info 中的真实回报，不受 reward scaling 影响)
        wealth.append(wealth[-1] * (1 + info['net_return']))
        
        if model:
            actions.append(action[0]) # 记录 RL 的动作
        regrets.append(info['regret'])
        
    # 获取 Ground Truth 权重 (从内部环境提取)
    # env -> VecNormalize -> DummyVecEnv -> MetaExecutionEnv
    raw_env = env.envs[0]
    gt_weights = raw_env.ground_truth_w
    real_returns = raw_env.real_returns
    
    return wealth, np.array(actions), np.sum(regrets), gt_weights, real_returns, raw_env

def main():
    print("🚀 加载测试环境...")
    
    # 1. 创建基础环境 (测试时不需要 RewardScale，我们要看真实的一分一毫)
    # 但必须使用 DummyVecEnv，因为 VecNormalize 需要它
    base_env = MetaExecutionEnv(n_assets=5, episode_length=100, cost_rate=0.0005)
    env = DummyVecEnv([lambda: base_env])
    
    # 2. 加载归一化参数 (关键步骤！)
    model_name = "meta_execution_sac_sbx"
    vecnorm_path = f"{model_name}_vecnorm.pkl"
    
    try:
        # 加载统计数据 (均值/方差)
        env = VecNormalize.load(vecnorm_path, env)
        # 测试模式：不要更新均值和方差
        env.training = False 
        # 测试模式：不要归一化 Reward (虽然训练时也没归一化，但这里显式关闭更安全)
        env.norm_reward = False
        print(f"✅ 成功加载 Observation 归一化参数: {vecnorm_path}")
    except Exception as e:
        print(f"❌ 无法加载归一化参数 ({e})。请确保 train.py 运行成功。")
        return

    # 3. 加载模型
    try:
        model = SAC.load(model_name)
        print(f"✅ 成功加载模型: {model_name}")
    except:
        print(f"❌ 无法加载模型 {model_name}")
        return

    print("\n⚔️ 开始对决：RL Agent vs Static Benchmark vs God Mode")

    # --- A. 运行 RL Agent ---
    wealth_rl, actions_rl, regret_rl, _, _, _ = run_strategy(
        env, model=model, label="Meta-RL Agent"
    )

    # --- B. 运行 Static Benchmark ---
    # Lambda=5.0 -> action[0] approx -0.5
    # Gamma=0.005 -> action[1] approx -0.9
    static_action = np.array([-0.5, -0.9, 0.0])
    wealth_static, _, regret_static, gt_weights, r_real, raw_env = run_strategy(
        env, static_action=static_action, label="Static Benchmark"
    )
    
    # --- C. 计算 Ground Truth 曲线 ---
    # 我们重新计算一遍 GT 净值，确保对齐
    gt_wealth = [1.0]
    w_prev = np.ones(5) / 5
    for t in range(len(r_real)):
        turnover = np.sum(np.abs(gt_weights[t] - w_prev))
        cost = turnover * 0.0005
        ret = np.dot(gt_weights[t], r_real[t]) - cost
        gt_wealth.append(gt_wealth[-1] * (1 + ret))
        w_prev = gt_weights[t]

    # --- 4. 绘图分析 ---
    print(f"\n📊 最终回测结果 (Episode Length: 100):")
    print(f"Ground Truth Final Wealth: {gt_wealth[-1]:.4f}")
    print(f"RL Agent Final Wealth:     {wealth_rl[-1]:.4f}")
    print(f"Static Bench Final Wealth: {wealth_static[-1]:.4f}")
    
    plt.figure(figsize=(16, 10))
    
    # 图1: 净值走势
    plt.subplot(2, 2, 1)
    plt.plot(gt_wealth, 'r--', label='Risk-Adjusted God (Ceiling)', alpha=0.6)
    plt.plot(wealth_rl, 'g-', label='Meta-RL Agent (Ours)', linewidth=2)
    plt.plot(wealth_static, 'b-', label='Static Benchmark', alpha=0.6)
    plt.title('Wealth Curve')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图2: RL 参数动态调整
    plt.subplot(2, 2, 2)
    if len(actions_rl) > 0:
        # 解析动作到物理参数
        risk_aversion = 0.1 + 19.9 * (0.5 * (actions_rl[:, 0] + 1))
        trade_penalty = 0.10 * (0.5 * (actions_rl[:, 1] + 1))
        
        plt.plot(risk_aversion, color='purple', label='Risk Aversion ($\lambda$)')
        plt.ylabel('Risk Aversion', color='purple')
        plt.legend(loc='upper left')
        
        ax2 = plt.gca().twinx()
        ax2.plot(trade_penalty * 10000, color='orange', label='Trade Penalty (bps)', alpha=0.7)
        ax2.set_ylabel('Trade Penalty (bps)', color='orange')
        ax2.legend(loc='upper right')
        plt.title('RL Dynamic Parameter Tuning')
    
    # 图3: 累积遗憾值 (越低越好)
    plt.subplot(2, 2, 3)
    rl_gap = np.array(gt_wealth) - np.array(wealth_rl)
    static_gap = np.array(gt_wealth) - np.array(wealth_static)
    plt.plot(rl_gap, 'g', label='RL Wealth Gap (to God)')
    plt.plot(static_gap, 'b', label='Static Wealth Gap (to God)')
    plt.title('Wealth Gap vs God Mode (Lower is Better)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 图4: 市场环境 (波动率)
    plt.subplot(2, 2, 4)
    vols = np.mean(np.sqrt(np.diagonal(raw_env.sigmas, axis1=1, axis2=2)), axis=1)
    # 只取前100步
    vols = vols[:100]
    plt.plot(vols, 'k-', alpha=0.5, label='Avg Market Volatility')
    plt.title('Market Context (Volatility)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()