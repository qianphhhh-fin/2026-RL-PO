import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import os

# 必须导入 PPO，因为我们换回了 PPO
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from meta_env import MetaExecutionEnv

# ------------------------------------------------------------------------
# 辅助函数：运行单次策略回测
# ------------------------------------------------------------------------
def run_strategy(env, model=None, static_action=None, seed=42, label="Strategy"):
    """
    运行策略并返回净值曲线和相关数据。
    关键：每次运行前重置相同的种子，保证所有策略面对的是完全相同的市场行情。
    """
    print(f"🔄 正在运行策略: {label} ...")
    
    # 1. 重置环境 (固定种子)
    # VecEnv 的 reset 不接受 seed，我们需要手动调用内部 env 的 reset
    # env.envs[0] 是 DummyVecEnv 包裹的原始环境 (MetaExecutionEnv)
    raw_env = env.envs[0]
    
    # 这里我们显式调用 raw_env.reset 来控制种子
    # 然后让 VecEnv 同步一下 obs (通过 env.reset() 再次触发，但因为 DataGenerator 是 rng 控制的，
    # 我们需要在 raw_env 层面重新生成 rng 或者确保 reset 逻辑一致)
    
    # 最稳妥的方法：直接调用 env.reset()，但在 MetaEnv 内部确保 seed 生效
    # 我们通过 hack 方式：
    raw_env.reset(seed=seed) 
    obs = env.reset() # 这会再次调用内部 reset，但如果是 DummyVecEnv，它只是获取返回值
    
    # 为了双重保险，再次强制重置 DataGenerator 的状态
    # (在 MetaExecutionEnv v2 中，reset(seed) 会重置 data_gen)
    
    terminated = False
    truncated = False
    
    wealth = [1.0]
    actions = []
    regrets = []
    net_returns = []
    
    while not (terminated or truncated):
        if model:
            # RL 预测 (确定性模式)
            action, _ = model.predict(obs, deterministic=True)
        else:
            # 静态策略：需要将 (3,) 扩展为 (1, 3) 适配 VecEnv
            action = np.array([static_action])
            
        obs, rewards, dones, infos = env.step(action)
        
        info = infos[0]
        terminated = dones[0]
        truncated = info.get("TimeLimit.truncated", False)
        
        # 记录数据
        r_net = info['net_return']
        wealth.append(wealth[-1] * (1 + r_net))
        net_returns.append(r_net)
        regrets.append(info['regret'])
        
        if model:
            actions.append(action[0])
            
    # 提取 Ground Truth (God Mode) 数据
    # 因为我们固定了种子，环境内部计算的 GT 也是针对当前行情的
    gt_weights = raw_env.ground_truth_w
    real_returns = raw_env.real_returns
    
    return wealth, np.array(actions), np.sum(regrets), gt_weights, real_returns

# ------------------------------------------------------------------------
# 主函数
# ------------------------------------------------------------------------
def main():
    print("🚀 初始化测试流程 (v2.0 PPO + LogSpace)...")
    
    # 配置路径
    MODEL_NAME = "meta_execution_ppo_v3" # 对应 train.py 中的名称
    VECNORM_PATH = f"{MODEL_NAME}_vecnorm.pkl"
    
    if not os.path.exists(f"{MODEL_NAME}.zip"):
        print(f"❌ 错误：找不到模型文件 {MODEL_NAME}.zip，请先运行 train.py")
        return

    # 1. 创建环境
    # 测试时使用 DummyVecEnv，因为 VecNormalize 需要它
    # 我们创建一个 lambda 工厂
    env_maker = lambda: MetaExecutionEnv(n_assets=5, episode_length=100, cost_rate=0.0005)
    env = DummyVecEnv([env_maker])
    
    # 2. 加载归一化参数 (VecNormalize)
    # 这是最关键的一步！如果没加载，模型就是瞎子
    try:
        env = VecNormalize.load(VECNORM_PATH, env)
        env.training = False     # 测试模式：不更新均值方差
        env.norm_reward = False  # 测试模式：不归一化奖励，我们要看真实收益
        print(f"✅ 成功加载 Observation 归一化参数: {VECNORM_PATH}")
    except Exception as e:
        print(f"❌ 警告：无法加载归一化参数 ({e})。如果是首次运行或没保存过pkl，请忽略。")
        # 如果加载失败，最好不要继续，因为观测空间分布完全不同
        return

    # 3. 加载 PPO 模型
    model = PPO.load(MODEL_NAME)
    print(f"✅ 成功加载 PPO 模型: {MODEL_NAME}")
    
    TEST_SEED = 2026 # 固定测试种子
    
    # -------------------------------------------------------
    # A. 运行 RL Agent (Ours)
    # -------------------------------------------------------
    wealth_rl, actions_rl, regret_rl, _, _ = run_strategy(
        env, model=model, seed=TEST_SEED, label="Meta-RL Agent"
    )
    
    # -------------------------------------------------------
    # B. 运行 Static Benchmark (Baseline)
    # -------------------------------------------------------
    # 我们需要把物理参数转换为 Log-Space 动作
    # 目标物理参数: Lambda=5.0, Gamma=0.005, Kappa=1.0
    # 映射公式回顾: 
    #   Lambda = exp(a[0] * 4.6)      => a[0] = ln(Lambda) / 4.6
    #   Gamma  = exp(a[1]*4.6 - 4.6)  => a[1] = (ln(Gamma) + 4.6) / 4.6
    #   Kappa  = a[2] + 1.0           => a[2] = Kappa - 1.0
    
    target_lambda = 5.0
    target_gamma = 0.005
    target_kappa = 1.0
    
    static_a0 = np.log(target_lambda) / 4.6
    static_a1 = (np.log(target_gamma) + 4.6) / 4.6
    static_a2 = target_kappa - 1.0
    
    static_action = np.array([static_a0, static_a1, static_a2])
    print(f"ℹ️  Static Benchmark Action (Log Space): {static_action}")
    
    wealth_static, _, regret_static, gt_weights, r_real = run_strategy(
        env, static_action=static_action, seed=TEST_SEED, label="Static Benchmark"
    )
    
    # -------------------------------------------------------
    # C. 计算 Ground Truth (God Mode) 净值曲线
    # -------------------------------------------------------
    # 我们重新计算一遍 GT 的复利净值，以确保时间轴对齐
    gt_wealth = [1.0]
    w_prev = np.ones(5) / 5
    cost_rate = 0.0005
    
    for t in range(len(r_real)):
        # 1. 计算换手成本
        turnover = np.sum(np.abs(gt_weights[t] - w_prev))
        cost = turnover * cost_rate
        # 2. 计算净收益
        ret = np.dot(gt_weights[t], r_real[t]) - cost
        # 3. 复利
        gt_wealth.append(gt_wealth[-1] * (1 + ret))
        w_prev = gt_weights[t]
        
    # -------------------------------------------------------
    # 4. 绘图分析
    # -------------------------------------------------------
    print(f"\n📊 最终回测统计 (Episode Length: 100):")
    print(f"God Mode Final Wealth:     {gt_wealth[-1]:.4f}")
    print(f"RL Agent Final Wealth:     {wealth_rl[-1]:.4f}")
    print(f"Static Bench Final Wealth: {wealth_static[-1]:.4f}")
    print(f"Efficiency (RL / God):     {wealth_rl[-1] / gt_wealth[-1] * 100:.2f}%")
    
    plt.figure(figsize=(16, 10))
    
    # --- 图 1: 净值曲线 ---
    plt.subplot(2, 2, 1)
    plt.plot(gt_wealth, 'r--', label='Risk-Adjusted God (Ceiling)', alpha=0.6)
    plt.plot(wealth_rl, 'g-', label='Meta-RL Agent (Ours)', linewidth=2.5)
    plt.plot(wealth_static, 'b-', label='Static Benchmark', alpha=0.6)
    plt.title('Wealth Curve Comparison')
    plt.ylabel('Net Worth')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # --- 图 2: RL 参数动态调整 (还原为物理意义) ---
    plt.subplot(2, 2, 2)
    if len(actions_rl) > 0:
        # 还原映射
        # Lambda = exp(a * 4.6)
        risk_aversion = np.exp(actions_rl[:, 0] * 4.6)
        # Gamma = exp(a * 4.6 - 4.6)
        trade_penalty = np.exp(actions_rl[:, 1] * 4.6 - 4.6)
        # Kappa = a + 1
        alpha_conf = actions_rl[:, 2] + 1.0
        
        # 双轴绘图
        ax1 = plt.gca()
        line1 = ax1.plot(risk_aversion, color='purple', label='Risk Aversion ($\lambda$)', alpha=0.8)
        ax1.set_ylabel('Risk Aversion ($\lambda$)', color='purple')
        ax1.set_yscale('log') # Lambda 变化范围大，用对数轴更好看
        
        ax2 = ax1.twinx()
        # 换手惩罚乘 10000 变成 bps 单位
        line2 = ax2.plot(trade_penalty * 10000, color='orange', label='Trade Penalty (bps)', alpha=0.8)
        ax2.set_ylabel('Trade Penalty (bps)', color='orange')
        
        # 图例合并
        lines = line1 + line2
        labels = [l.get_label() for l in lines]
        ax1.legend(lines, labels, loc='upper left')
        plt.title('RL Dynamic Parameter Tuning (Physical Scale)')
    
    # --- 图 3: 相对遗憾值 Gap ---
    plt.subplot(2, 2, 3)
    # 计算相对 God 的百分比差距
    rl_gap = (np.array(gt_wealth) - np.array(wealth_rl)) / np.array(gt_wealth) * 100
    static_gap = (np.array(gt_wealth) - np.array(wealth_static)) / np.array(gt_wealth) * 100
    
    plt.plot(rl_gap, 'g', label='RL Drawdown vs God (%)')
    plt.plot(static_gap, 'b', label='Static Drawdown vs God (%)')
    plt.title('Performance Gap to God Mode (Lower is Better)')
    plt.ylabel('Gap (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
# --- 图 4: 市场环境 (波动率) ---
    plt.subplot(2, 2, 4)
    
    # 修复：从 VecEnv 中提取原始环境实例
    raw_env = env.envs[0]
    
    # 从 raw_env 获取这一轮的波动率数据
    vols = np.mean(np.sqrt(np.diagonal(raw_env.sigmas, axis1=1, axis2=2)), axis=1)
    # 截取前 100 步
    vols = vols[:len(actions_rl)]
    
    plt.plot(vols, 'k-', alpha=0.6, label='Avg Market Volatility')
    plt.fill_between(range(len(vols)), vols, 0, color='gray', alpha=0.1)
    plt.title('Market Context: Volatility Regime')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()