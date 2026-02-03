import numpy as np
import gymnasium as gym
from env import PortfolioEnv
from dmpo_model import DMPOActionWrapper

def run_stress_test():
    print("🛡️ 开始硬约束压力测试 (Stress Test)...")
    
    # 1. 初始化环境和 Wrapper
    # 设定一个极其严格的换手率，比如 5%
    MAX_TURNOVER = 0.1 
    env = PortfolioEnv(n_assets=10, lookback=30, max_turnover=MAX_TURNOVER)
    env = DMPOActionWrapper(env, max_turnover=MAX_TURNOVER)
    
    obs, _ = env.reset()
    done = False
    
    total_steps = 0
    violations = 0
    max_violation_magnitude = 0.0
    
    print(f"设定硬约束: 单日换手率 <= {MAX_TURNOVER*100}%")
    
    # 2. 模拟 1000 步
    for t in range(1000):
        # --- 制造极端信号 (Extreme Signals) ---
        # 每一天都随机生成一个极端的 Mu，试图诱导 Agent 全仓切换
        # 比如：今天全仓买资产1，明天全仓买资产2
        # 这种信号如果没有约束，换手率会高达 200% (卖出100% + 买入100%)
        fake_signal_mu = np.random.randn(10) * 100 
        
        # 通过 Wrapper 执行
        # Wrapper 内部会调用 QP Solver 试图满足约束
        action = fake_signal_mu 
        obs, reward, done, _, info = env.step(action)
        
        # 3. 检查结果
        actual_turnover = info['turnover']
        
        # 允许极小的数值误差 (1e-5)
        if actual_turnover > MAX_TURNOVER + 1e-5:
            violations += 1
            magnitude = actual_turnover - MAX_TURNOVER
            max_violation_magnitude = max(max_violation_magnitude, magnitude)
            print(f"❌ 违规! Step {t}: 实际换手 {actual_turnover:.6f} > 阈值 {MAX_TURNOVER}")
        
        if done:
            obs, _ = env.reset()
            
        total_steps += 1

    print("\n" + "="*40)
    print(f"测试总结 (Total Steps: {total_steps})")
    print(f"违规次数: {violations}")
    print(f"最大违规幅度: {max_violation_magnitude:.8f}")
    
    if violations == 0:
        print("✅ 通过! 你的 QP 层坚不可摧。")
        print("结论: 无论神经网络输出多疯狂的信号，该架构都能保证合规。")
    else:
        print("❌ 失败! QP 层存在漏洞或精度问题。")
        print("建议: 检查 dmpo_model.py 中的 solver accuracy 或 np.maximum 裁剪逻辑。")

if __name__ == "__main__":
    run_stress_test()