import torch
import cvxpy as cp
import numpy as np
from cvxpylayers.torch import CvxpyLayer

def test_environment():
    print("="*30)
    print("🔍 开始环境检测...")
    
    # 1. 检测 PyTorch
    try:
        print(f"✅ PyTorch 版本: {torch.__version__}")
        if torch.cuda.is_available():
            print(f"   GPU 状态: 可用 ({torch.cuda.get_device_name(0)})")
        else:
            print("   GPU 状态: 不可用 (使用 CPU)")
    except ImportError:
        print("❌ PyTorch 未安装！")
        return

    # 2. 检测 CVXPY
    try:
        print(f"✅ CVXPY 版本: {cp.__version__}")
    except ImportError:
        print("❌ CVXPY 未安装！")
        return

    # 3. 检测 CVXPYLayers (核心测试)
    print("\n⚡ 正在测试 CvxpyLayer 的可微性 (梯度回传)...")
    try:
        # 定义一个简单的凸优化问题: min 0.5 * x^2   s.t. x >= theta
        # 理论解: x = theta (当 theta > 0)
        
        # 变量与参数
        x = cp.Variable(1)
        theta = cp.Parameter(1)
        
        # 问题定义
        objective = cp.Minimize(0.5 * cp.sum_squares(x))
        constraints = [x >= theta]
        problem = cp.Problem(objective, constraints)
        
        # 创建可微层
        layer = CvxpyLayer(problem, parameters=[theta], variables=[x])
        
        # PyTorch 输入 (需要求导)
        theta_tensor = torch.tensor([5.0], requires_grad=True, dtype=torch.float64)
        
        # 前向传播 (Forward)
        solution, = layer(theta_tensor)
        print(f"   前向传播结果 (x): {solution.item():.4f} (预期: 5.0000)")
        
        # 反向传播 (Backward)
        # Loss = x.sum(), 那么 dLoss/dtheta = dLoss/dx * dx/dtheta
        # 因为 x = theta, 所以 dx/dtheta = 1.0
        solution.sum().backward()
        
        grad = theta_tensor.grad.item()
        print(f"   反向传播梯度 (grad): {grad:.4f} (预期: 1.0000)")
        
        if np.isclose(grad, 1.0):
            print("\n🎉 恭喜！Pytorch + CVXPY + CvxpyLayers 安装成功且工作正常！")
        else:
            print("\n⚠️ 安装可能成功，但数值计算有偏差，请检查求解器。")
            
    except ImportError as e:
        print(f"❌ CvxpyLayers 导入失败: {e}")
    except Exception as e:
        print(f"❌ 运行时错误 (可能是 diffcp 编译问题): {e}")

if __name__ == "__main__":
    test_environment()