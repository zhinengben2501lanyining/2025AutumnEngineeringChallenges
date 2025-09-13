# Level2 实践日志

按照文档的操作流程，在安装步骤 `配置NVIDIA Container Toolkit` 处发现异常，经检查，相关bash命令应该替换为

```bash
#验证驱动状态
nvidia-smi

#安装 NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y curl
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

#安装 NVIDIA Container Toolkit
sudo apt-get update
sudo apt-get install -y nvidia-container-toolkit

# 更新 Docker 配置
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

#测试GPU访问

sudo docker run --rm --gpus all nvidia/cuda:12.6.0-base-ubuntu24.04 nvidia-smi
```

观察到文档并未提供 `gpu_test.py` 使用 `DeepSeek` 生成了一个测试文件

```python
"""
GPU测试脚本
用于验证Docker容器中的GPU加速功能
"""

import torch
import subprocess
import sys
import os

def check_nvidia_smi():
    """检查nvidia-smi命令是否正常工作"""
    print("=" * 50)
    print("1. 检查nvidia-smi命令输出")
    print("=" * 50)
    
    try:
        result = subprocess.run(['nvidia-smi'], 
                              capture_output=True, 
                              text=True, 
                              timeout=10)
        
        if result.returncode == 0:
            print("✓ nvidia-smi命令执行成功")
            print("\nGPU信息:")
            print(result.stdout)
            return True
        else:
            print("✗ nvidia-smi命令执行失败")
            print(f"错误信息: {result.stderr}")
            return False
            
    except Exception as e:
        print(f"✗ 执行nvidia-smi时发生错误: {e}")
        return False

def check_pytorch_gpu():
    """检查PyTorch是否能检测到GPU"""
    print("=" * 50)
    print("2. 检查PyTorch GPU支持")
    print("=" * 50)
    
    # 检查CUDA是否可用
    cuda_available = torch.cuda.is_available()
    print(f"CUDA可用: {'✓' if cuda_available else '✗'}")
    
    if not cuda_available:
        print("CUDA不可用，请检查安装")
        return False
    
    # 获取GPU数量
    gpu_count = torch.cuda.device_count()
    print(f"GPU数量: {gpu_count}")
    
    # 显示每个GPU的详细信息
    for i in range(gpu_count):
        print(f"\nGPU {i} 详细信息:")
        print(f"  设备名称: {torch.cuda.get_device_name(i)}")
        print(f"  CUDA计算能力: sm_{torch.cuda.get_device_capability(i)[0]}{torch.cuda.get_device_capability(i)[1]}")
        print(f"  总显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
    
    print(f"\n3. 执行兼容性测试")
    print("=" * 50)
    
    try:
        # 设置环境变量来捕获更详细的错误信息
        os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
        
        # 测试1: 简单的张量创建和移动
        print("测试1: 张量创建和移动...")
        x = torch.tensor([1.0, 2.0, 3.0]).cuda()
        print(f"✓ 张量创建成功: {x.device}")
        
        # 测试2: 简单的计算
        print("测试2: 简单计算...")
        y = x * 2
        print(f"✓ 计算成功: {y.cpu().numpy()}")
        
        # 测试3: 矩阵运算（小规模）
        print("测试3: 小规模矩阵运算...")
        a = torch.randn(100, 100).cuda()
        b = torch.randn(100, 100).cuda()
        c = torch.matmul(a, b)
        print(f"✓ 矩阵运算成功: {c.shape}")
        
        # 测试4: 检查CUDA功能
        print("测试4: CUDA功能检查...")
        print(f"  CUDA版本: {torch.version.cuda}")
        print(f"  当前设备: {torch.cuda.current_device()}")
        print(f"  设备属性: {torch.cuda.get_device_properties(0)}")
        
        return True
        
    except Exception as e:
        print(f"✗ 兼容性测试失败: {e}")
        print("\n详细诊断:")
        print(f"PyTorch版本: {torch.__version__}")
        print(f"CUDA版本: {torch.version.cuda if hasattr(torch.version, 'cuda') else 'N/A'}")
        print(f"计算能力: sm_{torch.cuda.get_device_capability(0)[0]}{torch.cuda.get_device_capability(0)[1]}")        
        return False

def check_system_info():
    """检查系统信息"""
    print("=" * 50)
    print("系统信息")
    print("=" * 50)
    
    print(f"Python版本: {sys.version.split()[0]}")
    print(f"PyTorch版本: {torch.__version__}")
    if hasattr(torch.version, 'cuda'):
        print(f"PyTorch CUDA版本: {torch.version.cuda}")
    print(f"CUDA可用: {torch.cuda.is_available()}")

def main():
    """主函数"""
    print("开始GPU兼容性测试...")
    print("=" * 60)
    print("目标设备: NVIDIA GeForce RTX 5060 (sm_120)")
    print("=" * 60)
    
    # 检查系统信息
    check_system_info()
    print()
    
    # 检查nvidia-smi
    nvidia_smi_success = check_nvidia_smi()
    print()
    
    # 检查PyTorch GPU支持
    pytorch_gpu_success = check_pytorch_gpu()
    print()
    
    # 输出测试结果摘要
    print("=" * 60)
    print("测试结果摘要:")
    print("=" * 60)
    
    print(f"nvidia-smi测试: {'通过 ✓' if nvidia_smi_success else '失败 ✗'}")
    print(f"PyTorch GPU测试: {'通过 ✓' if pytorch_gpu_success else '失败 ✗'}")
    
    if nvidia_smi_success and pytorch_gpu_success:
        print("\n🎉 所有测试通过！GPU加速环境配置成功！")
        print("您的RTX 5060 GPU现在可以正常使用PyTorch进行加速计算")
        return 0
    else:
        print("\n❌ 测试失败")
        return 1

if __name__ == "__main__":
    sys.exit(main())
```

```Dockerfile
FROM nvidia/cuda:12.6.0-base-ubuntu24.04

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    && rm -rf /var/lib/apt/lists/*

# 安装支持CUDA 12.4和最新架构的PyTorch
RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 复制测试脚本
COPY gpu_test.py /app/gpu_test.py

WORKDIR /app

# 设置执行权限
RUN chmod +x gpu_test.py

CMD ["python3", "gpu_test.py"]
```

报错，发现 `pip` 命令需要添加 `--break-system-packages` 参数(Ubuntu24.02的安全措施)

GPU测试失败，发现 `Pytorch` 对应的cuda版本过低，修改pip命令为 `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu130 --break-system-packages`

# 任务目标
1. nvidia-smi 命令在容器内正常运行 见 `nvidia-smi.png`
2. PyTorch能够检测到GPU设备 见 `pytorch1/2.png`
3. 显示正确的GPU型号和显存信息 见 `l2log/nvidia-smi.png` 或 `l2log.log`
