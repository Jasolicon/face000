"""
检查并安装缺失的依赖
"""
import subprocess
import sys


def check_and_install_dependencies():
    """检查并安装缺失的依赖"""
    print("=" * 70)
    print("检查项目依赖")
    print("=" * 70)
    
    # 必需的依赖列表
    required_deps = {
        "torch": "torch==2.2.2",
        "torchvision": "torchvision==0.17.2",
        "transformers": "transformers>=4.30.0,<5.0.0",
        "diffusers": "diffusers>=0.21.0,<1.0.0",
        "controlnet_aux": "controlnet-aux>=0.0.10",
        "accelerate": "accelerate>=0.20.0",
        "numpy": "numpy>=1.24.0,<2.0.0",
    }
    
    missing_deps = []
    installed_deps = []
    
    print("\n检查依赖...")
    for module_name, package_spec in required_deps.items():
        try:
            if module_name == "controlnet_aux":
                # controlnet-aux 的模块名是 controlnet_aux
                __import__(module_name)
            else:
                __import__(module_name)
            
            # 获取版本
            module = sys.modules[module_name]
            version = getattr(module, "__version__", "未知")
            print(f"  ✓ {module_name}: {version}")
            installed_deps.append(module_name)
        except ImportError:
            print(f"  ✗ {module_name}: 未安装")
            missing_deps.append((module_name, package_spec))
    
    if not missing_deps:
        print("\n✅ 所有依赖已安装！")
        return
    
    print(f"\n发现 {len(missing_deps)} 个缺失的依赖:")
    for module_name, package_spec in missing_deps:
        print(f"  - {module_name}: {package_spec}")
    
    # 询问是否安装
    print("\n是否安装缺失的依赖？(y/n): ", end="")
    response = input().strip().lower()
    
    if response != 'y':
        print("已取消")
        return
    
    # 安装缺失的依赖
    print("\n正在安装依赖...")
    for module_name, package_spec in missing_deps:
        print(f"\n安装 {module_name} ({package_spec})...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", package_spec],
                check=True
            )
            print(f"  ✓ {module_name} 安装成功")
        except subprocess.CalledProcessError as e:
            print(f"  ✗ {module_name} 安装失败: {e}")
    
    # 验证安装
    print("\n验证安装...")
    all_installed = True
    for module_name, _ in missing_deps:
        try:
            if module_name == "controlnet_aux":
                __import__(module_name)
            else:
                __import__(module_name)
            print(f"  ✓ {module_name} 已安装")
        except ImportError:
            print(f"  ✗ {module_name} 仍未安装")
            all_installed = False
    
    if all_installed:
        print("\n✅ 所有依赖安装完成！")
    else:
        print("\n⚠️  部分依赖安装失败，请手动安装")
        print("建议运行: pip install -r requirements.txt")


if __name__ == "__main__":
    try:
        check_and_install_dependencies()
    except KeyboardInterrupt:
        print("\n\n已取消")
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()

