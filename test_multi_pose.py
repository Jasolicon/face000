"""
多姿态人脸生成测试脚本
快速测试多姿态人脸生成功能
"""
import os
from pathlib import Path
from generate_multi_pose_faces import MultiPoseFaceGenerator


def test_multi_pose_generation():
    """测试多姿态人脸生成"""
    
    # 测试图像路径（请替换为实际路径）
    test_image_path = "test_face.jpg"  # 替换为你的测试图像路径
    
    # 检查测试图像是否存在
    if not os.path.exists(test_image_path):
        print(f"错误: 测试图像不存在: {test_image_path}")
        print("请将测试图像放在当前目录，或修改 test_image_path 变量")
        return
    
    print("=" * 70)
    print("多姿态人脸生成测试")
    print("=" * 70)
    
    try:
        # 初始化生成器
        print("\n正在初始化生成器...")
        generator = MultiPoseFaceGenerator(
            device="cuda" if os.environ.get("CUDA_VISIBLE_DEVICES") else None,
            use_fp16=True
        )
        
        # 生成多姿态人脸
        print(f"\n正在处理图像: {test_image_path}")
        output_paths = generator.generate_multi_pose_faces(
            input_image_path=test_image_path,
            output_dir="test_output",
            poses=["side", "down", "up"],  # 只生成3个姿态以加快测试
            prompt="a high quality portrait photo of a person, detailed face, realistic",
            num_inference_steps=15  # 减少步数以加快测试
        )
        
        # 输出结果
        print("\n" + "=" * 70)
        print("测试完成！")
        print("=" * 70)
        print("\n生成的文件:")
        for key, path in output_paths.items():
            if os.path.exists(path):
                print(f"  ✓ {key}: {path}")
            else:
                print(f"  ✗ {key}: {path} (文件不存在)")
        
    except ImportError as e:
        print(f"\n错误: 缺少依赖库")
        print(f"请运行: pip install diffusers controlnet-aux>=0.0.10 accelerate")
        print(f"详细错误: {str(e)}")
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_multi_pose_generation()

