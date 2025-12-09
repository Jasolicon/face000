"""
多姿态人脸生成脚本
使用 Stable Diffusion + ControlNet + OpenPose 生成同一人的多姿态人脸
"""
import os
import sys

# 检查 NumPy 版本（必须在导入 torch 之前）
try:
    import numpy as np
    numpy_version = np.__version__
    numpy_major = int(numpy_version.split('.')[0])
    if numpy_major >= 2:
        print(f"错误: NumPy 版本 {numpy_version} 不兼容")
        print("PyTorch 2.2.2 需要 NumPy < 2.0.0")
        print("解决方案:")
        print("  pip install 'numpy>=1.24.0,<2.0.0'")
        print("  或")
        print("  pip install numpy==1.26.4")
        sys.exit(1)
except ImportError:
    pass

import torch
import numpy as np
from PIL import Image
import cv2
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import json

try:
    # 检查 torch 和 torchvision 版本兼容性
    import torch
    import torchvision
    torch_version = torch.__version__
    torchvision_version = torchvision.__version__
    
    # 检查版本兼容性（固定版本：torch==2.2.2, torchvision==0.17.2）
    # 注意：PyTorch 版本可能包含 CUDA 后缀（如 2.2.2+cu118），需要提取基础版本号
    def extract_base_version(version_str):
        """提取版本号的基础部分（去除 +cu118 等后缀）"""
        # 移除 +cu118, +cpu 等后缀
        if '+' in version_str:
            return version_str.split('+')[0]
        return version_str
    
    expected_torch = "2.2.2"
    expected_torchvision = "0.17.2"
    
    torch_base = extract_base_version(torch_version)
    torchvision_base = extract_base_version(torchvision_version)
    
    if torch_base != expected_torch or torchvision_base != expected_torchvision:
        print(f"警告: torch ({torch_version}) 和 torchvision ({torchvision_version}) 版本不匹配")
        print(f"  基础版本: torch={torch_base}, torchvision={torchvision_base}")
        print(f"  期望版本: torch=={expected_torch}, torchvision=={expected_torchvision}")
        print("建议: 重新安装固定版本")
        print("  pip uninstall torch torchvision -y")
        print(f"  pip install torch=={expected_torch} torchvision=={expected_torchvision}")
    else:
        # 版本匹配，显示确认信息
        print(f"✓ 版本检查通过: torch={torch_version}, torchvision={torchvision_version}")
    
    from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
    from diffusers.utils import load_image
    DIFFUSERS_AVAILABLE = True
except ImportError as e:
    DIFFUSERS_AVAILABLE = False
    print(f"警告: diffusers 导入失败: {str(e)}")
    print("请运行: pip install diffusers")
except RuntimeError as e:
    DIFFUSERS_AVAILABLE = False
    error_msg = str(e)
    if "torchvision" in error_msg or "nms" in error_msg:
        print("错误: torch 和 torchvision 版本不兼容")
        print("解决方案:")
        print("  1. 重新安装固定版本:")
        print("     pip uninstall torch torchvision -y")
        print("     pip install torch==2.2.2 torchvision==0.17.2")
        print("  2. 或者使用修复脚本:")
        print("     python fix_versions.py")
        print("  3. 或者使用 conda 安装:")
        print("     conda install pytorch==2.2.2 torchvision==0.17.2 -c pytorch")
    else:
        print(f"错误: {error_msg}")
    raise

try:
    from controlnet_aux import OpenposeDetector
    OPENPOSE_AVAILABLE = True
except ImportError:
    OPENPOSE_AVAILABLE = False
    print("警告: controlnet-aux 未安装，请运行: pip install controlnet-aux")

from face_detector import FaceDetector


class MultiPoseFaceGenerator:
    """多姿态人脸生成器"""
    
    def __init__(
        self,
        model_id: str = "runwayml/stable-diffusion-v1-5",
        controlnet_model_id: str = "lllyasviel/sd-controlnet-openpose",
        device: Optional[str] = None,
        use_fp16: bool = True
    ):
        """
        初始化多姿态人脸生成器
        
        Args:
            model_id: Stable Diffusion 模型ID
            controlnet_model_id: ControlNet OpenPose 模型ID
            device: 计算设备 ('cuda' 或 'cpu')，如果为None则自动选择
            use_fp16: 是否使用半精度（FP16）以节省显存
        """
        if not DIFFUSERS_AVAILABLE:
            raise ImportError("请安装 diffusers: pip install diffusers")
        if not OPENPOSE_AVAILABLE:
            raise ImportError("请安装 controlnet-aux: pip install controlnet-aux")
        
        # 检查必需的依赖
        missing_deps = []
        
        try:
            import transformers
        except ImportError:
            missing_deps.append("transformers>=4.30.0,<5.0.0")
        
        try:
            import accelerate
        except ImportError:
            missing_deps.append("accelerate>=0.20.0")
        
        if missing_deps:
            deps_str = " ".join(missing_deps)
            raise ImportError(
                f"缺少必需的依赖库: {', '.join(missing_deps)}\n"
                f"请运行: pip install {deps_str}\n"
                f"或安装所有依赖: pip install -r requirements.txt"
            )
        
        if device is None:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"使用设备: {self.device}")
        
        # 初始化人脸检测器
        print("正在初始化人脸检测器...")
        self.face_detector = FaceDetector(device=str(self.device))
        
        # 初始化 OpenPose 检测器
        print("正在初始化 OpenPose 检测器...")
        self.openpose_detector = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
        
        # 检查 transformers 是否安装（在加载模型之前）
        try:
            import transformers
        except ImportError:
            raise ImportError(
                "transformers 库未安装。\n"
                "StableDiffusionControlNetPipeline 需要 transformers 库。\n"
                "请运行: pip install transformers>=4.30.0,<5.0.0\n"
                "或安装所有依赖: pip install -r requirements.txt"
            )
        
        # 加载 ControlNet 模型
        print(f"正在加载 ControlNet 模型: {controlnet_model_id}...")
        try:
            self.controlnet = ControlNetModel.from_pretrained(
                controlnet_model_id,
                torch_dtype=torch.float16 if use_fp16 and self.device.type == 'cuda' else torch.float32
            )
        except Exception as e:
            if "transformers" in str(e).lower():
                raise ImportError(
                    f"加载 ControlNet 模型失败: {str(e)}\n"
                    "请确保已安装 transformers: pip install transformers>=4.30.0,<5.0.0"
                )
            raise
        
        # 加载 Stable Diffusion 管道
        print(f"正在加载 Stable Diffusion 模型: {model_id}...")
        try:
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                model_id,
                controlnet=self.controlnet,
                torch_dtype=torch.float16 if use_fp16 and self.device.type == 'cuda' else torch.float32,
                safety_checker=None,  # 禁用安全检查器以加快速度
                requires_safety_checker=False
            )
        except Exception as e:
            if "transformers" in str(e).lower():
                raise ImportError(
                    f"加载 Stable Diffusion 模型失败: {str(e)}\n"
                    "请确保已安装 transformers: pip install transformers>=4.30.0,<5.0.0"
                )
            raise
        
        # 使用 UniPC 调度器（更快）
        self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config)
        
        # 移动到设备
        self.pipe = self.pipe.to(self.device)
        
        # 如果使用 CPU，启用内存优化
        if self.device.type == 'cpu':
            self.pipe.enable_attention_slicing()
        
        print("模型加载完成！")
    
    def extract_face_pose(self, image_path: str) -> Tuple[Image.Image, Image.Image]:
        """
        提取正面人脸的姿态关键点
        
        Args:
            image_path: 图像路径或PIL Image对象
            
        Returns:
            face_image: 裁剪的人脸图像
            pose_image: OpenPose 姿态关键点图像
        """
        # 加载图像
        if isinstance(image_path, str):
            image = Image.open(image_path).convert('RGB')
        else:
            image = image_path
        
        # 检测人脸
        faces, boxes, probs = self.face_detector.detect_faces(image)
        
        if not faces or len(faces) == 0:
            raise ValueError("未检测到人脸")
        
        # 使用第一个人脸
        face_image = faces[0]
        
        # 提取姿态关键点
        # 将人脸图像转换为 OpenPose 输入格式
        face_array = np.array(face_image)
        
        # 使用 OpenPose 检测关键点
        pose_image = self.openpose_detector(face_image)
        
        return face_image, pose_image
    
    def modify_pose_keypoints(
        self,
        pose_image: Image.Image,
        target_pose: str = "side",
        angle: float = 45.0
    ) -> Image.Image:
        """
        修改姿态关键点为目标姿态
        
        Args:
            pose_image: 原始姿态关键点图像
            target_pose: 目标姿态 ('side', 'down', 'up', 'left', 'right')
            angle: 旋转角度（度）
            
        Returns:
            modified_pose_image: 修改后的姿态关键点图像
        """
        pose_array = np.array(pose_image)
        
        # 根据目标姿态修改关键点
        if target_pose == "side":
            # 侧脸：旋转关键点
            h, w = pose_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            modified_pose = cv2.warpAffine(pose_array, M, (w, h))
        elif target_pose == "down":
            # 低头：向下移动关键点
            h, w = pose_array.shape[:2]
            M = np.float32([[1, 0, 0], [0, 1, h * 0.2]])
            modified_pose = cv2.warpAffine(pose_array, M, (w, h))
        elif target_pose == "up":
            # 抬头：向上移动关键点
            h, w = pose_array.shape[:2]
            M = np.float32([[1, 0, 0], [0, 1, -h * 0.2]])
            modified_pose = cv2.warpAffine(pose_array, M, (w, h))
        elif target_pose == "left":
            # 向左转
            h, w = pose_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            modified_pose = cv2.warpAffine(pose_array, M, (w, h))
        elif target_pose == "right":
            # 向右转
            h, w = pose_array.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, -angle, 1.0)
            modified_pose = cv2.warpAffine(pose_array, M, (w, h))
        else:
            modified_pose = pose_array
        
        return Image.fromarray(modified_pose)
    
    def generate_face_with_pose(
        self,
        face_image: Image.Image,
        pose_image: Image.Image,
        prompt: str = "a high quality portrait photo of a person",
        negative_prompt: str = "blurry, low quality, distorted, deformed",
        num_inference_steps: int = 20,
        guidance_scale: float = 7.5,
        strength: float = 0.8
    ) -> Image.Image:
        """
        使用 ControlNet（OpenPose 条件）+ 原正面人脸生成对应姿态的人脸图像
        
        Args:
            face_image: 原始正面人脸图像
            pose_image: 目标姿态关键点图像
            prompt: 生成提示词
            negative_prompt: 负面提示词
            num_inference_steps: 推理步数
            guidance_scale: 引导强度
            strength: 控制强度（0-1）
            
        Returns:
            generated_image: 生成的人脸图像
        """
        # 准备输入
        # 将人脸图像和姿态图像调整到相同尺寸
        target_size = (512, 512)
        face_resized = face_image.resize(target_size, Image.LANCZOS)
        pose_resized = pose_image.resize(target_size, Image.LANCZOS)
        
        # 构建提示词（包含人脸信息）
        enhanced_prompt = f"{prompt}, detailed face, realistic, professional photography"
        
        # 生成图像
        print(f"正在生成图像（推理步数: {num_inference_steps}）...")
        generated_image = self.pipe(
            prompt=enhanced_prompt,
            image=pose_resized,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            controlnet_conditioning_scale=strength,
            negative_prompt=negative_prompt
        ).images[0]
        
        return generated_image
    
    def generate_multi_pose_faces(
        self,
        input_image_path: str,
        output_dir: str = "generated_poses",
        poses: List[str] = ["side", "down", "up", "left", "right"],
        angles: Optional[Dict[str, float]] = None,
        prompt: str = "a high quality portrait photo of a person",
        num_inference_steps: int = 20
    ) -> Dict[str, str]:
        """
        生成同一人的多姿态人脸
        
        Args:
            input_image_path: 输入正面人脸图像路径
            output_dir: 输出目录
            poses: 目标姿态列表
            angles: 每个姿态的旋转角度字典
            prompt: 生成提示词
            num_inference_steps: 推理步数
            
        Returns:
            output_paths: 生成图像的路径字典
        """
        if angles is None:
            angles = {
                "side": 45.0,
                "down": 0.0,
                "up": 0.0,
                "left": 30.0,
                "right": -30.0
            }
        
        # 创建输出目录
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        # 提取原始人脸和姿态
        print(f"\n正在处理输入图像: {input_image_path}")
        face_image, original_pose_image = self.extract_face_pose(input_image_path)
        
        # 保存原始人脸和姿态
        original_face_path = output_path / "00_original_face.jpg"
        original_pose_path = output_path / "00_original_pose.jpg"
        face_image.save(original_face_path)
        original_pose_image.save(original_pose_path)
        
        output_paths = {
            "original_face": str(original_face_path),
            "original_pose": str(original_pose_path)
        }
        
        # 生成每个姿态的人脸
        for i, pose in enumerate(poses, 1):
            print(f"\n正在生成姿态 {i}/{len(poses)}: {pose}")
            
            try:
                # 修改姿态关键点
                modified_pose = self.modify_pose_keypoints(
                    original_pose_image,
                    target_pose=pose,
                    angle=angles.get(pose, 45.0)
                )
                
                # 生成人脸图像
                generated_image = self.generate_face_with_pose(
                    face_image,
                    modified_pose,
                    prompt=prompt,
                    num_inference_steps=num_inference_steps
                )
                
                # 保存结果
                output_filename = f"{i:02d}_{pose}_face.jpg"
                output_filepath = output_path / output_filename
                generated_image.save(output_filepath)
                
                output_paths[pose] = str(output_filepath)
                print(f"  ✓ 已保存: {output_filepath}")
                
            except Exception as e:
                print(f"  ✗ 生成失败: {str(e)}")
                import traceback
                traceback.print_exc()
        
        return output_paths


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="生成多姿态人脸")
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="输入正面人脸图像路径"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="generated_poses",
        help="输出目录（默认: generated_poses）"
    )
    parser.add_argument(
        "--poses",
        nargs="+",
        default=["side", "down", "up", "left", "right"],
        help="目标姿态列表（默认: side down up left right）"
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="a high quality portrait photo of a person",
        help="生成提示词"
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=20,
        help="推理步数（默认: 20）"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "cpu"],
        help="计算设备（默认: 自动选择）"
    )
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input):
        print(f"错误: 输入文件不存在: {args.input}")
        return
    
    # 初始化生成器
    print("=" * 70)
    print("多姿态人脸生成系统")
    print("=" * 70)
    
    try:
        generator = MultiPoseFaceGenerator(device=args.device)
        
        # 生成多姿态人脸
        output_paths = generator.generate_multi_pose_faces(
            input_image_path=args.input,
            output_dir=args.output_dir,
            poses=args.poses,
            prompt=args.prompt,
            num_inference_steps=args.steps
        )
        
        # 输出结果
        print("\n" + "=" * 70)
        print("生成完成！")
        print("=" * 70)
        print("\n生成的文件:")
        for key, path in output_paths.items():
            print(f"  {key}: {path}")
        
    except Exception as e:
        print(f"\n错误: {str(e)}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()

