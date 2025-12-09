"""
测试多角度人脸识别模型
"""
import sys
import torch
import torch.nn.functional as F
from PIL import Image
import numpy as np
from pathlib import Path
import argparse
from typing import List, Tuple

# 添加项目根目录和train目录到路径
train_dir = Path(__file__).parent
project_root = train_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(train_dir))

# 使用相对导入（从train目录内运行）
from model import MultiAngleFaceModel
from face_detector import FaceDetector


def load_image(image_path: str, image_size: int = 224) -> torch.Tensor:
    """加载并预处理图像"""
    image = Image.open(image_path).convert('RGB')
    
    # 检测并裁剪人脸
    face_detector = FaceDetector()
    faces, boxes, probs = face_detector.detect_faces(image)
    
    if len(faces) == 0:
        raise ValueError(f"未检测到人脸: {image_path}")
    
    face = faces[0]
    face = face.resize((image_size, image_size), Image.LANCZOS)
    
    # 转换为tensor
    face_array = np.array(face).astype(np.float32) / 255.0
    
    # 归一化 (CLIP标准)
    mean = np.array([0.48145466, 0.4578275, 0.40821073])
    std = np.array([0.26862954, 0.26130258, 0.27577711])
    face_array = (face_array - mean) / std
    
    # 转换为tensor [C, H, W]
    face_tensor = torch.from_numpy(face_array).permute(2, 0, 1)
    
    return face_tensor.unsqueeze(0)  # [1, C, H, W]


def extract_features(
    model: MultiAngleFaceModel,
    image_paths: List[str],
    device: torch.device
) -> torch.Tensor:
    """提取图像特征"""
    model.eval()
    features_list = []
    
    with torch.no_grad():
        for image_path in image_paths:
            image_tensor = load_image(image_path).to(device)
            feature = model.extract_feature(image_tensor)
            features_list.append(feature)
    
    return torch.stack(features_list)


def compare_features(
    query_feature: torch.Tensor,
    gallery_features: torch.Tensor,
    gallery_paths: List[str],
    top_k: int = 5
) -> List[Tuple[str, float]]:
    """
    对比特征并返回最相似的top_k结果
    
    Args:
        query_feature: 查询特征 [D]
        gallery_features: 图库特征 [N, D]
        gallery_paths: 图库图像路径列表
        top_k: 返回top_k个结果
        
    Returns:
        结果列表 [(path, similarity), ...]
    """
    # 计算相似度
    similarities = F.cosine_similarity(
        query_feature.unsqueeze(0),
        gallery_features,
        dim=1
    )
    
    # 获取top_k
    top_k_indices = similarities.topk(min(top_k, len(gallery_paths))).indices
    
    results = []
    for idx in top_k_indices:
        results.append((gallery_paths[idx], similarities[idx].item()))
    
    return results


def main():
    parser = argparse.ArgumentParser(description='测试多角度人脸识别模型')
    parser.add_argument('--checkpoint', type=str, required=True, help='模型检查点路径')
    parser.add_argument('--query', type=str, required=True, help='查询图像路径（多角度人脸）')
    parser.add_argument('--gallery', type=str, required=True, help='图库目录（包含正脸图片）')
    parser.add_argument('--top_k', type=int, default=5, help='返回top_k个结果')
    parser.add_argument('--device', type=str, default='cuda', help='计算设备')
    parser.add_argument('--threshold', type=float, default=0.6, help='相似度阈值')
    
    args = parser.parse_args()
    
    # 设备
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    # 加载模型
    print(f"加载模型: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location=device)
    
    model = MultiAngleFaceModel(
        feature_dim=512,
        num_classes=checkpoint.get('num_classes', None)
    ).to(device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print("✓ 模型加载完成")
    
    # 加载图库（正脸图片）
    print(f"加载图库: {args.gallery}")
    gallery_dir = Path(args.gallery)
    gallery_paths = []
    for ext in ['.jpg', '.jpeg', '.png', '.JPG', '.JPEG', '.PNG']:
        gallery_paths.extend(list(gallery_dir.glob(f'*{ext}')))
    
    if len(gallery_paths) == 0:
        print(f"错误: 图库目录中未找到图像")
        return
    
    print(f"  找到 {len(gallery_paths)} 张图像")
    
    # 提取图库特征
    print("提取图库特征...")
    gallery_features = extract_features(model, [str(p) for p in gallery_paths], device)
    print(f"✓ 图库特征提取完成")
    
    # 提取查询特征
    print(f"提取查询特征: {args.query}")
    query_feature = extract_features(model, [args.query], device)[0]
    print(f"✓ 查询特征提取完成")
    
    # 对比特征
    print("对比特征...")
    results = compare_features(
        query_feature,
        gallery_features,
        [str(p) for p in gallery_paths],
        top_k=args.top_k
    )
    
    # 显示结果
    print("\n" + "=" * 70)
    print("识别结果")
    print("=" * 70)
    print(f"查询图像: {args.query}")
    print(f"\nTop {len(results)} 匹配结果:")
    
    for i, (path, similarity) in enumerate(results, 1):
        status = "✓ 匹配" if similarity >= args.threshold else "✗ 不匹配"
        print(f"{i}. {Path(path).name}")
        print(f"   相似度: {similarity:.4f} {status}")
    
    # 最佳匹配
    if results:
        best_path, best_sim = results[0]
        if best_sim >= args.threshold:
            print(f"\n✓ 识别成功: {Path(best_path).name} (相似度: {best_sim:.4f})")
        else:
            print(f"\n✗ 未找到匹配 (最高相似度: {best_sim:.4f} < 阈值 {args.threshold})")


if __name__ == '__main__':
    main()

