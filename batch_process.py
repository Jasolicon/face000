"""
批量处理脚本 - 处理整个目录的图像
支持多个文件夹路径和ArcFace
"""
import os
import sys
import numpy as np
from pathlib import Path
from main import FaceRecognitionSystem
from face_detector import FaceDetector
from feature_extractor import DINOv2FeatureExtractor, FaceNetFeatureExtractor, ArcFaceFeatureExtractor
from feature_manager import FeatureManager


def batch_process_directory(
    image_dir,
    storage_dir='features',
    use_facenet=False,
    use_arcface=False,
    max_images=None,
    person_id_prefix='person'
):
    """
    批量处理目录中的所有图像
    
    Args:
        image_dir: 图像目录路径（字符串或路径列表）
        storage_dir: 特征存储目录
        use_facenet: 是否使用FaceNet（True）或DINOv2（False）
        use_arcface: 是否使用ArcFace（True）
        max_images: 最大处理图像数量，None表示处理所有
        person_id_prefix: 人员ID前缀
    """
    print("=" * 70)
    print("批量图像处理系统")
    print("=" * 70)
    
    # 支持多个目录
    if isinstance(image_dir, str):
        image_dirs = [image_dir]
    else:
        image_dirs = image_dir
    
    # 检查所有目录是否存在
    valid_dirs = []
    for dir_path in image_dirs:
        dir_path_obj = Path(dir_path)
        if dir_path_obj.exists():
            valid_dirs.append(dir_path_obj)
        else:
            print(f"警告: 目录不存在，跳过: {dir_path}")
    
    if not valid_dirs:
        print("错误: 没有有效的目录")
        return
    
    # 从所有目录中查找图像文件
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.JPG', '*.JPEG', '*.PNG']
    all_images = []
    for dir_path in valid_dirs:
        for ext in image_extensions:
            all_images.extend(list(dir_path.glob(ext)))
    
    if not all_images:
        print(f"错误: 在所有目录中未找到图像文件")
        return
    
    # 限制处理数量
    if max_images:
        all_images = all_images[:max_images]
    
    # 确定使用的模型
    if use_arcface:
        model_name = 'ArcFace'
    elif use_facenet:
        model_name = 'FaceNet'
    else:
        model_name = 'DINOv2'
    
    print(f"\n处理 {len(valid_dirs)} 个目录:")
    for dir_path in valid_dirs:
        print(f"  - {dir_path}")
    print(f"\n找到 {len(all_images)} 张图像")
    print(f"存储目录: {storage_dir}")
    print(f"使用模型: {model_name}")
    print("-" * 70)
    
    # 初始化系统
    print("\n正在初始化系统...")
    
    if use_arcface:
        # 使用ArcFace模式
        print("  初始化人脸检测器...")
        face_detector = FaceDetector()
        print("  初始化ArcFace特征提取器...")
        try:
            feature_extractor = ArcFaceFeatureExtractor()
            # 检查是否成功使用insightface
            if not hasattr(feature_extractor, 'use_insightface') or not feature_extractor.use_insightface:
                print("  注意: 使用fallback实现（timm模型），性能可能不如真正的ArcFace")
        except Exception as e:
            print(f"  错误: ArcFace初始化失败: {str(e)}")
            print(f"  提示: 将尝试使用fallback实现")
            # 即使失败也继续，使用fallback实现
            try:
                feature_extractor = ArcFaceFeatureExtractor()
            except:
                print(f"  fallback实现也失败，请检查依赖")
                return
        print("  初始化特征管理器...")
        feature_manager = FeatureManager(storage_dir=storage_dir)
        
        # 准备批量注册数据
        image_paths = [str(img) for img in all_images]
        person_ids = [f"{person_id_prefix}_{i+1:04d}" for i in range(len(all_images))]
        person_names = [img.stem for img in all_images]
        
        # 批量注册（使用ArcFace）
        print(f"\n开始批量注册 {len(image_paths)} 张图像（使用ArcFace）...")
        results = []
        total = len(image_paths)
        
        for idx, img_path in enumerate(image_paths):
            print(f"处理进度: {idx+1}/{total} - {os.path.basename(img_path)}", end='\r')
            
            try:
                # 检测人脸并提取对齐的人脸
                aligned_face, box, prob = face_detector.extract_aligned_face(img_path)
                if aligned_face is None:
                    results.append((img_path, False, "未检测到人脸"))
                    continue
                
                # 提取特征（ArcFace的extract_features已经可以处理Tensor输入）
                features = feature_extractor.extract_features(aligned_face)
                
                # 保存特征
                feature_manager.save_feature(
                    features, 
                    img_path, 
                    person_id=person_ids[idx] if idx < len(person_ids) else None,
                    person_name=person_names[idx] if idx < len(person_names) else None
                )
                
                results.append((img_path, True, f"成功注册: {img_path}"))
            except Exception as e:
                results.append((img_path, False, f"注册失败: {str(e)}"))
        
        print()  # 换行
    else:
        # 使用FaceRecognitionSystem（DINOv2或FaceNet）
        # FaceRecognitionSystem 现在默认使用 DINOv2（带 96×96 缩放）
        system = FaceRecognitionSystem(use_facenet=use_facenet, storage_dir=storage_dir)
        
        # 准备批量注册数据
        image_paths = [str(img) for img in all_images]
        person_ids = [f"{person_id_prefix}_{i+1:04d}" for i in range(len(all_images))]
        person_names = [img.stem for img in all_images]  # 使用文件名（不含扩展名）作为姓名
        
        # 批量注册
        print(f"\n开始批量注册 {len(image_paths)} 张图像...")
        results = system.register_batch(image_paths, person_ids=person_ids, person_names=person_names)
    
    # 统计结果
    success_count = sum(1 for _, success, _ in results if success)
    failed_count = len(results) - success_count
    
    print("\n" + "=" * 70)
    print("处理完成！")
    print("=" * 70)
    print(f"总计: {len(results)} 张")
    print(f"成功: {success_count} 张")
    print(f"失败: {failed_count} 张")
    
    # 显示失败的文件
    if failed_count > 0:
        print("\n失败的文件:")
        for img_path, success, message in results:
            if not success:
                print(f"  ✗ {os.path.basename(img_path)}: {message}")
    
    # 显示系统统计
    if use_arcface:
        stats_count = feature_manager.get_count()
        print(f"\n已保存特征数量: {stats_count}")
        print(f"存储目录: {storage_dir}")
    else:
        stats = system.get_statistics()
        print(f"\n已保存特征数量: {stats['total_features']}")
        print(f"存储目录: {stats['storage_dir']}")


def main():
    """主函数"""
    if len(sys.argv) < 2:
        print("使用方法:")
        print(f"  python {sys.argv[0]} <图像目录1> [图像目录2] ... [选项]")
        print("\n选项:")
        print("  --storage-dir <目录>    特征存储目录 (默认: features)")
        print("  --use-facenet           使用FaceNet而不是DINO")
        print("  --use-arcface           使用ArcFace（优先于FaceNet）")
        print("  --max-images <数量>     最大处理图像数量")
        print("\n示例:")
        print(f"  # 处理单个目录")
        print(f"  python {sys.argv[0]} C:\\AIXLAB\\DATA\\smartschool_student_face_NX\\NX_standard")
        print(f"  # 处理多个目录")
        print(f"  python {sys.argv[0]} dir1 dir2 dir3")
        print(f"  # 使用ArcFace")
        print(f"  python {sys.argv[0]} dir1 dir2 --use-arcface")
        print(f"  # 限制处理数量")
        print(f"  python {sys.argv[0]} dir1 --max-images 100")
        return
    
    # 解析参数：收集所有目录路径和选项
    image_dirs = []
    storage_dir = 'features'
    use_facenet = False
    use_arcface = False
    max_images = None
    
    i = 1
    while i < len(sys.argv):
        arg = sys.argv[i]
        
        if arg.startswith('--'):
            # 这是选项参数
            if arg == '--storage-dir' and i + 1 < len(sys.argv):
                storage_dir = sys.argv[i + 1]
                i += 2
            elif arg == '--use-facenet':
                use_facenet = True
                i += 1
            elif arg == '--use-arcface':
                use_arcface = True
                i += 1
            elif arg == '--max-images' and i + 1 < len(sys.argv):
                max_images = int(sys.argv[i + 1])
                i += 2
            else:
                print(f"未知参数: {arg}")
                i += 1
        else:
            # 这是目录路径
            image_dirs.append(arg)
            i += 1
    
    if not image_dirs:
        print("错误: 请至少提供一个图像目录路径")
        return
    
    # 如果指定了arcface，优先使用arcface
    if use_arcface:
        use_facenet = False
    
    # 执行批量处理
    batch_process_directory(
        image_dir=image_dirs if len(image_dirs) > 1 else image_dirs[0],
        storage_dir=storage_dir,
        use_facenet=use_facenet,
        use_arcface=use_arcface,
        max_images=max_images
    )


if __name__ == '__main__':
    main()

