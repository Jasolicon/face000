"""
主程序 - 人脸检测、特征提取、保存和比对示例
"""
import os
from pathlib import Path
from face_detector import FaceDetector
from feature_extractor import DINOv2FeatureExtractor, FaceNetFeatureExtractor
from feature_manager import FeatureManager
from feature_matcher import FaceMatcher
from PIL import Image


class FaceRecognitionSystem:
    """完整的人脸识别系统"""
    
    def __init__(self, use_facenet=False, storage_dir='features'):
        """
        初始化人脸识别系统
        
        Args:
            use_facenet: 是否使用FaceNet提取人脸特征（True）或DINOv2提取图像特征（False）
            storage_dir: 特征存储目录
        """
        self.use_facenet = use_facenet
        
        # 初始化组件
        print("正在初始化人脸检测器...")
        self.face_detector = FaceDetector()
        
        print("正在初始化特征提取器...")
        if use_facenet:
            self.feature_extractor = FaceNetFeatureExtractor()
        else:
            # 使用 DINOv2，并在提取特征前先缩放到 96*96
            self.feature_extractor = DINOv2FeatureExtractor(resize_to_96=False)
        
        print("正在初始化特征管理器...")
        self.feature_manager = FeatureManager(storage_dir=storage_dir)
        
        print("正在初始化特征比对器...")
        self.face_matcher = FaceMatcher(
            feature_manager=self.feature_manager,
            similarity_threshold=0.7
        )
        
        print("系统初始化完成！")
    
    def register_image(self, image_path, person_id=None, person_name=None):
        """
        注册图像（检测人脸、提取特征并保存）
        
        Args:
            image_path: 图像路径
            person_id: 人员ID（可选）
            person_name: 人员姓名（可选），如果为None则从文件名自动提取
            
        Returns:
            success: 是否成功注册
            message: 状态消息
        """
        # 如果未提供person_name，从文件名提取
        if person_name is None:
            person_name = os.path.basename(image_path).split('.')[0]
        try:
            # 检测人脸
            if self.use_facenet:
                # 使用FaceNet时，需要提取对齐的人脸
                aligned_face, box, prob = self.face_detector.extract_aligned_face(image_path)
                if aligned_face is None:
                    return False, "未检测到人脸"
                
                # 提取特征
                features = self.feature_extractor.extract_features(aligned_face)
            else:
                # 使用DINOv2时，直接提取整张图像的特征（会自动先缩放到96*96）
                features = self.feature_extractor.extract_features(image_path)
            
            # 保存特征
            self.feature_manager.save_feature(
                features, image_path, person_id, person_name
            )
            
            return True, f"成功注册图像: {image_path}"
            
        except Exception as e:
            return False, f"注册失败: {str(e)}"
    
    def register_batch(self, image_paths, person_ids=None, person_names=None, show_progress=True):
        """
        批量注册图像
        
        Args:
            image_paths: 图像路径列表
            person_ids: 人员ID列表（可选）
            person_names: 人员姓名列表（可选）
            show_progress: 是否显示进度
            
        Returns:
            results: 结果列表，每个元素为 (image_path, success, message)
        """
        results = []
        total = len(image_paths)
        
        # 处理默认值：如果未提供person_names，从文件名提取
        if person_names is None:
            person_names = [os.path.basename(img_path).split('.')[0] for img_path in image_paths]
        if person_ids is None:
            person_ids = [None] * len(image_paths)
        
        for idx, img_path in enumerate(image_paths):
            if show_progress:
                print(f"处理进度: {idx+1}/{total} - {os.path.basename(img_path)}", end='\r')
            
            # 传递person_id和person_name参数
            success, message = self.register_image(
                img_path,
                person_id=person_ids[idx] if idx < len(person_ids) else None,
                person_name=person_names[idx] if idx < len(person_names) else None
            )
            results.append((img_path, success, message))
        
        if show_progress:
            print()  # 换行
        
        return results
    
    def identify_image(self, image_path, top_k=5):
        """
        识别图像中的人员
        
        Args:
            image_path: 查询图像路径
            top_k: 返回最相似的top_k个结果
            
        Returns:
            matches: 匹配结果列表
        """
        try:
            # 提取特征
            if self.use_facenet:
                aligned_face, box, prob = self.face_detector.extract_aligned_face(image_path)
                if aligned_face is None:
                    return []
                features = self.feature_extractor.extract_features(aligned_face)
            else:
                # 使用DINOv2时，直接提取整张图像的特征（会自动先缩放到96*96）
                features = self.feature_extractor.extract_features(image_path)
            
            # 比对特征
            matches = self.face_matcher.match_face(features, top_k=top_k)
            
            return matches
            
        except Exception as e:
            print(f"识别失败: {str(e)}")
            return []
    
    def get_statistics(self):
        """获取系统统计信息"""
        count = self.feature_manager.get_count()
        return {
            'total_features': count,
            'storage_dir': str(self.feature_manager.storage_dir)
        }


def main():
    """主函数示例"""
    print("=" * 60)
    print("人脸识别系统示例")
    print("=" * 60)
    
    # 初始化系统（使用DINOv2提取图像特征，提取前会先缩放到96*96）
    system = FaceRecognitionSystem(use_facenet=False, storage_dir='features_96')
    
    # 示例1: 注册图像
    print("\n[示例1] 注册图像到数据库")
    print("-" * 60)
    
    # 假设有一些图像文件
    image_dir = r'C:\AIXLAB\DATA\smartschool_student_face_NX\NX_standard'
    sample_images = []
    if os.path.exists(image_dir):
        sample_images = list(Path(image_dir).glob('*.jpg')) + \
                       list(Path(image_dir).glob('*.png'))
    
    if sample_images:
        print(f"找到 {len(sample_images)} 张图像")
        print(f"将处理前 {min(3, len(sample_images))} 张图像\n")
        
        # 准备批量注册的数据
        image_paths = [str(img_path) for img_path in sample_images[:3]]
        person_ids = [f"person_{i+1}" for i in range(len(image_paths))]
        person_names = [os.path.basename(img_path).split('.')[0] for img_path in sample_images[:3]]
        
        # 批量注册
        results = system.register_batch(image_paths, person_ids=person_ids, person_names=person_names)
        
        # 显示结果
        success_count = sum(1 for _, success, _ in results if success)
        print(f"\n注册完成: 成功 {success_count}/{len(results)} 张")
        for img_path, success, message in results:
            status = "✓" if success else "✗"
            print(f"  {status} {os.path.basename(img_path)}: {message}")
    else:
        print(f"未找到图像文件在目录: {image_dir}")
        print("提示: 请检查图像路径，或使用以下代码注册图像:")
        print("  system.register_image('path/to/image.jpg', person_name='张三')")
    
    # 示例2: 识别图像
    print("\n[示例2] 识别图像")
    print("-" * 60)
    
    if sample_images:
        query_image = sample_images[0]
        print(f"查询图像: {query_image}")
        matches = system.identify_image(str(query_image), top_k=3)
        
        if matches:
            print(f"找到 {len(matches)} 个匹配:")
            for i, match in enumerate(matches, 1):
                print(f"  {i}. 相似度: {match['similarity']:.4f}")
                print(f"     图像: {match['metadata']['image_path']}")
                if match['metadata'].get('person_name'):
                    print(f"     姓名: {match['metadata']['person_name']}")
        else:
            print("未找到匹配")
    else:
        print("提示: 使用以下代码识别图像:")
        print("  matches = system.identify_image('path/to/query_image.jpg')")
    
    # 示例3: 系统统计
    print("\n[示例3] 系统统计")
    print("-" * 60)
    stats = system.get_statistics()
    print(f"已保存特征数量: {stats['total_features']}")
    print(f"存储目录: {stats['storage_dir']}")
    
    print("\n" + "=" * 60)
    print("示例完成！")
    print("=" * 60)


if __name__ == '__main__':
    main()

