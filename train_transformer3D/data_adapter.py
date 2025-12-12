"""
数据适配器：将不同来源的3D数据转换为统一格式
"""
import numpy as np
from typing import Dict, Optional, Union
from pathlib import Path
import sys

# 添加父目录到路径
sys.path.append(str(Path(__file__).parent.parent))

try:
    import mediapipe as mp
    MEDIAPIPE_AVAILABLE = True
except ImportError:
    MEDIAPIPE_AVAILABLE = False
    print("⚠️  MediaPipe未安装，无法使用MediaPipe适配器")


class Face3DDataAdapter:
    """
    将MediaPipe或InsightFace的3D输出适配为模型输入
    """
    
    @staticmethod
    def from_mediapipe(mediapipe_results, image_size=(640, 480)):
        """
        转换MediaPipe输出
        
        Args:
            mediapipe_results: MediaPipe FaceMesh.process()返回的结果
            image_size: 图像尺寸 (width, height)
        
        Returns:
            dict: 包含以下键的字典：
                - keypoints_3d: 3D关键点 [478, 3] 或 None
                - pose: 姿态向量 [3] 或 None
                - landmarks: MediaPipe landmarks对象
        """
        if not MEDIAPIPE_AVAILABLE:
            raise ImportError("MediaPipe未安装，无法使用此适配器")
        
        if not mediapipe_results.multi_face_landmarks:
            return None
        
        face_landmarks = mediapipe_results.multi_face_landmarks[0]
        
        # 提取3D关键点（MediaPipe有478个点）
        keypoints_3d = []
        for lm in face_landmarks.landmark:
            # MediaPipe的坐标是归一化的 [0, 1]
            # z是相对深度，需要适当缩放
            # 转换为实际像素坐标（如果需要）
            x = lm.x * image_size[0]
            y = lm.y * image_size[1]
            z = lm.z * image_size[0]  # z通常用宽度作为参考
            keypoints_3d.append([x, y, z])
        
        keypoints_3d = np.array(keypoints_3d, dtype=np.float32)
        
        # 提取头部姿态（需要额外计算，或使用MediaPipe的pose_estimation）
        # 这里返回一个示例姿态向量（实际应用中需要计算）
        # 可以使用关键点估计姿态，或使用专门的姿态估计模块
        pose = np.array([0.0, 0.0, 0.0], dtype=np.float32)  # 应替换为真实姿态
        
        return {
            'keypoints_3d': keypoints_3d,  # [478, 3]
            'pose': pose,  # [3]
            'landmarks': face_landmarks
        }
    
    @staticmethod
    def from_insightface(face_obj, landmarks_3d=None, euler_angles=None):
        """
        转换InsightFace输出（假设已通过utils_3d.py计算了pose和kps_3d）
        
        Args:
            face_obj: insightface.app.FaceAnalysis返回的face对象
            landmarks_3d: 3D关键点 [num_kp, 3]（可选，如果face_obj没有kps_3d属性）
            euler_angles: 欧拉角 [3]（可选，如果face_obj没有pose属性）
        
        Returns:
            dict: 包含以下键的字典：
                - keypoints_3d: 3D关键点 [num_kp, 3]
                - pose: 姿态向量 [3] (欧拉角)
                - bbox: 边界框 [4]
                - landmarks_2d: 2D关键点 [num_kp, 2]
        """
        # 检查是否有3D关键点和姿态
        if hasattr(face_obj, 'kps_3d') and face_obj.kps_3d is not None:
            keypoints_3d = face_obj.kps_3d
        elif landmarks_3d is not None:
            keypoints_3d = landmarks_3d
        else:
            raise ValueError("需要提供3D关键点（通过face_obj.kps_3d或landmarks_3d参数）")
        
        # 检查是否有姿态
        if hasattr(face_obj, 'pose') and face_obj.pose is not None:
            if isinstance(face_obj.pose, dict):
                pose = face_obj.pose.get('euler_angles', None)
            else:
                pose = face_obj.pose
        elif euler_angles is not None:
            pose = euler_angles
        else:
            raise ValueError("需要提供姿态（通过face_obj.pose或euler_angles参数）")
        
        # 确保是numpy数组
        if not isinstance(keypoints_3d, np.ndarray):
            keypoints_3d = np.array(keypoints_3d, dtype=np.float32)
        if not isinstance(pose, np.ndarray):
            pose = np.array(pose, dtype=np.float32)
        
        # 确保形状正确
        if len(keypoints_3d.shape) != 2 or keypoints_3d.shape[1] != 3:
            raise ValueError(f"keypoints_3d形状错误: {keypoints_3d.shape}，应为 [num_kp, 3]")
        if pose.shape != (3,):
            raise ValueError(f"pose形状错误: {pose.shape}，应为 [3]")
        
        return {
            'keypoints_3d': keypoints_3d,  # [num_kp, 3]
            'pose': pose,  # [3] 欧拉角
            'bbox': face_obj.bbox if hasattr(face_obj, 'bbox') else None,
            'landmarks_2d': face_obj.kps if hasattr(face_obj, 'kps') else None
        }
    
    @staticmethod
    def from_dict(data_dict: Dict):
        """
        从字典转换（用于从JSON文件加载的数据）
        
        Args:
            data_dict: 包含以下键的字典：
                - landmarks_3d: 3D关键点列表
                - euler_angles: 欧拉角列表
                - 其他可选字段
        
        Returns:
            dict: 标准格式的字典
        """
        keypoints_3d = np.array(data_dict.get('landmarks_3d', []), dtype=np.float32)
        euler_angles = np.array(data_dict.get('euler_angles', [0.0, 0.0, 0.0]), dtype=np.float32)
        
        if len(keypoints_3d) == 0:
            raise ValueError("缺少landmarks_3d字段")
        
        return {
            'keypoints_3d': keypoints_3d,
            'pose': euler_angles,
            'bbox': data_dict.get('box', None),
            'landmarks_2d': np.array(data_dict.get('landmarks_2d', []), dtype=np.float32) if 'landmarks_2d' in data_dict else None
        }


def normalize_keypoints_3d(keypoints_3d: np.ndarray, method: str = 'center_scale'):
    """
    归一化3D关键点
    
    Args:
        keypoints_3d: 3D关键点 [num_kp, 3]
        method: 归一化方法
            - 'center_scale': 中心化并缩放
            - 'unit_sphere': 投影到单位球面
            - 'min_max': 最小-最大归一化
    
    Returns:
        normalized_keypoints: 归一化后的关键点 [num_kp, 3]
    """
    if method == 'center_scale':
        # 中心化
        center = keypoints_3d.mean(axis=0)
        centered = keypoints_3d - center
        
        # 缩放（使用最大距离）
        max_dist = np.linalg.norm(centered, axis=1).max()
        if max_dist > 0:
            normalized = centered / max_dist
        else:
            normalized = centered
        
        return normalized
    
    elif method == 'unit_sphere':
        # 投影到单位球面
        norms = np.linalg.norm(keypoints_3d, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)  # 避免除零
        normalized = keypoints_3d / norms
        return normalized
    
    elif method == 'min_max':
        # 最小-最大归一化
        min_vals = keypoints_3d.min(axis=0)
        max_vals = keypoints_3d.max(axis=0)
        ranges = max_vals - min_vals
        ranges = np.where(ranges > 0, ranges, 1.0)  # 避免除零
        normalized = (keypoints_3d - min_vals) / ranges
        return normalized
    
    else:
        raise ValueError(f"未知的归一化方法: {method}")


if __name__ == "__main__":
    # 测试数据适配器
    print("=" * 70)
    print("测试 Face3DDataAdapter")
    print("=" * 70)
    
    # 测试从字典转换
    print("\n测试从字典转换...")
    test_dict = {
        'landmarks_3d': [[-36.0, 13.0, -42.0], [36.0, 13.0, -42.0], [0.0, 0.0, 0.0],
                         [-25.0, -30.0, -42.0], [25.0, -30.0, -42.0]],
        'euler_angles': [10.5, -5.2, 2.1],
        'box': [100, 100, 200, 200],
        'landmarks_2d': [[150, 120], [250, 120], [200, 150], [180, 200], [220, 200]]
    }
    
    try:
        result = Face3DDataAdapter.from_dict(test_dict)
        print(f"✓ 转换成功")
        print(f"  keypoints_3d形状: {result['keypoints_3d'].shape}")
        print(f"  pose形状: {result['pose'].shape}")
        print(f"  bbox: {result['bbox']}")
        print(f"  landmarks_2d形状: {result['landmarks_2d'].shape if result['landmarks_2d'] is not None else None}")
    except Exception as e:
        print(f"❌ 错误: {e}")
    
    # 测试归一化
    print("\n测试关键点归一化...")
    test_keypoints = np.array([
        [-36.0, 13.0, -42.0],
        [36.0, 13.0, -42.0],
        [0.0, 0.0, 0.0],
        [-25.0, -30.0, -42.0],
        [25.0, -30.0, -42.0]
    ], dtype=np.float32)
    
    for method in ['center_scale', 'unit_sphere', 'min_max']:
        try:
            normalized = normalize_keypoints_3d(test_keypoints, method=method)
            print(f"✓ {method}: 形状={normalized.shape}, 范围=[{normalized.min():.2f}, {normalized.max():.2f}]")
        except Exception as e:
            print(f"❌ {method}: 错误={e}")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)
