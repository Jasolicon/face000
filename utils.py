"""
工具函数模块 - 关键点检测和角度计算
"""
import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from insightface.app import FaceAnalysis
import math
import logging
import warnings
import os

# 设置日志级别为 ERROR，以屏蔽不必要的输出
logging.getLogger('insightface').setLevel(logging.ERROR)

# 抑制 NumPy 的 FutureWarning（来自 InsightFace 内部使用的 numpy.linalg.lstsq）
warnings.filterwarnings('ignore', category=FutureWarning, message='.*rcond.*')

# 全局检测器（单例模式，避免重复初始化）
_insightface_detector = None

def get_insightface_detector(use_cpu=False, max_retries=3, retry_delay=5):
    """
    获取 InsightFace 检测器（单例模式）
    支持下载重试机制
    
    Args:
        use_cpu: 是否使用 CPU
        max_retries: 最大重试次数（用于模型下载）
        retry_delay: 重试延迟（秒）
        
    Returns:
        detector: FaceAnalysis 检测器
    """
    global _insightface_detector
    if _insightface_detector is None:
        import time
        import sys
        
        # 设置模型下载镜像（在加载模型前）
        try:
            from model_utils import setup_model_mirrors
            setup_model_mirrors()
        except ImportError:
            # 如果 model_utils 不可用，直接设置环境变量
            if 'HF_ENDPOINT' not in os.environ:
                os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
        
        # 尝试下载模型，支持重试
        for attempt in range(max_retries):
            try:
                print(f"正在初始化 InsightFace 检测器（尝试 {attempt + 1}/{max_retries}）...")
                detector = FaceAnalysis(name='buffalo_l')
                
                # 准备检测器
                if use_cpu:
                    detector.prepare(ctx_id=-1, det_size=(640, 640))
                else:
                    try:
                        detector.prepare(ctx_id=0, det_size=(640, 640))
                    except Exception as e:
                        print(f"GPU 初始化失败，使用 CPU: {e}")
                        detector.prepare(ctx_id=-1, det_size=(640, 640))
                
                _insightface_detector = detector
                print("✓ InsightFace 检测器初始化成功")
                return _insightface_detector
                
            except Exception as e:
                error_msg = str(e)
                # 检查是否是下载相关的错误
                is_download_error = any(keyword in error_msg.lower() for keyword in [
                    'incompleteread', 'connection broken', 'chunkedencoding', 
                    'connection error', 'timeout', 'download'
                ])
                
                if is_download_error and attempt < max_retries - 1:
                    print(f"⚠️  模型下载失败（尝试 {attempt + 1}/{max_retries}）: {error_msg[:100]}")
                    print(f"   等待 {retry_delay} 秒后重试...")
                    time.sleep(retry_delay)
                    # 增加重试延迟（指数退避）
                    retry_delay *= 2
                    continue
                else:
                    # 如果是最后一次尝试或非下载错误，抛出异常
                    print(f"\n❌ InsightFace 检测器初始化失败: {error_msg}")
                    if is_download_error:
                        print("\n提示:")
                        print("1. 检查网络连接是否稳定")
                        print("2. 可以尝试手动下载模型:")
                        print("   - 模型位置: ~/.insightface/models/buffalo_l/")
                        print("   - 或设置环境变量: export INSIGHTFACE_ROOT=/path/to/models")
                        print("3. 如果网络不稳定，可以:")
                        print("   - 使用代理")
                        print("   - 多次运行脚本（支持断点续传）")
                        print("   - 手动下载模型文件到指定目录")
                    raise
    
    return _insightface_detector

def get_insightface_landmarks(detector, image_path):
    """
    使用 InsightFace 检测人脸关键点
    
    Args:
        detector: FaceAnalysis 检测器
        image_path: 图像路径（可以是 str 或 PIL Image）
        
    Returns:
        landmarks: 关键点坐标 [5, 2] (左眼、右眼、鼻子、左嘴角、右嘴角)
        box: 人脸边界框 [x1, y1, x2, y2]
    """
    # 处理输入：如果是 PIL Image，转换为 numpy 数组
    if isinstance(image_path, Image.Image):
        img_rgb = np.array(image_path.convert('RGB'))
        img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    else:
        # 直接使用 PIL 读取图像（支持中文路径），避免 cv2.imread 的中文路径问题
        try:
            pil_img = Image.open(image_path).convert('RGB')
            img = np.array(pil_img)
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        except Exception as e:
            # 如果 PIL 读取失败，尝试使用 cv2（可能路径不是中文）
            img = cv2.imread(str(image_path))
            if img is None:
                raise ValueError(f"无法读取图像: {image_path}, 错误: {e}")
    
    # InsightFace 检测人脸和关键点
    faces = detector.get(img)
    
    if len(faces) == 0:
        return None, None
    
    # 使用第一个人脸
    face = faces[0]
    
    # InsightFace 返回的关键点格式：[5, 2] - (左眼、右眼、鼻子、左嘴角、右嘴角)
    landmarks = face.kps  # [5, 2]
    
    # 边界框格式：[x1, y1, x2, y2]
    box = face.bbox  # [x1, y1, x2, y2]
    
    return landmarks, box

def landmarks_to_3d(landmarks, box, img_width, img_height):
    """
    将2D关键点转换为3D坐标（假设人脸在球面上）
    
    Args:
        landmarks: 2D关键点 [5, 2]
        box: 人脸边界框 [x1, y1, x2, y2]
        img_width: 图像宽度
        img_height: 图像高度
        
    Returns:
        landmarks_3d: 3D关键点 [5, 3]
    """
    # 计算人脸中心
    face_center_x = (box[0] + box[2]) / 2
    face_center_y = (box[1] + box[3]) / 2
    face_width = box[2] - box[0]
    face_height = box[3] - box[1]
    
    # 归一化坐标到 [-1, 1] 范围
    landmarks_normalized = landmarks.copy()
    landmarks_normalized[:, 0] = (landmarks[:, 0] - face_center_x) / (face_width / 2)
    landmarks_normalized[:, 1] = (landmarks[:, 1] - face_center_y) / (face_height / 2)
    
    # 转换为3D坐标（假设在单位球面上）
    # 使用球面坐标：x = sin(θ)cos(φ), y = sin(θ)sin(φ), z = cos(θ)
    landmarks_3d = np.zeros((5, 3))
    
    for i in range(5):
        x_norm = landmarks_normalized[i, 0]
        y_norm = landmarks_normalized[i, 1]
        
        # 计算球面坐标
        # 假设人脸在球面上，z坐标基于距离中心的距离
        r = np.sqrt(x_norm**2 + y_norm**2)
        if r > 1.0:
            r = 1.0
        
        # 计算角度
        theta = math.acos(1 - r)  # 从中心到边缘的角度
        phi = math.atan2(y_norm, x_norm) if x_norm != 0 else 0
        
        # 转换为3D坐标
        landmarks_3d[i, 0] = math.sin(theta) * math.cos(phi)
        landmarks_3d[i, 1] = math.sin(theta) * math.sin(phi)
        landmarks_3d[i, 2] = math.cos(theta)
    
    return landmarks_3d

def calculate_spherical_angle(landmarks1_3d, landmarks2_3d, landmarks1_2d=None, landmarks2_2d=None):
    """
    计算两组关键点之间的球面角度偏离（带正负号）
    
    Args:
        landmarks1_3d: 参考图像的3D关键点 [5, 3]
        landmarks2_3d: 当前图像的3D关键点 [5, 3]
        landmarks1_2d: 参考图像的2D关键点 [5, 2]（用于判断正负）
        landmarks2_2d: 当前图像的2D关键点 [5, 2]（用于判断正负）
        
    Returns:
        angles: 每个关键点的角度偏离（度，带正负号）[5]
               正数表示抬头，负数表示低头
        avg_angle: 平均角度偏离（度，带正负号）
    """
    angles = []
    
    for i in range(5):
        # 计算两个3D向量之间的角度
        v1 = landmarks1_3d[i]
        v2 = landmarks2_3d[i]
        
        # 归一化
        v1_norm = v1 / (np.linalg.norm(v1) + 1e-8)
        v2_norm = v2 / (np.linalg.norm(v2) + 1e-8)
        
        # 计算点积
        dot_product = np.clip(np.dot(v1_norm, v2_norm), -1.0, 1.0)
        
        # 计算角度（弧度转度）
        angle_magnitude = math.degrees(math.acos(dot_product))
        
        # 判断正负：根据y坐标变化
        # 如果提供了2D关键点，使用y坐标判断
        if landmarks1_2d is not None and landmarks2_2d is not None:
            # y坐标减小（向上移动）= 抬头 = 正数
            # y坐标增大（向下移动）= 低头 = 负数
            y_diff = landmarks2_2d[i, 1] - landmarks1_2d[i, 1]
            
            # 如果y坐标减小（向上），角度为正；如果y坐标增大（向下），角度为负
            if y_diff < 0:
                # 向上移动，抬头，正数
                angle = angle_magnitude
            elif y_diff > 0:
                # 向下移动，低头，负数
                angle = -angle_magnitude
            else:
                # 没有垂直移动，角度为0或很小的正数
                angle = angle_magnitude if angle_magnitude > 0.1 else 0.0
        else:
            # 如果没有2D关键点，使用3D向量的y分量判断
            # y分量减小 = 抬头 = 正数
            y_diff_3d = v2_norm[1] - v1_norm[1]
            if y_diff_3d < -0.01:  # 向上
                angle = angle_magnitude
            elif y_diff_3d > 0.01:  # 向下
                angle = -angle_magnitude
            else:
                angle = angle_magnitude if angle_magnitude > 0.1 else 0.0
        
        angles.append(angle)
    
    avg_angle = np.mean(angles)
    return np.array(angles), avg_angle

def detect_landmarks_and_calculate_angle(image, reference_landmarks_3d=None, reference_landmarks_2d=None, 
                                         reference_box=None, reference_img_size=None, detector=None, use_cpu=False):
    """
    检测图像中的关键点并计算与参考图像的角度偏离
    
    Args:
        image: 输入图像（可以是路径字符串或 PIL Image）
        reference_landmarks_3d: 参考图像的3D关键点 [5, 3]（可选）
        reference_landmarks_2d: 参考图像的2D关键点 [5, 2]（可选）
        reference_box: 参考图像的边界框 [x1, y1, x2, y2]（可选）
        reference_img_size: 参考图像的尺寸 (width, height)（可选）
        detector: InsightFace 检测器（可选，如果不提供则自动创建）
        use_cpu: 是否使用 CPU
        
    Returns:
        landmarks: 检测到的关键点 [5, 2] 或 None
        box: 检测到的边界框 [x1, y1, x2, y2] 或 None
        angles: 角度偏离 [5] 或 None
        avg_angle: 平均角度偏离（度）或 None
    """
    # 获取检测器
    if detector is None:
        detector = get_insightface_detector(use_cpu=use_cpu)
    
    # 检测关键点
    landmarks, box = get_insightface_landmarks(detector, image)
    
    if landmarks is None or box is None:
        return None, None, None, None
    
    # 如果提供了参考关键点，计算角度
    if reference_landmarks_3d is not None:
        # 获取当前图像尺寸
        if isinstance(image, Image.Image):
            img_width, img_height = image.size
        else:
            img = Image.open(image) if isinstance(image, (str, Path)) else image
            img_width, img_height = img.size
        
        # 转换为3D坐标
        landmarks_3d = landmarks_to_3d(landmarks, box, img_width, img_height)
        
        # 计算角度
        angles, avg_angle = calculate_spherical_angle(
            reference_landmarks_3d, landmarks_3d,
            reference_landmarks_2d, landmarks
        )
        
        return landmarks, box, angles, avg_angle
    else:
        return landmarks, box, None, None

def draw_landmarks_on_image(image, landmarks, box, angles=None, avg_angle=None, 
                            landmark_names=None, colors=None):
    """
    在图像上绘制关键点和角度信息
    
    Args:
        image: PIL Image 对象
        landmarks: 关键点 [5, 2]
        box: 边界框 [x1, y1, x2, y2]
        angles: 角度偏离 [5]（可选）
        avg_angle: 平均角度偏离（度）（可选）
        landmark_names: 关键点名称列表（可选）
        colors: 关键点颜色列表（可选）
        
    Returns:
        annotated_image: 标注后的 PIL Image 对象
    """
    # 创建可绘制的图像副本
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试加载中文字体（支持跨平台）
    try:
        from font_utils import get_chinese_font_pil
        font = get_chinese_font_pil(20)
    except ImportError:
        # 如果 font_utils 不可用，使用旧方法
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 20)  # 黑体
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 20)  # 微软雅黑
            except:
                font = ImageFont.load_default()  # 默认字体（可能不支持中文）
    
    # 默认关键点名称和颜色
    if landmark_names is None:
        landmark_names = ['左眼', '右眼', '鼻子', '左嘴角', '右嘴角']
    if colors is None:
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255)]
    
    # 绘制边界框（绿色）
    draw.rectangle([(int(box[0]), int(box[1])), (int(box[2]), int(box[3]))], 
                   outline=(0, 255, 0), width=2)
    
    # 绘制关键点和角度信息
    for i, (point, name, color) in enumerate(zip(landmarks, landmark_names, colors)):
        x, y = int(point[0]), int(point[1])
        # 绘制圆点
        draw.ellipse([(x-5, y-5), (x+5, y+5)], fill=color)
        
        # 准备文本
        if angles is not None and i < len(angles):
            angle_text = f"{name}: {angles[i]:+.1f}°"
        else:
            angle_text = name
        
        # 绘制文本
        draw.text((x+10, y-10), angle_text, fill=color, font=font)
    
    # 如果有平均角度，在图像顶部显示
    if avg_angle is not None:
        direction = "抬头" if avg_angle > 0 else "低头" if avg_angle < 0 else "平视"
        angle_text = f"平均角度: {avg_angle:+.2f}° ({direction})"
        # 绘制半透明背景
        text_bbox = draw.textbbox((0, 0), angle_text, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        overlay = Image.new('RGBA', draw_image.size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)
        overlay_draw.rectangle(
            [10, 10, 10 + text_width + 10, 10 + text_height + 10],
            fill=(0, 0, 0, 180)
        )
        draw_image = Image.alpha_composite(draw_image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(draw_image)
        draw.text((15, 15), angle_text, fill=(255, 255, 0), font=font)
    
    return draw_image

def deduplicate_recognition_results(results, confidence_key='similarity'):
    """
    处理识别结果，确保每个识别到的姓名只分配给相似度最高的检测框
    
    逻辑：
    1. 对于每个识别到的姓名（排除"未识别"、"检测失败"等），找到所有匹配这个姓名的检测框
    2. 在这些框中，选择相似度（similarity，余弦相似度）最高的那个
    3. 其他框就不能使用这个姓名了，改为"未识别"
    
    Args:
        results: 识别结果列表，每个元素包含：
            - 'box': [x1, y1, x2, y2] 检测框
            - 'name': 识别到的姓名
            - 'similarity': 特征比对相似度（余弦相似度，用于去重）
            - 'prob': 人脸检测置信度（不用于去重）
            - 其他字段（match, landmarks, angles, avg_angle等）
        confidence_key: 用于排序的字段名（默认'similarity'，即使用相似度去重）
        
    Returns:
        processed_results: 处理后的识别结果列表
    """
    if not results:
        return results
    
    # 复制结果列表
    processed_results = [r.copy() for r in results]
    
    # 收集所有识别到的姓名（排除"未识别"、"检测失败"等无效名称）
    invalid_names = {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小', '未知', ''}
    
    # 按姓名分组
    name_to_results = {}
    for i, result in enumerate(processed_results):
        name = result.get('name', '未识别')
        if name not in invalid_names:
            if name not in name_to_results:
                name_to_results[name] = []
            name_to_results[name].append((i, result))
    
    # 对于每个姓名，只保留相似度最高的检测框
    used_indices = set()
    
    for name, result_list in name_to_results.items():
        if len(result_list) > 1:
            # 多个框匹配同一个姓名，按相似度排序（使用 similarity 字段）
            result_list.sort(key=lambda x: x[1].get(confidence_key, 0.0), reverse=True)
            
            # 保留相似度最高的
            best_idx, best_result = result_list[0]
            best_similarity = best_result.get(confidence_key, 0.0)
            used_indices.add(best_idx)
            
            # 其他框改为"未识别"
            for other_idx, other_result in result_list[1:]:
                other_similarity = other_result.get(confidence_key, 0.0)
                processed_results[other_idx]['name'] = '未识别'
                processed_results[other_idx]['similarity'] = 0.0
                processed_results[other_idx]['match'] = None
                print(f"  去重: 框 {other_idx} 的识别结果 '{name}' 已被框 {best_idx} 使用（相似度更高: {best_similarity:.4f} > {other_similarity:.4f}）")
        else:
            # 只有一个框匹配这个姓名，直接保留
            used_indices.add(result_list[0][0])
    
    return processed_results

def draw_recognition_results(image, results, show_landmarks=True, show_angles=True):
    """
    绘制识别结果（包括检测框、姓名、关键点和角度）
    
    Args:
        image: PIL Image 对象
        results: 识别结果列表（已经过 deduplicate_recognition_results 处理）
        show_landmarks: 是否显示关键点
        show_angles: 是否显示角度信息
        
    Returns:
        annotated_image: 标注后的 PIL Image 对象
    """
    from PIL import ImageDraw, ImageFont
    import numpy as np
    
    # 创建可绘制的图像副本
    draw_image = image.copy()
    draw = ImageDraw.Draw(draw_image)
    
    # 尝试加载中文字体（支持跨平台）
    try:
        from font_utils import get_chinese_font_pil
        font = get_chinese_font_pil(24)
        small_font = get_chinese_font_pil(16)
    except ImportError:
        # 如果 font_utils 不可用，使用旧方法
        try:
            font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 24)  # 黑体
            small_font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 16)
        except:
            try:
                font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 24)  # 微软雅黑
                small_font = ImageFont.truetype("C:/Windows/Fonts/msyh.ttc", 16)
            except:
                font = ImageFont.load_default()
                small_font = ImageFont.load_default()
    
    # 准备绘制数据
    boxes = []
    names = []
    similarities = []
    
    for result in results:
        box = result.get('box', [0, 0, 0, 0])
        name = result.get('name', '未识别')
        similarity = result.get('similarity', 0.0)
        
        boxes.append(box)
        names.append(name)
        similarities.append(similarity)
    
    # 绘制检测框和姓名
    for i, (box, name, sim) in enumerate(zip(boxes, names, similarities)):
        x1, y1, x2, y2 = box
        
        # 扩展检测框25%
        width = x2 - x1
        height = y2 - y1
        center_x = (x1 + x2) / 2
        center_y = (y1 + y2) / 2
        
        new_width = width * 1.25
        new_height = height * 1.25
        
        new_x1 = int(center_x - new_width / 2)
        new_y1 = int(center_y - new_height / 2)
        new_x2 = int(center_x + new_width / 2)
        new_y2 = int(center_y + new_height / 2)
        
        # 确保边界框不超出图像范围
        img_width, img_height = image.size
        new_x1 = max(0, min(new_x1, img_width - 1))
        new_y1 = max(0, min(new_y1, img_height - 1))
        new_x2 = max(0, min(new_x2, img_width - 1))
        new_y2 = max(0, min(new_y2, img_height - 1))
        
        # 只绘制扩展的黄色框
        box_color = (255, 255, 0)  # 黄色
        draw.rectangle([new_x1, new_y1, new_x2, new_y2], outline=box_color, width=3)
        
        # 只有识别成功时才绘制标签（未识别不绘制标签）
        if name not in {'未识别', '检测失败', '数据库为空', '维度不匹配', '图像无效', '图像太小'}:
            # 准备文本
            if sim > 0:
                text = f"{name} ({sim:.3f})"
            else:
                text = name
            
            # 计算文本位置
            text_bbox = draw.textbbox((0, 0), text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_height = text_bbox[3] - text_bbox[1]
            
            # 绘制文本背景
            text_x = new_x1
            text_y = new_y1 - text_height - 5
            if text_y < 0:
                text_y = new_y2 + 5
            
            # 绘制半透明背景
            overlay = Image.new('RGBA', draw_image.size, (0, 0, 0, 0))
            overlay_draw = ImageDraw.Draw(overlay)
            overlay_draw.rectangle(
                [text_x - 2, text_y - 2, text_x + text_width + 2, text_y + text_height + 2],
                fill=(0, 0, 0, 180)
            )
            draw_image = Image.alpha_composite(draw_image.convert('RGBA'), overlay).convert('RGB')
            draw = ImageDraw.Draw(draw_image)
            
            # 绘制文本
            text_color = (255, 255, 0)  # 黄色文本
            draw.text((text_x, text_y), text, fill=text_color, font=font)
        
        # 绘制关键点和角度（如果启用）
        if show_landmarks and result.get('landmarks') is not None:
            landmarks = result['landmarks']
            angles = result.get('angles')
            avg_angle = result.get('avg_angle')
            
            # 使用 draw_landmarks_on_image 绘制关键点
            draw_image = draw_landmarks_on_image(
                draw_image,
                landmarks,
                np.array(box),
                angles=angles if show_angles else None,
                avg_angle=avg_angle if show_angles else None
            )
            draw = ImageDraw.Draw(draw_image)  # 重新创建 draw 对象
    
    return draw_image
