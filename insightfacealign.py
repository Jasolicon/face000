# *_*coding:utf-8 *_*
"""
视频人脸检测与对齐脚本

概述:
该脚本用于从视频中检测人脸并对齐。如果整个视频中都未检测到人脸，则从每一帧的中心裁剪一个方形区域并保存。脚本支持多 GPU 并行处理，以提高处理速度。

依赖项:
- Python > 3.9
- OpenCV
- insightface
- onnx
- onnxruntime-gpu

环境变量:
在运行脚本之前，可以设置 CUDA_VISIBLE_DEVICES 环境变量来指定可用的 GPU。例如：
export CUDA_VISIBLE_DEVICES=0,4

参数说明:
- video_dir: 包含视频文件的目录。
- out_dir: 保存处理后图像的输出目录。
- multi_process: 是否启用多进程处理，默认为 True。
- video_template_path: 视频文件的模板路径，默认为 *.mp4。
- img_size: 对齐后图像的尺寸，默认为 112。
- available_gpus: 可用 GPU 的列表，默认为 [0, 4]。

处理逻辑:
1. 检测并对齐人脸：从视频帧中检测人脸并对齐。
2. 处理单个视频：如果整个视频中都未检测到人脸，则从每一帧的中心裁剪一个方形区域并保存。
3. 主函数：遍历所有视频文件，并行处理每个视频。

输出:
处理后的图像将保存在指定的输出目录中。如果检测到人脸，保存对齐后的人脸图像；如果未检测到人脸，保存中心裁剪的图像。
"""

import os
import glob
import time
import shutil
import cv2
import logging
from joblib import Parallel, delayed
from tqdm import tqdm
from insightface.app import FaceAnalysis
from insightface.utils import face_align
from contextlib import redirect_stdout, redirect_stderr
import sys

# 设置日志级别为 ERROR，以屏蔽不必要的输出
logging.getLogger('insightface').setLevel(logging.ERROR)

def set_cuda_visible_devices(thread_idx, available_gpus):
    """
    根据线程索引设置 CUDA_VISIBLE_DEVICES 环境变量

    参数:
    - thread_idx: 线程索引
    - available_gpus: 可用 GPU 的列表
    """
    gpu_id = available_gpus[thread_idx % len(available_gpus)]
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)

def detect_and_align_faces(image, detector, aligned_size=224, video_file=None, frame_idx=None):
    """
    检测并对齐人脸

    参数:
    - image: 输入图像
    - detector: FaceAnalysis 对象
    - aligned_size: 对齐后图像的尺寸，默认为 224
    - video_file: 视频文件名，用于调试输出
    - frame_idx: 帧号，用于调试输出

    返回:
    - aligned_faces: 对齐后的人脸图像列表
    """
    # 检测人脸
    faces = detector.get(image)
    if not faces:
        print(f"未检测到人脸: 视频文件 {video_file}, 帧号 {frame_idx}")
        return []

    # 裁剪和对齐人脸
    aligned_faces = []
    for face in faces:
        bbox = face.bbox.astype(int)
        
        aligned_face = face_align.norm_crop(image, landmark=face.kps, image_size=aligned_size)
        if aligned_face is None or aligned_face.size == 0:
            print(f"对齐后的图像为空: 视频文件 {video_file}, 帧号 {frame_idx}, Bounding box: {bbox}")
            continue
        aligned_faces.append(aligned_face)

    return aligned_faces

def process_one_video(video_file, in_dir, out_dir, img_size=112, thread_idx=0, available_gpus=[0, 4], use_cpu=False):
    
    # 获取文件名（不带扩展名）
    file_name = os.path.splitext(os.path.basename(video_file))[0]
    out_dir = os.path.join(out_dir, file_name)
    if os.path.exists(out_dir):
        print(f'Note: "{out_dir}" already exist!')
        return video_file
    else:
        os.makedirs(out_dir)

    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    set_cuda_visible_devices(thread_idx, available_gpus)

    # 初始化检测器
    with open(os.devnull, 'w') as devnull:
        with redirect_stdout(devnull), redirect_stderr(devnull):
            detector = FaceAnalysis(name='buffalo_l')
            # ctx_id=-1 表示使用 CPU，ctx_id=0 表示使用 GPU
            # 如果遇到 cuDNN 错误，使用 CPU 模式：ctx_id=-1
            if use_cpu:
                # 强制使用 CPU 模式
                detector.prepare(ctx_id=-1, det_size=(640, 640))
            else:
                try:
                    # 尝试使用 GPU
                    detector.prepare(ctx_id=0, det_size=(640, 640))
                except Exception as e:
                    # 如果 GPU 失败，回退到 CPU
                    print(f"GPU 初始化失败，使用 CPU 模式: {e}")
                    detector.prepare(ctx_id=-1, det_size=(640, 640))

    # 读取视频帧并处理
    cap = cv2.VideoCapture(video_file)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detected_any_face = False
    frames = []

    for i in range(frame_count):
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
        aligned_faces = detect_and_align_faces(frame, detector, aligned_size=img_size, video_file=video_file, frame_idx=i)
        if aligned_faces:
            detected_any_face = True
            for idx, aligned_face in enumerate(aligned_faces):
                output_path = os.path.join(out_dir, f'{i:05d}_face_{idx}.png')
                cv2.imwrite(output_path, aligned_face)
                #print(f"保存对齐后的脸部图像: {output_path}")

    # 如果整个视频中都没有检测到人脸，则从每一帧的中心裁剪一个方形区域并保存
    if not detected_any_face:
        print(f"整个视频未检测到人脸: {video_file}")
        for i, frame in enumerate(frames):
            h, w = frame.shape[:2]
            min_dim = min(h, w)
            start_x = (w - min_dim) // 2
            start_y = (h - min_dim) // 2
            center_crop = frame[start_y:start_y + min_dim, start_x:start_x + min_dim]
            center_crop_resized = cv2.resize(center_crop, (img_size, img_size))
            output_path = os.path.join(out_dir, f'{i:05d}_center.png')
            cv2.imwrite(output_path, center_crop_resized)
            print(f"保存中心裁剪图像: {output_path}")

    cap.release()
    return video_file

def main(video_dir, out_dir, multi_process=True, video_template_path='*.mp4', img_size=112, available_gpus=[0, 4], use_cpu=False):
    print(f'out_dir: {out_dir}')

    #===============Need to be modified for different dataset===============
    video_files = glob.glob(os.path.join(video_dir, video_template_path))  # with emo dir

    n_files = len(video_files)
    print(f'Total videos: {n_files}.')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    
    start_time = time.time()
    if multi_process:
        Parallel(n_jobs=16)(delayed(process_one_video)(video_file, video_dir, out_dir, img_size, thread_idx, available_gpus, use_cpu) \
                     for thread_idx, video_file in enumerate(tqdm(video_files)))
    else:
        for i, video_file in enumerate(video_files, 1):
            print(f'Processing "{os.path.basename(video_file)}"...')
            process_one_video(video_file, video_dir, out_dir, img_size, thread_idx=i, available_gpus=available_gpus, use_cpu=use_cpu)
            print(f'"{os.path.basename(video_file)}" done, rate of progress: {100.0 * i / n_files:3.0f}% ({i}/{n_files})')

    end_time = time.time()
    print('Time used for video face extraction: {:.1f} s'.format(end_time - start_time))

if __name__ == "__main__":
    main(
        video_dir=r'C:\Codes\face000\train\datas\video',
        out_dir=r'C:\Codes\face000\train\datas_aligned',
        multi_process=True,
        video_template_path='*.mp4',
        img_size=112,
        available_gpus=[0, 4],
        use_cpu=True  # 如果遇到 cuDNN 错误，设置为 True 使用 CPU 模式
    )