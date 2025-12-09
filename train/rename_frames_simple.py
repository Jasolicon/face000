"""
简单重命名脚本：将 frame_* 重命名为 人名frame_*
"""
from pathlib import Path
import os
import re

# 获取脚本所在目录
script_dir = Path(__file__).parent

# 目标目录（相对于脚本目录）
target_dir = script_dir / "datas" / "video" / "袁润东"
person_name = "袁润东"

if not target_dir.exists():
    print(f"错误: 目录不存在: {target_dir}")
    exit(1)

print(f"处理目录: {target_dir}")
print(f"人名: {person_name}")

# 查找所有 frame_* 格式的文件
frame_files = [f for f in target_dir.iterdir() if f.is_file() and re.match(rf'.*frame_.*', f.name)]

if len(frame_files) == 0:
    print("未找到 frame_* 格式的文件")
    exit(0)

print(f"找到 {len(frame_files)} 个文件\n")

# 重命名文件
renamed_count = 0
for file in sorted(frame_files):
    old_name = file.name
    x = old_name.split('frame_')[1]
    new_name = f"{person_name}frame_{x}"
    new_path = file.parent / new_name
    
    # 检查新文件名是否已存在
    if new_path.exists():
        print(f"  跳过 {old_name} (目标文件已存在)")
        continue
    
    try:
        file.rename(new_path)
        print(f"  ✓ {old_name} -> {new_name}")
        renamed_count += 1
    except Exception as e:
        print(f"  ✗ 重命名失败 {old_name}: {e}")

print(f"\n完成！成功重命名 {renamed_count}/{len(frame_files)} 个文件")

