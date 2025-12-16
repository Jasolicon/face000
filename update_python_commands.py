"""
批量更新所有Markdown文件中的Python命令
将所有python命令改为使用完整路径，并改为单行格式
"""
import re
import os
from pathlib import Path

PYTHON_PATH = "C:/Users/62487/.conda/envs/llm/python.exe"
TARGET_DIR = Path("train_transformer3D")

def convert_multiline_command(content, python_path):
    """将多行Python命令转换为单行"""
    # 匹配代码块中的多行python命令
    # 匹配 ```bash, ```powershell, ```cmd 等代码块
    pattern = r'(```(?:bash|powershell|cmd)?\n)(python\s+[^\n]+(?:\\|`|\^)?\n(?:\s+[^\n]+(?:\\|`|\^)?\n?)*)(```)'
    
    def replace_command(match):
        code_block_start = match.group(1)
        command_lines = match.group(2)
        code_block_end = match.group(3)
        
        # 提取所有参数行
        lines = command_lines.strip().split('\n')
        if not lines:
            return match.group(0)
        
        # 第一行是python命令
        first_line = lines[0].strip()
        
        # 提取脚本路径和参数
        if first_line.startswith('python '):
            script_and_args = first_line[7:].strip()  # 移除 'python '
        else:
            return match.group(0)
        
        # 收集所有参数
        args = []
        for line in lines[1:]:
            line = line.strip()
            # 移除行尾的续行符
            line = line.rstrip('\\').rstrip('`').rstrip('^').strip()
            if line:
                args.append(line)
        
        # 构建单行命令
        if args:
            full_command = f"{python_path} {script_and_args} {' '.join(args)}"
        else:
            full_command = f"{python_path} {script_and_args}"
        
        return f"{code_block_start}{full_command}\n{code_block_end}"
    
    content = re.sub(pattern, replace_command, content, flags=re.MULTILINE)
    
    # 处理单行python命令（不在代码块中的）
    # 匹配单独一行的python命令
    single_line_pattern = r'^(\s*)python\s+([^\n]+)$'
    
    def replace_single_line(match):
        indent = match.group(1)
        command = match.group(2).strip()
        return f"{indent}{python_path} {command}"
    
    content = re.sub(single_line_pattern, replace_single_line, content, flags=re.MULTILINE)
    
    return content

def process_file(file_path, python_path):
    """处理单个文件"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        
        # 转换多行命令
        content = convert_multiline_command(content, python_path)
        
        # 如果内容有变化，写回文件
        if content != original_content:
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            print(f"✓ 已更新: {file_path}")
            return True
        else:
            print(f"- 无需更新: {file_path}")
            return False
    except Exception as e:
        print(f"✗ 错误 {file_path}: {e}")
        return False

def main():
    """主函数"""
    if not TARGET_DIR.exists():
        print(f"目录不存在: {TARGET_DIR}")
        return
    
    md_files = list(TARGET_DIR.glob("*.md"))
    print(f"找到 {len(md_files)} 个Markdown文件")
    print(f"Python路径: {PYTHON_PATH}\n")
    
    updated_count = 0
    for md_file in md_files:
        if process_file(md_file, PYTHON_PATH):
            updated_count += 1
    
    print(f"\n完成！更新了 {updated_count} 个文件")

if __name__ == "__main__":
    main()
