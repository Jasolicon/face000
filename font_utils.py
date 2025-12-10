"""
中文字体工具函数
支持 Windows、Linux、macOS
"""
import os
import platform
from pathlib import Path
from PIL import ImageFont
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

def get_chinese_font_pil(font_size=20):
    """
    获取 PIL/Pillow 中文字体
    
    Args:
        font_size: 字体大小
        
    Returns:
        font: PIL ImageFont 对象
    """
    system = platform.system()
    font_paths = []
    
    if system == 'Windows':
        # Windows 系统字体路径
        windows_fonts_dir = Path("C:/Windows/Fonts")
        font_paths = [
            windows_fonts_dir / "msyh.ttc",      # 微软雅黑
            windows_fonts_dir / "simhei.ttf",    # 黑体
            windows_fonts_dir / "simsun.ttc",    # 宋体
            windows_fonts_dir / "msyhbd.ttc",    # 微软雅黑 Bold
            windows_fonts_dir / "simkai.ttf",    # 楷体
        ]
    elif system == 'Darwin':  # macOS
        # macOS 系统字体路径
        mac_fonts_dir = Path("/System/Library/Fonts")
        font_paths = [
            mac_fonts_dir / "PingFang.ttc",       # 苹方
            mac_fonts_dir / "STHeiti Light.ttc",  # 黑体
            mac_fonts_dir / "STSong.ttc",         # 宋体
            Path("/Library/Fonts/Arial Unicode.ttf"),  # Arial Unicode
        ]
    else:  # Linux
        # Linux 系统字体路径（常见位置）
        linux_font_dirs = [
            Path("/usr/share/fonts"),
            Path("/usr/share/fonts/truetype"),
            Path("/usr/share/fonts/truetype/wqy"),  # 文泉驿
            Path("/usr/share/fonts/truetype/noto"),  # Noto
            Path("/usr/share/fonts/opentype/noto"), # Noto OpenType
            Path.home() / ".fonts",
            Path.home() / ".local/share/fonts",
        ]
        
        # 常见的 Linux 中文字体文件名
        linux_font_names = [
            "wqy-microhei.ttc",      # 文泉驿微米黑
            "wqy-zenhei.ttc",        # 文泉驿正黑
            "NotoSansCJK-Regular.ttc",  # Noto Sans CJK
            "NotoSansCJK-Bold.ttc",
            "NotoSerifCJK-Regular.ttc",
            "DroidSansFallbackFull.ttf",  # Droid Sans Fallback
            "AR PL UMing CN",        # 文鼎字体
            "AR PL UKai CN",
        ]
        
        # 构建 Linux 字体路径列表
        for font_dir in linux_font_dirs:
            if font_dir.exists():
                for font_name in linux_font_names:
                    # 尝试不同的扩展名
                    for ext in ['.ttc', '.ttf', '.otf']:
                        font_path = font_dir / (font_name + ext)
                        if font_path.exists():
                            font_paths.append(font_path)
                        # 也尝试直接匹配文件名
                        for font_file in font_dir.rglob(f"*{font_name}*"):
                            if font_file.suffix in ['.ttc', '.ttf', '.otf']:
                                font_paths.append(font_file)
        
        # 也尝试通过 fontconfig 查找
        try:
            import subprocess
            result = subprocess.run(
                ['fc-list', ':lang=zh'],
                capture_output=True,
                text=True,
                timeout=2
            )
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if ':' in line:
                        font_path = line.split(':')[0].strip()
                        if os.path.exists(font_path):
                            font_paths.append(Path(font_path))
        except:
            pass
    
    # 尝试加载字体
    for font_path in font_paths:
        if font_path.exists():
            try:
                return ImageFont.truetype(str(font_path), font_size)
            except Exception:
                continue
    
    # 如果找不到，尝试使用系统默认字体
    try:
        # 尝试使用 DejaVu Sans（通常支持基本 Unicode）
        return ImageFont.truetype("DejaVuSans.ttf", font_size)
    except:
        return ImageFont.load_default()


def setup_chinese_font_matplotlib():
    """
    设置 matplotlib 中文字体
    
    Returns:
        bool: 是否成功设置字体
    """
    system = platform.system()
    
    # 根据操作系统选择字体候选列表
    if system == 'Windows':
        font_candidates = [
            'Microsoft YaHei', 'SimHei', 'SimSun', 'KaiTi', 
            'FangSong', 'Microsoft JhengHei'
        ]
    elif system == 'Darwin':  # macOS
        font_candidates = [
            'PingFang SC', 'STHeiti', 'Arial Unicode MS', 
            'STSong', 'STKaiti'
        ]
    else:  # Linux
        font_candidates = [
            'WenQuanYi Micro Hei',      # 文泉驿微米黑
            'WenQuanYi Zen Hei',        # 文泉驿正黑
            'Noto Sans CJK SC',         # Noto Sans
            'Noto Serif CJK SC',        # Noto Serif
            'Droid Sans Fallback',      # Droid Sans
            'AR PL UMing CN',           # 文鼎字体
            'AR PL UKai CN',
            'Source Han Sans CN',       # 思源黑体
            'Source Han Serif CN',       # 思源宋体
            'DejaVu Sans',               # 备用字体
        ]
    
    # 尝试设置字体
    font_set = False
    for font_name in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            # 测试字体是否可用
            test_fig = plt.figure(figsize=(1, 1))
            test_ax = test_fig.add_subplot(111)
            test_ax.text(0.5, 0.5, '测试', fontsize=10)
            plt.close(test_fig)
            font_set = True
            print(f"✓ 已设置 matplotlib 中文字体: {font_name}")
            break
        except Exception:
            continue
    
    # 如果预设字体都不可用，尝试查找系统可用的中文字体
    if not font_set:
        try:
            fonts = [f.name for f in fm.fontManager.ttflist]
            # 查找中文字体关键词
            chinese_keywords = [
                'hei', 'song', 'kai', 'fang', 'yahei', 'simhei', 'simsun',
                'wqy', 'noto', 'source', 'droid', 'uming', 'ukai'
            ]
            chinese_fonts = [
                f for f in fonts 
                if any(keyword in f.lower() for keyword in chinese_keywords)
            ]
            
            if chinese_fonts:
                # 优先选择包含中文关键词的字体
                preferred_fonts = [
                    f for f in chinese_fonts 
                    if any(kw in f.lower() for kw in ['hei', 'noto', 'wqy', 'source'])
                ]
                selected_font = preferred_fonts[0] if preferred_fonts else chinese_fonts[0]
                
                plt.rcParams['font.sans-serif'] = [selected_font]
                plt.rcParams['axes.unicode_minus'] = False
                font_set = True
                print(f"✓ 已设置 matplotlib 中文字体: {selected_font}")
            else:
                print("⚠️  警告: 未找到中文字体，中文可能显示为方块")
                print("   提示: 在 Linux 上可以安装中文字体:")
                print("   sudo apt-get install fonts-wqy-microhei fonts-wqy-zenhei")
                print("   或: sudo yum install wqy-microhei-fonts wqy-zenhei-fonts")
                plt.rcParams['axes.unicode_minus'] = False
        except Exception as e:
            print(f"⚠️  警告: 设置中文字体失败: {e}")
            plt.rcParams['axes.unicode_minus'] = False
    
    return font_set


def list_available_chinese_fonts():
    """
    列出系统中可用的中文字体
    
    Returns:
        list: 可用字体列表
    """
    chinese_fonts = []
    
    # matplotlib 字体
    try:
        fonts = [f.name for f in fm.fontManager.ttflist]
        chinese_keywords = [
            'hei', 'song', 'kai', 'fang', 'yahei', 'simhei', 'simsun',
            'wqy', 'noto', 'source', 'droid', 'uming', 'ukai', 'pingfang'
        ]
        chinese_fonts = [
            f for f in fonts 
            if any(keyword in f.lower() for keyword in chinese_keywords)
        ]
    except:
        pass
    
    return sorted(set(chinese_fonts))


if __name__ == "__main__":
    # 测试字体功能
    print("=" * 70)
    print("中文字体工具测试")
    print("=" * 70)
    
    print(f"\n操作系统: {platform.system()}")
    
    print("\n1. PIL/Pillow 字体:")
    try:
        font = get_chinese_font_pil(20)
        print(f"   ✓ 成功加载字体: {font}")
    except Exception as e:
        print(f"   ✗ 加载失败: {e}")
    
    print("\n2. Matplotlib 字体:")
    setup_chinese_font_matplotlib()
    
    print("\n3. 可用的中文字体:")
    fonts = list_available_chinese_fonts()
    if fonts:
        print(f"   找到 {len(fonts)} 个中文字体:")
        for font in fonts[:10]:
            print(f"     - {font}")
    else:
        print("   未找到中文字体")

