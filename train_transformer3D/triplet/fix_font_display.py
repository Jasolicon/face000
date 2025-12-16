"""
修复字体显示问题的工具脚本
清除matplotlib字体缓存并重新生成
"""
import os
import shutil
import matplotlib
import matplotlib.font_manager as fm

def clear_matplotlib_cache():
    """清除matplotlib字体缓存"""
    cache_dir = matplotlib.get_cachedir()
    font_cache_dir = os.path.join(cache_dir, 'fontlist-v*.json')
    
    print(f"Matplotlib缓存目录: {cache_dir}")
    
    # 查找并删除字体缓存文件
    if os.path.exists(cache_dir):
        for file in os.listdir(cache_dir):
            if file.startswith('fontlist') and file.endswith('.json'):
                file_path = os.path.join(cache_dir, file)
                try:
                    os.remove(file_path)
                    print(f"✓ 已删除: {file}")
                except Exception as e:
                    print(f"✗ 删除失败 {file}: {e}")
    
    # 重建字体缓存
    try:
        fm._rebuild()
        print("✓ 已重建字体缓存")
    except Exception as e:
        print(f"⚠️  重建字体缓存失败: {e}")

def list_chinese_fonts():
    """列出系统中可用的中文字体"""
    fonts = [f.name for f in fm.fontManager.ttflist]
    
    chinese_keywords = [
        'yahei', 'simhei', 'simsun', 'hei', 'song', 'kai', 'fang',
        'wqy', 'noto', 'source', 'droid', 'uming', 'ukai', 'pingfang'
    ]
    
    chinese_fonts = [
        f for f in fonts 
        if any(keyword in f.lower() for keyword in chinese_keywords)
    ]
    
    print("\n系统中可用的中文字体:")
    for font in sorted(set(chinese_fonts)):
        print(f"  - {font}")
    
    return chinese_fonts

def test_chinese_display():
    """测试中文显示"""
    import matplotlib.pyplot as plt
    
    # 设置字体
    system = os.name
    if system == 'nt':  # Windows
        font_candidates = ['Microsoft YaHei', 'SimHei', 'SimSun']
    else:
        font_candidates = ['WenQuanYi Micro Hei', 'Noto Sans CJK SC']
    
    for font_name in font_candidates:
        try:
            plt.rcParams['font.sans-serif'] = [font_name]
            plt.rcParams['axes.unicode_minus'] = False
            
            # 测试
            fig, ax = plt.subplots(figsize=(6, 4))
            ax.text(0.5, 0.5, '测试中文显示：原始侧面、原始正面、模型生成正面', 
                   fontsize=14, ha='center', va='center')
            ax.set_title('中文字体测试', fontsize=16)
            ax.set_xlabel('X轴标签')
            ax.set_ylabel('Y轴标签')
            
            test_file = 'font_test.png'
            plt.savefig(test_file, dpi=150, bbox_inches='tight')
            plt.close()
            
            print(f"\n✓ 字体测试成功: {font_name}")
            print(f"  测试图片已保存: {test_file}")
            return True
        except Exception as e:
            print(f"✗ 字体测试失败 {font_name}: {e}")
            continue
    
    return False

if __name__ == '__main__':
    print("=" * 70)
    print("修复matplotlib中文字体显示")
    print("=" * 70)
    
    # 1. 列出可用字体
    chinese_fonts = list_chinese_fonts()
    
    # 2. 清除缓存
    print("\n清除matplotlib字体缓存...")
    clear_matplotlib_cache()
    
    # 3. 测试显示
    print("\n测试中文显示...")
    test_chinese_display()
    
    print("\n" + "=" * 70)
    print("完成！")
    print("=" * 70)

