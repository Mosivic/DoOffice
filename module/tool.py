import os
import argparse
from pathlib import Path

def rename_folders_by_doc(target_dir=None):
    """
    将指定目录下的文件夹重命名为其内部第一个 doc/docx 文件的名称
    Args:
        target_dir: 目标目录路径，默认为当前目录
    """
    # 使用指定目录或当前目录
    current_dir = Path(target_dir) if target_dir else Path('.')
    
    if not current_dir.exists():
        print(f'错误：目录 "{current_dir}" 不存在')
        return
        
    # 遍历当前目录下的所有文件夹
    for folder in current_dir.iterdir():
        if not folder.is_dir():
            continue
            
        # 搜索文件夹中的 doc/docx 文件
        doc_files = []
        for ext in ('*.doc', '*.docx'):
            doc_files.extend(folder.glob(ext))
            
        # 如果找到 doc 文件
        if doc_files:
            # 获取第一个 doc 文件的名称（不含扩展名）
            new_name = doc_files[0].stem
            
            try:
                # 构建新的文件夹路径
                new_folder_path = folder.parent / new_name
                # 重命名文件夹
                folder.rename(new_folder_path)
                print(f'成功将文件夹 "{folder.name}" 重命名为 "{new_name}"')
            except Exception as e:
                print(f'重命名文件夹 "{folder.name}" 失败: {str(e)}')
        else:
            print(f'文件夹 "{folder.name}" 中未找到 doc/docx 文件')

def main():
    parser = argparse.ArgumentParser(
        description='将文件夹重命名为其内部第一个 doc/docx 文件的名称'
    )
    parser.add_argument(
        '-d', '--directory',
        help='指定要处理的目录路径（默认为当前目录）',
        default=None
    )
    
    args = parser.parse_args()
    rename_folders_by_doc(args.directory)

if __name__ == '__main__':
    main()