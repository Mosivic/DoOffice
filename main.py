from module import format
from module import extract
from module import tool
import argparse
from pathlib import Path

VERSION = "v0.3.2"  

def main():
    # 创建主参数解析器
    parser = argparse.ArgumentParser(
        description=r'''
_________                         ________      ________                    
___  ___/___  ______________      ___  __ \________  __ \_______________  __
__  __   _  / / /_  ___/  __ \     __  / / /  __ \_  / / /  __ \  ___/_  |/_/
_  __/   / /_/ /_  /   / /_/ /     _  /_/ // /_/ /  /_/ // /_/ / /__ __>  <  
/_/      \__,_/ /_/    \____/      /_____/ \____//_____/ \____/\___/ /_/|_|                                                                           
''',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # 添加版本参数
    parser.add_argument(
        '--version', '-v',
        action='version',
        version=VERSION,
        help='显示版本号'
    )

    # 添加通用参数
    parser.add_argument(
        'path',
        help='指定要处理的Word文档路径或文件夹路径'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='递归处理子文件夹中的所有docx文件'
    )
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='安静模式，不显示处理进度'
    )
    
    # 添加命令选项
    parser.add_argument(
        '-m','--mode',
        choices=['format', 'extract-images', 'extract-markdown', 'rename-folder'],
        required=True,
        help='指定执行模式'
    )
    
    # format 命令特有参数
    parser.add_argument(
        '--rules',
        type=str,
        help='指定自定义规则文件路径 (仅用于 format 命令)'
    )
    
    # extract-images 命令特有参数
    parser.add_argument(
        '-o', '--output',
        help='输出文件夹路径 (用于 extract-images 和 extract-markdown 命令，默认为输入文件所在目录)'
    )
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不创建原文件的备份 (仅用于 extract-images 命令)'
    )
    parser.add_argument(
        '--remove-images',
        action='store_true',
        help='从文档中删除图片 (仅用于 extract-images 命令)'
    )

    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助信息
    if not args.mode:
        parser.print_help()
        return

    # 获取路径
    path = Path(args.path)
    
    # 加载自定义规则（如果需要）
    custom_rules = None
    if args.mode == 'format' and args.rules:
        rules_path = Path(args.rules)
        if not rules_path.exists():
            print(f"Error: Rules file {rules_path} does not exist")
            return
        custom_rules = format.load_rules(rules_path)
        if custom_rules is None:
            return

    # 处理单个文件
    if path.is_file():
        process_single_file(args, path, custom_rules)
    # 处理目录
    elif path.is_dir():
        if args.mode == 'rename-folder':
            tool.rename_folders_by_doc(str(path))
        else:
            process_directory(args, path, custom_rules)
    else:
        print(f"Error: Path {path} does not exist")

def process_single_file(args, path, custom_rules):
    if path.suffix.lower() != '.docx':
        print(f"Error: {path} is not a Word document")
        return

    quiet = hasattr(args, 'quiet') and args.quiet
    if not quiet:
        print(f"Processing file: {path}")

    # 如果没有指定输出路径，使用输入文件所在目录
    output_path = args.output if args.output else str(path.parent)

    if args.mode == 'format':
        format.format_title_section(str(path), custom_rules)
    elif args.mode == 'extract-images':
        extract.extract_images(str(path), output_path, not args.no_backup, args.remove_images)
    elif args.mode == 'extract-markdown':
        extract.extract_markdown(str(path),True,output_path)
    if not quiet:
        print(f"Done: {path}")

def process_directory(args, path, custom_rules):
    pattern = '**/*.docx' if args.recursive else '*.docx'
    files_processed = 0
    errors = []
    
    for doc_path in path.glob(pattern):
        quiet = hasattr(args, 'quiet') and args.quiet
        if not quiet:
            print(f"Processing: {doc_path}")
        try:
            # 如果没有指定输出路径，使用当前处理文件所在目录
            output_path = args.output if args.output else str(doc_path.parent)
            
            if args.mode == 'format':
                format.format_title_section(str(doc_path), custom_rules)
            elif args.mode == 'extract-images':
                extract.extract_images(str(doc_path), output_path, not args.no_backup, args.remove_images)
            elif args.mode == 'extract-markdown':
                extract.extract_markdown(str(doc_path),True,output_path)
            files_processed += 1
        except Exception as e:
            errors.append((doc_path, str(e)))
            if not quiet:
                print(f"Error with {doc_path}: {str(e)}")
    
    if not (hasattr(args, 'quiet') and args.quiet):
        if errors:
            print(f"- Failed: {len(errors)} files")
            print("\nFailed details:")
            for doc_path, error in errors:
                print(f"  - {doc_path}: {error}")

if __name__ == "__main__":
    main()
