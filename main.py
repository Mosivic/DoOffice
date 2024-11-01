import format
import extract
import argparse
from pathlib import Path

def main():
    # 创建主参数解析器
    parser = argparse.ArgumentParser(
        description='''
__________                         ________      ________                    
___  ____/___  ______________      ___  __ \________  __ \_______________  __
__  /_   _  / / /_  ___/  __ \     __  / / /  __ \_  / / /  __ \  ___/_  |/_/
_  __/   / /_/ /_  /   / /_/ /     _  /_/ // /_/ /  /_/ // /_/ / /__ __>  <  
/_/      \__,_/ /_/    \____/      /_____/ \____//_____/ \____/\___/ /_/|_|                                                                           
''',
        formatter_class=argparse.RawTextHelpFormatter,
    )
    
    # 添加版本参数
    parser.add_argument(
        '--version', '-v',
        action='version',
        version='%(prog)s 0.3.0',
        help='显示程序版本信息'
    )

    # 创建子命令解析器
    subparsers = parser.add_subparsers(dest='command', help='可用的命令')

    # format 命令
    format_parser = subparsers.add_parser('format', help='格式化文档')
    format_parser.add_argument(
        'path',
        help='指定要处理的Word文档路径或文件夹路径'
    )
    format_parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='递归处理子文件夹中的所有docx文件'
    )
    format_parser.add_argument(
        '--rules',
        type=str,
        help='指定自定义规则文件路径'
    )
    format_parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='安静模式，不显示处理进度'
    )

    # extract 命令
    extract_parser = subparsers.add_parser('extract', help='提取文档内容')
    extract_subparsers = extract_parser.add_subparsers(dest='command', help='可用的命令')
    extract_image_parser = extract_subparsers.add_parser('image',        
        description='从Word文档中提取图片',
        help='从Word文档中提取图片',
        formatter_class=argparse.RawTextHelpFormatter
    )
    extract_image_parser.add_argument(
        'path',
        help='指定要处理的Word文档路径或文件夹路径'
    )
    extract_image_parser.add_argument(
        '-o', '--output',
        default='extracted_images',
        help='图片保存的文件夹路径 (默认: extracted_images)'
    )
    extract_image_parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不创建原文件的备份'
    )
    extract_image_parser.add_argument(
        '--remove-images',
        action='store_true',
        help='从文档中删除图片'
    )

    extract_markdown_parser = extract_subparsers.add_parser('markdown', help='提取文档为markdown格式')
    extract_markdown_parser.add_argument(
        'path',
        help='指定要处理的Word文档路径或文件夹路径'
    )
    


    # 解析命令行参数
    args = parser.parse_args()
    
    # 如果没有指定命令，显示帮助信息
    if not args.command:
        parser.print_help()
        return

    # 获取路径
    path = Path(args.path)
    
    # 加载自定义规则（如果需要）
    custom_rules = None
    if hasattr(args, 'rules') and args.rules:
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

    if args.command == 'format':
        format.format_title_section(str(path), custom_rules)
    elif args.command == 'extract':
        if args.subcommand == 'image':
            extract.extract_images(str(path), args.output if hasattr(args, 'output') else None, not args.no_backup, args.remove_images)
        elif args.subcommand == 'markdown':
            extract.extract_markdown(str(path), args.output if hasattr(args, 'output') else None)
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
            if args.command == 'format':
                format.format_title_section(str(doc_path), custom_rules)
            elif args.command == 'extract':
                if args.subcommand == 'image':
                    extract.extract_images(str(doc_path), args.output if hasattr(args, 'output') else None, not args.no_backup, args.remove_images)
                elif args.subcommand == 'markdown':
                    extract.extract_markdown(str(doc_path), args.output if hasattr(args, 'output') else None)
            files_processed += 1
        except Exception as e:
            errors.append((doc_path, str(e)))
            if not quiet:
                print(f"Error with {doc_path}: {str(e)}")
    
    if not (hasattr(args, 'quiet') and args.quiet):
        print(f"\nDone:")
        print(f"- Successfully processed: {files_processed} files")
        if errors:
            print(f"- Failed: {len(errors)} files")
            print("\nFailed details:")
            for doc_path, error in errors:
                print(f"  - {doc_path}: {error}")

if __name__ == "__main__":
    main()
