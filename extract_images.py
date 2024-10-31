import argparse
from docx import Document
import os
from datetime import datetime

def extract_and_remove_images(docx_path, output_folder='extracted_images', backup=True):
    # 确保输入文件存在
    if not os.path.exists(docx_path):
        print(f"错误: 找不到文件 '{docx_path}'")
        return
    
    # 创建备份
    if backup:
        backup_path = f"{docx_path}.backup"
        try:
            import shutil
            shutil.copy2(docx_path, backup_path)
            print(f"已创建备份文件: {backup_path}")
        except Exception as e:
            print(f"警告: 创建备份失败: {str(e)}")
            return
    
    # 创建输出文件夹
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"创建输出文件夹: {output_folder}")
    
    # 打开word文档
    try:
        doc = Document(docx_path)
    except Exception as e:
        print(f"错误: 无法打开文件 '{docx_path}': {str(e)}")
        return
    
    # 用于记录提取的图片数量
    image_count = 0
    
    # 提取并删除图片
    rels_to_delete = []
    for rel in list(doc.part.rels.values()):
        # 检查是否是图片
        if "image" in rel.target_ref:
            try:
                # 获取图片数据
                image_data = rel.target_part.blob
                
                # 生成图片文件名
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                image_filename = f'image_{timestamp}_{image_count}.png'
                image_path = os.path.join(output_folder, image_filename)
                
                # 保存图片
                with open(image_path, 'wb') as f:
                    f.write(image_data)
                image_count += 1
                print(f'已保存图片: {image_filename}')
                
                # 记录需要删除的关系ID
                rels_to_delete.append(rel.rId)
                
            except Exception as e:
                print(f"警告: 处理图片时出错: {str(e)}")
    
    # 删除图片关系
    for rId in rels_to_delete:
        del doc.part.rels[rId]
    
    # 完全清理文档中的图片元素
    for paragraph in doc.paragraphs:
        p = paragraph._element
        
        # 使用 findall 替代 xpath
        ns = {
            'w': 'http://schemas.openxmlformats.org/wordprocessingml/2006/main',
            'v': 'urn:schemas-microsoft-com:vml',
            'wp': 'http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing',
            'a': 'http://schemas.openxmlformats.org/drawingml/2006/main',
            'pic': 'http://schemas.openxmlformats.org/drawingml/2006/picture'
        }
        
        # 删除 drawing 元素
        for element in p.findall('.//w:drawing', ns):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
        
        # 删除 pict 元素
        for element in p.findall('.//w:pict', ns):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
        
        # 删除 imagedata 元素
        for element in p.findall('.//v:imagedata', ns):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
                
        # 删除 shape 元素
        for element in p.findall('.//v:shape', ns):
            parent = element.getparent()
            if parent is not None:
                parent.remove(element)
        
        # 清理空的运行元素
        for run in p.findall('.//w:r', ns):
            if len(run) == 0 or (len(run) == 1 and run[0].tag.endswith('}rPr')):
                parent = run.getparent()
                if parent is not None:
                    parent.remove(run)
    
    # 保存修改后的文档
    try:
        doc.save(docx_path)
        print(f"\n文档已更新: {docx_path}")
    except Exception as e:
        print(f"错误: 保存文档失败: {str(e)}")
        return
    
    print(f'总共提取了 {image_count} 张图片')
    print(f'图片保存在文件夹: {output_folder}')

def main():
    # 创建参数解析器
    parser = argparse.ArgumentParser(
        description='从Word文档中提取图片并删除原文档中的图片',
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 添加命令行参数
    parser.add_argument(
        'docx_path',
        help='Word文档的路径'
    )
    
    parser.add_argument(
        '-o', '--output',
        default='extracted_images',
        help='图片保存的文件夹路径 (默认: extracted_images)'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='不创建原文件的备份'
    )

    
    # 解析命令行参数
    args = parser.parse_args()
    
    # 执行提取和删除操作
    extract_and_remove_images(args.docx_path, args.output, not args.no_backup)

if __name__ == '__main__':
    main()