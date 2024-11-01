# DoOffice

DoOffice 是一个专门用于处理中文 Word 文档的命令行工具。它能够自动调整文档格式，提取文档内容，支持多种中文段落样式，让文档处理更加高效和规范。

## ✨ 特性

- 文档格式化功能：
  - 自动识别并格式化文档标题和正文部分
  - 支持多种中文段落样式的智能识别和格式化：
    - 中文数字编号（一、二、三、）
    - 带括号的中文数字（（一）（二）（三））
    - "是"字句（一是、二是、三是）
    - 阿拉伯数字编号（1. 2. 3.）
  - 支持自定义格式规则（通过 YAML 配置文件）

- 内容提取功能：
  - 提取文档中的所有图片
  - 支持图片批量导出
  - 可选择是否在导出后删除文档中的图片
  - 将文档内容转换为 Markdown 格式
  - 智能识别标题层级和特殊格式

## 🛠️ 安装
1. 从 [Releases](https://github.com/mosivic/dooffice/releases) 页面下载最新版本
2. 解压下载的压缩包到任意目录
3. 将该目录添加到系统环境变量 PATH 中
4. 打开命令提示符或 PowerShell，输入 `dodocx --version` 验证安装

## 📖 使用说明

### 格式化文档
处理单个文档：
```bash
dodocx format document.docx
```

处理整个目录：
```bash
dodocx format /path/to/folder -r
```

### 提取图片
从文档中提取图片：
```bash
dodocx extract image document.docx -o output_folder
```

提取并删除文档中的图片：
```bash
dodocx extract image document.docx --remove-images
```

### 转换为 Markdown
将文档转换为 Markdown 格式：
```bash
dodocx extract markdown document.docx
```

### 命令行参数

```bash
usage: dodocx [-h] [--version] {format,extract} ...

子命令：
  format              格式化文档
  extract             提取文档内容

format 命令参数:
  -r, --recursive     递归处理子文件夹
  --rules RULES       指定自定义规则文件
  -q, --quiet        安静模式

extract image 命令参数:
  -o OUTPUT          指定输出文件夹
  --no-backup        不创建备份
  --remove-images    删除文档中的图片

extract markdown 命令参数:
  -o OUTPUT          指定输出文件夹
```

### 自定义格式规则
通过 YAML 文件自定义文档格式规则。创建 `rules.yaml` 文件：
```yaml
title:
  font: 宋体
  size: 22
  bold: true
  align: center
content:
  default:
    font: 仿宋
    size: 16
    bold: false
    align: justify
  number_dot:      # 一、二、等
    font: 黑体
    bold: false
  number_brackets: # （一）（二）等
    font: 楷体
    bold: true
  number_is:      # 一是 二是等
    font: 楷体
    bold: true
  arabic_number:  # 1. 2. 等
    font: 楷体
    bold: true
```

## 📝 注意事项

- 仅支持 .docx 格式文件
- 建议在处理重要文档前先创建备份
- 图片提取默认会自动创建备份文件
- Markdown 转换会保持文档的层级结构
- 格式化时会保持文档内容不变，仅调整格式

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情
