# DoOffice

DoOffice 是一个专门用于处理中文 Word 文档格式化的命令行工具。它能够自动调整文档的标题和正文格式，支持多种中文段落样式，让文档排版更加规范和美观。

## ✨ 特性

- 自动识别并格式化文档标题和正文部分
- 支持多种中文段落样式的智能识别和格式化：
  - 中文数字编号（一、二、三、）
  - 带括号的中文数字（（一）（二）（三））
  - "是"字句（一是、二是、三是）
  - 阿拉伯数字编号（1. 2. 3.）
- 支持批量处理整个目录下的 Word 文档
- 支持自定义格式规则（通过 YAML 配置文件）
- 保持文档内容的完整性，仅调整格式

## 🛠️ 安装
1. 从 [Releases](https://github.com/mosivic/dooffice/releases) 页面下载最新版本应用压缩包

2. 将下载的应用压缩包解压在你想要的目录下

3. 将该目录添加到系统环境变量 PATH 中：
   - 打开系统属性 > 环境变量
   - 在"系统变量"下找到 PATH
   - 点击"编辑"并添加第二步压缩包解压至的目录路径
   - 点击"确定"保存更改

4. 打开命令提示符或 PowerShell，输入 `dodocx --version` 验证安装是否成功

## 📖 使用说明
### 基本用法
处理单个文档：
``` bash
dodocx document.docx
```

处理整个目录：
``` bash
dodocx /path/to/folder
```

### 命令行参数

```bash
usage: dodocx [-h] [--recursive] [--version] [--quiet] [--rules RULES] path

位置参数:
path 指定要处理的Word文档路径或文件夹路径

可选参数:
-h, --help 显示帮助信息
--recursive, -r 递归处理子文件夹中的所有docx文件
--version, -v 显示程序版本信息
--quiet, -q 安静模式，不显示处理进度
--rules RULES 指定自定义规则文件路径
```



### 自定义格式规则
你可以通过 YAML 文件自定义文档格式规则。创建一个 `rules.yaml` 文件：
``` yaml
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
使用自定义规则：
``` bash 
dodocx --rules rules.yaml document.docx
```

## 📝 格式化规则说明

- **标题部分**：
  - 默认使用宋体，22磅（二号），加粗，居中对齐
  - 从文档开始到第一个空白段落之前的内容被视为标题

- **正文部分**：
  - 默认使用仿宋体，16磅（三号），两端对齐
  - 特殊段落样式：
    - 中文数字编号：使用黑体
    - 带括号的中文数字：使用楷体，加粗
    - "是"字句：使用楷体，加粗
    - 阿拉伯数字编号：使用楷体，加粗

## 📄 许可证

本项目采用 MIT 许可证 - 查看 [LICENSE](LICENSE) 文件了解详情
