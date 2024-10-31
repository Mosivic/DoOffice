默认格式规则:
  - 标题部分：宋体，22pt，加粗，居中对齐
  - 正文部分：
    * 默认格式：仿宋，16pt，两端对齐
    * "一、"开头：黑体，16pt（整段）
    * "（一）...。"：楷体，16pt，加粗
    * "一是...。"：楷体，16pt，加粗
    * "1....。"
  
  - 段落格式：
    * 行间距：固定值28.5磅
    * 段落缩进：首行缩进2个中文字符
    

自定义规则文件 (YAML):
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