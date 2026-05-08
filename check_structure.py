import ast
import json
import re
from pathlib import Path

kb_dir = Path('手册/建库优化_模型版')
kb_files = sorted(kb_dir.glob('*_章节化_建库版.txt'))

print(f"找到 {len(kb_files)} 个建库版文件\n")

for kb_path in kb_files:
    raw = kb_path.read_text(encoding='utf-8')
    try:
        data = json.loads(raw)
        text, ids = data

        # 检查文本结构
        lines = text.split('\n')
        total_lines = len(lines)
        non_empty_lines = sum(1 for l in lines if l.strip())

        # 统计 <PIC> 数量
        pic_count = text.count('<PIC>')
        bound_pic = len(re.findall(r'<PIC>\[', text))

        # 检查是否有多行（章节分割）
        heading_lines = [l for l in lines if l.startswith('# ')]
        h2_lines = [l for l in lines if l.startswith('## ')]

        # 检查第一行
        first_line = lines[0] if lines else ''

        print(f"{kb_path.name}:")
        print(f"  行数: {total_lines}, 非空行: {non_empty_lines}")
        print(f"  <PIC>: {pic_count}, 已绑定: {bound_pic}")
        print(f"  # 标题行: {len(heading_lines)}, ## 标题行: {len(h2_lines)}")
        print(f"  首行: {repr(first_line[:80])}")
        print()
    except Exception as e:
        print(f"{kb_path.name}: 解析错误 - {e}\n")
