import ast
import json
import re
from pathlib import Path

# 空气净化器手册 - 单行文本，按 # 分割
raw = Path('手册/空气净化器手册.txt').read_text(encoding='utf-8')
obj = ast.literal_eval(raw)
text, ids = obj

print(f"文本长度: {len(text)}")
print(f"换行符数量: {text.count(chr(10))}")

# 文本是单行，章节用 " # " 分隔
# 找 " # " 位置
pattern = r' # '
matches = list(re.finditer(pattern, text))

print(f"找到 {len(matches)} 个章节分隔符")

sections = []
for i, m in enumerate(matches):
    title_start = m.start() + 3  # skip " # "
    title_end = matches[i+1].start() if i+1 < len(matches) else len(text)
    # 找标题结束位置（下一个 " # " 或文本末尾）
    # 但标题后面紧跟正文，所以需要找第一个句号或问号后的大段文字

    # 简化处理：取到下一个 " # " 之前的文本
    chunk = text[title_start:title_end]

    # 从chunk中提取标题和正文
    # 标题通常在前30-50个字符，然后是正文
    # 用启发式方法：找第一个 "。" "？" "！" 后面的长段落

    # 找第一个完整句子结束
    # 先找第一个标点后的位置
    punct_pos = -1
    for p in ['。', '！', '？']:
        pos = chunk.find(p)
        if pos != -1 and (punct_pos == -1 or pos < punct_pos):
            punct_pos = pos

    if punct_pos > 10 and punct_pos < 80:
        title = chunk[:punct_pos+1].strip()
        body = chunk[punct_pos+1:].strip()
    else:
        # 标题取前40字
        title = chunk[:40].strip()
        body = chunk[40:].strip()

    pic_count = body.count('<PIC>')
    sections.append((title, body))
    print(f"\n章节{i+1}:")
    print(f"  标题: {repr(title[:60])}")
    print(f"  正文长度: {len(body)}, PIC: {pic_count}")
    if body:
        print(f"  正文开头: {repr(body[:60])}")

print(f"\n共 {len(sections)} 个章节")
