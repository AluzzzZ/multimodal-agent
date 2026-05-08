import ast
import json
import re
from pathlib import Path

def bind_pic_ids(text, image_ids):
    parts = []
    id_index = 0
    pos = 0
    while pos < len(text):
        pic_pos = text.find('<PIC>', pos)
        if pic_pos == -1:
            parts.append(text[pos:])
            break
        if pic_pos > pos:
            parts.append(text[pos:pic_pos])
        if id_index < len(image_ids):
            parts.append(f'<PIC>[{image_ids[id_index]}]')
            id_index += 1
        else:
            parts.append('<PIC>')
        pos = pic_pos + 5
    return ''.join(parts)

def split_sections(text):
    """按 ' # ' 分割章节"""
    parts = text.split(' # ')
    sections = []
    for i, part in enumerate(parts):
        part = part.strip()
        if not part:
            continue
        sections.append(part)
    return sections

def clean_title(title):
    """清理标题：截断太长的标题"""
    # 标题应该简洁，通常在第一个句号/问号/感叹号处截断
    # 但如果第一句太长（超过30字），就用前30字

    title = title.strip()

    # 检查是否标题后面直接跟正文（没有明显的句子分隔）
    # 如果标题包含句号但句号后还有内容，且句号在标题前半部分
    for punct in ['。', '！', '？']:
        pos = title.find(punct)
        if pos != -1 and pos < 60:
            # 句号在前面，可以在这里截断
            potential_title = title[:pos+1].strip()
            after_punct = title[pos+1:].strip()
            # 如果句号后的内容是正文（包含多个汉字或较长的英文），截断
            if len(after_punct) > 10:
                return potential_title

    # 如果没有找到合适的截断点，且标题太长，截断
    if len(title) > 40:
        return title[:40] + '...'
    return title

def split_title_content(section):
    """分离标题和正文"""
    # 找第一个换行符
    newline_pos = section.find('\n')
    if newline_pos != -1 and newline_pos < 100:
        title = section[:newline_pos].strip()
        content = section[newline_pos+1:].strip()
        return title, content

    # 单行格式：标题和正文在同一行
    # 用启发式方法找标题和正文的分界
    title = section

    # 找第一个标点后跟汉字的位置
    for punct in ['。', '！', '？', '.', '!', '?']:
        pos = title.find(punct)
        if pos != -1 and pos > 5 and pos < 80:
            after = title[pos+1:].strip()
            if len(after) > 5:
                return title[:pos+1], title[pos+1:]

    # 没有找到合适的分隔，正文为空
    return title, ''

def merge_sections(sections, ids):
    """合并空章节，保留图片ID顺序"""
    if not sections:
        return sections

    merged = []
    current_title = ''
    current_content = ''

    for section in sections:
        title, content = split_title_content(section)

        # 清理标题
        title = clean_title(title)

        # 如果正文为空或只有PIC标签，合并到上一节
        content_stripped = content.strip()
        content_clean = re.sub(r'<PIC>', '', content_stripped).strip()

        if not content_clean:
            # 空章节，保留标题信息但不单独创建块
            # 将其作为备注合并到当前块
            if merged:
                merged[-1] = (merged[-1][0], merged[-1][1] + ' ' + title)
            continue

        merged.append((title, content))

    return merged

# 空气净化器手册
raw = Path('手册/空气净化器手册.txt').read_text(encoding='utf-8')
obj = ast.literal_eval(raw)
text, ids = obj

# 按 ' # ' 分割
sections_raw = split_sections(text)
print(f"原始章节数: {len(sections_raw)}")

# 合并空章节
merged = merge_sections(sections_raw, ids)
print(f"合并后章节数: {len(merged)}")

# 构建输出文本
# 格式: # 标题\n正文内容<PIC>[ID]
chunks = []
id_index = 0

for title, content in merged:
    # 构建 chunk
    chunk = f'# {title}\n{content}'
    chunks.append(chunk)

output_text = '\n\n'.join(chunks)

# 绑定图片ID
final_text = bind_pic_ids_for_text(output_text, ids)

# 统计
print(f"\n输出文本长度: {len(final_text)}")
print(f"<PIC>数量: {final_text.count('<PIC>')}")
print(f"绑定ID: {len(re.findall(r'<PIC>\[', final_text))}")

# 显示章节结构
for title, content in merged[:10]:
    pic = content.count('<PIC>')
    print(f"  # {title[:40]} | {len(content)}字 | {pic} PIC")
