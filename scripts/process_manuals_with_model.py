"""
手册章节化与建库优化脚本 v4（修复版）
修复：
1. parse_merge 去重块内重复索引
2. 索引越界检查 (<= 改为 <)
3. 块内内容去重
4. 空块过滤
"""

import os, sys, re, ast
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


@dataclass
class ManualInfo:
    name: str
    raw_pic_count: int = 0
    raw_image_id_count: int = 0
    sectioned_count: int = 0
    chunk_count: int = 0
    empty_chunks: int = 0
    pic_ok: bool = False
    notes: str = ""


# ====== 解析工具 ======

def parse_manual_file(content: str) -> Tuple[str, List[str]]:
    content = content.strip()
    if not content:
        return "", []
    try:
        parsed = ast.literal_eval(content)
        if (isinstance(parsed, (list, tuple)) and len(parsed) >= 2
                and isinstance(parsed[0], str) and isinstance(parsed[1], list)):
            return parsed[0], [str(x) for x in parsed[1]]
    except Exception:
        pass
    return content, []


def count_pic(text: str) -> int:
    return len(re.findall(r'<PIC>(?:\[.*?\])?', text))


def bind_pic(text: str, image_ids: List[str]) -> str:
    """将 <PIC> 占位符与图片 ID 逐一绑定"""
    pic_pattern = re.compile(r'<PIC>(?!\[)')
    id_iter = iter(image_ids)
    def replace(m):
        try:
            return f'<PIC>[{next(id_iter)}]'
        except StopIteration:
            return '<PIC>'
    result = pic_pattern.sub(replace, text)
    remaining = list(id_iter)
    if remaining:
        result = result.rstrip() + '\n' + ' '.join(f'<PIC>[{i}]' for i in remaining)
    return result


def parse_titles_simple(output: str) -> List[str]:
    """从模型输出中提取章节标题（过滤纯数字页码）"""
    titles = []
    for line in output.split('\n'):
        line = line.strip()
        if not line:
            continue
        # 匹配 # 编号. 标题 格式
        m = re.match(r'^#?\s*(?:\d+[.、)]\s*)?(.+)$', line)
        if m:
            title = m.group(1).strip().strip('。.,，、、')
            # 过滤掉纯数字、纯页码、无意义标题
            if not title:
                continue
            # 过滤掉纯数字或太短的
            if re.match(r'^\d+$', title) or len(title) < 3:
                continue
            # 过滤掉纯英文标题（可能是页码）
            if re.match(r'^[a-zA-Z\s]+$', title) and len(title) < 10:
                continue
            # 截断过长的标题
            if len(title) > 25:
                title = title[:24]
            titles.append(title)
    return titles


def split_by_titles(text: str, titles: List[str]) -> List[Tuple[str, str]]:
    """
    按原文位置切分章节，返回 (标题, 内容) 列表
    
    策略：
    1. 跳过目录区域
    2. 直接从正文中提取 # 标题行
    3. 用这些实际标题来切分文本
    """
    if not text.strip():
        return []

    # 找到目录结束位置
    lines = text.split('\n')
    body_start = 0
    for i, line in enumerate(lines[:30]):
        stripped = line.strip()
        if re.search(r'\d+\s+[\u4e00-\u9fa5]{2,}', stripped) and len(stripped) > 50:
            body_start = i + 1
            break

    # 从正文中提取实际的 # 标题
    body_positions = []  # [(pos, title)]
    
    # 在正文区域搜索 # 标题
    body_pos = sum(len(lines[j]) + 1 for j in range(min(body_start, len(lines))))
    body_text = text[body_pos:]
    
    # 匹配 # 标题行（# 后面跟中文，且不是太长或太短）
    for m in re.finditer(r'^#\s+([^\n]{5,40})', body_text, re.MULTILINE):
        raw_title = m.group(1).strip()
        # 过滤掉太长的、包含链接的、或明显是描述性内容
        if len(raw_title) > 30:
            continue
        if '<PIC>' in raw_title or 'http' in raw_title:
            continue
        # 过滤掉纯数字页码（如 # 2, # 3, # 7）
        if re.match(r'^[\d\s.,、]+$', raw_title):
            continue
        # 过滤掉纯英文或太短的（至少5个字符）
        if re.match(r'^[a-zA-Z\s]+$', raw_title) and len(raw_title) < 8:
            continue
        # 取前20字作为标题
        title = raw_title[:20].strip()
        if title and len(title) >= 4:
            body_positions.append((body_pos + m.start(), title))
    
    if not body_positions:
        # 备选：使用模型输出的标题搜索
        return _split_by_model_titles(text, titles)
    
    # 如果正文标题太多（>50），说明手册结构特殊（如冰箱每个句子都有#）
    # 回退到模型标题搜索
    if len(body_positions) > 50:
        return _split_by_model_titles(text, titles)
    
    # 去重（相邻太近的合并标题）
    deduped = []
    for pos, title in body_positions:
        if not deduped or pos - deduped[-1][0] >= 10:
            deduped.append((pos, title))
        else:
            # 合并标题
            old_title = deduped[-1][1]
            deduped[-1] = (deduped[-1][0], f"{old_title}、{title}")
    
    if not deduped:
        return _split_by_model_titles(text, titles)
    
    # 切分内容
    chunks = []
    for i, (start_pos, title) in enumerate(deduped):
        end_pos = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        content = text[start_pos:end_pos].strip()
        if content:
            chunks.append((title, content))
    
    return chunks if chunks else _split_by_model_titles(text, titles)


def _split_by_model_titles(text: str, titles: List[str]) -> List[Tuple[str, str]]:
    """备选：用模型输出标题搜索（处理标题与正文不完全匹配的情况）"""
    if not titles:
        return [("全文", text)]

    lines = text.split('\n')
    body_start = 0
    for i, line in enumerate(lines[:30]):
        stripped = line.strip()
        if re.search(r'\d+\s+[\u4e00-\u9fa5]{2,}', stripped) and len(stripped) > 50:
            body_start = i + 1
            break

    body_pos = sum(len(lines[j]) + 1 for j in range(min(body_start, len(lines))))
    search_text = text[body_pos:]

    positions = []
    for title in titles:
        # 直接搜索
        idx = search_text.find(title)
        if idx >= 0:
            positions.append((body_pos + idx, title))
            continue
        
        # 部分匹配
        found = False
        for pl in range(min(10, len(title)), 4, -1):
            kw = title[:pl]
            idx = search_text.find(kw)
            if idx >= 0:
                positions.append((body_pos + idx, title))
                found = True
                break
        
        # 拆分"与"连接的标题
        if not found and "与" in title:
            for part in title.split("与"):
                part = part.strip()
                idx = search_text.find(part)
                if idx >= 0:
                    positions.append((body_pos + idx, part))
                    found = True
                    break
        
        if not found:
            positions.append((-1, title))

    valid = [(p, t) for p, t in positions if p >= 0]
    valid.sort(key=lambda x: x[0])

    # 去重
    deduped = []
    for pos, title in valid:
        if not deduped or pos - deduped[-1][0] >= 10:
            deduped.append((pos, title))
        else:
            old_title = deduped[-1][1]
            deduped[-1] = (deduped[-1][0], f"{old_title}、{title}")

    if not deduped:
        return [("全文", text)]

    chunks = []
    for i, (start_pos, title) in enumerate(deduped):
        end_pos = deduped[i + 1][0] if i + 1 < len(deduped) else len(text)
        content = text[start_pos:end_pos].strip()
        if content:
            chunks.append((title, content))

    # 如果匹配率太低（有效节少但总字数大），改用按字数均分
    if len(chunks) < 5 and len(text) > 3000:
        return _split_by_fixed_size(text)

    # 合并过短的节
    MIN_SIZE = 80
    merged = []
    i = 0
    while i < len(chunks):
        title, content = chunks[i]
        chars = len(re.sub(r'<PIC>', '', content))
        if chars < MIN_SIZE and len(chunks) > 1:
            if merged:
                prev_title, prev_content = merged.pop()
                merged.append((f"{prev_title}、{title}", prev_content + "\n" + content))
            elif i + 1 < len(chunks):
                next_title, next_content = chunks[i + 1]
                chunks[i + 1] = (f"{title}、{next_title}", content + "\n" + next_content)
            else:
                merged.append((title, content))
        else:
            merged.append((title, content))
        i += 1

    return merged if merged else [("全文", text)]


def _split_by_fixed_size(text: str, chunk_size: int = 800) -> List[Tuple[str, str]]:
    """按固定字数分块（用于标题匹配效果差的手册）"""
    # 找目录结束位置
    lines = text.split('\n')
    body_start = 0
    for i, line in enumerate(lines[:30]):
        stripped = line.strip()
        if re.search(r'\d+\s+[\u4e00-\u9fa5]{2,}', stripped) and len(stripped) > 50:
            body_start = i + 1
            break

    if body_start >= len(lines):
        body_start = 0
    body_pos = sum(len(lines[j]) + 1 for j in range(min(body_start, len(lines))))

    # 从正文开始按字数分段
    search_text = text[body_pos:]
    chunks = []
    pos = 0
    chunk_num = 1
    while pos < len(search_text):
        end_pos = min(pos + chunk_size, len(search_text))
        # 尽量在句号处截断
        last_period = search_text.rfind('。', pos, end_pos)
        if last_period > pos + 200:
            end_pos = last_period + 1

        content = search_text[pos:end_pos].strip()
        if content:
            # 从内容中提取第一个有效的章节标题
            first_title = _extract_first_title(content)
            chunks.append((first_title, content))
        pos = end_pos
        chunk_num += 1

    return chunks if chunks else [("全文", text)]


def _extract_first_title(content: str) -> str:
    """从内容中提取第一个有意义的章节标题"""
    lines = content.split('\n')
    for line in lines[:10]:  # 只看前10行
        stripped = line.strip()
        # 匹配 "# 标题" 格式（可能和其他内容在同一行）
        # 优先匹配独立的标题行
        m = re.match(r'^#\s+([^\n#]{4,30})(?:\s*[#\n]|$)', stripped)
        if m:
            title = m.group(1).strip()
            if len(title) >= 4:
                return title
        # 如果标题在行中间，尝试提取
        m = re.search(r'#\s+([^\n#]{4,30})(?:\s*[#]|$)', stripped)
        if m:
            title = m.group(1).strip()
            if len(title) >= 4 and not re.match(r'^[\d\s.,、]+$', title):
                return title
    # 如果没找到，返回通用标题
    return "内容部分"


def parse_merge(output: str) -> List[List[int]]:
    """解析合并方案，并对每块去重"""
    blocks = []
    for line in output.split('\n'):
        line = line.strip()
        if not line or line.startswith('```') or line.startswith('#'):
            continue
        m = re.match(r'块?\d+[:：]\s*(.+)', line)
        if m:
            nums = re.findall(r'\d+', m.group(1))
            # 关键：去重！防止模型输出重复索引
            seen = set()
            deduped = []
            for n in nums:
                n_int = int(n)
                if n_int not in seen:
                    seen.add(n_int)
                    deduped.append(n_int)
            if deduped:
                blocks.append(deduped)
    return blocks


def post_process(text: str) -> str:
    """清理标题格式"""
    if not text.strip():
        return text
    lines = text.split('\n')
    result = []
    for line in lines:
        stripped = line.strip()
        m = re.match(r'^# (.+)$', stripped)
        if m:
            raw_title = m.group(1).strip()
            # 如果标题中包含另一个 #（如 "# 安全说明  # 注意事项"），
            # 只保留第一个标题部分
            if ' #' in raw_title or raw_title.startswith('#'):
                # 提取第一个标题
                parts = raw_title.split('  # ')
                if parts:
                    raw_title = parts[0].strip()
            title = re.sub(r'\s+', ' ', raw_title).strip('。.，,')
            if len(title) > 20:
                title = title[:19] + '...'
            if title:
                result.append(f"# {title}")
        else:
            result.append(line)
    return '\n'.join(result)


def greedy_merge(sections: List[Tuple[str, str]], min_chunk_size: int = 200,
                  target_avg_size: int = 800) -> List[List[int]]:
    """
    基于目标块数的贪心合并策略：
    1. 计算目标块数 = ceil(总字数 / target_avg_size)
    2. 从每个 section 开始，贪心地扩展到目标块数
    3. 如果某些块 < min_chunk_size，强制合并
    """
    if not sections:
        return []

    # 计算每节字数（不含PIC标签）
    section_sizes = [
        len(re.sub(r'<PIC>(?:\[.*?\])?', '', content).strip())
        for _, content in sections
    ]
    total_size = sum(section_sizes)
    total = len(sections)

    if total <= 1:
        return [[1]]

    # 目标块数
    target_chunks = max(1, round(total_size / target_avg_size))
    target_chunks = min(target_chunks, total)

    # 初始化：每节一个块
    chunks = [[i] for i in range(1, total + 1)]
    chunk_sizes = list(section_sizes)

    # 合并直到达到目标块数
    while len(chunks) > target_chunks:
        # 找到最小的块
        min_idx = min(range(len(chunks)), key=lambda i: chunk_sizes[i])
        min_size = chunk_sizes[min_idx]

        # 选择合并代价最小的邻居
        left_ok = min_idx > 0
        right_ok = min_idx < len(chunks) - 1

        merged = False
        if left_ok and right_ok:
            left_inc = min_size
            right_inc = min_size
            # 合并到使结果最均衡的邻居
            left_new = chunk_sizes[min_idx - 1] + min_size
            right_new = min_size + chunk_sizes[min_idx + 1]
            if abs(left_new - target_avg_size) <= abs(right_new - target_avg_size):
                chunks[min_idx - 1].extend(chunks.pop(min_idx))
                chunk_sizes[min_idx - 1] = left_new
                chunk_sizes.pop(min_idx)
            else:
                chunks[min_idx + 1] = chunks[min_idx] + chunks[min_idx + 1]
                chunk_sizes[min_idx + 1] = right_new
                chunks.pop(min_idx)
                chunk_sizes.pop(min_idx)
            merged = True
        elif left_ok:
            chunks[min_idx - 1].extend(chunks.pop(min_idx))
            chunk_sizes[min_idx - 1] += min_size
            chunk_sizes.pop(min_idx)
            merged = True
        elif right_ok:
            chunks[min_idx + 1] = chunks[min_idx] + chunks[min_idx + 1]
            chunk_sizes[min_idx + 1] += min_size
            chunks.pop(min_idx)
            chunk_sizes.pop(min_idx)
            merged = True

        if not merged:
            break

    # 二次修复：合并仍然太小的块到邻居（贪心选择最优邻居）
    while True:
        sizes = [sum(section_sizes[i - 1] for i in blk) for blk in chunks]
        too_small = [idx for idx, s in enumerate(sizes) if 0 < s < min_chunk_size]
        if not too_small:
            break
        min_idx = too_small[0]
        left_ok = min_idx > 0
        right_ok = min_idx < len(chunks) - 1
        if left_ok and right_ok:
            left_inc = sizes[min_idx - 1]
            right_inc = sizes[min_idx + 1]
            if left_inc <= right_inc:
                chunks[min_idx - 1].extend(chunks.pop(min_idx))
            else:
                chunks[min_idx + 1] = chunks[min_idx] + chunks[min_idx + 1]
                chunks.pop(min_idx)
        elif left_ok:
            chunks[min_idx - 1].extend(chunks.pop(min_idx))
        elif right_ok:
            chunks[min_idx + 1] = chunks[min_idx] + chunks[min_idx + 1]
            chunks.pop(min_idx)
        else:
            break

    return chunks


# ====== Prompts ======

PROMPT_TITLES = """请仔细阅读以下手册，列出所有主要章节标题。

格式：每行一个，用 # 编号. 标题 的格式。
不要输出其他内容。

手册内容：
{text}
"""

PROMPT_MERGE = """以下是一个手册的 {total} 个章节（含标题和内容摘要）：

{sections}

请将这些章节合并成适合RAG检索的块，每块字数建议 400-1500 字。

要求：
- 结合标题和内容判断主题相关性，合并相关章节
- 不要孤零零留下单个很短的小节
- 不要让块太大（不超过2000字）
- 不要让块太小（少于200字的内容应合并到相邻块）
- 块数尽量少，但必须保证每块有实质内容

格式：
块1: 1, 2, 3
块2: 4, 5
...

只输出块规划，不要复制正文。"""


# ====== LLM 调用 ======

def get_llm_client():
    from config.settings import settings
    if settings.llm_provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=settings.llm_model,
            api_key=settings.llm_api_key,
            base_url=settings.llm_base_url,
            temperature=0.1,
            max_tokens=4000,
            request_timeout=180,
        )
    elif settings.llm_provider == "local":
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=settings.llm_model, temperature=0.1)
    return None


def call_model(client, prompt: str, max_tokens: int = 4000) -> str:
    old = client.max_tokens
    client.max_tokens = max_tokens
    try:
        resp = client.invoke(prompt)
        return resp.content if hasattr(resp, 'content') else str(resp)
    finally:
        client.max_tokens = old


# ====== 核心处理 ======

def process_manual(manual_path: Path, client) -> ManualInfo:
    name = manual_path.stem
    info = ManualInfo(name=name)

    with open(manual_path, 'r', encoding='utf-8') as f:
        raw = f.read()

    text, image_ids = parse_manual_file(raw)
    info.raw_pic_count = count_pic(text)
    info.raw_image_id_count = len(image_ids)
    expected_pic = min(info.raw_pic_count, info.raw_image_id_count)

    print(f"  手册: {len(text)} 字, text_PIC={info.raw_pic_count}, IDs={info.raw_image_id_count}")

    # 阶段1: 列出章节标题
    print(f"  阶段1: 识别章节标题...")
    titles_output = call_model(client, PROMPT_TITLES.format(text=text), max_tokens=3000)
    titles = parse_titles_simple(titles_output)
    print(f"  识别到 {len(titles)} 个标题: {titles[:5]}...")

    if not titles:
        # 备选：让模型直接输出带内容的格式
        alt_output = call_model(client,
            f"请将以下手册分成若干章节，每节先写标题（用 # 编号. 格式），再写摘要（50字以内）：\n\n{text[:8000]}",
            max_tokens=4000)
        titles = parse_titles_simple(alt_output)
        print(f"  备选识别到 {len(titles)} 个标题")

    # 如果模型无法识别标题，直接用固定字数分块
    if not titles:
        print("  模型无法识别标题，直接使用固定字数分块")
        valid_sections = _split_by_fixed_size(text)
        print(f"  固定分块: {len(valid_sections)} 个章节")
    else:
        # 阶段2: 从原文切分
        chunks = split_by_titles(text, titles)
        valid_sections = [(t, c) for t, c in chunks if c.strip()]
        print(f"  切分出 {len(valid_sections)} 个有效章节")

        # 检查：最大节是否过大（如冰箱可能有6000+字的节）
        max_section_size = max(
            (len(re.sub(r'<PIC>(?:\[.*?\])?', '', c)) for _, c in valid_sections),
            default=0
        )
        if max_section_size > 2000 and len(valid_sections) < 5:
            print(f"  警告: 最大节={max_section_size}字，节数={len(valid_sections)}，改用固定字数分块")
            valid_sections = _split_by_fixed_size(text)
            print(f"  重新切分: {len(valid_sections)} 个有效章节")

    # 确保 valid_sections 不为空
    if not valid_sections:
        valid_sections = _split_by_fixed_size(text)
        print(f"  备用固定分块: {len(valid_sections)} 个章节")

    # 阶段3: 贪心合并（不再依赖模型）
    print(f"  阶段3: 规划合并...")
    merge_blocks = greedy_merge(valid_sections, min_chunk_size=100, target_avg_size=600)
    print(f"  合并方案: {len(merge_blocks)} 块")

    # 阶段4: 构建块（带去重）
    lib_chunks = []
    for block in merge_blocks:
        # 过滤有效范围内的索引，并去重顺序
        seen_sidx = set()
        block_titles = []
        block_contents = []
        for n in block:
            sidx = n - 1  # 转为 0-based
            if 0 <= sidx < len(valid_sections) and sidx not in seen_sidx:
                seen_sidx.add(sidx)
                block_titles.append(valid_sections[sidx][0])
                block_contents.append(valid_sections[sidx][1])

        if not block_contents:
            continue

        # 合并标题
        if len(block_titles) >= 2:
            combined = f"{block_titles[0]}、{block_titles[1]}"
        elif block_titles:
            combined = block_titles[0]
        else:
            combined = "合并块"

        # 合并内容：去除内部子标题行
        combined_lines = []
        for c in block_contents:
            for line in c.split('\n'):
                stripped = line.strip()
                if stripped and not stripped.startswith('#'):
                    combined_lines.append(line)
                elif stripped.startswith('#') and not stripped.lstrip().startswith('#'):
                    pass  # 跳过子标题行
        combined_content = '\n'.join(combined_lines)

        # 过滤空块（内容少于20字的跳过）
        text_only = re.sub(r'<PIC>(?:\[.*?\])?', '', combined_content).strip()
        if len(text_only) >= 20:
            lib_chunks.append((combined, combined_content))

    print(f"  构建块: {len(lib_chunks)} 个（过滤空块后）")

    # 构建章节化版
    sec_lines = []
    for title, content in valid_sections:
        sec_lines.append(f"# {title}")
        sec_lines.append(content)
        sec_lines.append('')
    sec = post_process('\n'.join(sec_lines).strip())

    # 构建建库版
    lib_lines = []
    for title, content in lib_chunks:
        lib_lines.append(f"# {title}")
        lib_lines.append(content)
        lib_lines.append('')
    lib = post_process('\n'.join(lib_lines).strip())

    # 绑定 PIC
    sec_with_ids = bind_pic(sec, image_ids)
    lib_with_ids = bind_pic(lib, image_ids)

    # 验证
    sec_pic = count_pic(sec_with_ids)
    lib_pic = count_pic(lib_with_ids)
    empty_lib = sum(1 for _, c in lib_chunks
                    if len(re.sub(r'<PIC>(?:\[.*?\])?', '', c).strip()) < 10)

    info.sectioned_count = len(re.findall(r'^# ', sec_with_ids, re.MULTILINE))
    info.chunk_count = len(re.findall(r'^# ', lib_with_ids, re.MULTILINE))
    info.empty_chunks = empty_lib
    info.pic_ok = (sec_pic == expected_pic and lib_pic == expected_pic)

    if info.pic_ok:
        info.notes = f"OK: PIC={expected_pic}, 章节={info.sectioned_count}, 块={info.chunk_count}, 空块={info.empty_chunks}"
    else:
        info.notes = f"PIC: 章节化={sec_pic}/{expected_pic}, 建库={lib_pic}/{expected_pic}, 块={info.chunk_count}, 空块={info.empty_chunks}"
        print(f"  警告: {info.notes}")

    output_dir = PROJECT_ROOT / '手册' / '建库优化_模型版'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f'{name}_章节化.txt').write_text(sec_with_ids, encoding='utf-8')
    (output_dir / f'{name}_章节化_建库版.txt').write_text(lib_with_ids, encoding='utf-8')

    print(f"  完成: 章节化 {info.sectioned_count} 节, 建库 {info.chunk_count} 块, 空块 {info.empty_chunks}")
    return info


def main():
    import argparse
    parser = argparse.ArgumentParser(description="手册章节化 v4（修复重复索引问题）")
    parser.add_argument("--manual", type=str, help="指定处理某本手册")
    parser.add_argument("--all", action="store_true", help="重新处理全部")
    args = parser.parse_args()

    client = get_llm_client()
    if not client:
        print("错误: 无法初始化 LLM 客户端"); return

    from config.settings import settings
    print(f"使用模型: {settings.llm_model}")

    manual_dir = PROJECT_ROOT / '手册'
    EXCLUDE = {'相机手册_章节化.txt', '汇总英文手册.txt'}

    txt_files = sorted([
        f for f in manual_dir.glob('*.txt')
        if f.name not in EXCLUDE
    ])

    if args.manual:
        txt_files = [f for f in txt_files if f.name == args.manual]
        if not txt_files:
            print(f"未找到: {args.manual}"); return

    output_dir = PROJECT_ROOT / '手册' / '建库优化_模型版'

    to_process = []
    for f in txt_files:
        if args.all:
            to_process.append(f)
            continue
        lib_path = output_dir / f'{f.stem}_章节化.txt'
        if not lib_path.exists():
            to_process.append(f)

    print(f"\n待处理: {len(to_process)} 本\n")
    for f in to_process:
        print(f"  - {f.name}")

    infos = []
    for manual_path in to_process:
        print(f"\n处理: {manual_path.name}")
        try:
            infos.append(process_manual(manual_path, client))
        except Exception as e:
            print(f"  错误: {e}")
            import traceback; traceback.print_exc()

    if infos:
        report = output_dir / '汇总报告_v4.md'
        lines = [f"# 手册章节化汇总报告 v4\n使用模型: {settings.llm_model}\n"]
        for info in infos:
            lines.append(f"\n## {info.name}")
            lines.append(f"- PIC原={info.raw_pic_count}/{info.raw_image_id_count}")
            lines.append(f"- 章节化: {info.sectioned_count} 节, 建库: {info.chunk_count} 块, 空块: {info.empty_chunks}")
            lines.append(f"- {info.notes}")
        report.write_text('\n'.join(lines), encoding='utf-8')
        print(f"\n报告: {report}")


if __name__ == '__main__':
    main()
