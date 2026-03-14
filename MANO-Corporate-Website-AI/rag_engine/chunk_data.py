import os
import re
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(_HERE)  # project root

PAGES_DIR   = os.path.join(_ROOT, 'knowledge_base', 'pages')
OUTPUT_FILE = os.path.join(_ROOT, 'knowledge_base', 'chunks.json')

page_files = sorted(f for f in os.listdir(PAGES_DIR) if f.endswith('.txt'))

def parse_pages_to_chunks(pages_dir: str, page_files: list) -> list:
    all_chunks = []
    chunk_id   = 0

    for filename in page_files:
        filepath = os.path.join(pages_dir, filename)
        with open(filepath, 'r', encoding='utf-8') as f:
            raw = f.read()

        page_name = 'Unknown'
        page_url  = ''
        for line in raw.splitlines()[:5]:
            if line.startswith('PAGE:'):
                page_name = line.replace('PAGE:', '').strip()
            elif line.startswith('URL:'):
                page_url = line.replace('URL:', '').strip()

        raw_blocks = re.split(r'={3,}', raw)
        page_chunks_count = 0

        for block in raw_blocks:
            block = block.strip()
            block = re.sub(r'^\[\s*SECTION\s*\d+\s*\]\s*', '', block).strip()
            if not block or block.startswith('PAGE:') or len(block) < 30:
                continue

            lines = [ln.strip() for ln in block.splitlines() if ln.strip()]
            clean_content = '\n'.join(lines)

            if len(clean_content) < 30:
                continue

            all_chunks.append({
                'id':          f'chunk_{chunk_id:04d}',
                'page_name':   page_name,
                'url':         page_url,
                'source_file': filename,
                'section_num': page_chunks_count,
                'content':     clean_content,
            })
            chunk_id += 1
            page_chunks_count += 1

    return all_chunks

chunks = parse_pages_to_chunks(PAGES_DIR, page_files)
print(f'Total chunks created: {len(chunks)}')

os.makedirs(os.path.dirname(os.path.abspath(OUTPUT_FILE)), exist_ok=True)
with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
    json.dump(chunks, f, indent=2, ensure_ascii=False)
