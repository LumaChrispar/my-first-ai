#!/usr/bin/env python3
"""
Full Ingestion and Distillation of:
angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k (all 8,700 rows)
Produces large-scale training and validation sets for Aster.
"""

import urllib.request
import json
import ast
import re
import os
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

def clean_answer(asst_content):
    # Strip chain-of-thought <think>...</think>
    return re.sub(r'<think>.*?</think>', '', asst_content, flags=re.DOTALL).strip()

def get_good_pair(u, a):
    u = u.strip()
    if len(u) > 160 or len(u) < 8 or '\n' in u:
        return None
    
    ca = clean_answer(a)
    if not ca:
        return None

    # 1. Full clean answer is concise and complete
    if 30 <= len(ca) <= 280 and ca.count('```') % 2 == 0 and ca[-1] in '.!?\"\'`':
        clean_text = ' '.join(ca.split())
        return u, clean_text
            
    # 2. First paragraph is a complete, self-contained definition
    paragraphs = [p.strip() for p in ca.split('\n\n') if p.strip()]
    if paragraphs:
        p0 = paragraphs[0]
        # Remove bold headers like **Term**:
        p0_clean = re.sub(r'^\*\*[^*]+\*\*:\s*', '', p0)
        p0_clean = re.sub(r'^\#[#\s]*', '', p0_clean)
        if 40 <= len(p0_clean) <= 280 and p0_clean.endswith(('.', '!', '?')) and not p0_clean.startswith(('Step', '1.', 'Here', 'Below', 'To ', 'Sure')):
            clean_text = ' '.join(p0_clean.split())
            return u, clean_text

    return None

def fetch_batch(offset):
    url = f"https://datasets-server.huggingface.co/rows?dataset=angrygiraffe%2Fclaude-opus-4.6-4.7-reasoning-8.7k&config=default&split=train&offset={offset}&length=100"
    req = urllib.request.Request(url, headers={'User-Agent': 'Aster-Large-Corpus-Ingest/1.0'})
    for attempt in range(3):
        try:
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                return offset, data.get('rows', [])
        except Exception as e:
            time.sleep(1 + attempt)
    return offset, []

def main():
    os.makedirs('data', exist_ok=True)
    os.makedirs('models', exist_ok=True)

    # Base Aster identity turns (from demo_chat.jsonl.bak or demo_chat.jsonl)
    identity_turns = []
    base_file = os.path.join('data', 'demo_chat.jsonl.bak')
    if not os.path.exists(base_file):
        base_file = os.path.join('data', 'demo_chat.jsonl')
    
    if os.path.exists(base_file):
        with open(base_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                    if obj.get('role') in ('user', 'assistant'):
                        identity_turns.append(obj)
                except Exception:
                    pass

    print(f"Loaded {len(identity_turns)//2} core identity pairs from {base_file}.")

    # Collect all 8,700 rows
    offsets = list(range(0, 8700, 100))
    print(f"Fetching {len(offsets)} batches (8,700 total rows) from Hugging Face...")
    
    all_rows = []
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(fetch_batch, off): off for off in offsets}
        done = 0
        for f in as_completed(futures):
            off, rows = f.result()
            all_rows.extend(rows)
            done += 1
            if done % 15 == 0 or done == len(offsets):
                print(f"  Progress: {done}/{len(offsets)} batches completed ({len(all_rows)} rows downloaded)...")

    print(f"Download finished in {time.time() - start_time:.1f}s. Total rows: {len(all_rows)}.")

    # Process and filter
    print("Distilling high-quality reasoning Q&A pairs...")
    seen = set()
    pairs = []
    
    # Add identity prompts to seen set
    for i in range(0, len(identity_turns) - 1, 2):
        if identity_turns[i].get('role') == 'user':
            seen.add(identity_turns[i]['content'].strip().lower())

    for r in all_rows:
        try:
            row_data = r.get('row', {})
            raw_msgs = row_data.get('messages', [])
            msgs = ast.literal_eval(raw_msgs) if isinstance(raw_msgs, str) else raw_msgs
            u = next((m['content'] for m in msgs if m.get('role') == 'user'), None)
            a = next((m['content'] for m in msgs if m.get('role') == 'assistant'), None)
            if u and a:
                pair = get_good_pair(u, a)
                if pair:
                    norm = pair[0].lower()
                    if norm not in seen:
                        seen.add(norm)
                        pairs.append(pair)
        except Exception:
            continue

    print(f"Extracted {len(pairs)} unique high-quality reasoning pairs from Claude Opus dataset!")

    # 90% train / 10% validation split
    split_idx = int(len(pairs) * 0.90)
    train_reasoning = pairs[:split_idx]
    valid_reasoning = pairs[split_idx:]

    train_file = os.path.join('data', 'claude_opus_train.jsonl')
    valid_file = os.path.join('data', 'claude_opus_valid.jsonl')

    # Write training set
    with open(train_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"role": "system", "content": ""}) + '\n')
        # Include identity turns first
        for turn in identity_turns:
            f.write(json.dumps(turn, ensure_ascii=False) + '\n')
        # Then reasoning pairs
        for u, a in train_reasoning:
            f.write(json.dumps({"role": "user", "content": u}, ensure_ascii=False) + '\n')
            f.write(json.dumps({"role": "assistant", "content": a}, ensure_ascii=False) + '\n')

    # Write validation set
    with open(valid_file, 'w', encoding='utf-8') as f:
        f.write(json.dumps({"role": "system", "content": ""}) + '\n')
        for u, a in valid_reasoning:
            f.write(json.dumps({"role": "user", "content": u}, ensure_ascii=False) + '\n')
            f.write(json.dumps({"role": "assistant", "content": a}, ensure_ascii=False) + '\n')

    total_train_pairs = (len(identity_turns)//2) + len(train_reasoning)
    total_valid_pairs = len(valid_reasoning)
    train_size = os.path.getsize(train_file)
    valid_size = os.path.getsize(valid_file)

    print("\n" + "="*50)
    print("LARGE CORPUS GENERATION COMPLETE:")
    print(f"  Training File:   {train_file} ({train_size:,} bytes, {total_train_pairs:,} Q&A pairs)")
    print(f"  Validation File: {valid_file} ({valid_size:,} bytes, {total_valid_pairs:,} Q&A pairs)")
    print("="*50)

if __name__ == '__main__':
    main()
