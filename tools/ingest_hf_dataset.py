#!/usr/bin/env python3
"""
Ingest and distill Q&A pairs from Hugging Face dataset:
angrygiraffe/claude-opus-4.6-4.7-reasoning-8.7k
Tailored for the Aster small language model architecture.
"""

import urllib.request
import json
import ast
import re
import os
import shutil

def clean_answer(asst_content):
    # Remove <think>...</think> reasoning traces
    clean = re.sub(r'<think>.*?</think>', '', asst_content, flags=re.DOTALL).strip()
    return clean

def get_good_pair(u, a):
    u = u.strip()
    # Ensure user question is reasonably compact and self-contained
    if len(u) > 130 or len(u) < 10 or '\n' in u:
        return None
    
    ca = clean_answer(a)
    
    # 1. Full clean answer is concise and complete
    if 35 <= len(ca) <= 260 and ca.count('```') % 2 == 0 and ca[-1] in '.!?\"\'`':
        clean_text = ' '.join(ca.split())
        return u, clean_text
            
    # 2. First paragraph is a complete, self-contained definition
    paragraphs = [p.strip() for p in ca.split('\n\n') if p.strip()]
    if paragraphs:
        p0 = paragraphs[0]
        # Remove bold headers like **Term**:
        p0_clean = re.sub(r'^\*\*[^*]+\*\*:\s*', '', p0)
        p0_clean = re.sub(r'^\#[#\s]*', '', p0_clean)
        if 40 <= len(p0_clean) <= 260 and p0_clean.endswith('.') and not p0_clean.startswith(('Step', '1.', 'Here', 'Below', 'To ', 'Sure')):
            clean_text = ' '.join(p0_clean.split())
            return u, clean_text
            
    return None

def main():
    chat_file = os.path.join('data', 'demo_chat.jsonl')
    valid_file = os.path.join('data', 'demo_valid.jsonl')
    
    # Backup original files if backups don't exist
    if not os.path.exists(chat_file + '.bak'):
        shutil.copyfile(chat_file, chat_file + '.bak')
        print(f"Backed up {chat_file} -> {chat_file}.bak")
    if not os.path.exists(valid_file + '.bak'):
        shutil.copyfile(valid_file, valid_file + '.bak')
        print(f"Backed up {valid_file} -> {valid_file}.bak")

    # Read existing prompts to avoid duplicates
    existing_prompts = set()
    with open(chat_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('role') == 'user':
                    existing_prompts.add(obj['content'].strip().lower())
            except Exception:
                pass
                
    with open(valid_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                obj = json.loads(line)
                if obj.get('role') == 'user':
                    existing_prompts.add(obj['content'].strip().lower())
            except Exception:
                pass

    print(f"Loaded {len(existing_prompts)} existing prompts.")

    pairs = []
    seen = set(existing_prompts)
    
    # Fetch 1,000 rows across 10 batches
    print("Fetching dataset rows from Hugging Face...")
    for offset in range(0, 1000, 100):
        url = f"https://datasets-server.huggingface.co/rows?dataset=angrygiraffe%2Fclaude-opus-4.6-4.7-reasoning-8.7k&config=default&split=train&offset={offset}&length=100"
        req = urllib.request.Request(url, headers={'User-Agent': 'Aster-Dataset-Ingest/1.0'})
        try:
            with urllib.request.urlopen(req) as resp:
                data = json.loads(resp.read().decode('utf-8'))
                for r in data.get('rows', []):
                    msgs = ast.literal_eval(r['row']['messages']) if isinstance(r['row']['messages'], str) else r['row']['messages']
                    u = next((m['content'] for m in msgs if m['role'] == 'user'), None)
                    a = next((m['content'] for m in msgs if m['role'] == 'assistant'), None)
                    if u and a:
                        pair = get_good_pair(u, a)
                        if pair:
                            norm_q = pair[0].lower()
                            if norm_q not in seen:
                                seen.add(norm_q)
                                pairs.append(pair)
            print(f"  Batch offset {offset:4d}: Total candidate pairs so far: {len(pairs)}")
        except Exception as e:
            print(f"  Error at offset {offset}: {e}")

    print(f"\nSuccessfully collected {len(pairs)} high-quality unique Q&A pairs.")

    # Split into train (85%) and validation (15%)
    split_idx = int(len(pairs) * 0.85)
    train_pairs = pairs[:split_idx]
    valid_pairs = pairs[split_idx:]

    print(f"Appending {len(train_pairs)} pairs to {chat_file}...")
    with open(chat_file, 'a', encoding='utf-8') as f:
        for u, a in train_pairs:
            f.write(json.dumps({"role": "user", "content": u}, ensure_ascii=False) + '\n')
            f.write(json.dumps({"role": "assistant", "content": a}, ensure_ascii=False) + '\n')

    print(f"Appending {len(valid_pairs)} pairs to {valid_file}...")
    with open(valid_file, 'a', encoding='utf-8') as f:
        for u, a in valid_pairs:
            f.write(json.dumps({"role": "user", "content": u}, ensure_ascii=False) + '\n')
            f.write(json.dumps({"role": "assistant", "content": a}, ensure_ascii=False) + '\n')

    print("Ingestion complete!")

if __name__ == '__main__':
    main()
