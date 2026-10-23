#!/usr/bin/env python3
"""
GPU Trainer for Aster Architecture using PyTorch + CUDA.
Trains Aster on an NVIDIA GPU (e.g. GTX 1050) and exports binary
checkpoints directly loadable by Aster's C engine and server.
"""

import os
import sys
import math
import struct
import zlib
import hashlib
import json
import argparse
import time

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:
    print("Error: PyTorch is required. Install with: pip install torch")
    sys.exit(1)

# Structural turn markers (from Aster's tokenizer.h)
TOK_SYSTEM = 256
TOK_USER = 257
TOK_ASSISTANT = 258
TOK_END = 259
TOK_PAD = 260
ASTER_BASE_VOCAB = 261

class AsterVocab:
    """Parses an Aster binary .vocab file (ASTERVB1 format)."""
    def __init__(self, vocab_path):
        self.vocab_path = vocab_path
        self.merges = []
        self.bytes_per_token = 3.0
        self.load(vocab_path)

    def load(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        if len(data) < 20 or data[:8] != b"ASTERVB1":
            raise ValueError(f"Not a valid Aster binary vocabulary file: {path}")
        nm, bpt = struct.unpack('<Id', data[8:20])
        self.bytes_per_token = bpt
        self.merges = []
        self.bpe_ranks = {}
        for i in range(nm):
            a, b = struct.unpack('<HH', data[20 + i*4 : 24 + i*4])
            self.merges.append((a, b))
            self.bpe_ranks[(a, b)] = (i, ASTER_BASE_VOCAB + i)
        self.n_merges = len(self.merges)
        self.vocab_size = ASTER_BASE_VOCAB + self.n_merges

    def encode(self, text):
        tokens = [int(b) for b in text.encode('utf-8')]
        if len(tokens) < 2:
            return tokens
        while len(tokens) >= 2:
            best_pair = None
            best_rank = float('inf')
            best_id = None
            for i in range(len(tokens) - 1):
                p = (tokens[i], tokens[i+1])
                if p in self.bpe_ranks:
                    rank, mid = self.bpe_ranks[p]
                    if rank < best_rank:
                        best_rank = rank
                        best_pair = p
                        best_id = mid
            if best_pair is None:
                break
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i+1]) == best_pair:
                    new_tokens.append(best_id)
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            tokens = new_tokens
        return tokens

class CausalSelfAttention(nn.Module):
    def __init__(self, d_model=64, n_head=4, context=128):
        super().__init__()
        self.d_model = d_model
        self.n_head = n_head
        self.head_dim = d_model // n_head
        self.context = context

        self.wq = nn.Linear(d_model, d_model, bias=False)
        self.wk = nn.Linear(d_model, d_model, bias=False)
        self.wv = nn.Linear(d_model, d_model, bias=False)
        self.wo = nn.Linear(d_model, d_model, bias=False)

        # Causal mask buffer
        self.register_buffer(
            "mask",
            torch.tril(torch.ones(context, context)).view(1, 1, context, context)
        )

    def forward(self, x):
        B, T, C = x.size()
        q = self.wq(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(x).view(B, T, self.n_head, self.head_dim).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(self.head_dim))
        att = att.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)

        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.wo(y)

class AsterBlock(nn.Module):
    def __init__(self, d_model=64, n_head=4, d_ff=256, context=128, n_layer=2):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model, eps=1e-5)
        self.attn = CausalSelfAttention(d_model, n_head, context)
        self.ln2 = nn.LayerNorm(d_model, eps=1e-5)
        self.ff1 = nn.Linear(d_model, d_ff, bias=True)
        self.ff2 = nn.Linear(d_ff, d_model, bias=False)
        self.res_scale = 1.0 / math.sqrt(2.0 * n_layer)

    def forward(self, x):
        x = x + self.res_scale * self.attn(self.ln1(x))
        h = F.gelu(self.ff1(self.ln2(x)), approximate='tanh')
        x = x + self.res_scale * self.ff2(h)
        return x

class AsterTransformer(nn.Module):
    def __init__(self, vocab_size=1285, context=128, d_model=64, n_layer=2, n_head=4, d_ff=256):
        super().__init__()
        self.vocab_size = vocab_size
        self.context = context
        self.d_model = d_model
        self.n_layer = n_layer
        self.n_head = n_head
        self.d_ff = d_ff

        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(context, d_model)
        self.layers = nn.ModuleList([
            AsterBlock(d_model, n_head, d_ff, context, n_layer) for _ in range(n_layer)
        ])
        self.ln_f = nn.LayerNorm(d_model, eps=1e-5)

        self.apply(self._init_weights)

    def _init_weights(self, module):
        emb_std = 0.02
        res_std = emb_std / math.sqrt(2.0 * self.n_layer)
        if isinstance(module, nn.Linear):
            if module is getattr(self, 'wo', None) or module is getattr(self, 'ff2', None):
                nn.init.normal_(module.weight, mean=0.0, std=res_std)
            else:
                nn.init.normal_(module.weight, mean=0.0, std=emb_std)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=emb_std)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, idx, targets=None):
        B, T = idx.size()
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for layer in self.layers:
            x = layer(x)
        x = self.ln_f(x)

        # Output projection tied to tok_emb weight
        logits = F.linear(x, self.tok_emb.weight)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.vocab_size), targets.view(-1), ignore_index=-1)

        return logits, loss

def load_dataset_windows(jsonl_path, vocab, block_size=64):
    """Encodes JSONL conversations into causal training token windows."""
    windows = []
    sys_tokens = [TOK_SYSTEM]
    last_user = []

    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line)
            except Exception:
                continue

            role = msg.get('role')
            content = msg.get('content', '')

            if role == 'system':
                sys_tokens = [TOK_SYSTEM]
                if content:
                    sys_tokens.extend(vocab.encode(content))
            elif role == 'user':
                last_user = [TOK_USER] + vocab.encode(content) + [TOK_END]
            elif role == 'assistant':
                asst = [TOK_ASSISTANT] + vocab.encode(content) + [TOK_END]
                turn = sys_tokens + last_user + asst
                if len(turn) >= 2:
                    windows.append(turn[:block_size])

    return windows

def export_to_aster_binary(model, vocab, out_path, step=1000, seed=1234):
    """
    Exports PyTorch model weights into Aster's exact ASTERMD2 binary checkpoint format
    with CRC-32 and SHA-256 integrity block.
    """
    model.eval()
    cpu_model = model.cpu()

    n_layer = cpu_model.n_layer
    n_head = cpu_model.n_head
    d_model = cpu_model.d_model
    d_ff = cpu_model.d_ff
    context = cpu_model.context
    vocab_size = cpu_model.vocab_size
    n_merges = len(vocab.merges)

    per_layer = (2 * d_model) + (4 * d_model * d_model) + (2 * d_model) + (d_model * d_ff + d_ff) + (d_ff * d_model)
    total_params = (vocab_size * d_model) + (context * d_model) + (n_layer * per_layer) + (2 * d_model)

    flat_params = []
    # 1. Token Embedding: [V * C]
    flat_params.extend(cpu_model.tok_emb.weight.detach().numpy().flatten().tolist())
    # 2. Position Embedding: [T * C]
    flat_params.extend(cpu_model.pos_emb.weight.detach().numpy().flatten().tolist())
    # 3. Layers
    for layer in cpu_model.layers:
        flat_params.extend(layer.ln1.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.ln1.bias.detach().numpy().flatten().tolist())
        flat_params.extend(layer.attn.wq.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.attn.wk.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.attn.wv.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.attn.wo.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.ln2.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.ln2.bias.detach().numpy().flatten().tolist())
        flat_params.extend(layer.ff1.weight.detach().numpy().flatten().tolist())
        flat_params.extend(layer.ff1.bias.detach().numpy().flatten().tolist())
        flat_params.extend(layer.ff2.weight.detach().numpy().flatten().tolist())
    # 4. Final LN: [2 * C]
    flat_params.extend(cpu_model.ln_f.weight.detach().numpy().flatten().tolist())
    flat_params.extend(cpu_model.ln_f.bias.detach().numpy().flatten().tolist())

    assert len(flat_params) == total_params, f"Param count mismatch: {len(flat_params)} vs {total_params}"

    magic = b"ASTERMD2"
    header = struct.pack(
        "<8s12IIIfI",
        magic,
        2, 1, 2,  # format_version, arch_version, tokenizer_version
        n_layer, n_head, d_model, d_ff, context, vocab_size,
        total_params,
        step, seed,
        0,   # has_optimizer
        60,  # max_new_tokens_default
        0.0, # temperature_default
        n_merges
    )

    bpt_bytes = struct.pack("<d", vocab.bytes_per_token)
    merge_bytes = bytearray()
    for a, b in vocab.merges:
        merge_bytes.extend(struct.pack("<HH", a, b))

    param_bytes = struct.pack(f"<{len(flat_params)}f", *flat_params)
    body = header + bpt_bytes + bytes(merge_bytes) + param_bytes

    # Integrity block: CRC-32 (4 bytes LE) + SHA-256 (32 bytes raw)
    crc = zlib.crc32(body) & 0xFFFFFFFF
    sha = hashlib.sha256(body).digest()
    full_checkpoint = body + struct.pack("<I", crc) + sha

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    with open(out_path, 'wb') as f:
        f.write(full_checkpoint)

    print(f"Successfully exported Aster binary checkpoint:")
    print(f"  File: {out_path} ({len(full_checkpoint):,} bytes)")

def train_gpu(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 60)
    print("  ASTER GPU TRAINER (PyTorch + CUDA)")
    print("=" * 60)
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU Hardware: {torch.cuda.get_device_name(0)}")

    if not os.path.exists(args.vocab):
        print(f"Error: vocabulary file {args.vocab} not found.")
        return

    vocab = AsterVocab(args.vocab)
    print(f"Vocabulary: {vocab.vocab_size} tokens ({vocab.n_merges} merges)")

    model = AsterTransformer(vocab_size=vocab.vocab_size).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Load data
    print(f"Encoding dataset {args.data}...")
    windows = load_dataset_windows(args.data, vocab, block_size=64)
    print(f"Loaded {len(windows):,} training sequence windows.")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

    print(f"Starting training for {args.steps} steps (batch {args.batch})...")
    start_t = time.time()
    model.train()

    import random
    for step in range(1, args.steps + 1):
        batch_samples = random.sample(windows, min(args.batch, len(windows)))
        
        # Pad batch
        max_len = max(len(w) for w in batch_samples)
        if max_len < 2:
            continue

        inputs, targets = [], []
        for w in batch_samples:
            inp = w[:-1] + [TOK_PAD] * (max_len - len(w))
            tgt = w[1:] + [-1] * (max_len - len(w))
            inputs.append(inp)
            targets.append(tgt)

        x = torch.tensor(inputs, dtype=torch.long, device=device)
        y = torch.tensor(targets, dtype=torch.long, device=device)

        logits, loss = model(x, y)
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % 50 == 0 or step == args.steps:
            elapsed = time.time() - start_t
            tok_s = (step * args.batch * max_len) / max(elapsed, 0.001)
            print(f"Step {step:5d}/{args.steps} | Loss: {loss.item():.4f} | Speed: {tok_s:.0f} tok/s | Elapsed: {elapsed:.1f}s")

    print(f"Training completed in {time.time() - start_t:.1f}s!")
    export_to_aster_binary(model, vocab, args.out, step=args.steps, seed=1234)

def main():
    parser = argparse.ArgumentParser(description="GPU Trainer for Aster")
    parser.add_argument("--data", default="data/claude_opus_train.jsonl")
    parser.add_argument("--vocab", default="models/aster-claude.vocab")
    parser.add_argument("--out", default="models/aster-gpu.bin")
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_gpu(args)

if __name__ == '__main__':
    main()
