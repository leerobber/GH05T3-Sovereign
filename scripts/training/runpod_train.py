#!/usr/bin/env python3
"""
SovereignNation — Avery 3B LoRA Training
Run this on RunPod (T4/A100/3090) or any Linux GPU box.

Usage:
    python runpod_train.py --hf_token hf_xxx
    python runpod_train.py  # reads HF_TOKEN from env
"""
import os, sys, argparse, subprocess

# ── Args ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--hf_token',   default=os.environ.get('HF_TOKEN', ''))
parser.add_argument('--base_model', default='Qwen/Qwen2.5-3B-Instruct')
parser.add_argument('--hf_dataset', default='tastytator/sovereign-economy')
parser.add_argument('--hf_repo_out',default='tastytator/avery-sovereign-lora')
parser.add_argument('--output_dir', default='/workspace/sovereign-lora')
parser.add_argument('--epochs',     type=int,   default=1)
parser.add_argument('--max_seq',    type=int,   default=2048)
parser.add_argument('--batch_size', type=int,   default=2)
parser.add_argument('--grad_acc',   type=int,   default=4)
parser.add_argument('--lora_r',     type=int,   default=16)
parser.add_argument('--no_push',    action='store_true', help='Skip HF push (local only)')
args = parser.parse_args()

HF_TOKEN = args.hf_token
if not HF_TOKEN:
    print('ERROR: HF_TOKEN not set. Pass --hf_token or set HF_TOKEN env var.')
    sys.exit(1)

# ── Install deps ──────────────────────────────────────────────────────────────
print('=' * 60)
print('Installing dependencies...')
# Pin transformers first with --no-deps to avoid version conflicts with pod's existing packages
subprocess.run(
    'pip install -q --force-reinstall --no-deps '
    '"transformers==4.47.1" "tokenizers>=0.20,<0.22" safetensors',
    shell=True, check=True
)
subprocess.run(
    'pip install -q '
    '"trl==0.12.2" "peft==0.14.0" "bitsandbytes>=0.43.0" '
    '"accelerate==1.2.1" datasets huggingface_hub',
    shell=True, check=True
)
print('Deps ready.')

# ── GPU check ─────────────────────────────────────────────────────────────────
import torch
if not torch.cuda.is_available():
    print('ERROR: No CUDA GPU found.')
    sys.exit(1)

gpu_name = torch.cuda.get_device_name(0)
vram_gb  = torch.cuda.get_device_properties(0).total_memory / 1e9
major, minor = torch.cuda.get_device_capability(0)
compute_cap = major + minor / 10
has_bf16    = torch.cuda.is_bf16_supported()
torch_dtype = torch.bfloat16 if has_bf16 else torch.float16
load_4bit   = compute_cap >= 7.5  # bitsandbytes NF4 needs sm_7.5+

print(f'\nGPU : {gpu_name}')
print(f'VRAM: {vram_gb:.1f} GB')
print(f'sm  : {compute_cap}  BF16={has_bf16}  4bit={load_4bit}')
print(f'dtype: {torch_dtype}')

# ── Load dataset ──────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print('Loading economy dataset from HuggingFace...')
from datasets import load_dataset

eco_raw = load_dataset(args.hf_dataset, name='sft', split='train', token=HF_TOKEN)
print(f'Raw rows: {len(eco_raw):,}  cols: {eco_raw.column_names}')

def normalize(row):
    msgs  = row.get('messages', [])
    users = [m['content'] for m in msgs if m.get('role') == 'user']
    astts = [m['content'] for m in msgs if m.get('role') == 'assistant']
    return {
        'instruction': users[0].strip() if users else '',
        'response':    astts[-1].strip() if astts else '',
    }

ds = eco_raw.map(normalize, remove_columns=eco_raw.column_names, num_proc=4)
ds = ds.filter(lambda r: len(r['instruction']) > 20 and len(r['response']) > 20)
print(f'Filtered rows: {len(ds):,}')

# ── Load model ────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print(f'Loading {args.base_model}...')
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

tokenizer = AutoTokenizer.from_pretrained(args.base_model, token=HF_TOKEN)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = 'right'

if load_4bit:
    from transformers import BitsAndBytesConfig
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch_dtype,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, quantization_config=bnb,
        token=HF_TOKEN, device_map='auto',
    )
    model = prepare_model_for_kbit_training(model)
    print('Loaded in 4-bit NF4')
else:
    # FP16 fallback (V100/P100 with compatible PyTorch)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, dtype=torch_dtype,
        token=HF_TOKEN, device_map='auto',
    )
    print('Loaded in FP16')

model.config.use_cache = False

lora_cfg = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=args.lora_r, lora_alpha=args.lora_r * 2, lora_dropout=0.05,
    target_modules=['q_proj','k_proj','v_proj','o_proj','gate_proj','up_proj','down_proj'],
    bias='none',
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# ── Format dataset ────────────────────────────────────────────────────────────
SYSTEM = (
    'You are Avery, the sovereign business strategist for SovereignNation — '
    'a fixed-cost AI platform built for lower and middle class families, '
    'children education, and affordable connectivity. '
    'Use the KAIROS framework: Kickoff, Alignment, Implementation, '
    'Refinement, Optimization, Scaling. Be direct, structured, actionable.'
)

def fmt(row):
    g = str(row.get('instruction') or '')
    r = str(row.get('response') or '')
    text = (
        '<|im_start|>system\n' + SYSTEM + '<|im_end|>\n'
        '<|im_start|>user\n' + g + '<|im_end|>\n'
        '<|im_start|>assistant\n' + r + '<|im_end|>'
    )
    return {'text': text}

ds_fmt = ds.map(fmt, remove_columns=ds.column_names)
print(f'\nFormatted {len(ds_fmt):,} examples for SFT')

# ── Train ─────────────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
print(f'Training: epochs={args.epochs}  batch={args.batch_size}  grad_acc={args.grad_acc}')

from trl import SFTTrainer, SFTConfig

use_bf16 = has_bf16
use_fp16 = not has_bf16

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=ds_fmt,
    args=SFTConfig(
        dataset_text_field='text',
        max_seq_length=args.max_seq,
        packing=False,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_acc,
        num_train_epochs=args.epochs,
        learning_rate=2e-4,
        bf16=use_bf16, fp16=use_fp16,
        logging_steps=50,
        output_dir=args.output_dir,
        warmup_ratio=0.1,
        lr_scheduler_type='cosine',
        save_strategy='epoch',
        report_to='none',
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={'use_reentrant': False},
        dataloader_pin_memory=False,
    ),
)

stats = trainer.train()
loss = stats.metrics.get('train_loss', 0)
rt   = stats.metrics.get('train_runtime', 0)
print(f'\nTraining complete!  Loss={loss:.4f}  Time={rt/60:.1f} min')

# ── Save & push ───────────────────────────────────────────────────────────────
print('\n' + '=' * 60)
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f'Saved to {args.output_dir}')

if not args.no_push:
    print(f'Pushing to {args.hf_repo_out}...')
    model.push_to_hub(args.hf_repo_out, token=HF_TOKEN, private=False)
    tokenizer.push_to_hub(args.hf_repo_out, token=HF_TOKEN, private=False)
    print(f'\nhttps://huggingface.co/{args.hf_repo_out}')

print('\n' + '=' * 60)
print(f'  DONE — Loss={loss:.4f}  Time={rt/60:.1f} min')
print('=' * 60)

