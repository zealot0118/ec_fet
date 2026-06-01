"""
EC-FET Linear Attention: WikiText-103 Convergence Experiment
wikitext_experiment.py 뼈대 기반

변경사항:
    - Dataset: wikitext-103-raw-v1 (~103M chars)
    - EP=5 (수렴 경향 확인용)
    - BS=32, D=256, NH=8, SEQ=256
    - configs: 4개 (Softmax, ELU+1, EC r=200, GLA EC r=200)
    - 논문용 convergence plot 자동 생성
"""

import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from datasets import load_dataset
import numpy as np, math, sys, time, os, argparse, json, glob
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('method', nargs='?', default='all')
parser.add_argument('--seed', type=int, default=42)
args = parser.parse_args()
METHOD = args.method
SEED   = args.seed

torch.manual_seed(SEED); np.random.seed(SEED)
EPS = 1e-10

# ── 설정 (GPU에 맞게 조정) ──
# 8GB GPU: D=256, NH=8, SEQ=256
# 16GB GPU: D=512, NH=8, SEQ=512
# 24GB GPU: D=768, NH=12, SEQ=512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f}GB")

D   = int(os.environ.get('D',   256))
NH  = int(os.environ.get('NH',  8))
SEQ = int(os.environ.get('SEQ', 256))
BS  = int(os.environ.get('BS',  32))
EP  = int(os.environ.get('EP',  5))
LR  = float(os.environ.get('LR', 3e-4))
DH  = D // NH

print(f"\nConfig: D={D}, NH={NH}, SEQ={SEQ}, BS={BS}, EP={EP}")

# ── WikiText-103 데이터 ──
print("\nLoading WikiText-103...")
ds = load_dataset('Salesforce/wikitext', 'wikitext-103-raw-v1')
train_text = ' '.join([t for t in ds['train']['text']      if t.strip()])
val_text   = ' '.join([t for t in ds['validation']['text'] if t.strip()])
test_text  = ' '.join([t for t in ds['test']['text']       if t.strip()])

# char-level
chars = sorted(set(train_text + val_text + test_text))
c2i   = {c: i for i, c in enumerate(chars)}
vocab = len(chars)
print(f"Train: {len(train_text):,} chars, Val: {len(val_text):,}, vocab: {vocab}")

class DS(Dataset):
    def __init__(self, t, stride=SEQ):
        self.d = torch.tensor([c2i.get(c, 0) for c in t], dtype=torch.long)
        self.stride = stride
    def __len__(self):  return max(0, (len(self.d) - SEQ) // self.stride)
    def __getitem__(self, i):
        s = i * self.stride
        return self.d[s:s+SEQ], self.d[s+1:s+SEQ+1]

trdl = DataLoader(DS(train_text), BS, shuffle=True,  drop_last=True,  num_workers=4, pin_memory=True)
vdl  = DataLoader(DS(val_text),   BS, shuffle=False, drop_last=True,  num_workers=4, pin_memory=True)
tedl = DataLoader(DS(test_text),  BS, shuffle=False, drop_last=True,  num_workers=4, pin_memory=True)
print(f"Batches: train={len(trdl)}, val={len(vdl)}, test={len(tedl)}")

# ── Feature maps ──
def phi_ec(x, ratio=7):
    # normalize by G_max: output in [1/ratio, 1], score ~ DH regardless of ratio
    sc = x.abs().amax(dim=-1, keepdim=True).clamp(1e-10)
    return (1./ratio) + (1. - 1./ratio) * ((x + sc) / (2 * sc)).clamp(0, 1)

def phi_ec_quantized(x, ratio=7, n_states=256):
    # HW precision 제약: n_states 개의 discrete conductance level
    sc = x.abs().amax(dim=-1, keepdim=True).clamp(1e-10)
    norm_x = ((x + sc) / (2 * sc)).clamp(0, 1)
    idx = torch.round(norm_x * (n_states - 1))
    return 1./ratio + (1. - 1./ratio) * idx / (n_states - 1)

def quantize_gate(gate, n_bits):
    """gate ∈ (0,1) → n_bits discrete levels. HW: DAC n_bits precision으로 V_WL 인가."""
    n_levels = 2 ** n_bits
    step = 1.0 / (n_levels - 1)
    return (torch.round(gate / step) * step).clamp(0., 1.)

def quantize_vwl(gated_pQ, n_bits):
    """
    V_WL = gate × phi_EC(Q) 전체를 n_bits로 quantize

    gate 단독 quantize 대비 장점:
        gate ∈ [0.82, 1.0] → [0,1] 256등분 시 47 levels만 사용
        V_WL = gate × phi_EC(Q): phi_EC(Q)가 넓게 분포
        → 256 levels를 90~100% 활용
        → 같은 8bit DAC로 더 정밀한 표현
    HW 의미:
        SW에서 gate × phi_EC(Q) 계산 후
        DAC가 이 값을 n_bits로 양자화하여 V_WL 인가
        → gate와 Q를 별도로 DAC에 넣는 대신
          곱해진 V_WL을 단일 DAC로 인가
    """
    n_levels = 2 ** n_bits
    # gated_pQ ∈ (0, 1) (gate와 phi_ec 모두 (0,1))
    step = 1.0 / (n_levels - 1)
    return (torch.round(gated_pQ / step) * step).clamp(0., 1.)

# ── Attention implementations ──
def la_fwd(Q, K, V, phi_fn):
    """Parallel causal linear attention"""
    pQ = phi_fn(Q); pK = phi_fn(K); T = Q.shape[2]
    s  = torch.matmul(pQ, pK.transpose(-2, -1))
    s  = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=Q.device)), 0.)
    return torch.matmul(s / s.sum(-1, keepdim=True).clamp(EPS), V)

def gla_fwd(Q, K, V, phi_fn, gate_logit):
    """Gated Linear Attention with log-cumsum decay"""
    B, H, T, d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)
    scores = torch.matmul(pQ, pK.transpose(-2, -1))
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=Q.device))
    scores = scores.masked_fill(~causal, 0.)
    log_gate  = -F.softplus(-gate_logit.squeeze(-1))
    log_cum   = torch.cumsum(log_gate, dim=-1)
    log_cum_s = torch.cat([torch.zeros(B, H, 1, device=Q.device), log_cum[:,:,:-1]], dim=-1)
    decay = torch.exp(
        (log_cum_s.unsqueeze(-1) - log_cum_s.unsqueeze(-2)).clamp(-50, 0))
    scores = (scores * decay).masked_fill(~causal, 0.)
    return torch.matmul(scores / scores.sum(-1, keepdim=True).clamp(EPS), V)

def la_fwd_harmonic(Q, K, V, phi_fn, chunk=32):
    """Chunked harmonic LA: (B,H,chunk,klen,d) 단위로 처리해 OOM 회피."""
    pQ = phi_fn(Q); pK = phi_fn(K)
    B, H, T, d = pQ.shape
    out = torch.zeros_like(V)
    for i in range(0, T, chunk):
        pQ_c = pQ[:,:,i:i+chunk,:]
        clen = pQ_c.shape[2]; klen = i + clen
        a = pK[:,:,:klen,:].unsqueeze(2) * pQ_c.unsqueeze(3)   # (B,H,clen,klen,d)
        h = 1. / (1./a.float().clamp(EPS)).sum(-1)                # (B,H,clen,klen)
        t_q = torch.arange(i, i+clen, device=Q.device)
        t_k = torch.arange(klen, device=Q.device)
        h = h * (t_q.unsqueeze(1) >= t_k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        h = h / h.sum(-1, keepdim=True).clamp(EPS)
        out[:,:,i:i+clen,:] = torch.matmul(h, V[:,:,:klen,:])
    return out

def gla_fwd_harmonic(Q, K, V, phi_fn, gate_logit, chunk=32):
    """Chunked GLA harmonic: (B,H,chunk,klen,d) 단위로 처리해 OOM 회피."""
    B, H, T, d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)
    gate     = torch.sigmoid(gate_logit.squeeze(-1))
    gated_pQ = gate.unsqueeze(-1) * pQ
    log_gate  = -F.softplus(-gate_logit.squeeze(-1))
    log_cum   = torch.cumsum(log_gate, dim=-1)
    log_cum_s = torch.cat([torch.zeros(B, H, 1, device=Q.device), log_cum[:,:,:-1]], dim=-1)
    out = torch.zeros_like(V)
    for i in range(0, T, chunk):
        gpQ_c = gated_pQ[:,:,i:i+chunk,:]
        clen = gpQ_c.shape[2]; klen = i + clen
        a    = pK[:,:,:klen,:].unsqueeze(2) * gpQ_c.unsqueeze(3)   # (B,H,clen,klen,d)
        harm = 1. / (1./a.float().clamp(EPS)).sum(-1)                # (B,H,clen,klen)
        lcs_q = log_cum_s[:,:,i:i+clen]
        lcs_k = log_cum_s[:,:,:klen]
        decay = torch.exp((lcs_q.unsqueeze(-1) - lcs_k.unsqueeze(-2)).clamp(-50, 0))
        t_q = torch.arange(i, i+clen, device=Q.device)
        t_k = torch.arange(klen, device=Q.device)
        mask = (t_q.unsqueeze(1) >= t_k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        scores = (harm * decay).masked_fill(~mask, 0.)
        out[:,:,i:i+clen,:] = torch.matmul(scores / scores.sum(-1, keepdim=True).clamp(EPS), V[:,:,:klen,:])
    return out

# ── Analysis용 attention weight 반환 버전 (학습 코드와 분리) ──
def la_fwd_attn(Q, K, V, phi_fn):
    pQ = phi_fn(Q); pK = phi_fn(K); T = Q.shape[2]
    s = torch.matmul(pQ, pK.transpose(-2, -1))
    s = s.masked_fill(~torch.tril(torch.ones(T, T, dtype=torch.bool, device=Q.device)), 0.)
    attn_w = s / s.sum(-1, keepdim=True).clamp(EPS)
    return torch.matmul(attn_w, V), attn_w

def gla_fwd_attn(Q, K, V, phi_fn, gate_logit):
    B, H, T, d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)
    scores = torch.matmul(pQ, pK.transpose(-2, -1))
    causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=Q.device))
    scores = scores.masked_fill(~causal, 0.)
    log_gate  = -F.softplus(-gate_logit.squeeze(-1))
    log_cum   = torch.cumsum(log_gate, dim=-1)
    log_cum_s = torch.cat([torch.zeros(B, H, 1, device=Q.device), log_cum[:,:,:-1]], dim=-1)
    decay = torch.exp(
        (log_cum_s.unsqueeze(-1) - log_cum_s.unsqueeze(-2)).clamp(-50, 0))
    scores = (scores * decay).masked_fill(~causal, 0.)
    attn_w = scores / scores.sum(-1, keepdim=True).clamp(EPS)
    return torch.matmul(attn_w, V), attn_w

def la_fwd_attn_harmonic(Q, K, V, phi_fn, chunk=32):
    pQ = phi_fn(Q); pK = phi_fn(K)
    B, H, T, d = pQ.shape
    out = torch.zeros_like(V)
    attn_w_full = torch.zeros(B, H, T, T, device=Q.device, dtype=pQ.dtype)
    for i in range(0, T, chunk):
        pQ_c = pQ[:,:,i:i+chunk,:]
        clen = pQ_c.shape[2]; klen = i + clen
        a = pK[:,:,:klen,:].unsqueeze(2) * pQ_c.unsqueeze(3)
        h = 1. / (1./a.clamp(EPS)).sum(-1)
        t_q = torch.arange(i, i+clen, device=Q.device)
        t_k = torch.arange(klen, device=Q.device)
        h = h * (t_q.unsqueeze(1) >= t_k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        h = h / h.sum(-1, keepdim=True).clamp(EPS)
        out[:,:,i:i+clen,:] = torch.matmul(h, V[:,:,:klen,:])
        attn_w_full[:,:,i:i+clen,:klen] = h
    return out, attn_w_full

def gla_fwd_attn_harmonic(Q, K, V, phi_fn, gate_logit, chunk=32):
    B, H, T, d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)
    gate     = torch.sigmoid(gate_logit.squeeze(-1))
    gated_pQ = gate.unsqueeze(-1) * pQ
    log_gate  = -F.softplus(-gate_logit.squeeze(-1))
    log_cum   = torch.cumsum(log_gate, dim=-1)
    log_cum_s = torch.cat([torch.zeros(B, H, 1, device=Q.device), log_cum[:,:,:-1]], dim=-1)
    out = torch.zeros_like(V)
    attn_w_full = torch.zeros(B, H, T, T, device=Q.device, dtype=pQ.dtype)
    for i in range(0, T, chunk):
        gpQ_c = gated_pQ[:,:,i:i+chunk,:]
        clen = gpQ_c.shape[2]; klen = i + clen
        a    = pK[:,:,:klen,:].unsqueeze(2) * gpQ_c.unsqueeze(3)
        harm = 1. / (1./a.clamp(EPS)).sum(-1)
        lcs_q = log_cum_s[:,:,i:i+clen]
        lcs_k = log_cum_s[:,:,:klen]
        decay = torch.exp((lcs_q.unsqueeze(-1) - lcs_k.unsqueeze(-2)).clamp(-50, 0))
        t_q = torch.arange(i, i+clen, device=Q.device)
        t_k = torch.arange(klen, device=Q.device)
        mask = (t_q.unsqueeze(1) >= t_k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        scores = (harm * decay).masked_fill(~mask, 0.)
        scores = scores / scores.sum(-1, keepdim=True).clamp(EPS)
        out[:,:,i:i+clen,:] = torch.matmul(scores, V[:,:,:klen,:])
        attn_w_full[:,:,i:i+clen,:klen] = scores
    return out, attn_w_full

def gla_fwd_harmonic_dim(Q, K, V, phi_fn, gate_logit, chunk=32, gate_bits=None, vwl_bits=None):
    """gla_fwd_harmonic과 동일하나 gate_logit: (B,H,T,d) — dimension별 gate."""
    B, H, T, d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)
    gate     = torch.sigmoid(gate_logit)               # (B,H,T,d)
    if gate_bits is not None:
        gate = quantize_gate(gate, gate_bits)
    gated_pQ = gate * pQ                               # V_WL continuous
    if vwl_bits is not None:
        gated_pQ = quantize_vwl(gated_pQ, vwl_bits)   # V_WL quantize
    if gate_bits is not None:
        log_gate = torch.log(gate.mean(-1).clamp(EPS))
    elif vwl_bits is not None:
        log_gate = -F.softplus(-gate_logit).mean(-1)
    else:
        log_gate = -F.softplus(-gate_logit).mean(-1)
    log_cum   = torch.cumsum(log_gate, dim=-1)
    log_cum_s = torch.cat([torch.zeros(B, H, 1, device=Q.device), log_cum[:,:,:-1]], dim=-1)
    out = torch.zeros_like(V)
    for i in range(0, T, chunk):
        gpQ_c = gated_pQ[:,:,i:i+chunk,:]
        clen = gpQ_c.shape[2]; klen = i + clen
        a    = pK[:,:,:klen,:].unsqueeze(2) * gpQ_c.unsqueeze(3)   # (B,H,clen,klen,d)
        harm = 1. / (1./a.float().clamp(EPS)).sum(-1)               # (B,H,clen,klen)
        lcs_q = log_cum_s[:,:,i:i+clen]
        lcs_k = log_cum_s[:,:,:klen]
        decay = torch.exp((lcs_q.unsqueeze(-1) - lcs_k.unsqueeze(-2)).clamp(-50, 0))
        t_q = torch.arange(i, i+clen, device=Q.device)
        t_k = torch.arange(klen, device=Q.device)
        mask = (t_q.unsqueeze(1) >= t_k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        scores = (harm * decay).masked_fill(~mask, 0.)
        out[:,:,i:i+clen,:] = torch.matmul(scores / scores.sum(-1, keepdim=True).clamp(EPS), V[:,:,:klen,:])
    return out

def gla_fwd_attn_harmonic_dim(Q, K, V, phi_fn, gate_logit, chunk=32, gate_bits=None, vwl_bits=None):
    B, H, T, d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)
    gate     = torch.sigmoid(gate_logit)
    if gate_bits is not None:
        gate = quantize_gate(gate, gate_bits)
    gated_pQ = gate * pQ
    if vwl_bits is not None:
        gated_pQ = quantize_vwl(gated_pQ, vwl_bits)
    if gate_bits is not None:
        log_gate = torch.log(gate.mean(-1).clamp(EPS))
    elif vwl_bits is not None:
        log_gate = -F.softplus(-gate_logit).mean(-1)
    else:
        log_gate = -F.softplus(-gate_logit).mean(-1)
    log_cum   = torch.cumsum(log_gate, dim=-1)
    log_cum_s = torch.cat([torch.zeros(B, H, 1, device=Q.device), log_cum[:,:,:-1]], dim=-1)
    out = torch.zeros_like(V)
    attn_w_full = torch.zeros(B, H, T, T, device=Q.device, dtype=pQ.dtype)
    for i in range(0, T, chunk):
        gpQ_c = gated_pQ[:,:,i:i+chunk,:]
        clen = gpQ_c.shape[2]; klen = i + clen
        a    = pK[:,:,:klen,:].unsqueeze(2) * gpQ_c.unsqueeze(3)
        harm = 1. / (1./a.float().clamp(EPS)).sum(-1)
        lcs_q = log_cum_s[:,:,i:i+clen]
        lcs_k = log_cum_s[:,:,:klen]
        decay = torch.exp((lcs_q.unsqueeze(-1) - lcs_k.unsqueeze(-2)).clamp(-50, 0))
        t_q = torch.arange(i, i+clen, device=Q.device)
        t_k = torch.arange(klen, device=Q.device)
        mask = (t_q.unsqueeze(1) >= t_k.unsqueeze(0)).unsqueeze(0).unsqueeze(0)
        scores = (harm * decay).masked_fill(~mask, 0.)
        scores = scores / scores.sum(-1, keepdim=True).clamp(EPS)
        out[:,:,i:i+clen,:] = torch.matmul(scores, V[:,:,:klen,:])
        attn_w_full[:,:,i:i+clen,:klen] = scores
    return out, attn_w_full

# ── 모델 ──
class GPT(nn.Module):
    def __init__(self, at='softmax', ratio=7, n_layers=4):
        super().__init__()
        self.at = at; self.ratio = ratio
        self.emb  = nn.Embedding(vocab, D)
        self.pos  = nn.Embedding(SEQ, D)
        self.drop = nn.Dropout(0.1)

        self.blocks = nn.ModuleList([
            nn.ModuleDict({
                'ln1':  nn.LayerNorm(D),
                'qkv':  nn.Linear(D, 3*D, bias=False),
                'proj': nn.Linear(D, D,   bias=False),
                'ln2':  nn.LayerNorm(D),
                'ff':   nn.Sequential(
                    nn.Linear(D, 4*D), nn.GELU(), nn.Dropout(0.1),
                    nn.Linear(4*D, D), nn.Dropout(0.1)),
                **({'gp': nn.Linear(D, NH * DH, bias=True)} if 'gla_harm_dim' in at
                   else {'gp': nn.Linear(D, NH, bias=True)} if 'gla' in at
                   else {}),
            })
            for _ in range(n_layers)
        ])
        self.lnf  = nn.LayerNorm(D)
        self.head = nn.Linear(D, vocab, bias=False)
        # weight tying
        self.head.weight = self.emb.weight

        # init
        self.apply(self._init_weights)
        if 'gla' in at:
            for blk in self.blocks:
                if 'gp' in blk:
                    nn.init.constant_(blk['gp'].bias, 4.0)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.02)
            if m.bias is not None: nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, std=0.02)

    def phi(self, x):
        if self.at in ('softmax',): return x
        if 'elu' in self.at: return F.elu(x) + 1. + 1e-6
        if 'q256' in self.at: return phi_ec_quantized(x, self.ratio, n_states=256)
        return phi_ec(x, self.ratio)

    def block_fwd(self, x, blk):
        B, T, _ = x.shape
        h = blk['ln1'](x)
        Q, K, V = blk['qkv'](h).chunk(3, -1)
        Q = Q.view(B,T,NH,DH).transpose(1,2)
        K = K.view(B,T,NH,DH).transpose(1,2)
        V = V.view(B,T,NH,DH).transpose(1,2)

        if self.at == 'softmax':
            o = F.scaled_dot_product_attention(Q, K, V, is_causal=True, dropout_p=0.1 if self.training else 0.)
        elif 'gla_harm_dim_q256_qvwl8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=8)
        elif 'gla_harm_dim_q256_qvwl4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=4)
        elif 'gla_harm_dim_qvwl8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=8)
        elif 'gla_harm_dim_qvwl4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=4)
        elif 'gla_harm_dim_q256_gq8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=8)
        elif 'gla_harm_dim_q256_gq4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=4)
        elif 'gla_harm_dim_gq8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=8)
        elif 'gla_harm_dim_gq4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=4)
        elif 'gla_harm_dim' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)   # (B,NH,T,DH)
            o = gla_fwd_harmonic_dim(Q, K, V, self.phi, gate_logit)
        elif 'gla_harm' in self.at:
            gate_logit = blk['gp'](h).permute(0,2,1).unsqueeze(-1)
            o = gla_fwd_harmonic(Q, K, V, self.phi, gate_logit)
        elif 'gla' in self.at:
            gate_logit = blk['gp'](h).permute(0,2,1).unsqueeze(-1)
            o          = gla_fwd(Q, K, V, self.phi, gate_logit)
        elif 'ec_harm' in self.at:
            o = la_fwd_harmonic(Q, K, V, self.phi)
        else:
            o = la_fwd(Q, K, V, self.phi)

        return blk['proj'](o.transpose(1,2).reshape(B,T,D))

    def forward(self, x):
        B, T = x.shape
        h = self.drop(self.emb(x) + self.pos(torch.arange(T, device=x.device)))
        for blk in self.blocks:
            h = h + self.block_fwd(h, blk)
            h = h + blk['ff'](blk['ln2'](h))
        return self.head(self.lnf(h))

    def count_params(self):
        return sum(p.numel() for p in self.parameters())

    def block_fwd_attn(self, x, blk):
        """block_fwd와 동일하지만 attention weight도 반환 (analysis 전용)"""
        B, T, _ = x.shape
        h = blk['ln1'](x)
        Q, K, V = blk['qkv'](h).chunk(3, -1)
        Q = Q.view(B,T,NH,DH).transpose(1,2)
        K = K.view(B,T,NH,DH).transpose(1,2)
        V = V.view(B,T,NH,DH).transpose(1,2)

        if self.at == 'softmax':
            causal = torch.tril(torch.ones(T, T, dtype=torch.bool, device=x.device))
            attn_w = torch.matmul(Q, K.transpose(-2,-1)) * (DH ** -0.5)
            attn_w = attn_w.masked_fill(~causal, float('-inf'))
            attn_w = F.softmax(attn_w, dim=-1).nan_to_num(0.)
            o = torch.matmul(attn_w, V)
        elif 'gla_harm_dim_q256_qvwl8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=8)
        elif 'gla_harm_dim_q256_qvwl4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=4)
        elif 'gla_harm_dim_qvwl8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=8)
        elif 'gla_harm_dim_qvwl4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, vwl_bits=4)
        elif 'gla_harm_dim_q256_gq8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=8)
        elif 'gla_harm_dim_q256_gq4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=4)
        elif 'gla_harm_dim_gq8' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=8)
        elif 'gla_harm_dim_gq4' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit, gate_bits=4)
        elif 'gla_harm_dim' in self.at:
            gate_logit = blk['gp'](h).view(B,T,NH,DH).permute(0,2,1,3)   # (B,NH,T,DH)
            o, attn_w  = gla_fwd_attn_harmonic_dim(Q, K, V, self.phi, gate_logit)
        elif 'gla_harm' in self.at:
            gate_logit = blk['gp'](h).permute(0,2,1).unsqueeze(-1)
            o, attn_w  = gla_fwd_attn_harmonic(Q, K, V, self.phi, gate_logit)
        elif 'gla' in self.at:
            gate_logit = blk['gp'](h).permute(0,2,1).unsqueeze(-1)
            o, attn_w  = gla_fwd_attn(Q, K, V, self.phi, gate_logit)
        elif 'ec_harm' in self.at:
            o, attn_w = la_fwd_attn_harmonic(Q, K, V, self.phi)
        else:
            o, attn_w = la_fwd_attn(Q, K, V, self.phi)

        return blk['proj'](o.transpose(1,2).reshape(B,T,D)), attn_w

    def forward_attn(self, x):
        """attention weight list를 함께 반환 (analysis 전용, dropout 없음)"""
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T, device=x.device))
        attn_list = []
        for blk in self.blocks:
            h_out, attn_w = self.block_fwd_attn(h, blk)
            h = h + h_out
            h = h + blk['ff'](blk['ln2'](h))
            attn_list.append(attn_w)  # (B, H, T, T)
        return self.head(self.lnf(h)), attn_list

# ── 학습 / 평가 ──
def train_eval(at, ratio=7, n_layers=4):
    torch.manual_seed(SEED)
    m = GPT(at, ratio, n_layers).to(device)
    print(f"\n{'='*55}")
    ratio_str = f"ratio={ratio}" if at not in ('softmax', 'elu') else "ratio N/A"
    print(f"Method: {at} ({ratio_str}), params={m.count_params():,}")

    opt    = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.1)
    sched  = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP * len(trdl))
    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    history = {'train': [], 'val': []}
    t0 = time.time()

    for ep in range(EP):
        # train
        m.train(); tl = tn = 0
        pbar = tqdm(trdl, desc=f"[{at}] Ep{ep+1:02d}/train", leave=False)
        for xb, yb in pbar:
            xb, yb = xb.to(device), yb.to(device)
            if scaler:
                with torch.amp.autocast('cuda'):
                    loss = F.cross_entropy(m(xb).view(-1, vocab), yb.view(-1))
                opt.zero_grad(); scaler.scale(loss).backward()
                scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                scaler.step(opt); scaler.update()
            else:
                loss = F.cross_entropy(m(xb).view(-1, vocab), yb.view(-1))
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
            sched.step()
            tl += loss.item(); tn += 1
            pbar.set_postfix_str(f"loss={tl/tn:.3f}")
        pbar.close()

        # val
        m.eval(); vl = vn = 0
        with torch.no_grad():
            for xb, yb in tqdm(vdl, desc=f"[{at}] Ep{ep+1:02d}/val  ", leave=False):
                xb, yb = xb.to(device), yb.to(device)
                vl += F.cross_entropy(m(xb).view(-1, vocab), yb.view(-1)).item()
                vn += 1

        tr_ppl = math.exp(tl/tn); vl_ppl = math.exp(vl/vn)
        history['train'].append(tr_ppl); history['val'].append(vl_ppl)
        elapsed = time.time() - t0
        print(f"  Ep {ep+1:2d}: train={tr_ppl:.2f}, val={vl_ppl:.2f}  [{elapsed:.0f}s]")

    # test PPL
    m.eval(); tet = ten = 0
    with torch.no_grad():
        for xb, yb in tedl:
            xb, yb = xb.to(device), yb.to(device)
            tet += F.cross_entropy(m(xb).view(-1, vocab), yb.view(-1)).item()
            ten += 1
    test_ppl = math.exp(tet/ten)
    print(f"  Test PPL: {test_ppl:.3f}")

    # 모델 저장 (analysis용)
    mfname = f'wt103_model_{at}_r{ratio}_s{SEED}.pt'
    torch.save(m.state_dict(), mfname)
    print(f"  Saved: {mfname}")

    return history, test_ppl

# ── 실행 ──
print(f"Seed: {SEED}")

configs = [
    ('softmax',               None, 'Softmax'),
    # ('gla',                   200,  'GLA EC r=200'),
    # ('elu',                   None, 'ELU+1'),
    # ('ec',                    200,  'EC r=200'),
    # ── HW harmonic 구현 ──
    ('ec_harm',               200,  'EC r=200 (harm)'),
    # ('gla_harm',              200,  'GLA r=200 (harm)'),
    ('gla_harm_dim',          200,  'GLA r=200 (harm-dim)'),
    # ── Quantized 256 states (HW precision 제약) ──
    # ('ec_harm_q256',          200,  'EC r=200 (harm-q256)'),
    # ('gla_harm_q256',         200,  'GLA r=200 (harm-q256)'),
    ('gla_harm_dim_q256',     200,  'GLA r=200 (harm-dim-q256)'),
    # ── V_WL 통합 quantize (gate × phi_EC(Q) 전체 DAC 양자화) ──
    ('gla_harm_dim_q256_qvwl8',   200,  'GLA r=200 (harm-dim-q256-qvwl8)'),
    # ('gla_harm_dim_qvwl8',        200,  'GLA r=200 (harm-dim-qvwl8)'),
    # ('gla_harm_dim_qvwl4',        200,  'GLA r=200 (harm-dim-qvwl4)'),
    # ('gla_harm_dim_q256_qvwl4',   200,  'GLA r=200 (harm-dim-q256-qvwl4)'),
    # ── Gate quantized (HW DAC precision) ──
    # ('gla_harm_dim_q256_gq8', 200,  'GLA r=200 (harm-dim-q256-gq8)'),
    # ('gla_harm_dim_gq8',      200,  'GLA r=200 (harm-dim-gq8)'),
    # ('gla_harm_dim_gq4',      200,  'GLA r=200 (harm-dim-gq4)'),
    # ('gla_harm_dim_q256_gq4', 200,  'GLA r=200 (harm-dim-q256-gq4)'),
]

# analyze: 학습 없이 분석만
if METHOD == 'all':
    run_configs = configs
elif METHOD == 'analyze':
    run_configs = []
else:
    run_configs = [(at, r, name) for at, r, name in configs
                   if METHOD in name or METHOD == at]

def result_fname(name):
    return f'wt103_result_{name.replace(" ","_")}_s{SEED}.json'

# 이미 완료된 결과 로드 (JSON 파일이 있으면 재사용)
results = {}
for _, _, name in configs:
    fname = result_fname(name)
    if os.path.exists(fname):
        with open(fname) as f:
            results[name] = json.load(f)

for at, ratio, name in run_configs:
    if name in results:
        print(f"  Skip {name} (JSON exists)")
        continue
    hist, test_ppl = train_eval(at, ratio)
    results[name] = {'val': hist['val'], 'test': test_ppl}
    with open(result_fname(name), 'w') as f:
        json.dump({'val': hist['val'], 'test': test_ppl}, f)

# ── 최종 요약 ──
print("\n" + "="*55)
print("WikiText-103 Results")
print("="*55)
sm = results.get('Softmax', {}).get('test', None)
for _, _, name in configs:
    if name not in results:
        continue
    tp = results[name]['test']
    dsm = f"{tp-sm:+.3f}" if sm else "--"
    print(f"  {name:<32}: test PPL={tp:.3f}  vs SM={dsm}")

# ── Learning Curve 저장 ──
def save_learning_curves(results, configs, SEED):
    import csv, matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # ── 1. 통합 CSV ──
    csv_all = f'wt103_all_val_ppl_s{SEED}.csv'
    all_names = [name for _, _, name in configs if name in results]
    ep_count = max(len(results[n]['val']) for n in all_names)

    with open(csv_all, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['epoch'] + all_names)
        for ep_i in range(1, ep_count + 1):
            row = [ep_i]
            for name in all_names:
                vals = results[name]['val']
                row.append(f'{vals[ep_i-1]:.4f}' if ep_i <= len(vals) else '')
            writer.writerow(row)
    print(f'Saved: {csv_all}')

    # ── 2. Val PPL learning curve plot ──
    fig, ax = plt.subplots(figsize=(14, 7))
    colors = plt.cm.tab20.colors
    for i, (_, _, name) in enumerate(configs):
        if name not in results: continue
        vals = results[name]['val']
        epochs = list(range(1, len(vals)+1))
        ax.plot(epochs, vals, label=name,
                color=colors[i % len(colors)], marker='o', ms=3, linewidth=1.5)
    ax.set_xlabel('Epoch'); ax.set_ylabel('Val PPL')
    ax.set_title(f'WikiText-103 Validation PPL (seed={SEED})')
    ax.legend(fontsize=7, ncol=2, loc='upper right')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'wt103_learning_curves_s{SEED}.png', dpi=130, bbox_inches='tight')
    plt.savefig(f'wt103_learning_curves_s{SEED}.pdf', bbox_inches='tight')
    print(f'Saved: wt103_learning_curves_s{SEED}.png / .pdf')
    plt.close()

# ── Convergence Plot ──
def plot_convergence():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    # 폴더에서 wt103_result_*_s*.json 전부 로드
    all_files = glob.glob('wt103_result_*_s*.json')
    if not all_files:
        print("  결과 파일 없음 — convergence plot 스킵"); return

    # {method_name: [[val_ppl per epoch per seed], ...]} 수집
    method_vals = {}
    for fpath in all_files:
        fname = os.path.basename(fpath)
        # wt103_result_{name}_s{seed}.json 파싱
        inner = fname[len('wt103_result_'):-len('.json')]   # e.g. "Softmax_s42"
        seed_tag = '_s' + inner.split('_s')[-1]
        name = inner[:-len(seed_tag)].replace('_', ' ')
        with open(fpath) as f:
            data = json.load(f)
        if 'val' not in data: continue
        method_vals.setdefault(name, []).append(data['val'])

    if not method_vals:
        print("  유효한 결과 없음 — convergence plot 스킵"); return

    # configs 순서대로 정렬
    config_names = [name for _, _, name in configs]
    plot_order   = [n for n in config_names if n in method_vals]
    # configs에 없는 추가 method도 포함
    for n in method_vals:
        if n not in plot_order:
            plot_order.append(n)

    colors  = plt.cm.tab10.colors
    markers = ['o', 's', '^', 'D', 'v', 'P']

    fig, ax = plt.subplots(figsize=(7, 5))

    for i, name in enumerate(plot_order):
        seed_curves = np.array(method_vals[name])  # (n_seeds, n_epochs)
        n_ep   = seed_curves.shape[1]
        epochs = np.arange(1, n_ep + 1)
        mean   = seed_curves.mean(axis=0)
        color  = colors[i % len(colors)]
        marker = markers[i % len(markers)]

        ax.plot(epochs, mean, label=name, color=color,
                marker=marker, markersize=5, linewidth=2)

        if seed_curves.shape[0] > 1:
            std = seed_curves.std(axis=0)
            ax.fill_between(epochs, mean - std, mean + std,
                            alpha=0.15, color=color)

    ax.set_xlabel('Epoch', fontsize=12)
    ax.set_ylabel('Val PPL', fontsize=12)
    ax.set_title('WikiText-103 Convergence (5 epochs)', fontsize=13)
    ax.set_xticks(range(1, seed_curves.shape[1] + 1))
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out_png = 'wikitext103_convergence.png'
    out_pdf = 'wikitext103_convergence.pdf'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"\n  Saved: {out_png} / {out_pdf}")

    # 수치 요약
    print("\n  Val PPL (mean over seeds):")
    for name in plot_order:
        mean = np.array(method_vals[name]).mean(axis=0)
        print(f"    {name:<16}: {' → '.join(f'{v:.2f}' for v in mean)}")

# ── Attention 분석 ──
def analyze_attention(n_batches=50):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    n_layers = 4
    print(f"\n{'='*55}")
    print(f"Attention Analysis  (val {n_batches} batches)")
    print('='*55)

    # 모델 로드
    models_dict = {}
    for at, ratio, name in configs:
        mfname = f'wt103_model_{at}_r{ratio}_s{SEED}.pt'
        if os.path.exists(mfname):
            mdl = GPT(at, ratio if ratio is not None else 7).to(device)
            mdl.load_state_dict(torch.load(mfname, map_location=device, weights_only=True))
            mdl.eval()
            models_dict[name] = mdl
            print(f"  Loaded {mfname}")

    if 'Softmax' not in models_dict or len(models_dict) < 2:
        print("  Softmax 모델 없음 — 분석 스킵"); return

    method_names = list(models_dict.keys())
    non_sm = [n for n in method_names if n != 'Softmax']

    # 누적 통계: {name: {layer: [scalar values]}}
    kl_acc   = {n: [[] for _ in range(n_layers)] for n in non_sm}
    ent_acc  = {n: [[] for _ in range(n_layers)] for n in method_names}
    topk_acc = {n: [[] for _ in range(n_layers)] for n in non_sm}

    val_iter = iter(vdl)
    for _ in tqdm(range(n_batches), desc='Analyzing', leave=False):
        try:
            xb, _ = next(val_iter)
        except StopIteration:
            break
        xb = xb.to(device)

        with torch.no_grad():
            _, sm_attn = models_dict['Softmax'].forward_attn(xb)
            # softmax entropy
            for l, sa in enumerate(sm_attn):
                sa_f = sa.float().cpu().clamp(1e-10)
                ent_acc['Softmax'][l].append(-(sa_f * sa_f.log()).sum(-1).mean().item())

            for name in non_sm:
                _, m_attn = models_dict[name].forward_attn(xb)
                for l, (sa, ma) in enumerate(zip(sm_attn, m_attn)):
                    sa_f = sa.float().cpu().clamp(1e-10)
                    ma_f = ma.float().cpu().clamp(1e-10)

                    # KL(softmax || method)
                    kl = (sa_f * (sa_f.log() - ma_f.log())).sum(-1)
                    kl_acc[name][l].append(kl.mean().item())

                    # entropy
                    ent_acc[name][l].append(-(ma_f * ma_f.log()).sum(-1).mean().item())

                    # top-5 overlap
                    k = min(5, sa_f.shape[-1])
                    top_sm = sa_f.topk(k, dim=-1).indices
                    top_m  = ma_f.topk(k, dim=-1).indices
                    match = (top_sm.unsqueeze(-1) == top_m.unsqueeze(-2)).any(-1).float()
                    topk_acc[name][l].append(match.mean().item())

    # 평균 계산
    kl_mean   = {n: [np.mean(kl_acc[n][l])   for l in range(n_layers)] for n in non_sm}
    ent_mean  = {n: [np.mean(ent_acc[n][l])  for l in range(n_layers)] for n in method_names}
    topk_mean = {n: [np.mean(topk_acc[n][l]) for l in range(n_layers)] for n in non_sm}

    # ── 출력 ──
    print("\nKL(Softmax||Method) — mean over layers:")
    for n in non_sm:
        print(f"  {n:<16}: {np.mean(kl_mean[n]):.4f}  {[f'{v:.3f}' for v in kl_mean[n]]}")
    print("\nEntropy — mean over layers:")
    for n in method_names:
        print(f"  {n:<16}: {np.mean(ent_mean[n]):.4f}  {[f'{v:.3f}' for v in ent_mean[n]]}")
    print("\nTop-5 Overlap with Softmax:")
    for n in non_sm:
        print(f"  {n:<16}: {np.mean(topk_mean[n]):.4f}")

    # ── Plot ──
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    layer_labels = [f'L{l+1}' for l in range(n_layers)]
    colors = plt.cm.tab10.colors

    # 1) KL divergence heatmap
    ax = axes[0]
    kl_mat = np.array([[kl_mean[n][l] for l in range(n_layers)] for n in non_sm])
    im = ax.imshow(kl_mat, aspect='auto', cmap='YlOrRd')
    ax.set_xticks(range(n_layers)); ax.set_xticklabels(layer_labels)
    ax.set_yticks(range(len(non_sm))); ax.set_yticklabels(non_sm, fontsize=9)
    ax.set_title('KL(Softmax || Method)', fontsize=11)
    fig.colorbar(im, ax=ax)
    for i in range(len(non_sm)):
        for j in range(n_layers):
            ax.text(j, i, f'{kl_mat[i,j]:.2f}', ha='center', va='center', fontsize=8,
                    color='white' if kl_mat[i,j] > kl_mat.max()*0.6 else 'black')

    # 2) Entropy bar chart (per layer, grouped by method)
    ax = axes[1]
    x = np.arange(n_layers)
    w = 0.8 / len(method_names)
    for i, n in enumerate(method_names):
        ax.bar(x + i*w - 0.4 + w/2, ent_mean[n], w, label=n,
               color=colors[i % len(colors)], alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(layer_labels)
    ax.set_ylabel('Entropy H(attn)'); ax.set_title('Attention Entropy per Layer', fontsize=11)
    ax.legend(fontsize=7, ncol=2)

    # 3) Top-5 overlap bar chart
    ax = axes[2]
    vals = [np.mean(topk_mean[n]) for n in non_sm]
    bars = ax.bar(range(len(non_sm)), vals,
                  color=[colors[method_names.index(n) % len(colors)] for n in non_sm], alpha=0.85)
    ax.set_xticks(range(len(non_sm)))
    ax.set_xticklabels(non_sm, rotation=30, ha='right', fontsize=8)
    ax.set_ylabel('Top-5 Overlap Rate'); ax.set_title('Top-5 Token Overlap w/ Softmax', fontsize=11)
    ax.set_ylim(0, 1.05)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width()/2, v + 0.01, f'{v:.2f}',
                ha='center', va='bottom', fontsize=9)

    plt.suptitle(f'WikiText-103 Attention Analysis  (seed={SEED})', fontsize=12, y=1.01)
    plt.tight_layout()
    out_png = f'wt103_attention_analysis_s{SEED}.png'
    out_pdf = f'wt103_attention_analysis_s{SEED}.pdf'
    plt.savefig(out_png, dpi=150, bbox_inches='tight')
    plt.savefig(out_pdf, bbox_inches='tight')
    print(f"\n  Saved: {out_png} / {out_pdf}")

save_learning_curves(results, configs, SEED)
# METHOD=='analyze': 학습 없이 분석만 / 그 외: 학습 후 자동 실행
plot_convergence()
analyze_attention()
