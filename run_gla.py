import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np, math, sys

torch.manual_seed(42); np.random.seed(42)
EPS=1e-10

with open('/home/claude/full_corpus_raw.txt') as f: raw=f.read()
text=re.sub(r'<[^>]+>|https?://\S+|[^\x20-\x7e\n]',' ',raw)
text=re.sub(r'[ \t]+',' ',text); text=re.sub(r'\n{3,}','\n\n',text)
lines=[l.strip() for l in text.split('\n') if len(l.strip())>15]
text='\n'.join(lines)[:150000]
chars=sorted(set(text)); c2i={c:i for i,c in enumerate(chars)}; vocab=len(chars)
train_t=text[:int(len(text)*0.85)]; val_t=text[int(len(text)*0.85):]

SEQ=32; D=48; NH=3; DH=16; BS=1024; LR=2e-3; EP=5

class DS(Dataset):
    def __init__(self,t): self.d=torch.tensor([c2i.get(c,0) for c in t],dtype=torch.long)
    def __len__(self): return max(0,len(self.d)-SEQ)
    def __getitem__(self,i): return self.d[i:i+SEQ],self.d[i+1:i+SEQ+1]

trdl=DataLoader(DS(train_t),BS,shuffle=True,drop_last=True)
vdl =DataLoader(DS(val_t),BS,drop_last=True)

NAME = sys.argv[1] if len(sys.argv)>1 else 'gla_r7'
RATIO = int(sys.argv[2]) if len(sys.argv)>2 else 7

def phi_ec(x, ratio):
    gmin=1e-6; gmax=gmin*ratio
    sc=x.abs().amax(dim=-1,keepdim=True).clamp(1e-10)
    return gmin+(gmax-gmin)*((x+sc)/(2*sc)).clamp(0,1)

def gla_causal(Q, K, V, phi_fn, gate):
    """
    Gated Linear Attention (parallel causal via decay mask)

    gate: (B, H, T, 1) per-token forget gate ∈ (0,1)

    score[t,s] = phi(Q_t) · phi(K_s) × decay(s→t)
    decay(s,t) = Π_{i=s}^{t-1} gate[i]  (누적 forget)

    병렬 근사: log-cumsum trick
    log_decay[t,s] = Σ_{i=s}^{t-1} log(gate[i])
    """
    B,H,T,d = Q.shape
    pQ = phi_fn(Q); pK = phi_fn(K)   # (B,H,T,d)

    # base scores: (B,H,T,T)
    scores = torch.matmul(pQ, pK.transpose(-2,-1))

    # causal mask
    causal = torch.tril(torch.ones(T,T,device=Q.device,dtype=torch.bool))
    scores = scores.masked_fill(~causal, 0.0)

    # gate decay: log-cumsum trick
    # gate: (B,H,T,1) → log_gate: (B,H,T)
    log_gate = torch.log(gate.squeeze(-1).clamp(EPS))  # (B,H,T)

    # cumsum of log_gate: log_cum[t] = Σ_{i=0}^{t} log_gate[i]
    log_cum = torch.cumsum(log_gate, dim=-1)  # (B,H,T)

    # decay[t,s] = exp(log_cum[t-1] - log_cum[s-1])
    # = exp(Σ_{i=s}^{t-1} log_gate[i])
    # numerically: shift by 1
    log_cum_shifted = torch.cat([
        torch.zeros(B,H,1,device=Q.device),
        log_cum[:,:,:-1]
    ], dim=-1)  # (B,H,T): cum[t] = Σ_{i=0}^{t-1}

    # decay matrix: (B,H,T,T)
    # decay[t,s] = exp(log_cum[t] - log_cum[s])  for t>=s
    decay = torch.exp(
        log_cum_shifted.unsqueeze(-1) -   # (B,H,T,1): query position
        log_cum_shifted.unsqueeze(-2)      # (B,H,1,T): key position
    )  # (B,H,T,T)

    # apply decay to scores
    scores = scores * decay
    scores = scores.masked_fill(~causal, 0.0)

    # normalize
    z = scores.sum(-1, keepdim=True).clamp(EPS)
    return torch.matmul(scores/z, V)


class GLA_GPT(nn.Module):
    def __init__(self, ratio=7):
        super().__init__()
        self.ratio = ratio
        self.emb  = nn.Embedding(vocab, D)
        self.pos  = nn.Embedding(SEQ, D)
        self.ln1  = nn.LayerNorm(D); self.ln2 = nn.LayerNorm(D); self.lnf = nn.LayerNorm(D)
        self.qkv  = nn.Linear(D, 3*D, bias=False)
        self.proj = nn.Linear(D, D,   bias=False)
        # Gate projection: (B,T,D) → (B,T,NH) → per-head gate
        self.gate_proj = nn.Linear(D, NH, bias=True)
        self.ff   = nn.Sequential(nn.Linear(D,2*D), nn.GELU(), nn.Linear(2*D,D))
        self.head = nn.Linear(D, vocab, bias=False)

    def phi(self, x):
        return phi_ec(x, self.ratio)

    def forward(self, x):
        B, T = x.shape
        h = self.emb(x) + self.pos(torch.arange(T))
        hln = self.ln1(h)

        Q, K, V = self.qkv(hln).chunk(3, -1)
        Q = Q.view(B,T,NH,DH).transpose(1,2)   # (B,H,T,d)
        K = K.view(B,T,NH,DH).transpose(1,2)
        V = V.view(B,T,NH,DH).transpose(1,2)

        # Gate: sigmoid → (0,1)
        # gate_proj: (B,T,NH) → (B,NH,T,1)
        gate = torch.sigmoid(self.gate_proj(hln))   # (B,T,NH)
        gate = gate.permute(0,2,1).unsqueeze(-1)    # (B,NH,T,1)

        o = gla_causal(Q, K, V, self.phi, gate)
        h = h + self.proj(o.transpose(1,2).reshape(B,T,D))
        return self.head(self.lnf(h + self.ff(self.ln2(h))))


def run(model):
    torch.manual_seed(42); m = model
    opt = torch.optim.AdamW(m.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, EP*len(trdl))
    ppls = []
    for ep in range(EP):
        m.train()
        for xb, yb in trdl:
            l = F.cross_entropy(m(xb).view(-1,vocab), yb.view(-1))
            opt.zero_grad(); l.backward()
            nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step(); sched.step()
        m.eval(); vl=vn=0
        with torch.no_grad():
            for xb,yb in vdl:
                vl+=F.cross_entropy(m(xb).view(-1,vocab),yb.view(-1)).item(); vn+=1
        ppls.append(math.exp(vl/vn))
        sys.stdout.write(f'{ppls[-1]:.2f} '); sys.stdout.flush()
    print()
    return ppls

print(f'Running: GLA EC r={RATIO}', flush=True)
pp = run(GLA_GPT(RATIO))
final = pp[-1]
print(f'RESULT GLA_r{RATIO} {final:.4f}')
with open(f'/tmp/ppl_gla_r{RATIO}.txt','w') as f:
    f.write(f'{final:.4f}\n')
    f.write(' '.join([f'{p:.2f}' for p in pp])+'\n')
