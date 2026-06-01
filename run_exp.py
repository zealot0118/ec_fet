import re, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import numpy as np, math, sys, os

torch.manual_seed(42); np.random.seed(42)
EPS=1e-10

with open('/home/claude/full_corpus_raw.txt') as f: raw=f.read()
text=re.sub(r'<[^>]+>|https?://\S+|[^\x20-\x7e\n]',' ',raw)
text=re.sub(r'[ \t]+',' ',text); text=re.sub(r'\n{3,}','\n\n',text)
lines=[l.strip() for l in text.split('\n') if len(l.strip())>15]; text='\n'.join(lines)[:150000]
chars=sorted(set(text)); c2i={c:i for i,c in enumerate(chars)}; vocab=len(chars)
train_t=text[:int(len(text)*0.85)]; val_t=text[int(len(text)*0.85):]

SEQ=32; D=48; NH=3; DH=16; BS=1024; LR=2e-3; EP=5

class DS(Dataset):
    def __init__(self,t): self.d=torch.tensor([c2i.get(c,0) for c in t],dtype=torch.long)
    def __len__(self): return max(0,len(self.d)-SEQ)
    def __getitem__(self,i): return self.d[i:i+SEQ],self.d[i+1:i+SEQ+1]

trdl=DataLoader(DS(train_t),BS,shuffle=True,drop_last=True)
vdl =DataLoader(DS(val_t),BS,drop_last=True)

def la(Q,K,V,fn):
    pQ=fn(Q); pK=fn(K); T=Q.shape[2]
    s=torch.matmul(pQ,pK.transpose(-2,-1))
    s=s.masked_fill(~torch.tril(torch.ones(T,T,dtype=torch.bool)),0.)
    return torch.matmul(s/s.sum(-1,keepdim=True).clamp(EPS),V)

class GPT(nn.Module):
    def __init__(self,phi=None):
        super().__init__(); self.phi_fn=phi
        self.emb=nn.Embedding(vocab,D); self.pos=nn.Embedding(SEQ,D)
        self.ln1=nn.LayerNorm(D); self.ln2=nn.LayerNorm(D); self.lnf=nn.LayerNorm(D)
        self.qkv=nn.Linear(D,3*D,bias=False); self.proj=nn.Linear(D,D,bias=False)
        self.ff=nn.Sequential(nn.Linear(D,2*D),nn.GELU(),nn.Linear(2*D,D))
        self.head=nn.Linear(D,vocab,bias=False)
    def forward(self,x):
        B,T=x.shape; h=self.emb(x)+self.pos(torch.arange(T))
        Q,K,V=self.qkv(self.ln1(h)).chunk(3,-1)
        Q=Q.view(B,T,NH,DH).transpose(1,2); K=K.view(B,T,NH,DH).transpose(1,2); V=V.view(B,T,NH,DH).transpose(1,2)
        if self.phi_fn is None: o=F.scaled_dot_product_attention(Q,K,V,is_causal=True)
        else: o=la(Q,K,V,self.phi_fn)
        h=h+self.proj(o.transpose(1,2).reshape(B,T,D))
        return self.head(self.lnf(h+self.ff(self.ln2(h))))

def run(phi):
    torch.manual_seed(42); m=GPT(phi)
    opt=torch.optim.AdamW(m.parameters(),lr=LR,weight_decay=0.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,EP*len(trdl)); ppls=[]
    for ep in range(EP):
        m.train()
        for xb,yb in trdl:
            l=F.cross_entropy(m(xb).view(-1,vocab),yb.view(-1))
            opt.zero_grad(); l.backward(); nn.utils.clip_grad_norm_(m.parameters(),1.); opt.step(); sched.step()
        m.eval(); vl=vn=0
        with torch.no_grad():
            for xb,yb in vdl:
                vl+=F.cross_entropy(m(xb).view(-1,vocab),yb.view(-1)).item(); vn+=1
        ppls.append(math.exp(vl/vn))
        sys.stdout.write(f'{ppls[-1]:.2f} '); sys.stdout.flush()
    print()
    return ppls

NAME = sys.argv[1] if len(sys.argv)>1 else 'softmax'

phi_map = {
    'softmax': None,
    'elu':     lambda x: F.elu(x)+1.+1e-6,
    'ec_r7':   lambda x: 1e-6+6e-6*((x+x.abs().amax(-1,keepdim=True).clamp(1e-10))/(2*x.abs().amax(-1,keepdim=True).clamp(1e-10))).clamp(0,1),
    'ec_r200': lambda x: 1e-6+199e-6*((x+x.abs().amax(-1,keepdim=True).clamp(1e-10))/(2*x.abs().amax(-1,keepdim=True).clamp(1e-10))).clamp(0,1),
    'ec_znorm_r200': lambda x: 1e-6+199e-6*torch.sigmoid((x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True).clamp(1e-6)),
    'ec_znorm_r7':   lambda x: 1e-6+6e-6*torch.sigmoid((x-x.mean(-1,keepdim=True))/x.std(-1,keepdim=True).clamp(1e-6)),
}

print(f'Running: {NAME}', flush=True)
pp = run(phi_map[NAME])
final = pp[-1]
print(f'RESULT {NAME} {final:.4f}')

# 결과 저장
with open(f'/tmp/ppl_{NAME}.txt','w') as f:
    f.write(f'{final:.4f}\n')
    f.write(' '.join([f'{p:.2f}' for p in pp])+'\n')
