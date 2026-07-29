#!/usr/bin/env python3
from __future__ import annotations
import argparse,csv,json,math,os,pickle,random,re,time,unicodedata,zipfile
from collections import Counter,defaultdict
from dataclasses import dataclass,asdict
from difflib import SequenceMatcher
from pathlib import Path
from typing import Dict,List,Optional,Sequence,Tuple
import numpy as np,pandas as pd,torch
import torch.nn as nn,torch.nn.functional as F
from torch.utils.data import Dataset,DataLoader
from tqdm.auto import tqdm
try:
    from rapidfuzz.distance import Levenshtein as RFLev
except Exception:
    RFLev=None
PAD_CH='<pad>'; UNK_CH='<unk>'; PAD=0; UNK=1; IGNORE=-100

def nfc(x):
    if pd.isna(x): return ''
    return unicodedata.normalize('NFC',str(x))
def light_cleanup(s):
    s=nfc(s).replace('\u200b','')
    s=re.sub(r'[ \t]+',' ',s); s=re.sub(r'\s+([,.;:!?%\]\)\}])',r'\1',s)
    s=re.sub(r'([\[\(\{])\s+',r'\1',s); s=re.sub(r'\s*/\s*','/',s)
    return s.strip()
def base_ch(ch):
    if ch=='đ': return 'd'
    if ch=='Đ': return 'd'
    d=unicodedata.normalize('NFD',ch)
    return unicodedata.normalize('NFC',''.join(c for c in d if unicodedata.category(c)!='Mn')).lower() or ch.lower()
def type_id(ch):
    if not ch: return 0
    if ch.isspace(): return 1
    if ch.isdigit(): return 2
    if ch.isalpha() and ch.islower(): return 3
    if ch.isalpha() and ch.isupper(): return 4
    cat=unicodedata.category(ch)
    if cat.startswith('P'): return 5
    if cat.startswith('S'): return 6
    return 7
def seed_all(seed):
    random.seed(seed); np.random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark=True
def load_train_csv(path,limit=None):
    df=pd.read_csv(path); need={'id','input','corrected_text'}; miss=need-set(df.columns)
    if miss: raise ValueError(f'train.csv missing columns: {sorted(miss)}')
    if limit: df=df.head(limit).copy()
    df=df.dropna(subset=['input','corrected_text']).copy(); df['input']=df['input'].map(nfc); df['corrected_text']=df['corrected_text'].map(nfc)
    df=df[(df['input'].str.len()>0)&(df['corrected_text'].str.len()>0)]
    return df.drop_duplicates(subset=['input','corrected_text']).reset_index(drop=True)
def load_test_csv(path):
    df=pd.read_csv(path); need={'id','input'}; miss=need-set(df.columns)
    if miss: raise ValueError(f'test.csv missing columns: {sorted(miss)}')
    df['input']=df['input'].map(nfc); return df
def lev(a,b):
    a=nfc(a); b=nfc(b)
    if RFLev is not None: return RFLev.distance(a,b)
    if a==b: return 0
    if len(a)<len(b): a,b=b,a
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]
        for j,cb in enumerate(b,1): cur.append(min(prev[j]+1,cur[-1]+1,prev[j-1]+(ca!=cb)))
        prev=cur
    return prev[-1]
def cer(preds,refs):
    e=t=0
    for p,r in zip(preds,refs): e+=lev(p,r); t+=max(1,len(nfc(r)))
    return e/max(1,t)
class Vocab:
    def __init__(self,stoi):
        self.stoi=stoi; self.itos=[None]*len(stoi)
        for k,v in stoi.items(): self.itos[v]=k
    @classmethod
    def build(cls,items):
        stoi={PAD_CH:PAD,UNK_CH:UNK}
        for x in sorted(set(items)):
            if x not in stoi: stoi[x]=len(stoi)
        return cls(stoi)
    def save(self,p): Path(p).write_text(json.dumps({'stoi':self.stoi},ensure_ascii=False,indent=2),encoding='utf-8')
    @classmethod
    def load(cls,p): return cls(json.loads(Path(p).read_text(encoding='utf-8'))['stoi'])
    def __len__(self): return len(self.itos)
class CharVocab(Vocab):
    @classmethod
    def from_texts(cls,texts):
        s=set(); [s.update(nfc(t)) for t in texts]; return cls.build(s)
    def enc(self,text): return [self.stoi.get(ch,UNK) for ch in nfc(text)]
class BaseVocab(Vocab):
    @classmethod
    def from_texts(cls,texts):
        s=set()
        for t in texts:
            for ch in nfc(t): s.add(base_ch(ch))
        return cls.build(s)
    def enc(self,text): return [self.stoi.get(base_ch(ch),UNK) for ch in nfc(text)]
class EditVocab:
    def __init__(self,seg_to_id):
        self.seg_to_id=seg_to_id; self.id_to_seg=[None]*len(seg_to_id)
        for k,v in seg_to_id.items(): self.id_to_seg[v]=k
        self.pad_id=seg_to_id[PAD_CH]; self.empty_id=seg_to_id.get('',1)
    @classmethod
    def build(cls,counter,chars,min_count=2,max_len=5,max_labels=6144):
        d={PAD_CH:0,'':1}
        for ch in sorted(set(chars)):
            if ch and ch not in d: d[ch]=len(d)
        items=[(s,c) for s,c in counter.items() if s not in d and 0<len(s)<=max_len and c>=min_count]
        items.sort(key=lambda x:(-x[1],len(x[0]),x[0]))
        for s,c in items:
            if len(d)>=max_labels: break
            d[s]=len(d)
        return cls(d)
    def enc(self,ch,seg):
        if seg in self.seg_to_id: return self.seg_to_id[seg]
        if ch in self.seg_to_id: return self.seg_to_id[ch]
        return self.empty_id
    def save(self,p): Path(p).write_text(json.dumps({'seg_to_id':self.seg_to_id},ensure_ascii=False,indent=2),encoding='utf-8')
    @classmethod
    def load(cls,p): return cls(json.loads(Path(p).read_text(encoding='utf-8'))['seg_to_id'])
    def __len__(self): return len(self.id_to_seg)

def align_to_segments(src,tgt):
    src=nfc(src); tgt=nfc(tgt)
    if not src: return []
    segs=['']*len(src); sm=SequenceMatcher(a=src,b=tgt,autojunk=False)
    for tag,i1,i2,j1,j2 in sm.get_opcodes():
        if tag=='equal':
            for i in range(i1,i2): segs[i]=src[i]
        elif tag=='delete':
            for i in range(i1,i2): segs[i]=''
        elif tag=='insert':
            ins=tgt[j1:j2]
            if i1>0: segs[i1-1]+=ins
            else: segs[0]=ins+segs[0]
        elif tag=='replace':
            n=i2-i1; b=tgt[j1:j2]
            if n==len(b):
                for k in range(n): segs[i1+k]=b[k]
            else:
                for k in range(n):
                    s0=round(k*len(b)/max(1,n)); s1=round((k+1)*len(b)/max(1,n)); segs[i1+k]=b[s0:s1]
    return segs
def split_text_soft(text,max_chars):
    text=nfc(text)
    if len(text)<=max_chars: return [text]
    out=[]; i=0
    while i<len(text):
        end=min(len(text),i+max_chars)
        if end==len(text): out.append(text[i:end]); break
        win=text[i:end]; cut=-1
        for pat in [r'[\.\!\?;:]\s+',r'[,\)]\s+',r'\s+']:
            ms=list(re.finditer(pat,win))
            if ms: cut=ms[-1].end(); break
        if cut<max(64,max_chars//3): cut=max_chars
        out.append(text[i:i+cut]); i+=cut
    return [x for x in out if x]
def pair_chunks(src,tgt,max_chars):
    src=nfc(src); tgt=nfc(tgt)
    if max(len(src),len(tgt))<=max_chars: return [(src,tgt)]
    chunks=split_text_soft(src,max_chars); out=[]; pos=0
    for sc in chunks:
        s0=pos; s1=pos+len(sc); pos=s1
        t0=round(s0/max(1,len(src))*len(tgt)); t1=round(s1/max(1,len(src))*len(tgt))
        out.append((sc,tgt[max(0,min(len(tgt),t0)):max(0,min(len(tgt),t1))]))
    return out
_TOKEN_RE=re.compile(r'\w+|[^\w\s]+|\s+',re.UNICODE); _WORD_RE=re.compile(r'\w+',re.UNICODE)
def toks(s): return _TOKEN_RE.findall(nfc(s))
def isword(t): return bool(_WORD_RE.fullmatch(t))
def build_exact_map(df,min_conf=0.999):
    counts=defaultdict(Counter)
    for src,tgt in df[['input','corrected_text']].itertuples(index=False): counts[nfc(src)][nfc(tgt)]+=1
    exact={}
    for s,ctr in counts.items():
        t,c=ctr.most_common(1)[0]
        if c/sum(ctr.values())>=min_conf: exact[s]=t
    return exact
def build_word_lexicon(df,min_count=3,min_conf=0.80,max_token_len=42,max_tokens=450):
    counts=defaultdict(Counter); total=Counter()
    for src,tgt in tqdm(df[['input','corrected_text']].itertuples(index=False),total=len(df),desc='Build word lexicon'):
        a=[t for t in toks(src) if not t.isspace()]; b=[t for t in toks(tgt) if not t.isspace()]
        if len(a)>max_tokens or len(b)>max_tokens: continue
        sm=SequenceMatcher(a=a,b=b,autojunk=False)
        for tag,i1,i2,j1,j2 in sm.get_opcodes():
            if tag=='equal':
                for tok in a[i1:i2]:
                    if isword(tok) and len(tok)<=max_token_len: total[tok]+=1; counts[tok][tok]+=1
            else:
                aa=a[i1:i2]; bb=b[j1:j2]
                if len(aa)==len(bb):
                    for x,y in zip(aa,bb):
                        if isword(x) and isword(y) and len(x)<=max_token_len and len(y)<=max_token_len:
                            total[x]+=1; counts[x][y]+=1
    lex={}
    for noisy,ctr in counts.items():
        clean,c=ctr.most_common(1)[0]; conf=c/max(1,total[noisy])
        if clean!=noisy and c>=min_count and conf>=min_conf: lex[noisy]=clean
    return lex
def apply_word_lexicon(text,lex):
    if not lex: return text
    return ''.join(lex.get(t,t) if isword(t) else t for t in toks(text))
@dataclass
class TrainConfig:
    train_csv:str; test_csv:str='NLP/test.csv'; output_dir:str='runs/ocr_transducer_v2'; seed:int=42; val_ratio:float=0.035; limit_rows:Optional[int]=None
    max_chars:int=896; hard_align_chars:int=1400; min_seg_count:int=2; max_segment_len:int=5; max_labels:int=6144; identity_augment_ratio:float=0.18
    d_model:int=448; nhead:int=8; layers:int=7; conv_layers:int=2; ffn_dim:int=1792; dropout:float=0.12
    batch_size:int=16; accum_steps:int=2; epochs:int=40; lr:float=4e-4; min_lr_ratio:float=0.03; warmup_ratio:float=0.06; weight_decay:float=0.035; grad_clip:float=1.0; amp:bool=True; num_workers:int=2; ema_decay:float=0.9994
    rewrite_weight:float=4.0; delete_insert_weight:float=4.8; copy_weight:float=1.0; gate_loss_weight:float=0.25; label_smoothing:float=0.015
    eval_samples:int=1200; eval_thresholds:str='0.55,0.65,0.75,0.85,0.90'; eval_copy_margins:str='0.04,0.08,0.12'; eval_gate_thresholds:str='0.20,0.35,0.50'; eval_use_lexicon:str='0,1'
    max_edit_frac:float=0.45; max_len_delta_frac:float=0.35; keep_epoch_ckpts:int=6; resume:Optional[str]=None
    use_word_lexicon:bool=True; lex_min_count:int=3; lex_min_conf:float=0.80
@dataclass
class DecodeParams:
    threshold:float=0.75; copy_margin:float=0.08; gate_threshold:float=0.35; use_lexicon:bool=True; max_edit_frac:float=0.45; max_len_delta_frac:float=0.35
class TransducerDataset(Dataset):
    def __init__(self,examples,char_vocab,base_vocab,max_chars,training):
        self.examples=examples; self.char_vocab=char_vocab; self.base_vocab=base_vocab; self.max_chars=max_chars; self.training=training
    def __len__(self): return len(self.examples)
    def __getitem__(self,i):
        src,y,g,w=self.examples[i]
        if len(src)>self.max_chars:
            st=random.randint(0,len(src)-self.max_chars) if self.training else 0
            src=src[st:st+self.max_chars]; y=y[st:st+self.max_chars]; g=g[st:st+self.max_chars]; w=w[st:st+self.max_chars]
        return {'x':self.char_vocab.enc(src),'base':self.base_vocab.enc(src),'typ':[type_id(ch) for ch in src],'y':y,'gate':g,'w':w}
def collate(batch):
    B=len(batch); L=max(len(b['x']) for b in batch)
    x=torch.full((B,L),PAD,dtype=torch.long); base=torch.full((B,L),PAD,dtype=torch.long); typ=torch.zeros((B,L),dtype=torch.long)
    y=torch.full((B,L),IGNORE,dtype=torch.long); gate=torch.full((B,L),IGNORE,dtype=torch.long); w=torch.zeros((B,L),dtype=torch.float32)
    for i,b in enumerate(batch):
        n=len(b['x']); x[i,:n]=torch.tensor(b['x']); base[i,:n]=torch.tensor(b['base']); typ[i,:n]=torch.tensor(b['typ']); y[i,:n]=torch.tensor(b['y']); gate[i,:n]=torch.tensor(b['gate']); w[i,:n]=torch.tensor(b['w'],dtype=torch.float32)
    return {'x':x,'base':base,'typ':typ,'y':y,'gate':gate,'w':w}
def train_val_split(df,val_ratio,seed):
    rng=np.random.default_rng(seed); idx=np.arange(len(df)); rng.shuffle(idx); n=max(1,int(round(len(df)*val_ratio))) if val_ratio>0 else 0
    return df.iloc[idx[n:]].reset_index(drop=True),df.iloc[idx[:n]].reset_index(drop=True)
def build_segment_counter(df,cfg):
    ctr=Counter()
    for src,tgt in tqdm(df[['input','corrected_text']].itertuples(index=False),total=len(df),desc='Scan edit segments'):
        for sc,tc in pair_chunks(src,tgt,cfg.hard_align_chars):
            for seg in align_to_segments(sc,tc): ctr[seg]+=1
    return ctr
def build_examples(df,edit_vocab,cfg,training=True):
    ex=[]; rng=random.Random(cfg.seed+(1 if training else 999))
    for src,tgt in tqdm(df[['input','corrected_text']].itertuples(index=False),total=len(df),desc='Build examples'):
        for sc,tc in pair_chunks(src,tgt,cfg.hard_align_chars):
            if not sc: continue
            segs=align_to_segments(sc,tc); y=[]; g=[]; w=[]
            for ch,seg in zip(sc,segs):
                y.append(edit_vocab.enc(ch,seg)); edited=int(seg!=ch); g.append(edited)
                w.append(int(round((cfg.copy_weight if seg==ch else cfg.delete_insert_weight if seg=='' or len(seg)!=1 else cfg.rewrite_weight)*100)))
            for pos in range(0,len(sc),cfg.max_chars): ex.append((sc[pos:pos+cfg.max_chars],y[pos:pos+cfg.max_chars],g[pos:pos+cfg.max_chars],w[pos:pos+cfg.max_chars]))
            if training and cfg.identity_augment_ratio>0 and rng.random()<cfg.identity_augment_ratio:
                clean=tc if tc else sc
                for ic in split_text_soft(clean,cfg.max_chars):
                    iy=[edit_vocab.enc(ch,ch) for ch in ic]; ig=[0]*len(ic); iw=[int(round(cfg.copy_weight*100))]*len(ic)
                    if ic: ex.append((ic,iy,ig,iw))
    return ex

def prepare_cache(df_train,df_val,cfg,out_dir,rebuild=False):
    char_path=out_dir/'char_vocab.json'; base_path=out_dir/'base_vocab.json'; edit_path=out_dir/'edit_vocab.json'; trp=out_dir/'train_examples.pkl'; vap=out_dir/'val_examples.pkl'
    if all(p.exists() for p in [char_path,base_path,edit_path,trp,vap]) and not rebuild:
        print('[*] Loading cached vocab/examples')
        return CharVocab.load(char_path),BaseVocab.load(base_path),EditVocab.load(edit_path),pickle.loads(trp.read_bytes()),pickle.loads(vap.read_bytes())
    texts=list(df_train['input'])+list(df_train['corrected_text']); char_vocab=CharVocab.from_texts(texts); base_vocab=BaseVocab.from_texts(texts)
    char_vocab.save(char_path); base_vocab.save(base_path); print(f'[*] char={len(char_vocab)} base={len(base_vocab)}')
    ctr=build_segment_counter(df_train,cfg); chars=set(); [chars.update(t) for t in texts]
    edit_vocab=EditVocab.build(ctr,chars,cfg.min_seg_count,cfg.max_segment_len,cfg.max_labels); edit_vocab.save(edit_path)
    print(f'[*] edit labels={len(edit_vocab)} common={ctr.most_common(20)}')
    train_examples=build_examples(df_train,edit_vocab,cfg,True); val_examples=build_examples(df_val,edit_vocab,cfg,False)
    trp.write_bytes(pickle.dumps(train_examples,pickle.HIGHEST_PROTOCOL)); vap.write_bytes(pickle.dumps(val_examples,pickle.HIGHEST_PROTOCOL))
    print(f'[*] examples train={len(train_examples):,} val={len(val_examples):,}')
    return char_vocab,base_vocab,edit_vocab,train_examples,val_examples
class PE(nn.Module):
    def __init__(self,d,max_len=8192,dropout=0.1):
        super().__init__(); self.drop=nn.Dropout(dropout); pe=torch.zeros(max_len,d); pos=torch.arange(max_len,dtype=torch.float32).unsqueeze(1); div=torch.exp(torch.arange(0,d,2).float()*(-math.log(10000.0)/d)); pe[:,0::2]=torch.sin(pos*div); pe[:,1::2]=torch.cos(pos*div); self.register_buffer('pe',pe.unsqueeze(0),persistent=False)
    def forward(self,x): return self.drop(x+self.pe[:,:x.size(1)].to(dtype=x.dtype))
class LocalConv(nn.Module):
    def __init__(self,d,drop=0.1,k=5):
        super().__init__(); self.norm=nn.LayerNorm(d); self.dw=nn.Conv1d(d,d,k,padding=k//2,groups=d); self.pw1=nn.Conv1d(d,d*2,1); self.pw2=nn.Conv1d(d*2,d,1); self.drop=nn.Dropout(drop)
    def forward(self,x):
        h=self.norm(x).transpose(1,2); h=self.dw(h); h=F.gelu(self.pw1(h)); h=self.pw2(h).transpose(1,2); return x+self.drop(h)
class CharEditTaggerV2(nn.Module):
    def __init__(self,char_n,base_n,label_n,cfg):
        super().__init__(); self.cfg=cfg
        self.ce=nn.Embedding(char_n,cfg.d_model,padding_idx=PAD); self.be=nn.Embedding(base_n,cfg.d_model,padding_idx=PAD); self.te=nn.Embedding(8,cfg.d_model)
        self.in_norm=nn.LayerNorm(cfg.d_model); self.pos=PE(cfg.d_model,cfg.max_chars+256,cfg.dropout); self.local=nn.ModuleList([LocalConv(cfg.d_model,cfg.dropout) for _ in range(cfg.conv_layers)])
        layer=nn.TransformerEncoderLayer(cfg.d_model,cfg.nhead,cfg.ffn_dim,cfg.dropout,batch_first=True,norm_first=True,activation='gelu')
        self.enc=nn.TransformerEncoder(layer,cfg.layers); self.norm=nn.LayerNorm(cfg.d_model); self.edit=nn.Linear(cfg.d_model,label_n); self.gate=nn.Linear(cfg.d_model,1); self.scale=math.sqrt(cfg.d_model)
    def forward(self,x,base,typ):
        pad=x.eq(PAD); h=(self.ce(x)+self.be(base)+self.te(typ))*self.scale; h=self.in_norm(h); h=self.pos(h)
        for blk in self.local: h=blk(h)
        h=self.enc(h,src_key_padding_mask=pad); h=self.norm(h); return self.edit(h),self.gate(h).squeeze(-1)
class EMA:
    def __init__(self,model,decay):
        self.decay=decay; self.shadow={k:v.detach().clone() for k,v in model.state_dict().items() if v.dtype.is_floating_point}; self.non={k:v.detach().clone() for k,v in model.state_dict().items() if not v.dtype.is_floating_point}
    @torch.no_grad()
    def update(self,model):
        for k,v in model.state_dict().items():
            if k in self.shadow: self.shadow[k].mul_(self.decay).add_(v.detach(),alpha=1-self.decay)
            elif not v.dtype.is_floating_point: self.non[k]=v.detach().clone()
    def state_dict(self):
        d={k:v.detach().clone() for k,v in self.shadow.items()}; d.update({k:v.detach().clone() for k,v in self.non.items()}); return d
    def load_state_dict(self,state):
        self.shadow={k:v.detach().clone() for k,v in state.items() if v.dtype.is_floating_point}; self.non={k:v.detach().clone() for k,v in state.items() if not v.dtype.is_floating_point}
def loss_fn(logits,gate_logits,y,gate_t,w,label_smoothing,gate_w):
    B,L,C=logits.shape; valid=y.ne(IGNORE); ww=w.float()/100.0
    ce=F.cross_entropy(logits.reshape(B*L,C),y.reshape(B*L),ignore_index=IGNORE,reduction='none',label_smoothing=label_smoothing).view(B,L)
    ce=(ce*ww*valid.float()).sum()/torch.clamp((ww*valid.float()).sum(),min=1.0)
    gv=gate_t.ne(IGNORE); gt=gate_t.float().clamp(0,1); pos_weight=torch.tensor(3.0,device=logits.device,dtype=logits.dtype)
    bce=F.binary_cross_entropy_with_logits(gate_logits,gt,reduction='none',pos_weight=pos_weight)
    bce=(bce*ww*gv.float()).sum()/torch.clamp((ww*gv.float()).sum(),min=1.0)
    return ce+gate_w*bce,ce.detach(),bce.detach()
def enc_tensors(text,char_vocab,base_vocab,device):
    return (torch.tensor(char_vocab.enc(text),dtype=torch.long,device=device).unsqueeze(0),torch.tensor(base_vocab.enc(text),dtype=torch.long,device=device).unsqueeze(0),torch.tensor([type_id(ch) for ch in text],dtype=torch.long,device=device).unsqueeze(0))
@torch.no_grad()
def raw_chunk(models,text,char_vocab,base_vocab,edit_vocab,device):
    if not text: return []
    x,b,t=enc_tensors(text,char_vocab,base_vocab,device); ps=None; gs=None
    for m in models:
        logits,gl=m(x,b,t); p=F.softmax(logits[0],dim=-1); g=torch.sigmoid(gl[0]); ps=p if ps is None else ps+p; gs=g if gs is None else gs+g
    p=ps/len(models); g=gs/len(models); bp,bid=p.max(dim=-1); out=[]
    for i,ch in enumerate(text):
        seg=edit_vocab.id_to_seg[int(bid[i])]; cid=edit_vocab.seg_to_id.get(ch); cp=float(p[i,cid]) if cid is not None else 0.0; out.append((ch,seg,float(bp[i]),cp,float(g[i])))
    return out
def decode_raw(raw,params):
    out=[]
    for ch,seg,bp,cp,gp in raw:
        if seg is None or seg==PAD_CH: seg=ch
        if seg!=ch and not (bp>=params.threshold and gp>=params.gate_threshold and (bp-cp)>=params.copy_margin): seg=ch
        out.append(seg)
    pred=''.join(out); src=''.join(x[0] for x in raw)
    if src:
        ef=lev(pred,src)/max(1,len(src)); lf=abs(len(pred)-len(src))/max(1,len(src))
        if ef>params.max_edit_frac or lf>params.max_len_delta_frac: pred=src
    return pred
def raw_text(models,text,char_vocab,base_vocab,edit_vocab,cfg,device):
    raw=[]
    for c in split_text_soft(text,cfg.max_chars): raw.extend(raw_chunk(models,c,char_vocab,base_vocab,edit_vocab,device))
    return raw
def finalize(raw,params,lex,cleanup=True):
    y=decode_raw(raw,params)
    if cleanup: y=light_cleanup(y)
    if params.use_lexicon and lex:
        y=apply_word_lexicon(y,lex)
        if cleanup: y=light_cleanup(y)
    return y
@torch.no_grad()
def eval_grid(models,df_val,char_vocab,base_vocab,edit_vocab,cfg,device,lex):
    sample=df_val.head(cfg.eval_samples) if cfg.eval_samples and len(df_val)>cfg.eval_samples else df_val
    inputs=sample['input'].tolist(); refs=sample['corrected_text'].tolist(); raws=[raw_text(models,x,char_vocab,base_vocab,edit_vocab,cfg,device) for x in tqdm(inputs,desc='Eval forward')]
    ths=[float(x) for x in cfg.eval_thresholds.split(',') if x.strip()]; cms=[float(x) for x in cfg.eval_copy_margins.split(',') if x.strip()]; gts=[float(x) for x in cfg.eval_gate_thresholds.split(',') if x.strip()]; lexs=[bool(int(x)) for x in cfg.eval_use_lexicon.split(',') if x.strip()]
    best=999.0; bestp=DecodeParams(max_edit_frac=cfg.max_edit_frac,max_len_delta_frac=cfg.max_len_delta_frac); bestpred=[]; total=len(ths)*len(cms)*len(gts)*len(lexs); k=0
    for th in ths:
      for cm in cms:
       for gt in gts:
        for ul in lexs:
            k+=1; p=DecodeParams(th,cm,gt,ul,cfg.max_edit_frac,cfg.max_len_delta_frac); preds=[finalize(r,p,lex) for r in raws]; c=cer(preds,refs)
            print(f'[*] Val {k:03d}/{total}: CER={c:.6f} score={1-c:.6f} th={th:.2f} cm={cm:.2f} gate={gt:.2f} lex={int(ul)}')
            if c<best: best=c; bestp=p; bestpred=preds
    return best,bestp,list(zip(inputs[:3],bestpred[:3],refs[:3]))
def scheduler(optim,total,warm,min_ratio):
    def f(step):
        if step<warm: return max(1e-8,step/max(1,warm))
        prog=(step-warm)/max(1,total-warm); cos=0.5*(1+math.cos(math.pi*min(1.0,prog))); return min_ratio+(1-min_ratio)*cos
    return torch.optim.lr_scheduler.LambdaLR(optim,f)
def save_ckpt(path,model,ema,opt,sch,scaler,epoch,best,params,cfg):
    tmp=Path(str(path)+'.tmp'); path=Path(path); path.parent.mkdir(parents=True,exist_ok=True)
    torch.save({'model':model.state_dict(),'ema':ema.state_dict() if ema else None,'optimizer':opt.state_dict() if opt else None,'scheduler':sch.state_dict() if sch else None,'scaler':scaler.state_dict() if scaler else None,'epoch':epoch,'best_cer':best,'best_params':asdict(params),'cfg':asdict(cfg)},tmp); tmp.replace(path)
def load_ckpt(path,model,ema=None,opt=None,sch=None,scaler=None,device='cpu',prefer_ema=True):
    
    try:
        ck=torch.load(path,map_location=device,weights_only=False)
    except TypeError:
        ck=torch.load(path,map_location=device)
    state=ck.get('ema') if prefer_ema and ck.get('ema') is not None else ck.get('model',ck); model.load_state_dict(state,strict=True)
    if ema and ck.get('ema') is not None: ema.load_state_dict(ck['ema'])
    if opt and ck.get('optimizer') is not None: opt.load_state_dict(ck['optimizer'])
    if sch and ck.get('scheduler') is not None: sch.load_state_dict(ck['scheduler'])
    if scaler and ck.get('scaler') is not None: scaler.load_state_dict(ck['scaler'])
    return int(ck.get('epoch',0)),float(ck.get('best_cer',999.0)),DecodeParams(**ck.get('best_params',{}))
def prune(out_dir,keep):
    fs=sorted(Path(out_dir).glob('epoch_*.pt'),key=lambda p:p.stat().st_mtime)
    while len(fs)>keep: fs.pop(0).unlink(missing_ok=True)

def inspect_main(args):
    tr=load_train_csv(Path(args.train_csv),args.limit_rows); te=load_test_csv(Path(args.test_csv)); print('train',tr.shape,'test',te.shape); print(tr['input'].str.len().describe()); print(te['input'].str.len().describe()); sm=tr.sample(min(len(tr),args.sample),random_state=42); c=cer(sm['input'].tolist(),sm['corrected_text'].tolist()); print(f'copy CER={c:.6f} score={1-c:.6f}')
def train_main(args):
    cfg=TrainConfig(**{k:getattr(args,k) for k in TrainConfig.__dataclass_fields__ if hasattr(args,k)})
    seed_all(cfg.seed); out=Path(cfg.output_dir); out.mkdir(parents=True,exist_ok=True); (out/'config.json').write_text(json.dumps(asdict(cfg),ensure_ascii=False,indent=2),encoding='utf-8')
    df=load_train_csv(Path(cfg.train_csv),cfg.limit_rows); df_train,df_val=train_val_split(df,cfg.val_ratio,cfg.seed); print(f'[*] rows train={len(df_train):,} val={len(df_val):,}')
    exact=build_exact_map(df_train); (out/'exact_map.json').write_text(json.dumps(exact,ensure_ascii=False),encoding='utf-8'); print(f'[*] exact={len(exact):,}')
    lex={}
    if cfg.use_word_lexicon:
        lp=out/'word_lexicon.json'
        if lp.exists() and not args.rebuild_cache: lex=json.loads(lp.read_text(encoding='utf-8'))
        else:
            lex=build_word_lexicon(df_train,cfg.lex_min_count,cfg.lex_min_conf); lp.write_text(json.dumps(lex,ensure_ascii=False,indent=2),encoding='utf-8')
        print(f'[*] lex={len(lex):,}')
    char_vocab,base_vocab,edit_vocab,train_ex,val_ex=prepare_cache(df_train,df_val,cfg,out,args.rebuild_cache)
    ds=TransducerDataset(train_ex,char_vocab,base_vocab,cfg.max_chars,True); loader=DataLoader(ds,batch_size=cfg.batch_size,shuffle=True,collate_fn=collate,num_workers=cfg.num_workers,pin_memory=True)
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'); print('[*] device',device)
    if device.type=='cuda': torch.backends.cuda.matmul.allow_tf32=True; torch.backends.cudnn.allow_tf32=True
    model=CharEditTaggerV2(len(char_vocab),len(base_vocab),len(edit_vocab),cfg).to(device); opt=torch.optim.AdamW(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay,betas=(0.9,0.98))
    total=math.ceil(len(loader)/cfg.accum_steps)*cfg.epochs; sch=scheduler(opt,total,max(10,int(total*cfg.warmup_ratio)),cfg.min_lr_ratio); scaler=torch.cuda.amp.GradScaler(enabled=(cfg.amp and device.type=='cuda')); ema=EMA(model,cfg.ema_decay) if cfg.ema_decay>0 else None
    start=1; best=999.0; bestp=DecodeParams(max_edit_frac=cfg.max_edit_frac,max_len_delta_frac=cfg.max_len_delta_frac)
    if cfg.resume:
        ep,best,bestp=load_ckpt(cfg.resume,model,ema,opt,sch,scaler,device,prefer_ema=False); start=ep+1; print('[*] resumed',ep,best,bestp)
    log=out/'train_log.csv'
    if not log.exists(): log.write_text('epoch,loss,ce,gate,val_cer,score,threshold,copy_margin,gate_threshold,use_lexicon,lr,sec\n',encoding='utf-8')
    for epoch in range(start,cfg.epochs+1):
        model.train(); t0=time.time(); tl=tc=tg=0.0; n=0; opt.zero_grad(set_to_none=True); pbar=tqdm(loader,desc=f'Epoch {epoch}/{cfg.epochs}')
        for it,batch in enumerate(pbar,1):
            x=batch['x'].to(device,non_blocking=True); ba=batch['base'].to(device,non_blocking=True); typ=batch['typ'].to(device,non_blocking=True); y=batch['y'].to(device,non_blocking=True); gt=batch['gate'].to(device,non_blocking=True); w=batch['w'].to(device,non_blocking=True)
            with torch.cuda.amp.autocast(enabled=(cfg.amp and device.type=='cuda')):
                logits,gl=model(x,ba,typ); loss,ce_l,gate_l=loss_fn(logits,gl,y,gt,w,cfg.label_smoothing,cfg.gate_loss_weight); loss=loss/cfg.accum_steps
            scaler.scale(loss).backward(); tl+=float(loss.detach().cpu())*cfg.accum_steps; tc+=float(ce_l.cpu()); tg+=float(gate_l.cpu()); n+=1
            if it%cfg.accum_steps==0 or it==len(loader):
                scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(),cfg.grad_clip); scaler.step(opt); scaler.update(); opt.zero_grad(set_to_none=True); sch.step();
                if ema: ema.update(model)
            pbar.set_postfix(loss=f'{tl/max(1,n):.4f}',lr=f'{sch.get_last_lr()[0]:.2e}')
        raw_state=None
        if ema:
            raw_state={k:v.detach().cpu().clone() for k,v in model.state_dict().items()}; model.load_state_dict(ema.state_dict(),strict=True)
        val,bp,examples=eval_grid([model],df_val,char_vocab,base_vocab,edit_vocab,cfg,device,lex if cfg.use_word_lexicon else {})
        if raw_state is not None: model.load_state_dict(raw_state,strict=True)
        print(f'[*] Epoch {epoch} val CER={val:.6f} score={1-val:.6f} params={bp}')
        for i,(src,pred,ref) in enumerate(examples[:2]): print(f'--- val {i} ---\nINP: {src[:500]}\nPRD: {pred[:500]}\nREF: {ref[:500]}')
        sec=time.time()-t0; log.open('a',encoding='utf-8').write(f'{epoch},{tl/max(1,n):.8f},{tc/max(1,n):.8f},{tg/max(1,n):.8f},{val:.8f},{1-val:.8f},{bp.threshold},{bp.copy_margin},{bp.gate_threshold},{int(bp.use_lexicon)},{sch.get_last_lr()[0]:.8g},{sec:.2f}\n')
        save_ckpt(out/'last.pt',model,ema,opt,sch,scaler,epoch,min(best,val),bp,cfg); save_ckpt(out/f'epoch_{epoch:03d}.pt',model,ema,opt,sch,scaler,epoch,min(best,val),bp,cfg); prune(out,cfg.keep_epoch_ckpts)
        if val<best:
            best=val; bestp=bp; save_ckpt(out/'best.pt',model,ema,opt,sch,scaler,epoch,best,bestp,cfg); (out/'best_params.json').write_text(json.dumps(asdict(bestp),ensure_ascii=False,indent=2),encoding='utf-8'); print(f'[*] new best {best:.6f}')
    print('[*] done')
def load_run(run_dir):
    run=Path(run_dir); data=json.loads((run/'config.json').read_text(encoding='utf-8')); valid=set(TrainConfig.__dataclass_fields__.keys()); cfg=TrainConfig(**{k:v for k,v in data.items() if k in valid})
    char=CharVocab.load(run/'char_vocab.json'); base=BaseVocab.load(run/'base_vocab.json'); edit=EditVocab.load(run/'edit_vocab.json'); exact=json.loads((run/'exact_map.json').read_text(encoding='utf-8')) if (run/'exact_map.json').exists() else {}; lex=json.loads((run/'word_lexicon.json').read_text(encoding='utf-8')) if (run/'word_lexicon.json').exists() else {}; return cfg,char,base,edit,exact,lex
def infer_main(args):
    paths=[]
    if args.checkpoint: paths.append(Path(args.checkpoint))
    if args.checkpoints: paths += [Path(p) for p in args.checkpoints]
    if not paths: raise ValueError('need --checkpoint or --checkpoints')
    run=Path(args.run_dir) if args.run_dir else paths[0].parent; cfg,char,base,edit,exact,lex=load_run(run)
    if args.max_chars: cfg.max_chars=args.max_chars
    device=torch.device('cuda' if torch.cuda.is_available() and not args.cpu else 'cpu'); models=[]; best=999.0; params=None
    for p in paths:
        m=CharEditTaggerV2(len(char),len(base),len(edit),cfg).to(device); _,bc,bp=load_ckpt(p,m,device=device,prefer_ema=not args.no_ema); m.eval(); models.append(m); print(f'[*] loaded {p} best={bc:.6f} params={bp}')
        if bc<best: best=bc; params=bp
    params=params or DecodeParams()
    for name in ['threshold','copy_margin','gate_threshold','max_edit_frac','max_len_delta_frac']:
        v=getattr(args,name)
        if v is not None: setattr(params,name,v)
    if args.no_word_lexicon: params.use_lexicon=False
    if args.force_word_lexicon: params.use_lexicon=True
    print('[*] decode params',params,'device',device)
    df=load_test_csv(Path(args.test_csv or cfg.test_csv)); preds=[]
    for text in tqdm(df['input'].tolist(),desc='Infer'):
        if not args.no_exact and text in exact: pred=exact[text]
        else: pred=finalize(raw_text(models,text,char,base,edit,cfg,device),params,{} if args.no_word_lexicon else lex,cleanup=not args.no_cleanup)
        preds.append(pred if pred or not str(text).strip() else light_cleanup(str(text)))
    sub=pd.DataFrame({'id':df['id'],'corrected_text':preds}); out=Path(args.submission); out.parent.mkdir(parents=True,exist_ok=True); sub.to_csv(out,index=False,encoding='utf-8',quoting=csv.QUOTE_MINIMAL); print('[*] saved',out,'rows',len(sub))
    if args.zip:
        zp=Path(args.zip); zp.unlink(missing_ok=True)
        with zipfile.ZipFile(zp,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf: zf.write(out,arcname='submission.csv')
        print('[*] saved zip',zp,f'{zp.stat().st_size/1024/1024:.2f} MB')
    print(sub.head().to_string())
def rules_main(args):
    tr=load_train_csv(Path(args.train_csv),args.limit_rows); te=load_test_csv(Path(args.test_csv)); exact=build_exact_map(tr); lex=build_word_lexicon(tr,args.lex_min_count,args.lex_min_conf); preds=[exact[x] if x in exact else light_cleanup(apply_word_lexicon(x,lex)) for x in tqdm(te['input'],desc='Rules')]; sub=pd.DataFrame({'id':te['id'],'corrected_text':preds}); sub.to_csv(args.submission,index=False,encoding='utf-8')
    if args.zip:
        with zipfile.ZipFile(args.zip,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as zf: zf.write(args.submission,arcname='submission.csv')

def parser():
    p=argparse.ArgumentParser('Vietnamese OCR correction transducer-v2'); sub=p.add_subparsers(dest='mode',required=True)
    a=sub.add_parser('inspect'); a.add_argument('--train-csv',default='NLP/train.csv'); a.add_argument('--test-csv',default='NLP/test.csv'); a.add_argument('--limit-rows',type=int); a.add_argument('--sample',type=int,default=1000); a.set_defaults(func=inspect_main)
    t=sub.add_parser('train')
    t.add_argument('--train-csv',default='NLP/train.csv'); t.add_argument('--test-csv',default='NLP/test.csv'); t.add_argument('--output-dir',default='runs/ocr_transducer_v2_seed42'); t.add_argument('--seed',type=int,default=42); t.add_argument('--val-ratio',type=float,default=0.035); t.add_argument('--limit-rows',type=int)
    t.add_argument('--max-chars',type=int,default=896); t.add_argument('--hard-align-chars',type=int,default=1400); t.add_argument('--min-seg-count',type=int,default=2); t.add_argument('--max-segment-len',type=int,default=5); t.add_argument('--max-labels',type=int,default=6144); t.add_argument('--identity-augment-ratio',type=float,default=0.18)
    t.add_argument('--d-model',type=int,default=448); t.add_argument('--nhead',type=int,default=8); t.add_argument('--layers',type=int,default=7); t.add_argument('--conv-layers',type=int,default=2); t.add_argument('--ffn-dim',type=int,default=1792); t.add_argument('--dropout',type=float,default=0.12)
    t.add_argument('--batch-size',type=int,default=16); t.add_argument('--accum-steps',type=int,default=2); t.add_argument('--epochs',type=int,default=40); t.add_argument('--lr',type=float,default=4e-4); t.add_argument('--min-lr-ratio',type=float,default=0.03); t.add_argument('--warmup-ratio',type=float,default=0.06); t.add_argument('--weight-decay',type=float,default=0.035); t.add_argument('--grad-clip',type=float,default=1.0); t.add_argument('--amp',action=argparse.BooleanOptionalAction,default=True); t.add_argument('--num-workers',type=int,default=2); t.add_argument('--ema-decay',type=float,default=0.9994)
    t.add_argument('--rewrite-weight',type=float,default=4.0); t.add_argument('--delete-insert-weight',type=float,default=4.8); t.add_argument('--copy-weight',type=float,default=1.0); t.add_argument('--gate-loss-weight',type=float,default=0.25); t.add_argument('--label-smoothing',type=float,default=0.015)
    t.add_argument('--eval-samples',type=int,default=1200); t.add_argument('--eval-thresholds',default='0.55,0.65,0.75,0.85,0.90'); t.add_argument('--eval-copy-margins',default='0.04,0.08,0.12'); t.add_argument('--eval-gate-thresholds',default='0.20,0.35,0.50'); t.add_argument('--eval-use-lexicon',default='0,1'); t.add_argument('--max-edit-frac',type=float,default=0.45); t.add_argument('--max-len-delta-frac',type=float,default=0.35); t.add_argument('--keep-epoch-ckpts',type=int,default=6); t.add_argument('--resume')
    t.add_argument('--use-word-lexicon',action=argparse.BooleanOptionalAction,default=True); t.add_argument('--lex-min-count',type=int,default=3); t.add_argument('--lex-min-conf',type=float,default=0.80); t.add_argument('--cpu',action='store_true'); t.add_argument('--rebuild-cache',action='store_true'); t.set_defaults(func=train_main)
    i=sub.add_parser('infer'); i.add_argument('--checkpoint'); i.add_argument('--checkpoints',nargs='*'); i.add_argument('--run-dir'); i.add_argument('--test-csv'); i.add_argument('--submission',default='submission.csv'); i.add_argument('--zip',default='submission.zip'); i.add_argument('--threshold',type=float); i.add_argument('--copy-margin',type=float); i.add_argument('--gate-threshold',type=float); i.add_argument('--max-edit-frac',type=float); i.add_argument('--max-len-delta-frac',type=float); i.add_argument('--max-chars',type=int); i.add_argument('--cpu',action='store_true'); i.add_argument('--no-word-lexicon',action='store_true'); i.add_argument('--force-word-lexicon',action='store_true'); i.add_argument('--no-exact',action='store_true'); i.add_argument('--no-cleanup',action='store_true'); i.add_argument('--no-ema',action='store_true'); i.set_defaults(func=infer_main)
    r=sub.add_parser('rules'); r.add_argument('--train-csv',default='NLP/train.csv'); r.add_argument('--test-csv',default='NLP/test.csv'); r.add_argument('--submission',default='submission_rules.csv'); r.add_argument('--zip',default='submission_rules.zip'); r.add_argument('--limit-rows',type=int); r.add_argument('--lex-min-count',type=int,default=3); r.add_argument('--lex-min-conf',type=float,default=0.80); r.set_defaults(func=rules_main)
    return p
if __name__=='__main__':
    args=parser().parse_args(); args.func(args)
