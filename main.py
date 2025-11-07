import sys,subprocess,importlib,os,threading,time,random,contextlib,json,ctypes,math,site
_pkg_modules={"opencv-python":"cv2","open-clip-torch":"open_clip","pillow":"PIL","screeninfo":"screeninfo"}
def _pip(x,base_dir):
    names=[x,_pkg_modules.get(x,x.replace("-","_"))]
    for name in names:
        try:
            importlib.import_module(name)
            return
        except:
            continue
    cmds=[[sys.executable,"-m","pip","install","--upgrade","--no-input","--timeout","30","--target",base_dir,x]]
    if os.name=="nt":
        cmds.append([sys.executable,"-m","pip","install","--upgrade","--no-input","--timeout","30","--user",x])
    else:
        cmds.append([sys.executable,"-m","pip","install","--upgrade","--no-input","--timeout","30",x])
    for cmd in cmds:
        try:
            subprocess.check_call(cmd)
            for name in names:
                try:
                    importlib.invalidate_caches();importlib.import_module(name);return
                except:
                    continue
        except:
            continue
    try:
        subprocess.check_call([sys.executable,"-m","pip","download","-d",base_dir,"--no-input","--timeout","30",x])
    except:
        pass
home=os.path.expanduser("~");desk=os.path.join(home,"Desktop");base_dir=os.path.join(desk,"GameAI");models_dir=os.path.join(base_dir,"models");os.makedirs(base_dir,exist_ok=True);os.makedirs(models_dir,exist_ok=True);site.addsitedir(base_dir);sys.path.insert(0,base_dir) if base_dir not in sys.path else None
for p in ["psutil","pillow","numpy","opencv-python","mss","pynput","pyautogui","torch","torchvision","GPUtil","pynvml","pygetwindow","screeninfo","requests","ultralytics","open-clip-torch","segment-anything","networkx"]:
    _pip(p,base_dir)
import psutil,pyautogui,torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim,torchvision.models as models,GPUtil,cv2,numpy as np,mss,requests,open_clip,networkx as nx
from ultralytics import YOLO
from segment_anything import sam_model_registry,SamPredictor
from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo,nvmlDeviceGetCount
from pynput import mouse,keyboard
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk
import pygetwindow as gw
from screeninfo import get_monitors
pyautogui.FAILSAFE=False;pyautogui.PAUSE=0.0;pyautogui.MINIMUM_DURATION=0.0
if os.name=="nt":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except:
            pass
try:
    torch.set_num_threads(max(1,(os.cpu_count() or 4)-1))
except:
    pass
try:
    torch.set_float32_matmul_precision("high")
except:
    pass
try:
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark=True
except:
    pass
device="cuda" if torch.cuda.is_available() else "cpu"
model_path=os.path.join(models_dir,"model.pt");exp_dir=os.path.join(base_dir,"experience");os.makedirs(exp_dir,exist_ok=True)
scaler=torch.amp.GradScaler("cuda") if device=="cuda" else None
pre_url="https://download.pytorch.org/models/resnet18-f37072fd.pth";pre_path=os.path.join(models_dir,"resnet18.pth");policy_url="https://download.pytorch.org/models/squeezenet1_1-b8a52dc0.pth";yolo_url="https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt";yolo_path=os.path.join(models_dir,"yolov8n.pt");clip_name="ViT-B-32";clip_pretrained="laion2b_s34b_b79k";clip_path=os.path.join(models_dir,"clip_vitb32.pt");sam_url="https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth";sam_path=os.path.join(models_dir,"sam_vit_b.pth")
try:
    nvmlInit();_gpu_count=nvmlDeviceGetCount()
except:
    _gpu_count=0
def _gpu_load():
    try:
        if _gpu_count<1:
            g=GPUtil.getGPUs()
            if not g:
                return 0.0,0.0
            g=g[0];return float(g.load*100.0),float(g.memoryUtil*100.0)
        h=nvmlDeviceGetHandleByIndex(0);u=nvmlDeviceGetUtilizationRates(h);m=nvmlDeviceGetMemoryInfo(h);return float(u.gpu),float(m.used*100.0/max(m.total,1))
    except:
        try:
            g=GPUtil.getGPUs()
            if not g:
                return 0.0,0.0
            g=g[0];return float(g.load*100.0),float(g.memoryUtil*100.0)
        except:
            return 0.0,0.0
def _sys_usage():
    c=float(psutil.cpu_percent(interval=None));mem=float(psutil.virtual_memory().percent);g,vr=_gpu_load();return c,mem,g,vr
def _clamp(a,lo,hi):
    return lo if a<lo else hi if a>hi else a
def _now():
    return time.time()
class Replay:
    def __init__(self,cap=50000,seq=4):
        self.cap=cap;self.seq=seq;self.buf=[];self.src=[];self.rewards=[];self.logp=[]
    def push(self,frame,act,source,reward=0.0,logp=0.0):
        if len(self.buf)>=self.cap:
            self.buf.pop(0);self.src.pop(0);self.rewards.pop(0);self.logp.pop(0)
        self.buf.append((frame,act,_now()));self.src.append(int(source));self.rewards.append(float(reward));self.logp.append(float(logp))
    def can_sample(self,batch):
        return len(self.buf)>self.seq+batch
    def sample(self,batch):
        idx=[];mx=len(self.buf)-self.seq-1
        for _ in range(batch):
            idx.append(random.randint(0,mx))
        seqs=[];acts=[];rewards=[];masks=[];sources=[];logps=[]
        for k in idx:
            fr=[self.buf[k+i][0] for i in range(self.seq)]
            ac=self.buf[k+self.seq-1][1]
            pre=self.buf[k+self.seq-2][0];post=self.buf[k+self.seq-1][0]
            pre=cv2.Canny(pre,32,128);post=cv2.Canny(post,32,128)
            r=float(np.mean(np.abs(post.astype(np.float32)-pre.astype(np.float32)))/255.0)+self.rewards[k+self.seq-1]
            seqs.append(np.stack(fr,0));acts.append(np.array(ac,dtype=np.float32));rewards.append(r);masks.append(1.0);sources.append(self.src[k+self.seq-1]);logps.append(self.logp[k+self.seq-1])
        return np.stack(seqs,0),np.stack(acts,0),np.array(rewards,dtype=np.float32),np.array(masks,dtype=np.float32),np.array(sources,dtype=np.float32),np.array(logps,dtype=np.float32)
class ExperienceWriter:
    def __init__(self,path):
        self.path=path;self.learn_dir=os.path.join(path,"learn");self.train_dir=os.path.join(path,"train");self.meta=os.path.join(path,"log.jsonl");os.makedirs(self.learn_dir,exist_ok=True);os.makedirs(self.train_dir,exist_ok=True);self.lock=threading.Lock()
    def record(self,frame,action,pos,source,rect,title,mode):
        if frame is None or rect is None:
            return
        if pos is None:
            pos=(0.5,0.5)
        ts=int(_now()*1000);sub=self.learn_dir if int(source)==1 else self.train_dir
        try:
            os.makedirs(sub,exist_ok=True)
        except:
            pass
        name=os.path.join(sub,f"{ts}_{int(source)}.npz")
        data={"time":ts,"action":[float(_clamp(action[0],-1.0,1.0)),float(_clamp(action[1],-1.0,1.0)),float(_clamp(action[2],0.0,1.0))],"pos":[float(_clamp(pos[0],0.0,1.0)),float(_clamp(pos[1],0.0,1.0))],"rect":[int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3])],"source":int(source),"window":title or "","mode":mode}
        arr=np.array(frame,dtype=np.uint8)
        with self.lock:
            try:
                np.savez_compressed(name,frame=arr,meta=data)
            except:
                pass
            try:
                with open(self.meta,"a",encoding="utf-8") as f:
                    f.write(json.dumps(data,ensure_ascii=False)+"\n")
            except:
                pass
class SpikeEncoder(nn.Module):
    def __init__(self,dim,steps=6):
        super().__init__();self.steps=steps;self.fc=nn.Linear(dim,dim);self.decay=nn.Parameter(torch.full((dim,),0.8));self.thresh=nn.Parameter(torch.ones(dim));self.reset=nn.Parameter(torch.zeros(dim))
    def forward(self,x):
        b=x.shape[0];h=self.fc(x);mem=torch.zeros(b,h.shape[1],device=x.device);spikes=[]
        for _ in range(self.steps):
            mem=mem*torch.sigmoid(self.decay)+h
            spk=torch.sigmoid((mem-self.thresh)*8.0)
            mem=mem*(1.0-spk)+self.reset
            spikes.append(spk)
        s=torch.stack(spikes,1)
        return s,s.mean(1)
class DifferentiableMemory(nn.Module):
    def __init__(self,slots,dim):
        super().__init__();self.slots=slots;self.dim=dim;self.mem=nn.Parameter(torch.randn(slots,dim)*0.01);self.controller=nn.GRUCell(dim,dim);self.write=nn.Linear(dim,dim);self.read=nn.Linear(dim,dim)
    def forward(self,query,context):
        b=query.shape[0];memory=self.mem.unsqueeze(0).expand(b,-1,-1);state=torch.zeros(b,self.dim,device=query.device);reads=[]
        for i in range(context.shape[1]):
            state=self.controller(context[:,i],state)
            erase=torch.sigmoid(self.write(state))
            memory=memory*(1.0-erase.unsqueeze(1))+erase.unsqueeze(1)*context[:,i].unsqueeze(1)
            key=self.read(state).unsqueeze(1)
            att=torch.softmax((key*memory).sum(-1)/math.sqrt(self.dim),-1)
            read=torch.sum(att.unsqueeze(-1)*memory,1)
            reads.append(read)
        read_final=reads[-1] if reads else query
        return query+state+read_final
class BrainInspired(nn.Module):
    def __init__(self,dim,steps=6):
        super().__init__();self.dim=dim;self.steps=steps;self.spike=SpikeEncoder(dim,steps);self.rec=nn.GRU(dim,dim,batch_first=True);self.mem=DifferentiableMemory(32,dim);self.attn=nn.MultiheadAttention(dim,8,batch_first=True);self.register_buffer("trace",torch.zeros(dim,dim));self.last=None
    def forward(self,x,context=None):
        spikes,avg=self.spike(x)
        seq=torch.cumsum(spikes,1)
        if context is not None:
            seq=torch.cat([context,seq],1)
        out,_=self.rec(seq)
        key=out;value=out
        query=avg.unsqueeze(1)
        attn,_=self.attn(query,key,value,need_weights=False)
        mem=self.mem(attn.squeeze(1),seq)
        self.last=(spikes.detach(),mem.detach(),x.detach())
        return mem
    def hebbian_loss(self,lam=1e-3):
        if not self.last:
            return torch.tensor(0.0,device=self.trace.device)
        spikes,mem,x=self.last
        corr=torch.einsum("bsi,bsj->ij",spikes,spikes)/(spikes.shape[0]*spikes.shape[1]+1e-6)
        stdp=torch.einsum("bi,bj->ij",x,mem)/(x.shape[0]+1e-6)
        self.trace=self.trace.mul(0.9).add_(0.1*corr)
        return lam*(self.trace.pow(2).mean()+stdp.pow(2).mean())
class Net(nn.Module):
    def __init__(self,seq=4):
        super().__init__();self.seq=seq;self.frame_enc=nn.Sequential(nn.Conv2d(3,32,5,2,2),nn.ReLU(),nn.BatchNorm2d(32),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Dropout2d(0.1),nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.Dropout2d(0.1));self.pool=nn.AdaptiveAvgPool2d((4,4));self.proj=nn.Linear(128*16,256);self.temporal=nn.TransformerEncoder(nn.TransformerEncoderLayer(256,8,512,dropout=0.1,batch_first=True),num_layers=2);self.brain=BrainInspired(256,6);self.value_head=nn.Linear(256,1);self.policy_head=nn.Linear(256,3)
    def forward(self,x):
        b,t,c,h,w=x.shape;frames=x.reshape(b*t,c,h,w);feat=self.frame_enc(frames);feat=self.pool(feat).flatten(1);feat=self.proj(feat);feat=feat.view(b,t,256);temp=self.temporal(feat);context=temp;brain=self.brain(temp[:,-1],context);p=self.policy_head(brain);v=self.value_head(brain).squeeze(-1);dx=torch.tanh(p[:,0]);dy=torch.tanh(p[:,1]);cl=torch.sigmoid(p[:,2]);return torch.stack([dx,dy,cl],1),v
def _download_file(url,path,timeout=45):
    try:
        with requests.get(url,timeout=timeout,stream=True) as r:
            r.raise_for_status()
            with open(path,"wb") as f:
                for chunk in r.iter_content(8192):
                    if chunk:
                        f.write(chunk)
        return True
    except:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
        return False
def _generate_local_model(path,seq):
    try:
        net=Net(seq)
        torch.save(net.state_dict(),path)
        return True
    except:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
        return False
def _generate_resnet_local(path):
    try:
        state=models.resnet18().state_dict()
        torch.save(state,path)
        return True
    except:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
        return False
def _generate_yolo_local(path):
    try:
        torch.save({"model":"yolov8n","weights":torch.randn(16,16)},path)
        return True
    except:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
        return False
def _generate_clip_local(path):
    try:
        model,_,_=open_clip.create_model_and_transforms(clip_name,pretrained=clip_pretrained,cache_dir=models_dir,device=device if device=="cuda" else "cpu")
        torch.save(model.state_dict(),path)
        return True
    except:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
        return False
def _generate_sam_local(path):
    try:
        torch.save({"model":"sam_vit_b","weights":torch.randn(4,4)},path)
        return True
    except:
        try:
            if os.path.exists(path):
                os.remove(path)
        except:
            pass
        return False
class VisualPerception:
    def __init__(self,seq):
        self.seq=seq;self.lock=threading.Lock();self.detector=None;self.clip_model=None;self.clip_preprocess=None;self.clip_tokenizer=None;self.sam=None;self.text_prompts=["button","score","health","message","target","danger","bonus"];self.temporal_feats=[];self.temporal_len=24
        try:
            if os.path.exists(yolo_path):
                self.detector=YOLO(yolo_path)
            else:
                self.detector=YOLO("yolov8n.pt")
        except:
            self.detector=None
        try:
            model,preprocess,_=open_clip.create_model_and_transforms(clip_name,pretrained=clip_pretrained,cache_dir=models_dir,device=device if device=="cuda" else "cpu")
            if os.path.exists(clip_path):
                try:
                    state=torch.load(clip_path,map_location=device)
                    model.load_state_dict(state,strict=False)
                except:
                    pass
            model.eval()
            self.clip_model=model
            self.clip_preprocess=preprocess
            self.clip_tokenizer=open_clip.get_tokenizer(clip_name)
        except:
            self.clip_model=None;self.clip_preprocess=None;self.clip_tokenizer=None
        try:
            if os.path.exists(sam_path):
                sam=sam_model_registry.get("vit_b")(checkpoint=sam_path)
                sam.to(device if device=="cuda" else "cpu")
                self.sam=SamPredictor(sam)
        except:
            self.sam=None
    def analyze(self,frames):
        with self.lock:
            if not frames:
                return {"detections":[],"segments":[],"clip":{},"temporal":np.zeros(512,dtype=np.float32)}
            frame=frames[-1]
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            detections=[]
            if self.detector is not None:
                try:
                    res=self.detector.predict(rgb,conf=0.2,verbose=False,stream=False)
                    if res:
                        r=res[0]
                        if hasattr(r,"boxes") and r.boxes is not None:
                            boxes=r.boxes.xyxy.detach().cpu().numpy()
                            confs=r.boxes.conf.detach().cpu().numpy()
                            cls=r.boxes.cls.detach().cpu().numpy()
                            for i in range(min(len(boxes),12)):
                                detections.append({"box":boxes[i].tolist(),"conf":float(confs[i]),"cls":int(cls[i])})
                except:
                    detections=[]
            segments=[]
            if self.sam is not None:
                try:
                    self.sam.set_image(rgb)
                    h,w,_=rgb.shape
                    pts=np.array([[w*0.25,h*0.25],[w*0.75,h*0.25],[w*0.5,h*0.75]],dtype=np.float32)
                    labels=np.ones(len(pts))
                    masks,_=self.sam.predict(point_coords=pts,point_labels=labels,multimask_output=True)
                    for m in masks[:3]:
                        segments.append(m.astype(np.uint8))
                except:
                    segments=[]
            clip_scores={}
            if self.clip_model is not None and self.clip_preprocess is not None:
                try:
                    image=self.clip_preprocess(Image.fromarray(rgb)).unsqueeze(0)
                    image=image.to(device if device=="cuda" else "cpu")
                    text=open_clip.tokenize(self.text_prompts).to(device if device=="cuda" else "cpu")
                    with torch.no_grad():
                        img_feat=self.clip_model.encode_image(image)
                        txt_feat=self.clip_model.encode_text(text)
                    img_feat=img_feat/img_feat.norm(dim=-1,keepdim=True)
                    txt_feat=txt_feat/txt_feat.norm(dim=-1,keepdim=True)
                    sims=(img_feat@txt_feat.T).detach().cpu().numpy()[0]
                    for i,prompt in enumerate(self.text_prompts):
                        clip_scores[prompt]=float(sims[i])
                    self.temporal_feats.append(img_feat.detach().cpu().numpy()[0])
                    if len(self.temporal_feats)>self.temporal_len:
                        self.temporal_feats.pop(0)
                except:
                    clip_scores={}
            temp_vec=np.mean(self.temporal_feats,0) if self.temporal_feats else np.zeros(512,dtype=np.float32)
            return {"detections":detections,"segments":segments,"clip":clip_scores,"temporal":temp_vec}
class GraphReasoner(nn.Module):
    def __init__(self,dim=64,steps=3):
        super().__init__();self.dim=dim;self.steps=steps;self.fc=nn.Linear(6,dim);self.msg=nn.Linear(dim,dim);self.out=nn.Linear(dim,1)
    def forward(self,states,adj):
        x=self.fc(states)
        for _ in range(self.steps):
            agg=torch.matmul(adj,x)
            deg=torch.clamp(adj.sum(-1,keepdim=True),min=1e-6)
            agg=agg/deg
            x=torch.relu(self.msg(agg+x))
        return torch.sigmoid(self.out(x)).squeeze(-1)
class KnowledgeBase:
    def __init__(self,seq,perception):
        self.seq=seq;self.perception=perception;self.lock=threading.Lock();self.buttons=[];self.max_buttons=128;self.goal_feat=None;self.goal_strength=0.15;self.prev_feat=None;self.dim=512;self.encoder=self._build_encoder();self.prev_regions={"score":None,"health":None,"message":None};self.state_model={};self.state_counts={};self.transition_counts={};self.last_state=None;self.event_stats={"score_up":0,"score_down":0,"health_up":0,"health_down":0,"victory":0,"defeat":0,"message":0,"target":0};self.action_rules={k:{} for k in self.event_stats};self.event_values={"score_up":1.0,"score_down":-0.5,"health_up":0.6,"health_down":-1.2,"victory":6.0,"defeat":-6.0,"message":0.4,"target":1.5};self.event_history=[];self.max_event_history=800;self.graph=nx.DiGraph();self.graph_counts={};self.graph_reasoner=GraphReasoner();self.percept_temporal=np.zeros(512,dtype=np.float32);self.last_percepts=None
    def _build_encoder(self):
        try:
            enc=models.resnet18()
            if os.path.exists(pre_path):
                state=torch.load(pre_path,map_location="cpu")
                enc.load_state_dict(state,strict=False)
            enc.fc=nn.Identity();enc.to(device);enc.eval();return enc
        except:
            net=nn.Sequential(nn.Conv2d(3,16,3,2,1),nn.ReLU(),nn.Conv2d(16,32,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((1,1)),nn.Flatten(),nn.Linear(32,self.dim));net.to(device);return net
    def _prep(self,img):
        t=torch.from_numpy(img.astype(np.float32)/255.0);t=t.permute(2,0,1).unsqueeze(0);t=F.interpolate(t,size=(224,224),mode="bilinear",align_corners=False);return t
    def _encode(self,img):
        try:
            with torch.no_grad():
                feat=self.encoder(self._prep(img).to(device))
            return feat.detach().cpu().numpy()[0]
        except:
            return np.zeros(self.dim,dtype=np.float32)
    def _norm(self,v):
        return max(float(np.linalg.norm(v)),1e-6)
    def _store_button(self,feat,pos):
        best=-1.0;idx=-1
        for i,btn in enumerate(self.buttons):
            sim=float(np.dot(btn["feat"],feat)/(self._norm(btn["feat"])*self._norm(feat)))
            if sim>best:
                best=sim;idx=i
        if idx>=0 and best>0.8:
            btn=self.buttons[idx];cnt=btn["count"]+1;btn["feat"]=(btn["feat"]*btn["count"]+feat)/max(cnt,1);px=0.7*btn["pos"][0]+0.3*pos[0];py=0.7*btn["pos"][1]+0.3*pos[1];btn["pos"]=(float(_clamp(px,0.0,1.0)),float(_clamp(py,0.0,1.0)));btn["count"]=cnt
        else:
            if len(self.buttons)>=self.max_buttons:
                self.buttons.pop(0)
            self.buttons.append({"feat":feat,"pos":(float(_clamp(pos[0],0.0,1.0)),float(_clamp(pos[1],0.0,1.0))),"count":1})
    def _crop(self,frame,rect,pos):
        if frame is None or rect is None:
            return None
        x1,y1,x2,y2=rect;w=max(1,x2-x1);h=max(1,y2-y1);px=int(x1+pos[0]*w);py=int(y1+pos[1]*h);sz=max(10,int(min(w,h)*0.12));lx=int(_clamp(px-sz,x1,x2-1));rx=int(_clamp(px+sz,x1+1,x2));ly=int(_clamp(py-sz,y1,y2-1));ry=int(_clamp(py+sz,y1+1,y2))
        if rx<=lx or ry<=ly:
            return None
        sub=frame[ly-y1:ry-y1,lx-x1:rx-x1]
        if sub.size==0:
            return None
        return cv2.cvtColor(sub,cv2.COLOR_BGR2RGB)
    def _region(self,frame,rect,center,size):
        if frame is None:
            return None
        if rect is None:
            h,w=frame.shape[:2];x1,y1,ww,hh=0,0,w,h
        else:
            x1,y1,x2,y2=rect;ww=max(1,x2-x1);hh=max(1,y2-y1)
        cx=float(_clamp(center[0],0.0,1.0));cy=float(_clamp(center[1],0.0,1.0));bw=float(_clamp(size[0],0.0,1.0));bh=float(_clamp(size[1],0.0,1.0));px=int(x1+cx*ww);py=int(y1+cy*hh);hx=max(2,int(bw*ww*0.5));hy=max(2,int(bh*hh*0.5));lx=int(_clamp(px-hx,x1,x1+ww-1));rx=int(_clamp(px+hx,x1+1,x1+ww));ly=int(_clamp(py-hy,y1,y1+hh-1));ry=int(_clamp(py+hy,y1+1,y1+hh))
        if rx<=lx or ry<=ly:
            return None
        sub=frame[ly-y1:ry-y1,lx-x1:rx-x1]
        if sub.size==0:
            return None
        return sub
    def _signature(self,patch,shape=(16,16)):
        if patch is None:
            return None
        gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY)
        sig=cv2.resize(gray,shape,interpolation=cv2.INTER_AREA).astype(np.float32)/255.0
        return sig.flatten()
    def _signature_delta(self,a,b):
        if a is None or b is None:
            return 0.0
        n=min(len(a),len(b))
        if n==0:
            return 0.0
        return float(np.linalg.norm(a[:n]-b[:n])/np.sqrt(n))
    def _extract_regions(self,frame,rect,percepts):
        reg={};h,w=frame.shape[:2];boxes=[];confs=[]
        if percepts and percepts.get("detections"):
            for det in percepts["detections"]:
                box=det.get("box")
                if not box or len(box)<4:
                    continue
                x1=float(_clamp(box[0],0.0,w-1));y1=float(_clamp(box[1],0.0,h-1));x2=float(_clamp(box[2],x1+1,w));y2=float(_clamp(box[3],y1+1,h))
                boxes.append((x1,y1,x2,y2));confs.append(float(det.get("conf",0.0)))
        base=[("score",(0.2,0.08),(0.4,0.2)),("health",(0.2,0.9),(0.45,0.2)),("message",(0.5,0.5),(0.6,0.35))]
        segments=percepts.get("segments") if percepts else []
        clip_scores=percepts.get("clip") if percepts else {}
        for idx,(name,center,size) in enumerate(base):
            bias=float(clip_scores.get(name,0.0)) if clip_scores else 0.0
            bias=(bias+1.0)*0.5
            if boxes:
                order=np.argsort(np.array(confs))[::-1]
                pick=order[int(_clamp(int(round(bias*(len(order)-1))),0,len(order)-1))]
                bx=boxes[pick];cx=((bx[0]+bx[2])*0.5)/max(w,1);cy=((bx[1]+bx[3])*0.5)/max(h,1);bw=max(size[0],(bx[2]-bx[0])/max(w,1));bh=max(size[1],(bx[3]-bx[1])/max(h,1))
                center=(center[0]*(1.0-bias)+cx*bias,center[1]*(1.0-bias)+cy*bias);size=(float(_clamp(bw,0.05,1.0)),float(_clamp(bh,0.05,1.0)))
            patch=self._region(frame,rect,center,size)
            if patch is None:
                reg[name]={"sig":None,"mean":0.0,"edges":0.0,"contrast":0.0,"segment":0.0,"clip":bias}
            else:
                sig=self._signature(patch);arr=patch.astype(np.float32)/255.0;mean=float(np.mean(arr));edges=float(np.mean(cv2.Canny(cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY),32,128))/255.0);contrast=float(np.std(arr));coverage=0.0
                if segments:
                    for seg in segments:
                        try:
                            mask=cv2.resize(seg.astype(np.float32),(patch.shape[1],patch.shape[0]),interpolation=cv2.INTER_NEAREST)
                            coverage=max(coverage,float(np.mean(mask>0.5)))
                        except:
                            continue
                reg[name]={"sig":sig,"mean":mean,"edges":edges,"contrast":contrast,"segment":coverage,"clip":bias}
        return reg
    def _button_similarity(self,feat):
        if feat is None or not self.buttons:
            return 0.0
        best=0.0
        for btn in self.buttons:
            sim=float(np.dot(btn["feat"],feat)/(self._norm(btn["feat"])*self._norm(feat)))
            if sim>best:
                best=sim
        return max(0.0,best)
    def _goal_alignment(self,feat):
        if self.goal_feat is None:
            return 0.0
        return float(np.dot(self.goal_feat,feat)/(self._norm(self.goal_feat)*self._norm(feat)))
    def _bucket(self,val,levels):
        v=float(val)
        if np.isnan(v):
            v=0.0
        v=_clamp(v,0.0,1.0)
        return int(_clamp(int(v*levels),0,levels-1)) if levels>1 else 0
    def _make_state(self,frame_feat,regions):
        score_lvl=self._bucket(regions["score"]["mean"],8)
        health_lvl=self._bucket(regions["health"]["mean"],8)
        msg_lvl=1 if regions["message"]["edges"]>0.25 else 0
        segment_lvl=self._bucket(regions["message"].get("segment",0.0),6)
        clip_lvl=self._bucket(regions["message"].get("clip",0.0),6)
        align=float(_clamp((self._goal_alignment(frame_feat)+1.0)*0.5,0.0,1.0))
        align_lvl=self._bucket(align,6)
        return (score_lvl,health_lvl,msg_lvl,segment_lvl,clip_lvl,align_lvl)
    def _register_transition(self,prev_state,state):
        key=(prev_state,state)
        self.transition_counts[key]=self.transition_counts.get(key,0)+1
        self.graph_counts[state]=self.graph_counts.get(state,0)+1
        self.graph.add_node(prev_state);self.graph.add_node(state)
        w=self.graph.get_edge_data(prev_state,state,{}).get("weight",0.0)+1.0
        self.graph.add_edge(prev_state,state,weight=w)
    def _action_key(self,action):
        if action is None:
            return "none"
        try:
            dx=int(round(float(_clamp(action[0],-1.0,1.0))*5))
            dy=int(round(float(_clamp(action[1],-1.0,1.0))*5))
            cl=int(float(action[2])>0.5)
        except:
            dx=dy=0;cl=0
        return f"{dx}:{dy}:{cl}"
    def _log_event(self,event,intensity,state,action_key,source):
        val=float(self.event_values.get(event,0.0))*float(intensity)
        self.event_stats[event]=self.event_stats.get(event,0)+1
        if state not in self.state_model:
            self.state_model[state]=0.0
        self.state_model[state]+=val
        if action_key not in self.action_rules[event]:
            self.action_rules[event][action_key]=0.0
        weight=1.0+0.5*source
        self.action_rules[event][action_key]+=val*weight
        self.event_history.append((event,_now(),val,state,action_key,source))
        if len(self.event_history)>self.max_event_history:
            self.event_history.pop(0)
    def _rule_action_hint(self,state):
        best=None;best_score=-1e9
        for event,acts in self.action_rules.items():
            if self.event_values.get(event,0.0)<=0:
                continue
            for key,val in acts.items():
                score=val
                if state in self.state_model:
                    score*=1.0+max(0.0,self.state_model[state])
                if score>best_score:
                    best_score=score;best=key
        if best is None:
            return None
        try:
            dx,dy,cl=best.split(":")
            dx=float(_clamp(int(dx)/5.0,-1.0,1.0));dy=float(_clamp(int(dy)/5.0,-1.0,1.0));cl=float(int(cl))
            return [dx,dy,cl]
        except:
            return None
    def update(self,frames,rect,action,pos,source):
        if not frames:
            return 0.0
        frame=frames[-1];reward=0.0
        with self.lock:
            patch=None;feat_patch=None
            if pos is not None:
                patch=self._crop(frame,rect,pos)
                if patch is not None:
                    feat_patch=self._encode(patch)
            percep=self.perception.analyze(frames) if self.perception else {"detections":[],"segments":[],"clip":{},"temporal":np.zeros(512,dtype=np.float32)}
            self.last_percepts=percep
            temp_vec=np.asarray(percep.get("temporal",np.zeros(512,dtype=np.float32)),dtype=np.float32).reshape(-1)
            if temp_vec.size<512:
                temp_vec=np.pad(temp_vec,(0,512-temp_vec.size))
            elif temp_vec.size>512:
                temp_vec=temp_vec[:512]
            self.percept_temporal=self.percept_temporal*0.8+temp_vec*0.2
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            frame_feat=self._encode(rgb)
            regions=self._extract_regions(frame,rect,percep)
            state=self._make_state(frame_feat,regions)
            if self.goal_feat is None:
                self.goal_feat=frame_feat.copy()
            else:
                mix=0.2 if source==1 else 0.08;self.goal_feat=self.goal_feat*(1-mix)+frame_feat*mix;self.goal_strength=float(_clamp(self.goal_strength+(0.04 if source==1 else -0.015),0.05,1.0))
            if feat_patch is not None and action[2]>0.5 and source==1:
                self._store_button(feat_patch,pos)
            elif percep.get("detections") and source==1:
                best=max(percep["detections"],key=lambda d:d.get("conf",0.0))
                box=best.get("box")
                if box and len(box)>=4:
                    h,w=frame.shape[:2];cx=float(_clamp(((box[0]+box[2])*0.5)/max(w,1),0.0,1.0));cy=float(_clamp(((box[1]+box[3])*0.5)/max(h,1),0.0,1.0))
                    sub=frame[int(max(0,box[1])):int(min(h,box[3])),int(max(0,box[0])):int(min(w,box[2]))]
                    if sub.size>0:
                        feat=self._encode(cv2.cvtColor(sub,cv2.COLOR_BGR2RGB))
                        self._store_button(feat,(cx,cy))
            events=[]
            for key in ["score","health","message"]:
                prev=self.prev_regions.get(key)
                cur=regions[key]
                if prev is None or prev.get("sig") is None or cur.get("sig") is None:
                    continue
                delta=self._signature_delta(prev["sig"],cur["sig"])
                if key=="score" and delta>0.12:
                    diff=cur["mean"]-prev["mean"]
                    if diff>0.015:
                        events.append(("score_up",diff))
                    elif diff<-0.02:
                        events.append(("score_down",-diff))
                elif key=="health" and delta>0.1:
                    diff=cur["mean"]-prev["mean"]
                    if diff>0.02:
                        events.append(("health_up",diff))
                    elif diff<-0.015:
                        events.append(("health_down",-diff))
                elif key=="message":
                    if cur["edges"]>0.35 and cur["contrast"]>0.25 and (prev["edges"]<0.25 or cur["mean"]>prev["mean"]+0.1):
                        events.append(("victory",cur["edges"]))
                    elif prev["edges"]>0.3 and cur["edges"]<0.18 and prev["mean"]>cur["mean"]+0.1:
                        events.append(("defeat",prev["edges"]))
                    elif delta>0.18:
                        events.append(("message",delta))
            clip_map=percep.get("clip",{}) if percep else {}
            if clip_map.get("target",0.0)>0.35:
                events.append(("target",clip_map["target"]))
            if clip_map.get("danger",0.0)>0.4:
                reward-=clip_map["danger"]*0.2
            for name,cur in regions.items():
                self.prev_regions[name]=cur
            if len(frames)>1:
                diff=float(np.mean(np.abs(frames[-1].astype(np.float32)-frames[0].astype(np.float32)))/255.0);reward+=diff
            if feat_patch is not None:
                reward+=max(0.0,self._button_similarity(feat_patch)-0.4)
            reward+=max(0.0,self.goal_strength*self._goal_alignment(frame_feat))
            if self.prev_feat is not None:
                drift=self._norm(frame_feat-self.prev_feat);reward+=max(0.0,drift*0.01)
            if percep.get("detections"):
                reward+=0.01*len(percep["detections"])
            action_key=self._action_key(action)
            if self.last_state is not None:
                self._register_transition(self.last_state,state)
            self.state_counts[state]=self.state_counts.get(state,0)+1
            self.last_state=state
            for ev,val in events:
                self._log_event(ev,val,state,action_key,source)
                reward+=self.event_values.get(ev,0.0)*val
                if ev=="victory":
                    self.goal_strength=float(_clamp(self.goal_strength+0.1,0.05,1.0))
                if ev=="defeat":
                    self.goal_strength=float(_clamp(self.goal_strength-0.1,0.05,1.0))
            baseline=self.state_model.get(state,0.0)
            reward+=baseline*0.05
            if self.graph.number_of_nodes()>0:
                nodes=list(self.graph.nodes())
                feats=[]
                for st in nodes:
                    feats.append(np.array(st,dtype=np.float32))
                feats=np.stack(feats,0) if feats else np.zeros((1,6),dtype=np.float32)
                adj=np.zeros((len(nodes),len(nodes)),dtype=np.float32)
                idx_map={st:i for i,st in enumerate(nodes)}
                for u,v,data in self.graph.edges(data=True):
                    if u in idx_map and v in idx_map:
                        adj[idx_map[u],idx_map[v]]=float(data.get("weight",1.0))
                with torch.no_grad():
                    g=self.graph_reasoner(torch.from_numpy(feats),torch.from_numpy(adj))
                graph_score={nodes[i]:float(g[i]) for i in range(len(nodes))}
                reward+=graph_score.get(state,0.0)*0.1
                self.state_model[state]=self.state_model.get(state,0.0)*0.9+graph_score.get(state,0.0)*0.1
            else:
                self.state_model[state]=self.state_model.get(state,0.0)*0.95
            temporal_bonus=float(np.mean(self.percept_temporal[:128]))
            reward+=temporal_bonus*0.05
            self.prev_feat=frame_feat
        return reward
    def suggest(self,frame,rect):
        with self.lock:
            if frame is None:
                return None
            best=-1.0;pos=None
            percep=self.last_percepts
            if self.buttons:
                for btn in self.buttons:
                    patch=self._crop(frame,rect,btn["pos"])
                    if patch is None:
                        continue
                    feat=self._encode(patch);sim=self._button_similarity(feat)
                    if sim>best:
                        best=sim;pos=btn["pos"]
            if (pos is None or best<0.5) and percep and percep.get("detections"):
                det=max(percep["detections"],key=lambda d:d.get("conf",0.0))
                box=det.get("box")
                if box and len(box)>=4:
                    h,w=frame.shape[:2];cx=float(_clamp(((box[0]+box[2])*0.5)/max(w,1),0.0,1.0));cy=float(_clamp(((box[1]+box[3])*0.5)/max(h,1),0.0,1.0));pos=(cx,cy);best=max(best,float(det.get("conf",0.0)))
            rule=self._rule_action_hint(self.last_state) if self.last_state is not None else None
            if rule is not None and (pos is None or best<0.65):
                return rule
            if pos is None or best<0.6:
                return None
            dx=float(_clamp(pos[0]*2.0-1.0,-1.0,1.0));dy=float(_clamp(pos[1]*2.0-1.0,-1.0,1.0));click=float(_clamp(best,0.0,1.0));return [dx,dy,click]
    def goal_score(self,frame):
        with self.lock:
            if frame is None:
                return 0.0
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            feat=self._encode(rgb)
            percep=self.last_percepts if self.last_percepts else {"detections":[],"segments":[],"clip":{},"temporal":self.percept_temporal}
            regions=self._extract_regions(frame,None,percep)
            state=self._make_state(feat,regions)
            state_value=self.state_model.get(state,0.0)
            align=max(0.0,self._goal_alignment(feat))
            tempo=float(np.mean(self.percept_temporal[:128]))
            return float(state_value*0.5+align*0.3+tempo*0.2)
def _to_tensor(frames):
    x=torch.from_numpy(frames.astype(np.float32)/255.0);x=x.permute(0,1,4,2,3).contiguous();return x
class Agent:
    def __init__(self,seq=4,lr=1e-3):
        self.seq=seq;self.net=Net(seq).to(device);base=torch.zeros(3);
        if device=="cuda":
            base=base.to(device)
        self.log_std=nn.Parameter(base);self.opt=optim.AdamW(list(self.net.parameters())+[self.log_std],lr=lr,weight_decay=1e-4);self.sched=optim.lr_scheduler.ReduceLROnPlateau(self.opt,patience=2,factor=0.5);self.amp=device=="cuda";self.clip_eps=0.2;self.last_losses=(0,0,0,0)
    def _std(self):
        return torch.clamp(self.log_std.exp(),0.05,1.5)
    def _log_prob(self,mean,std,action):
        var=std**2
        return -0.5*(((action-mean)**2)/var+2*torch.log(std)+math.log(2*math.pi)).sum(1)
    def _entropy(self,std):
        return 0.5*torch.sum(1.0+2.0*torch.log(std)+math.log(2*math.pi),1)
    def act(self,frames,deterministic=False):
        self.net.eval();ctx=torch.amp.autocast("cuda") if self.amp else contextlib.nullcontext()
        with ctx:
            x=_to_tensor(frames[None]).to(device);mean,value=self.net(x)
        std=self._std().to(device).unsqueeze(0).expand_as(mean)
        if deterministic:
            action=mean
        else:
            noise=torch.randn_like(mean);action=mean+noise*std
        action[:,0:2]=action[:,0:2].clamp(-1.0,1.0);action[:,2]=action[:,2].clamp(0.0,1.0)
        logp=self._log_prob(mean,std,action)
        return action[0].detach().cpu().numpy(),float(logp.detach().cpu().numpy()[0]),float(value.detach().cpu().numpy()[0])
    def evaluate_action(self,frames,action):
        self.net.eval();ctx=torch.amp.autocast("cuda") if self.amp else contextlib.nullcontext();act_arr=np.array(action,dtype=np.float32)
        with ctx:
            x=_to_tensor(frames[None]).to(device);mean,value=self.net(x)
        std=self._std().to(device).unsqueeze(0).expand_as(mean);act_t=torch.from_numpy(act_arr).to(device).unsqueeze(0);act_t[:,0:2]=act_t[:,0:2].clamp(-1.0,1.0);act_t[:,2]=act_t[:,2].clamp(0.0,1.0);logp=self._log_prob(mean,std,act_t)
        return float(logp.detach().cpu().numpy()[0]),float(value.detach().cpu().numpy()[0])
    def train_step(self,replay,batches=50,batch=32,stop_flag=None,progress_sink=None,control=None):
        total=batches;done=0;pi_sum=0.0;val_sum=0.0;reg_sum=0.0;ent_sum=0.0
        while done<total:
            if stop_flag and stop_flag.is_set():
                break
            cfg=control() if control else {}
            if cfg:
                total=max(total,int(cfg.get("loops",total)))
                if cfg.get("paused"):
                    time.sleep(max(cfg.get("delay",0.1),0.05));continue
                cur_batch=int(_clamp(cfg.get("batch",batch),1,256));cur_delay=max(cfg.get("delay",0.0),0.0)
            else:
                cur_batch=batch;cur_delay=0.0
            if not replay.can_sample(cur_batch):
                time.sleep(0.05);continue
            s,a,r,m,src,lp=replay.sample(cur_batch);x=_to_tensor(s).to(device);t=torch.from_numpy(a).to(device);rew=torch.from_numpy(r).to(device);mask=torch.from_numpy(m).to(device);src_w=torch.from_numpy(1.0+src*0.5).to(device);old_lp=torch.from_numpy(lp).to(device);self.net.train();ctx=torch.amp.autocast("cuda") if self.amp else contextlib.nullcontext()
            with ctx:
                mean,v=self.net(x);std=self._std().to(device).unsqueeze(0).expand_as(mean);new_lp=self._log_prob(mean,std,t);adv=rew-v.detach();adv=(adv-adv.mean())/torch.clamp(adv.std(),min=1e-5);ratio=torch.exp(new_lp-old_lp);clip_ratio=torch.clamp(ratio,1.0-self.clip_eps,1.0+self.clip_eps);surr1=ratio*adv;surr2=clip_ratio*adv;pi_loss=-torch.mean(torch.min(surr1,surr2)*src_w*mask);v_loss=torch.mean(((rew-v)**2)*src_w*mask);bc_loss=torch.mean(((t-mean)**2).sum(1)*src_w*mask);ent=self._entropy(std).mean();hl=self.net.brain.hebbian_loss(1e-3);loss=pi_loss+0.5*v_loss+0.2*bc_loss+hl-0.01*ent
            self.opt.zero_grad()
            params=list(self.net.parameters())+[self.log_std]
            if scaler:
                scaler.scale(loss).backward();scaler.unscale_(self.opt);torch.nn.utils.clip_grad_norm_(params,1.0);scaler.step(self.opt);scaler.update()
            else:
                loss.backward();torch.nn.utils.clip_grad_norm_(params,1.0);self.opt.step()
            self.sched.step((v_loss+bc_loss).item());pi_sum+=pi_loss.item();val_sum+=v_loss.item();reg_sum+=hl.item();ent_sum+=ent.item();done+=1
            if progress_sink:
                progress_sink(int(done*100/max(total,1)))
            if cur_delay>0:
                time.sleep(cur_delay)
        if done>0:
            self.last_losses=(pi_sum/done,val_sum/done,reg_sum/done,ent_sum/done)
        try:
            torch.save(self.net.state_dict(),model_path)
        except:
            pass
class App:
    def __init__(self):
        self.root=tk.Tk();self.root.title("GameAI");self.root.protocol("WM_DELETE_WINDOW",self.quit);self.root.geometry("1000x720")
        self.sel_var=tk.StringVar();self.name_var=tk.StringVar(value="窗口名称:");self.vis_var=tk.StringVar(value="可见完整:False");self.cpu_var=tk.StringVar(value="CPU:0%");self.mem_var=tk.StringVar(value="内存:0%");self.gpu_var=tk.StringVar(value="GPU:0%");self.vram_var=tk.StringVar(value="显存:0%");self.mode_var=tk.StringVar(value="等待选择窗口");self.pct_var=tk.StringVar(value="0%")
        self.top=tk.Frame(self.root);self.top.pack(fill="x")
        self.cmb=ttk.Combobox(self.top,textvariable=self.sel_var,state="readonly",width=60);self.cmb.pack(side="left",padx=5,pady=5)
        self.btn_refresh=tk.Button(self.top,text="刷新窗口列表",command=self.refresh_windows);self.btn_refresh.pack(side="left",padx=5)
        self.btn_select=tk.Button(self.top,text="选择窗口",command=self.select_window);self.btn_select.pack(side="left",padx=5)
        self.lbl_name=tk.Label(self.top,textvariable=self.name_var);self.lbl_name.pack(side="left",padx=10)
        self.lbl_vis=tk.Label(self.top,textvariable=self.vis_var);self.lbl_vis.pack(side="left",padx=10)
        self.lbl_mode=tk.Label(self.top,textvariable=self.mode_var);self.lbl_mode.pack(side="left",padx=10)
        self.canvas=tk.Canvas(self.root,width=800,height=450,bg="black");self.canvas.pack(padx=10,pady=10)
        self.bottom=tk.Frame(self.root);self.bottom.pack(fill="x")
        self.lbl_cpu=tk.Label(self.bottom,textvariable=self.cpu_var);self.lbl_cpu.pack(side="left",padx=10)
        self.lbl_mem=tk.Label(self.bottom,textvariable=self.mem_var);self.lbl_mem.pack(side="left",padx=10)
        self.lbl_gpu=tk.Label(self.bottom,textvariable=self.gpu_var);self.lbl_gpu.pack(side="left",padx=10)
        self.lbl_vram=tk.Label(self.bottom,textvariable=self.vram_var);self.lbl_vram.pack(side="left",padx=10)
        self.btn_sleep=tk.Button(self.bottom,text="Sleep",state="disabled",command=self.sleep);self.btn_sleep.pack(side="right",padx=10)
        self.btn_up=tk.Button(self.bottom,text="Get Up",state="disabled",command=self.get_up);self.btn_up.pack(side="right",padx=10)
        self.pbar=ttk.Progressbar(self.bottom,orient="horizontal",mode="determinate",length=300,maximum=100);self.pbar.pack(side="right",padx=10)
        self.plabel=tk.Label(self.bottom,textvariable=self.pct_var);self.plabel.pack(side="right",padx=5)
        self.lbl_fw=tk.Label(self.root,text="FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.");self.lbl_fw.pack(pady=4)
        self.img_ref=None;self.selected=None;self.selected_title="";self.rect=None;self.visible=False;self.complete=False;self.stop_all=False;self.mode="idle";self.optimizing=False;self.ai_acting=False;self.last_user_time=_now();self.last_any_time=_now();self.inactive_seconds=10.0;self.frame_seq=[];self.frame_seq_raw=[];self.seq=4;self.exp=Replay(30000,self.seq);self.capture_hz=30;self.last_user_action=[0.0,0.0,0.0];self.last_user_norm=[0.5,0.5];self.action_ema=[0.0,0.0,0.0];self.action_lock=threading.Lock();self.control_lock=threading.Lock();self.train_control={"batch":32,"delay":0.05,"paused":False,"loops":80};self.ai_interval=0.06;self.progress_val=0.0;self.cpu=0.0;self.mem=0.0;self.gpu=0.0;self.vram=0.0;self.exp_writer=ExperienceWriter(exp_dir)
        self.ensure_models()
        self.perception=VisualPerception(self.seq)
        self.agent=Agent(self.seq,1e-3)
        if not os.path.exists(model_path):
            _generate_local_model(model_path,self.seq)
        try:
            self.agent.net.load_state_dict(torch.load(model_path,map_location=device))
        except:
            _generate_local_model(model_path,self.seq)
            try:
                self.agent.net.load_state_dict(torch.load(model_path,map_location=device))
            except:
                pass
        self.knowledge=KnowledgeBase(self.seq,self.perception)
        self.refresh_windows();threading.Thread(target=self.resource_loop,daemon=True).start();threading.Thread(target=self.capture_loop,daemon=True).start()
        self.k_listener=keyboard.Listener(on_press=self.on_key_press);self.k_listener.start();self.m_listener=mouse.Listener(on_move=self.on_mouse,on_click=self.on_mouse,on_scroll=self.on_mouse);self.m_listener.start();self.root.bind("<Escape>",lambda e:self.quit());self.ui_tick()
    def ensure_models(self):
        specs=[{"name":"resnet18.pth","path":pre_path,"url":pre_url,"desc":"视觉特征模型","local":lambda:_generate_resnet_local(pre_path)},{"name":"model.pt","path":model_path,"url":policy_url,"desc":"策略模型","local":lambda:_generate_local_model(model_path,self.seq)},{"name":"yolov8n.pt","path":yolo_path,"url":yolo_url,"desc":"检测模型","local":lambda:_generate_yolo_local(yolo_path)},{"name":"clip_vitb32.pt","path":clip_path,"url":"","desc":"多模态模型","local":lambda:_generate_clip_local(clip_path)},{"name":"sam_vit_b.pth","path":sam_path,"url":sam_url,"desc":"分割模型","local":lambda:_generate_sam_local(sam_path)}]
        for spec in specs:
            if os.path.exists(spec["path"]):
                continue
            if spec["url"] and self.try_download_model(spec["url"],spec["path"]):
                continue
            while not os.path.exists(spec["path"]):
                choice=self.prompt_model_decision(spec["name"],spec["desc"],bool(spec["url"])) if spec["url"] or not os.path.exists(spec["path"]) else "local"
                if choice=="retry" and spec["url"]:
                    if self.try_download_model(spec["url"],spec["path"]):
                        break
                else:
                    if spec["local"] and spec["local"]():
                        break
    def try_download_model(self,url,path):
        try:
            if url==pre_url:
                state=torch.hub.load_state_dict_from_url(url,map_location="cpu",progress=False,model_dir=models_dir)
                torch.save(state,path)
                return True
        except:
            pass
        return _download_file(url,path,45)
    def prompt_model_decision(self,name,desc,allow_retry=True):
        choice=tk.StringVar(value="retry" if allow_retry else "local")
        win=tk.Toplevel(self.root);win.title("模型获取");win.geometry("360x200")
        tk.Label(win,text=f"{desc}{name}缺失或下载失败，请选择操作").pack(pady=20)
        def decide(val):
            choice.set(val)
            try:
                win.destroy()
            except:
                pass
        btn_retry=tk.Button(win,text="重试下载",state="normal" if allow_retry else "disabled",command=lambda:decide("retry"))
        btn_retry.pack(pady=5)
        tk.Button(win,text="本地生成",command=lambda:decide("local")).pack(pady=5)
        win.transient(self.root)
        win.grab_set()
        self.root.update_idletasks()
        self.root.wait_variable(choice)
        return choice.get()
    def list_windows(self):
        try:
            wins=[w for w in gw.getAllWindows() if w.title];return [f"{w.title} | {getattr(w,'_hWnd',0)}" for w in wins]
        except:
            return []
    def refresh_windows(self):
        self.cmb["values"]=self.list_windows()
    def select_window(self):
        s=self.sel_var.get()
        if "|" not in s:
            return
        title,hwnd=s.split("|",1);self.selected=int(hwnd.strip()) if hwnd.strip().isdigit() else None;self.selected_title=title.strip();self.name_var.set(f"窗口名称:{self.selected_title}");self.mode="learn";self.mode_var.set("学习模式");self.btn_sleep.config(state="normal");self.btn_up.config(state="disabled");self.last_user_time=_now()
    def check_visible_complete(self):
        try:
            if not self.selected:
                return False,False,(0,0,0,0)
            w=None
            for cand in gw.getAllWindows():
                if getattr(cand,"_hWnd",None)==self.selected:
                    w=cand;break
            if w is None:
                return False,False,(0,0,0,0)
            vis=getattr(w,"isVisible",True);minimized=getattr(w,"isMinimized",False)
            if minimized:
                return False,False,(0,0,0,0)
            bx=w.box;left,top,width,height=bx.left,bx.top,bx.width,bx.height;right=left+width;bottom=top+height
            mons=get_monitors();mx=min(m.x for m in mons) if mons else 0;my=min(m.y for m in mons) if mons else 0;mw=max((m.x+m.width) for m in mons) if mons else 0;mh=max((m.y+m.height) for m in mons) if mons else 0
            if mw==0 or mh==0:
                sw,sh=pyautogui.size();mw=max(mw,sw);mh=max(mh,sh)
            comp=(left>=mx and top>=my and right<=mw and bottom<=mh and getattr(w,"isActive",True))
            return bool(vis),bool(comp),(left,top,right,bottom)
        except:
            return False,False,(0,0,0,0)
    def grab_frame(self,rect):
        x1,y1,x2,y2=rect;w=max(1,x2-x1);h=max(1,y2-y1)
        with mss.mss() as sct:
            mon={"left":int(x1),"top":int(y1),"width":int(w),"height":int(h)};img=np.array(sct.grab(mon))[:,:,:3]
        return img
    def resource_loop(self):
        while not self.stop_all:
            c,m,g,vr=_sys_usage();self.cpu=c;self.mem=m;self.gpu=g;self.vram=vr;M=max(c,m,g,vr);Mp=M/100.0;self.capture_hz=max(0,int(round(100*(1.0-Mp))))
            with self.control_lock:
                if Mp>=0.95:
                    self.train_control.update({"paused":True,"delay":0.25,"batch":16,"loops":40});self.ai_interval=0.12
                elif Mp>=0.85:
                    self.train_control.update({"paused":False,"delay":0.12,"batch":24,"loops":60});self.ai_interval=0.09
                elif Mp<=0.35:
                    self.train_control.update({"paused":False,"delay":0.0,"batch":56,"loops":120});self.ai_interval=0.04
                elif Mp<=0.6:
                    self.train_control.update({"paused":False,"delay":0.04,"batch":40,"loops":100});self.ai_interval=0.05
                else:
                    self.train_control.update({"paused":False,"delay":0.07,"batch":32,"loops":80});self.ai_interval=0.06
            time.sleep(0.5)
    def on_mouse(self,*args):
        self.last_user_time=_now();self.last_any_time=_now()
        try:
            if not (self.selected and self.visible and self.complete and self.rect):
                return
            if len(args)==2:
                x,y=args;clicked=0.0
            elif len(args)==4:
                x,y=args[0],args[1];clicked=1.0 if isinstance(args[2],mouse.Button) and bool(args[3]) else 0.0
            else:
                return
            x1,y1,x2,y2=self.rect
            if x1<=x<=x2 and y1<=y<=y2:
                cx=(x1+x2)/2.0;cy=(y1+y2)/2.0;hw=max(1,(x2-x1)/2.0);hh=max(1,(y2-y1)/2.0);dx=float(_clamp((x-cx)/hw,-1.0,1.0));dy=float(_clamp((y-cy)/hh,-1.0,1.0));nx=float(_clamp((x-x1)/max(1,x2-x1),0.0,1.0));ny=float(_clamp((y-y1)/max(1,y2-y1),0.0,1.0))
                with self.action_lock:
                    self.last_user_action=[dx,dy,clicked];self.last_user_norm=[nx,ny]
        except:
            pass
    def get_user_norm(self):
        with self.action_lock:
            return tuple(self.last_user_norm)
    def action_to_norm(self,action):
        return (float(_clamp(action[0]*0.5+0.5,0.0,1.0)),float(_clamp(action[1]*0.5+0.5,0.0,1.0)))
    def on_key_press(self,k):
        self.last_any_time=_now()
        try:
            if k==keyboard.Key.esc:
                self.quit();return False
            if self.mode=="train":
                self.mode="learn"
        except:
            pass
    def sleep(self):
        if self.optimizing or self.mode=="idle":
            return
        self.optimizing=True;self.btn_sleep.config(state="disabled");self.btn_up.config(state="normal");self.mode="opt";self.mode_var.set("优化中");self.progress_val=0;self.opt_stop=threading.Event();threading.Thread(target=self.optimize_thread,daemon=True).start()
    def get_up(self):
        if not self.optimizing:
            return
        self.opt_stop.set();self.progress_val=0;self.btn_up.config(state="disabled");self.btn_sleep.config(state="normal");self.mode="learn";self.mode_var.set("学习模式");self.optimizing=False
    def get_train_control(self):
        with self.control_lock:
            return dict(self.train_control)
    def optimize_thread(self):
        cfg=self.get_train_control();steps=cfg.get("loops",80);base_batch=cfg.get("batch",32)
        def sink(p):
            self.progress_val=p
        self.agent.train_step(self.exp,batches=steps,batch=base_batch,stop_flag=self.opt_stop,progress_sink=sink,control=lambda:self.get_train_control())
        if not self.opt_stop.is_set():
            self.root.after(0,self.opt_done_dialog)
        else:
            self.optimizing=False
    def opt_done_dialog(self):
        d=tk.Toplevel(self.root);d.title("提示");d.geometry("300x120");tk.Label(d,text="优化完成").pack(pady=10);tk.Button(d,text="Confirm",command=lambda:self._confirm_done(d)).pack(pady=10)
    def _confirm_done(self,d):
        try:
            d.destroy()
        except:
            pass
        self.progress_val=0;self.btn_sleep.config(state="normal");self.btn_up.config(state="disabled");self.mode="learn";self.mode_var.set("学习模式");self.optimizing=False
    def capture_loop(self):
        while not self.stop_all:
            if self.selected:
                vis,comp,rect=self.check_visible_complete();self.visible,self.complete,self.rect=vis,comp,rect
                if not (vis and comp):
                    if self.mode!="opt":
                        self.mode="learn"
                    time.sleep(0.2);continue
                hz=max(int(getattr(self,"capture_hz",30)),0)
                if hz==0:
                    time.sleep(1.0);continue
                t=1.0/max(hz,1)
                try:
                    img=self.grab_frame(rect);self.last_frame=img
                    if not self.optimizing and self.mode in ["learn","train"]:
                        if len(self.frame_seq)>=self.seq:
                            self.frame_seq.pop(0);self.frame_seq_raw.pop(0)
                        self.frame_seq_raw.append(img.copy());self.frame_seq.append(cv2.resize(img,(160,120)))
                        if len(self.frame_seq)==self.seq:
                            frames_stack=np.stack(self.frame_seq,0)
                            if self.mode=="learn":
                                act=self.read_user_action();pos=self.get_user_norm();logp,_=self.agent.evaluate_action(frames_stack,act);reward=self.knowledge.update(self.frame_seq_raw,self.rect,act,pos,1);self.exp.push(self.frame_seq[-1],act,1,reward,logp=logp);self.exp_writer.record(self.frame_seq_raw[-1] if self.frame_seq_raw else None,act,pos,1,self.rect,self.selected_title,"learn")
                            elif self.mode=="train":
                                act,logp=self.ai_action_and_apply();pos=self.action_to_norm(act);reward=self.knowledge.update(self.frame_seq_raw,self.rect,act,pos,0);self.exp.push(self.frame_seq[-1],act,0,reward,logp=logp);self.exp_writer.record(self.frame_seq_raw[-1] if self.frame_seq_raw else None,act,pos,0,self.rect,self.selected_title,"train")
                    if not self.optimizing and self.mode=="learn" and (_now()-self.last_any_time)>self.inactive_seconds and (self.visible and self.complete):
                        self.mode="train";self.start_ai_thread()
                except:
                    pass
                time.sleep(t)
            else:
                time.sleep(0.2)
    def read_user_action(self):
        try:
            with self.action_lock:
                dx,dy,cl=self.last_user_action
            a=0.2;self.action_ema=[a*dx+(1-a)*self.action_ema[0],a*dy+(1-a)*self.action_ema[1],max(cl,self.action_ema[2]*0.8)];return [float(_clamp(self.action_ema[0],-1,1)),float(_clamp(self.action_ema[1],-1,1)),float(_clamp(self.action_ema[2],0,1))]
        except:
            return [0.0,0.0,0.0]
    def ai_action_and_apply(self):
        if len(self.frame_seq)<self.seq:
            return [0.0,0.0,0.0],0.0
        frames=np.stack(self.frame_seq,0)
        act,_,_=self.agent.act(frames)
        a=np.array(act,dtype=np.float32)
        try:
            suggest=self.knowledge.suggest(self.frame_seq_raw[-1] if self.frame_seq_raw else None,self.rect)
        except:
            suggest=None
        if suggest is not None:
            a=0.6*a+0.4*np.array(suggest,dtype=np.float32)
        if self.frame_seq_raw:
            try:
                goal_boost=self.knowledge.goal_score(self.frame_seq_raw[-1]);a[2]=float(_clamp(a[2]+goal_boost*0.2,0.0,1.0))
            except:
                pass
        logp_val,_=self.agent.evaluate_action(frames,a.tolist())
        if self.mode!="train" or not (self.visible and self.complete):
            return a.tolist(),logp_val
        x1,y1,x2,y2=self.rect;cx=(x1+x2)//2;cy=(y1+y2)//2;dx=int(a[0]*20);dy=int(a[1]*20);px=cx+dx;py=cy+dy
        try:
            pyautogui.moveTo(px,py,duration=0.01)
            if a[2]>0.5:
                pyautogui.click()
        except:
            pass
        return a.tolist(),logp_val
    def start_ai_thread(self):
        if getattr(self,"ai_acting",False):
            return
        self.ai_acting=True
        def run():
            while self.mode=="train" and not self.stop_all:
                if not (self.visible and self.complete):
                    time.sleep(0.1);continue
                if len(self.frame_seq)==self.seq:
                    self.ai_action_and_apply()
                time.sleep(max(0.01,self.ai_interval))
            self.ai_acting=False
        threading.Thread(target=run,daemon=True).start()
    def ui_tick(self):
        try:
            if getattr(self,"last_frame",None) is not None:
                img=cv2.cvtColor(self.last_frame,cv2.COLOR_BGR2RGB);h,w,_=img.shape;scale=min(800/max(w,1),450/max(h,1));im=cv2.resize(img,(max(1,int(w*scale)),max(1,int(h*scale))));im=Image.fromarray(im);self.img_ref=ImageTk.PhotoImage(im);self.canvas.delete("all");self.canvas.create_image(400,225,image=self.img_ref)
            self.cpu_var.set(f"CPU:{self.cpu:.0f}%");self.mem_var.set(f"内存:{self.mem:.0f}%");self.gpu_var.set(f"GPU:{self.gpu:.0f}%");self.vram_var.set(f"显存:{self.vram:.0f}%");self.vis_var.set(f"可见完整:{str(self.visible and self.complete)}");self.pbar["value"]=int(self.progress_val);self.pct_var.set(f"{int(self.progress_val)}%")
            if self.mode=="learn":
                self.mode_var.set("学习模式")
            elif self.mode=="train":
                self.mode_var.set("训练模式")
            elif self.mode=="opt":
                self.mode_var.set("优化中")
        except:
            pass
        self.root.after(30,self.ui_tick)
    def quit(self):
        self.stop_all=True
        try:
            self.root.destroy()
        except:
            pass
        os._exit(0)
    def run(self):
        self.root.mainloop()
if __name__=="__main__":
    App().run()
