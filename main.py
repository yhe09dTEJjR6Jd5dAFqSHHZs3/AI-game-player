import os,sys,threading,time,queue,math,random,ctypes,json,subprocess,ctypes.wintypes,webbrowser,hashlib,datetime,re,traceback
import psutil,pyautogui
try:
 import torch,torch.nn as nn,torch.optim as optim,torch.nn.functional as F
except OSError as e:
 import tkinter as _tk
 from tkinter import messagebox as _mb
 try:
  _r=_tk.Tk();_r.withdraw()
  _mb.showerror('PyTorch错误','检测到CUDA版PyTorch但缺少依赖\n请安装CPU版PyTorch后重试:\nhttps://pytorch.org/get-started/locally/\n建议命令: pip install torch --index-url https://download.pytorch.org/whl/cpu')
 except Exception:
  pass
 sys.exit(1)
import GPUtil,cv2,numpy as np,mss,requests,pygetwindow as gw
from pynput import mouse,keyboard
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk,messagebox
from screeninfo import get_monitors
import os,sys,threading,time,queue,math,random,ctypes,json,subprocess,ctypes.wintypes,webbrowser,hashlib,datetime,re,traceback
from pynput import mouse,keyboard
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk,messagebox
from screeninfo import get_monitors
torch.backends.cudnn.benchmark=True
OCCLUSION_THR=0.01
WS_EX_TRANSPARENT=0x00000020
WS_EX_LAYERED=0x00080000
LWA_ALPHA=0x2
GWL_EXSTYLE=-20
GW_HWNDPREV=3
PW_RENDERFULLCONTENT=0x00000002
pyautogui.FAILSAFE=False
pyautogui.PAUSE=0.0
pyautogui.MINIMUM_DURATION=0.0
if os.name=="nt":
    try:
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except Exception:
        try:
            ctypes.windll.user32.SetProcessDPIAware()
        except Exception:
            pass
    try:
        dwmapi=ctypes.windll.dwmapi
    except Exception:
        dwmapi=None
    user32=ctypes.windll.user32
    gdi32=ctypes.windll.gdi32
else:
    dwmapi=None
    user32=None
    gdi32=None
home=os.path.expanduser("~")
base_dir=os.path.join(home,"Desktop","GameAI")
models_dir=os.path.join(base_dir,"models")
experience_dir=os.path.join(base_dir,"experience")
os.makedirs(models_dir,exist_ok=True)
os.makedirs(experience_dir,exist_ok=True)
model_path=os.path.join(models_dir,"policy.pt")
resnet_path=os.path.join(models_dir,"resnet18-f37072fd.pth")
yolo_path=os.path.join(models_dir,"yolov8n.pt")
sam_path=os.path.join(models_dir,"sam_vit_b_01ec64.pth")
device="cuda" if torch.cuda.is_available() else "cpu"
scaler=torch.amp.GradScaler("cuda") if device=="cuda" else None
try:
    from pynvml import nvmlInit,nvmlDeviceGetCount,nvmlDeviceGetHandleByIndex,nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo,nvmlDeviceGetTemperature,nvmlDeviceGetPowerUsage,nvmlDeviceGetEnforcedPowerLimit
    nvmlInit();_gpu_count=nvmlDeviceGetCount()
except Exception:
    _gpu_count=0
def _rect_intersection(a,b):
    l=max(a[0],b[0]);t=max(a[1],b[1]);r=min(a[2],b[2]);btm=min(a[3],b[3])
    if r>l and btm>t:
        return (l,t,r,btm)
    return None
def _rect_area(rc):
    return max(0,rc[2]-rc[0])*max(0,rc[3]-rc[1])
def _rect_union_area(rects):
    if not rects:
        return 0
    xs=sorted({r[0] for r in rects}|{r[2] for r in rects})
    area=0
    for i in range(len(xs)-1):
        x1=xs[i];x2=xs[i+1]
        if x2<=x1:
            continue
        seg=[]
        for l,t,r,b in rects:
            if l<x2 and r>x1:
                seg.append((t,b))
        if not seg:
            continue
        seg.sort()
        cs,ce=seg[0]
        total=0
        for s,e in seg[1:]:
            if e<=cs:
                continue
            if s>ce:
                total+=ce-cs;cs,ce=s,e
            else:
                if e>ce:ce=e
        total+=ce-cs
        area+=total*(x2-x1)
    return area
def _poly_simplify(points,eps=2.0):
    if len(points)<3:
        return points
    pts=np.array(points,dtype=np.float32).reshape(-1,1,2)
    eps=float(eps)
    simp=cv2.approxPolyDP(pts,eps,False).reshape(-1,2).tolist()
    return simp if len(simp)>=2 else points
def _now_ms():
    return int(time.time()*1000)
class UIElementPool:
    def __init__(self,embedder=None,pre=None):
        self.embedder=embedder
        self.pre=pre
        self.lock=threading.Lock()
        self.protos=[]
        self.next_id=1
        self.last_state=""
        self.transitions={}
    def _embed(self,patch):
        try:
            if patch is None or patch.size==0:
                return None
            import torchvision.transforms as T
            if self.embedder is not None and self.pre is not None:
                img=Image.fromarray(cv2.cvtColor(patch,cv2.COLOR_BGR2RGB))
                x=self.pre(img).unsqueeze(0).to(device)
                with torch.no_grad():
                    e=self.embedder.encode_image(x).float().detach().cpu().numpy().reshape(-1)
                n=np.linalg.norm(e)+1e-8
                return e/n
        except Exception:
            pass
        try:
            hist=[]
            for ch in cv2.split(patch):
                h=cv2.calcHist([ch],[0],None,[64],[0,256]).flatten()
                hist.append(h)
            e=np.concatenate(hist,0).astype(np.float32)
            e/=np.linalg.norm(e)+1e-8
            return e
        except Exception:
            return None
    def _nn(self,emb):
        if emb is None:
            return -1,0.0
        best=-1;bestsim=-1.0
        with self.lock:
            for i,p in enumerate(self.protos):
                sim=float(np.dot(emb,p["emb"]))
                if sim>bestsim:
                    bestsim=sim;best=i
        return best,bestsim
    def _extract_patch(self,frame,rect,nx,ny):
        H,W=frame.shape[:2];x=int(nx*W);y=int(ny*H);sz=max(16,min(H,W)//10);x1=max(0,x-sz);y1=max(0,y-sz);x2=min(W,x+sz);y2=min(H,y+sz);return frame[y1:y2,x1:x2]
    def add_click(self,frame,rect,nx,ny):
        patch=self._extract_patch(frame,rect,nx,ny);emb=self._embed(patch)
        with self.lock:
            idx,sim=self._nn(emb)
            if idx<0 or sim<0.8:
                pid=self.next_id;self.next_id+=1;self.protos.append({"id":pid,"emb":emb,"count":1})
            else:
                pid=self.protos[idx]["id"];self.protos[idx]["emb"]=((self.protos[idx]["emb"]*self.protos[idx]["count"])+emb)/(self.protos[idx]["count"]+1e-8);self.protos[idx]["emb"]/=np.linalg.norm(self.protos[idx]["emb"])+1e-8;self.protos[idx]["count"]+=1
        return pid
    def add_drag(self,frame,rect,path):
        if not path:
            return -1
        nx,ny=path[-1]
        return self.add_click(frame,rect,nx,ny)
    def update_state(self,state_text):
        st=(state_text or "").strip()
        if st and self.last_state and st!=self.last_state:
            key=(self.last_state,"*")
            self.transitions[key]=self.transitions.get(key,0)+1
        if st:
            self.last_state=st
    def matrix(self):
        with self.lock:
            if not self.protos:
                return None,{}
            M=np.stack([p["emb"] for p in self.protos],0).astype(np.float32)
            ids=[p["id"] for p in self.protos]
        T=torch.from_numpy(M)
        T=T/torch.clamp(T.norm(dim=1,keepdim=True),min=1e-6)
        id2idx={pid:i for i,pid in enumerate(ids)}
        return T.to(device),id2idx
class ExperienceWriter:
    def __init__(self,root_dir,max_bytes=8589934592):
        self.root_dir=root_dir
        self.learn_dir=os.path.join(root_dir,"learn")
        self.train_dir=os.path.join(root_dir,"train")
        os.makedirs(self.learn_dir,exist_ok=True)
        os.makedirs(self.train_dir,exist_ok=True)
        self.lock=threading.Lock()
        self.max_bytes=int(max_bytes)
    def _subdir(self,base,ts_ms):
        dt=datetime.datetime.fromtimestamp(ts_ms/1000.0)
        d=os.path.join(base,dt.strftime("%Y%m%d"),dt.strftime("%H"))
        os.makedirs(d,exist_ok=True)
        return d
    def _ensure_quota(self):
        try:
            total=0
            files=[]
            for root,_,fns in os.walk(self.root_dir):
                for fn in fns:
                    if fn.endswith(".npz"):
                        p=os.path.join(root,fn)
                        try:
                            s=os.path.getsize(p)
                            total+=s
                            files.append((p,os.path.getmtime(p),s))
                        except Exception:
                            pass
            if total>self.max_bytes:
                files.sort(key=lambda x:x[1])
                i=0
                while total>self.max_bytes and i<len(files):
                    p,_,s=files[i]
                    try:
                        os.remove(p)
                        total-=s
                    except Exception:
                        pass
                    i+=1
        except Exception:
            pass
    def record(self,frame,action,pos,source,rect,title,mode,event="",extra=None):
        if frame is None or rect is None:
            return
        arr=np.asarray(frame,dtype=np.uint8)
        norm_action=[float(action[0]),float(action[1]),float(action[2])]
        if isinstance(pos,(list,tuple)) and len(pos)==2:
            norm_pos=[float(pos[0]),float(pos[1])]
        else:
            norm_pos=[0.0,0.0]
        data={"time":int(time.time()*1000),"action":norm_action,"pos":norm_pos,"source":int(source),"rect":[int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3])],"window":title or "","mode":mode,"event":event,"extra":extra if extra is not None else {}}
        base=self.learn_dir if int(source)==1 else self.train_dir
        target_dir=self._subdir(base,data["time"])
        os.makedirs(target_dir,exist_ok=True)
        name=os.path.join(target_dir,f"{data['time']}_{int(source)}.npz")
        with self.lock:
            try:
                ok,buf=cv2.imencode(".jpg",arr,[int(cv2.IMWRITE_JPEG_QUALITY),85])
                if ok:
                    np.savez_compressed(name,frame_jpg=buf,meta=json.dumps(data,ensure_ascii=False))
                else:
                    np.savez_compressed(name,frame=arr,meta=json.dumps(data,ensure_ascii=False))
                self._ensure_quota()
            except Exception:
                pass
class ReplayBuffer:
    def __init__(self,cap=50000):
        self.cap=cap
        self.data=[]
        self.lock=threading.Lock()
    def add(self,frame,action,source,event_id=0,atype=0,ctrl=-1,txt=""):
        with self.lock:
            if len(self.data)>=self.cap:
                self.data.pop(0)
            self.data.append((frame.astype(np.uint8),np.array(action,dtype=np.float32),int(source),int(event_id),int(atype),int(ctrl),str(txt)))
    def size(self):
        with self.lock:
            return len(self.data)
    def clear(self):
        with self.lock:
            self.data.clear()
    def sample(self,batch,seq=4):
        with self.lock:
            if len(self.data)<seq+batch:
                return None
            idx=[random.randint(0,len(self.data)-seq) for _ in range(batch)]
            frames=[];actions=[];sources=[];events=[];atypes=[];ctrls=[];txts=[]
            for i in idx:
                seq_frames=[self.data[i+j][0] for j in range(seq)]
                frames.append(np.stack(seq_frames,0))
                actions.append(self.data[i+seq-1][1])
                sources.append(self.data[i+seq-1][2])
                events.append(self.data[i+seq-1][3])
                atypes.append(self.data[i+seq-1][4])
                ctrls.append(self.data[i+seq-1][5])
                txts.append(self.data[i+seq-1][6])
            return np.stack(frames,0),np.stack(actions,0),np.array(sources),np.array(events),np.array(atypes),np.array(ctrls),txts
class DiskExperienceDataset(torch.utils.data.Dataset):
    def __init__(self,root_dir,seq=4,aug=True):
        self.files=[]
        for sub in ("learn","train"):
            d=os.path.join(root_dir,sub)
            if os.path.isdir(d):
                for root,_,fns in os.walk(d):
                    for fn in fns:
                        if fn.endswith(".npz"):
                            self.files.append(os.path.join(root,fn))
        self.files.sort()
        self.seq=seq
        self.aug=aug
    def __len__(self):
        return max(1,len(self.files))
    def _aug(self,img):
        if not self.aug:
            return img
        h,w=img.shape[:2]
        sx=int(w*0.9)
        sy=int(h*0.9)
        x=random.randint(0,max(0,w-sx))
        y=random.randint(0,max(0,h-sy))
        img=img[y:y+sy,x:x+sx]
        img=cv2.resize(img,(w,h))
        if random.random()<0.5:
            img=cv2.flip(img,1)
        if random.random()<0.5:
            img=cv2.GaussianBlur(img,(3,3),0)
        if random.random()<0.5:
            g=random.uniform(0.9,1.1);b=random.uniform(-10,10)
            img=np.clip(img.astype(np.float32)*g+b,0,255).astype(np.uint8)
        return img
    def _encode_path(self,extra):
        if not isinstance(extra,dict):
            return np.zeros((64,),np.float32)
        path=extra.get("path",[])
        if not path:
            return np.zeros((64,),np.float32)
        pts=np.array(path,dtype=np.float32)
        pts=np.clip(pts,0.0,1.0)
        if len(pts)<2:
            pts=np.vstack([pts,pts])
        L=32
        idxs=np.linspace(0,len(pts)-1,L).astype(np.int32)
        smp=pts[idxs]
        smp=np.diff(np.vstack([smp[:1],smp]),axis=0)
        vec=smp.reshape(-1)
        if vec.shape[0]<64:
            pad=np.zeros((64-vec.shape[0],),np.float32)
            vec=np.concatenate([vec,pad],0)
        return vec.astype(np.float32)[:64]
    def __getitem__(self,idx):
        if len(self.files)==0:
            blank=np.zeros((256,256,3),np.uint8)
            seq=np.stack([blank]*self.seq,0)
            act=np.zeros(3,np.float32)
            return seq,act,np.array(0,dtype=np.int64),np.array(-1,dtype=np.int64),np.zeros((64,),np.float32),"",0
        i=random.randint(0,len(self.files)-1)
        try:
            arr=np.load(self.files[i],allow_pickle=True)
            if "frame_jpg" in arr.files:
                frame=cv2.imdecode(arr["frame_jpg"],cv2.IMREAD_COLOR)
            else:
                frame=arr["frame"]
            meta=json.loads(str(arr["meta"]))
            act=np.array(meta.get("action",[0,0,0]),dtype=np.float32)
            ev=str(meta.get("event",""))
            extra=meta.get("extra",{})
            atype=0
            if ev=="press":
                atype=1
            elif ev=="drag":
                atype=2
            ctrl=int(extra.get("proto",-1)) if isinstance(extra,dict) else -1
            path_vec=self._encode_path(extra)
            text=str(meta.get("event_text",""))
            success=1 if any(k in text.lower() for k in ["win","victory","success","complete"]) else 0
        except Exception:
            frame=np.zeros((256,256,3),np.uint8);act=np.zeros(3,np.float32);atype=0;ctrl=-1;path_vec=np.zeros((64,),np.float32);text="";success=0
        seq=np.stack([self._aug(frame.copy()) for _ in range(self.seq)],0)
        return seq,act,np.array(atype,dtype=np.int64),np.array(ctrl,dtype=np.int64),path_vec,text,success
class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv=nn.Sequential(nn.Conv2d(12,32,5,2,2),nn.ReLU(),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.AdaptiveAvgPool2d((8,8)))
        self.fc=nn.Linear(128*8*8,256)
    def forward(self,x):
        b,t,c,h,w=x.shape
        x=x.reshape(b,t*c,h,w)
        x=self.conv(x)
        x=x.view(b,-1)
        return torch.tanh(self.fc(x))
class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=Encoder()
        self.policy=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,3))
        self.value=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,1))
        self.atype=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,3))
        self.elem=nn.Linear(256,128)
        self.path_head=nn.Sequential(nn.Linear(256,128),nn.ReLU(),nn.Linear(128,64))
    def forward(self,x):
        emb=self.encoder(x)
        logits=self.policy(emb)
        value=self.value(emb).squeeze(-1)
        atype_logits=self.atype(emb)
        elem_emb=F.normalize(self.elem(emb),dim=1)
        path_pred=self.path_head(emb)
        dx=torch.tanh(logits[:,0])
        dy=torch.tanh(logits[:,1])
        click=torch.sigmoid(logits[:,2])
        return torch.stack([dx,dy,click],1),value,atype_logits,elem_emb,path_pred
def generate_policy(path):
    try:
        net=PolicyNet()
        torch.save(net.state_dict(),path)
        return True
    except Exception:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception:
            pass
        return False
class ModelSpec:
    def __init__(self,name,path,urls,generator,sha256=None,expected_size=None,max_retries=3):
        self.name=name
        self.path=path
        self.urls=urls if isinstance(urls,(list,tuple)) else ([urls] if urls is not None else [])
        self.generator=generator
        self.sha256=sha256
        self.expected_size=expected_size
        self.max_retries=max_retries
class ModelManager:
    def __init__(self,app):
        self.app=app
        self.specs=[ModelSpec("policy",model_path,None,generate_policy,None,None,0),ModelSpec("resnet18",resnet_path,"https://download.pytorch.org/models/resnet18-f37072fd.pth",None,None,None,3),ModelSpec("yolov8n",yolo_path,["https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt","https://cdn.jsdelivr.net/gh/ultralytics/assets@main/yolov8/yolov8n.pt"],None,None,None,3),ModelSpec("sam_vit_b",sam_path,["https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth","https://huggingface.co/facebook/sam/resolve/main/sam_vit_b_01ec64.pth?download=true"],None,None,None,3)]
        self.retry_counts={}
    def ensure(self):
        threading.Thread(target=self._worker,daemon=True).start()
    def _sha256(self,path):
        try:
            h=hashlib.sha256()
            with open(path,"rb") as f:
                for chunk in iter(lambda:f.read(1024*1024),b""):
                    h.update(chunk)
            return h.hexdigest()
        except Exception:
            return ""
    def _validate(self,spec):
        try:
            if spec.name=="resnet18":
                _=torch.load(spec.path,map_location="cpu");return True
            if spec.name=="yolov8n":
                from ultralytics import YOLO;_=YOLO(spec.path);return True
            if spec.name=="sam_vit_b":
                from segment_anything import sam_model_registry;_=sam_model_registry.get("vit_b")(checkpoint=spec.path);return True
            return True
        except Exception:
            try:
                os.remove(spec.path)
            except Exception:
                pass
            return False
    def _alert(self,msg):
        try:
            messagebox.showerror("Error",msg)
        except Exception:
            pass
    def _download(self,url,path,max_retries):
        tmp=path+".download"
        try:
            from requests.adapters import HTTPAdapter
            try:
                from urllib3.util.retry import Retry
            except Exception:
                from urllib3.util import Retry
            sess=requests.Session()
            retry=Retry(total=max(0,int(max_retries)),read=max(0,int(max_retries)),connect=max(0,int(max_retries)),backoff_factor=1,status_forcelist=[429,500,502,503,504],allowed_methods=frozenset(["GET","HEAD"]))
            sess.mount("http://",HTTPAdapter(max_retries=retry))
            sess.mount("https://",HTTPAdapter(max_retries=retry))
            headers={}
            exist=0
            if os.path.exists(tmp):
                try:
                    exist=os.path.getsize(tmp)
                    if exist>0:
                        headers["Range"]=f"bytes={exist}-"
                except Exception:
                    exist=0
            t0=time.time();bytes0=exist
            with sess.get(url,timeout=45,stream=True,headers=headers) as r:
                if r.status_code in (200,206):
                    mode="ab" if "Range" in headers and r.status_code==206 else "wb"
                    if mode=="wb" and os.path.exists(tmp):
                        try:os.remove(tmp)
                        except Exception:pass
                    with open(tmp,mode) as f:
                        for chunk in r.iter_content(65536):
                            if chunk:
                                f.write(chunk)
                                t=time.time()
                                if t-t0>=1.0:
                                    sz=os.path.getsize(tmp);spd=(sz-bytes0)/(t-t0);self.app.schedule(lambda s=int(spd):self.app.pause_var.set(f"下载中 {os.path.basename(path)} {s}B/s(超时45秒)"));t0=t;bytes0=sz
                    size=os.path.getsize(tmp)
                    if size<=0:
                        try:os.remove(tmp)
                        except Exception:pass
                        return False,"下载大小为0"
                    os.replace(tmp,path)
                    return True,""
                else:
                    return False,f"HTTP {r.status_code}"
        except Exception as e:
            try:
                if os.path.exists(tmp):os.remove(tmp)
            except Exception:
                pass
            return False,str(e)
    def prompt_retry_or_local(self,name,url,reason,path_hint):
        result=queue.Queue(maxsize=1)
        def dialog():
            top=tk.Toplevel(self.app.root);top.title(f"Model {name}")
            ttk.Label(top,text=f"下载 {name} 失败：{reason}").grid(row=0,column=0,columnspan=3,sticky="w")
            ttk.Label(top,text=f"可手动下载链接:").grid(row=1,column=0,sticky="w")
            link=tk.Entry(top);link.insert(0,url);link.configure(state="readonly");link.grid(row=1,column=1,columnspan=2,sticky="ew")
            ttk.Label(top,text=f"请将文件放到:").grid(row=2,column=0,sticky="w")
            path=tk.Entry(top);path.insert(0,path_hint);path.configure(state="readonly");path.grid(row=2,column=1,columnspan=2,sticky="ew")
            ttk.Label(top,text=f"超时45秒，最大重试{3}次").grid(row=3,column=0,columnspan=3,sticky="w")
            top.columnconfigure(1,weight=1)
            def choose(val):
                if result.empty():result.put(val);top.destroy()
            ttk.Button(top,text="Open Link",command=lambda:choose("open")).grid(row=4,column=0,sticky="ew")
            ttk.Button(top,text="Retry",command=lambda:choose("retry")).grid(row=4,column=1,sticky="ew")
            ttk.Button(top,text="Local",command=lambda:choose("local")).grid(row=4,column=2,sticky="ew")
            top.grab_set();top.protocol("WM_DELETE_WINDOW",lambda:choose("retry"))
        self.app.schedule(dialog);return result.get()
    def _worker(self):
        for spec in self.specs:
            if os.path.exists(spec.path):
                if spec.sha256:
                    try:
                        if self._sha256(spec.path)!=spec.sha256:
                            try:os.remove(spec.path)
                            except Exception:pass
                    except Exception:
                        pass
                if os.path.exists(spec.path):
                    continue
            done=False;attempt=0;delay=2.0
            while not done and self.app.running and (spec.max_retries==0 or attempt<spec.max_retries):
                attempt+=1
                if spec.urls==[]:
                    try:
                        if spec.generator and spec.generator(spec.path):
                            done=True;break
                        else:
                            raise RuntimeError("本地生成失败")
                    except Exception as e:
                        reason=str(e);self.app.schedule(lambda n=spec.name:self._alert(f"{n} 生成失败:{reason}"));break
                ok_any=False;last_reason=""
                for url in spec.urls:
                    try:
                        self.app.schedule(lambda n=spec.name,a=attempt,m=spec.max_retries:self.app.pause_var.set(f"正在下载 {n} 第{a}次尝试"))
                        ok,reason=self._download(url,spec.path,spec.max_retries)
                        if ok:
                            if spec.expected_size and os.path.getsize(spec.path)!=spec.expected_size:
                                raise RuntimeError("文件大小校验失败")
                            if spec.sha256 and self._sha256(spec.path)!=spec.sha256:
                                raise RuntimeError("哈希校验失败")
                            if not self._validate(spec):
                                last_reason="文件校验失败";ok=False
                            else:
                                ok_any=True;break
                        else:
                            last_reason=reason
                    except Exception as e:
                        last_reason=str(e)
                if ok_any:
                    self.app.schedule(lambda:self.app.pause_var.set(""));self.app.reload_perception();self.app.update_model_ready_ui();done=True;break
                self.app.schedule(lambda n=spec.name,a=attempt,r=last_reason:self.app.pause_var.set(f"{n} 下载失败：{r}"))
                choice=self.prompt_retry_or_local(spec.name,(spec.urls[0] if spec.urls else ""),last_reason or "网络错误或超时",spec.path)
                if choice=="open":
                    try:webbrowser.open(spec.urls[0] if spec.urls else "")
                    except Exception:pass
                if choice=="retry":
                    time.sleep(delay);delay=min(delay*2.0,30.0);continue
                if choice=="local":
                    if spec.generator and spec.name=="policy":
                        try:
                            if spec.generator(spec.path):
                                done=True
                        except Exception as e:
                            self.app.schedule(lambda msg=str(e):self._alert(f"本地生成失败:{msg}"))
                    else:
                        self.app.schedule(lambda:self._alert(f"{spec.name} 未就绪，请手动下载放置到指定路径"))
                else:
                    continue
class PerceptionEngine:
    def __init__(self):
        self.last_text=""
        try:
            from ultralytics import YOLO
            self.yolo=YOLO(yolo_path) if os.path.exists(yolo_path) else None
        except Exception:
            self.yolo=None
        try:
            import pytesseract
            self.ocr=pytesseract
        except Exception:
            self.ocr=None
    def detect(self,frame):
        H,W=frame.shape[:2]
        dets=[]
        try:
            if self.yolo is not None:
                res=self.yolo.predict(source=frame,verbose=False,device=device if device=="cuda" else None,conf=0.25,imgsz=640)
                for r in res:
                    if hasattr(r,"boxes"):
                        for b in r.boxes:
                            x1,y1,x2,y2=b.xyxy[0].tolist()
                            dets.append({"bbox":[max(0,int(x1)),max(0,int(y1)),min(W-1,int(x2)),min(H-1,int(y2))],"score":float(b.conf[0].item()),"label":str(int(b.cls[0].item()))})
        except Exception:
            pass
        if not dets:
            gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);edges=cv2.Canny(gray,50,150);cnts,_=cv2.findContours(edges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            for c in cnts:
                x,y,w,h=cv2.boundingRect(c)
                if w*h>2000 and 10<=w<=600 and 10<=h<=300:
                    dets.append({"bbox":[x,y,x+w,y+h],"score":0.3,"label":"edge"})
        txt=""
        if self.ocr is not None:
            try:
                rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);txt=self.ocr.image_to_string(rgb,lang="eng")
            except Exception:
                txt=""
        else:
            try:
                gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY);thr=cv2.threshold(gray,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
                cnts,_=cv2.findContours(thr,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE);numbers=0
                for c in cnts:
                    x,y,w,h=cv2.boundingRect(c)
                    if 8<w<80 and 12<h<80 and w*h<2000:
                        numbers+=1
                txt=f"NUM:{numbers}"
            except Exception:
                txt=""
        self.last_text=txt.strip()
        out=[]
        for d in dets[:20]:
            x1,y1,x2,y2=d["bbox"];cx=(x1+x2)/2/W;cy=(y1+y2)/2/H
            out.append((cx,cy,d.get("label","0"),d.get("score",0.0)))
        return out,txt
class GoalGraph:
    def __init__(self):
        self.lock=threading.Lock()
        self.events=[]
        self.proto_success={}
        self.proto_counts={}
        self.last_score=None
    def _extract_score(self,txt):
        try:
            nums=[int(x) for x in re.findall(r"\d+",txt)]
            return max(nums) if nums else None
        except Exception:
            return None
    def add(self,ts,proto_id,text,atype,path_vec):
        score=self._extract_score(text.lower())
        success=1 if any(k in text.lower() for k in ["win","victory","success","complete"]) else 0
        with self.lock:
            self.events.append((ts,proto_id,score,success,atype))
            if proto_id is not None and proto_id>=0:
                self.proto_counts[proto_id]=self.proto_counts.get(proto_id,0)+1
                self.proto_success[proto_id]=self.proto_success.get(proto_id,0)+success
    def causal_score(self,proto_id):
        with self.lock:
            c=self.proto_counts.get(proto_id,0)
            if c==0:return 0.0
            s=self.proto_success.get(proto_id,0)
            p1=s/max(c,1)
            p0=(sum(self.proto_success.values())-s)/max(sum(self.proto_counts.values())-c,1)
            return float(p1-p0)
    def granger(self,lag=1):
        with self.lock:
            if len(self.events)<lag+5:return {}
            ids=list(set([e[1] for e in self.events if e[1]>=0]))
            res={}
            for pid in ids:
                x=[];y=[]
                for i in range(lag,len(self.events)):
                    prev=self.events[i-lag];curr=self.events[i]
                    x.append(1.0 if prev[1]==pid else 0.0)
                    y.append(1.0 if curr[3]>0 else 0.0)
                X=np.array(x,dtype=np.float32).reshape(-1,1);Y=np.array(y,dtype=np.float32).reshape(-1,1)
                try:
                    w=np.linalg.lstsq(X,Y,rcond=None)[0];res[pid]=float(w[0][0])
                except Exception:
                    res[pid]=0.0
            return res
class App:
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("Game AI")
        if device!="cuda":
            try:self.root.after(100,lambda:messagebox.showwarning("GPU不可用","未检测到可用的GPU，将使用CPU运行"))
            except Exception:pass
        self.ui_queue=queue.Queue()
        self.running=True
        self.writer=ExperienceWriter(experience_dir)
        self.buffer=ReplayBuffer()
        self.model=PolicyNet().to(device)
        if os.path.exists(model_path):
            try:self.model.load_state_dict(torch.load(model_path,map_location=device),strict=False)
            except Exception:pass
        self.optimizer=optim.Adam(self.model.parameters(),lr=1e-4)
        self.capture_interval=0.05
        self.metrics={"cpu":0.0,"mem":0.0,"gpu":0.0,"vram":0.0,"freq":20.0,"temp":0.0,"pwr":0.0}
        self.gpu_source="不可用"
        self.frame=None
        self.photo=None
        self.frame_lock=threading.Lock()
        self.frame_history=[]
        self.history_lock=threading.Lock()
        self.mode="init"
        self.resource_paused=False
        self.capture_enabled=False
        self.recording_enabled=False
        self.optimize_event=None
        self.optimize_thread=None
        self.ai_thread=None
        self.ai_stop=threading.Event()
        self.hold_state=False
        self.ai_interval=0.05
        self.selected_title=tk.StringVar()
        self.window_state=tk.StringVar(value="No window")
        self.visible_var=tk.StringVar(value='可见: 未检测')
        self.full_var=tk.StringVar(value='完整: 未检测')
        self.pause_var=tk.StringVar(value="未选择窗口，功能锁定")
        self.mode_var=tk.StringVar(value="Init")
        self.cpu_var=tk.StringVar(value="CPU:0.0%")
        self.mem_var=tk.StringVar(value="Memory:0.0%")
        self.gpu_var=tk.StringVar(value="GPU:0.0%")
        self.vram_var=tk.StringVar(value="VRAM:0.0%")
        self.gpu_src_var=tk.StringVar(value="GPU指标来源: 不可用")
        self.freq_var=tk.StringVar(value="Capture:0.0 Hz")
        self.progress_var=tk.DoubleVar(value=0.0)
        self.progress_text=tk.StringVar(value="0%")
        self.last_user_input=time.time()
        self.window_obj=None
        self.window_rect=None
        self.window_hwnd=None
        self.window_visible=False
        self.window_full=False
        self.window_occ_state=0
        self.window_visible_reason='未知'
        self.window_full_reason='未知'
        self.last_occl_check=0.0
        self.status_lock=threading.Lock()
        self.monitor_bounds=[(m.x,m.y,m.width,m.height) for m in get_monitors()]
        self.mss_ctx=mss.mss()
        self.cap_queue=queue.Queue(maxsize=2)
        self.prep_queue=queue.Queue(maxsize=2)
        self.drag_active=False
        self.drag_points=[]
        self.drag_start=None
        self.capture_thread=threading.Thread(target=self._capture_loop,daemon=True)
        self.prep_thread=threading.Thread(target=self._preprocess_loop,daemon=True)
        self.write_thread=threading.Thread(target=self._writer_loop,daemon=True)
        self.monitor_thread=threading.Thread(target=self._monitor_loop,daemon=True)
        self.listener_mouse=mouse.Listener(on_move=self._on_mouse_move,on_click=self._on_mouse_click)
        self.listener_keyboard=keyboard.Listener(on_press=self._on_key_press)
        self._build_ui()
        self.model_manager=ModelManager(self)
        self.model_manager.ensure()
        self.capture_thread.start()
        self.prep_thread.start()
        self.write_thread.start()
        self.monitor_thread.start()
        self.listener_mouse.start()
        self.listener_keyboard.start()
        self.root.protocol("WM_DELETE_WINDOW",self.stop)
        self.root.after(50,self._update_ui)
        self.root.after(200,self._check_mode_switch)
        self.perception=PerceptionEngine()
        try:
            import clip as openai_clip
            self.clip_model,self.clip_preprocess=openai_clip.load("ViT-B/32",device=device,download_root=models_dir)
        except Exception:
            self.clip_model=None;self.clip_preprocess=None
        self.pool=UIElementPool(self.clip_model,self.clip_preprocess)
        self.rule_text=""
        self.goal_graph=GoalGraph()
        try:
            if os.cpu_count():
                torch.set_num_threads(max(1,min(os.cpu_count(),8)))
        except Exception:
            pass
    def reload_perception(self):
        try:self.perception=PerceptionEngine()
        except Exception:pass
    def models_ready(self):
        ok=True
        for p in [model_path]:
            ok=ok and os.path.exists(p)
        return ok
    def update_model_ready_ui(self):
        try:
            if self.models_ready():
                self.sleep_btn.configure(state="normal" if self.window_obj is not None else "disabled")
            else:
                self.sleep_btn.configure(state="disabled")
        except Exception:
            pass
    def _build_ui(self):
        self.root.columnconfigure(0,weight=1)
        top=tk.Frame(self.root);top.grid(row=0,column=0,sticky="nsew");top.columnconfigure(1,weight=1)
        ttk.Label(top,text="Window:").grid(row=0,column=0,sticky="w")
        self.window_combo=ttk.Combobox(top,textvariable=self.selected_title,state="readonly",width=50);self.window_combo.grid(row=0,column=1,sticky="ew")
        ttk.Button(top,text="Refresh",command=self.refresh_windows).grid(row=0,column=2,sticky="ew")
        ttk.Button(top,text="Select",command=self.select_window).grid(row=0,column=3,sticky="ew")
        ttk.Label(top,textvariable=self.window_state).grid(row=1,column=0,columnspan=2,sticky="w")
        ttk.Label(top,textvariable=self.visible_var).grid(row=1,column=2,sticky="e")
        ttk.Label(top,textvariable=self.full_var).grid(row=1,column=3,sticky="e")
        mid=tk.Frame(self.root);mid.grid(row=1,column=0,sticky="nsew");mid.rowconfigure(0,weight=1);mid.columnconfigure(0,weight=1)
        self.frame_label=tk.Label(mid);self.frame_label.grid(row=0,column=0,sticky="nsew")
        stats=tk.Frame(self.root);stats.grid(row=2,column=0,sticky="ew")
        for i in range(6):stats.columnconfigure(i,weight=1)
        ttk.Label(stats,textvariable=self.cpu_var).grid(row=0,column=0,sticky="w")
        ttk.Label(stats,textvariable=self.mem_var).grid(row=0,column=1,sticky="w")
        ttk.Label(stats,textvariable=self.gpu_var).grid(row=0,column=2,sticky="w")
        ttk.Label(stats,textvariable=self.vram_var).grid(row=0,column=3,sticky="w")
        ttk.Label(stats,textvariable=self.gpu_src_var).grid(row=0,column=4,sticky="e");ttk.Label(stats,textvariable=self.freq_var).grid(row=0,column=5,sticky="e")
        control=tk.Frame(self.root);control.grid(row=3,column=0,sticky="ew");control.columnconfigure(0,weight=1);control.columnconfigure(1,weight=1)
        ttk.Label(control,textvariable=self.mode_var).grid(row=0,column=0,columnspan=2,sticky="w")
        self.sleep_btn=ttk.Button(control,text="Sleep",command=self.on_sleep,state="disabled");self.sleep_btn.grid(row=1,column=0,sticky="ew")
        self.getup_btn=ttk.Button(control,text="Get Up",command=self.on_getup,state="disabled");self.getup_btn.grid(row=1,column=1,sticky="ew")
        prog_row=tk.Frame(control);prog_row.grid(row=2,column=0,columnspan=2,sticky="ew");prog_row.columnconfigure(0,weight=1)
        self.progress=ttk.Progressbar(prog_row,variable=self.progress_var,maximum=100);self.progress.grid(row=0,column=0,sticky="ew")
        ttk.Label(prog_row,textvariable=self.progress_text,width=6).grid(row=0,column=1,sticky="e")
        ttk.Label(self.root,textvariable=self.pause_var,foreground="#888").grid(row=4,column=0,sticky="w")
        self.refresh_windows()
    def refresh_windows(self):
        titles=[w.title for w in gw.getAllWindows() if w.title.strip()]
        self.window_combo["values"]=titles
        if titles and not self.selected_title.get():
            self.selected_title.set(titles[0])
    def select_window(self):
        title=self.selected_title.get()
        windows=[w for w in gw.getAllWindows() if w.title==title]
        if windows:
            self.window_obj=windows[0]
            try:self.window_hwnd=int(self.window_obj._hWnd)
            except Exception:self.window_hwnd=None
            self.window_state.set(f"Selected:{title}")
            self._update_window_status(force=True)
            self.buffer.clear()
            self.capture_enabled=True
            self.recording_enabled=True
            self.sleep_btn.configure(state="normal" if self.models_ready() else "disabled")
            self.getup_btn.configure(state="disabled")
            self.pause_var.set("" if self.models_ready() else "模型未就绪，已暂停且 10 秒切换规则失效")
            self.set_mode("learn")
    def _get_ext_frame_rect(self,hwnd):
        try:
            if hwnd is None:
                return None
            rect=ctypes.wintypes.RECT()
            if dwmapi is not None:
                try:
                    DWMWA_EXTENDED_FRAME_BOUNDS=9
                    if dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(int(hwnd)),ctypes.c_int(DWMWA_EXTENDED_FRAME_BOUNDS),ctypes.byref(rect),ctypes.sizeof(rect))==0:
                        return (rect.left,rect.top,rect.right,rect.bottom)
                except Exception:
                    pass
            try:
                if user32 is not None and user32.GetWindowRect(ctypes.wintypes.HWND(int(hwnd)),ctypes.byref(rect))!=0:
                    return (rect.left,rect.top,rect.right,rect.bottom)
            except Exception:
                pass
            try:
                rc=ctypes.wintypes.RECT()
                if user32 is not None and user32.GetClientRect(ctypes.wintypes.HWND(int(hwnd)),ctypes.byref(rc))!=0:
                    pt1=ctypes.wintypes.POINT(rc.left,rc.top)
                    pt2=ctypes.wintypes.POINT(rc.right,rc.bottom)
                    user32.ClientToScreen(ctypes.wintypes.HWND(int(hwnd)),ctypes.byref(pt1))
                    user32.ClientToScreen(ctypes.wintypes.HWND(int(hwnd)),ctypes.byref(pt2))
                    return (pt1.x,pt1.y,pt2.x,pt2.y)
            except Exception:
                pass
            try:
                if self.window_obj is not None:
                    self.window_obj.refresh()
                    left=int(getattr(self.window_obj,'left',0));top=int(getattr(self.window_obj,'top',0));right=int(getattr(self.window_obj,'right',left+int(getattr(self.window_obj,'width',0))));bottom=int(getattr(self.window_obj,'bottom',top+int(getattr(self.window_obj,'height',0))))
                    if right>left and bottom>top:
                        return (left,top,right,bottom)
            except Exception:
                pass
            try:
                title=getattr(self.window_obj,'title','') or self.selected_title.get()
                if title:
                    wins=[w for w in gw.getAllWindows() if w.title==title or title in w.title]
                    if wins:
                        w=wins[0];w.refresh();left=int(getattr(w,'left',0));top=int(getattr(w,'top',0));right=int(getattr(w,'right',left+int(getattr(w,'width',0))));bottom=int(getattr(w,'bottom',top+int(getattr(w,'height',0))))
                        if right>left and bottom>top:
                            return (left,top,right,bottom)
            except Exception:
                pass
            return None
        except Exception:
            return None
    def _resolve_window_handle(self):
        if self.window_obj is None:
            return None
        try:
            self.window_obj.refresh()
            return int(self.window_obj._hWnd)
        except Exception:
            pass
        try:
            title=getattr(self.window_obj,'title','') or self.selected_title.get()
            if title:
                windows=[w for w in gw.getAllWindows() if w.title==title]
                if windows:
                    self.window_obj=windows[0]
                    self.window_obj.refresh()
                    return int(self.window_obj._hWnd)
        except Exception:
            pass
        return None
    def _dwm_occlusion_state(self,hwnd):
        if dwmapi is None or hwnd is None:
            return 0
        try:
            DWMWA_OCCLUSION_STATE=18
            state=ctypes.c_uint(0)
            res=dwmapi.DwmGetWindowAttribute(ctypes.wintypes.HWND(int(hwnd)),ctypes.c_int(DWMWA_OCCLUSION_STATE),ctypes.byref(state),ctypes.sizeof(state))
            if res==0:
                return int(state.value)
        except Exception:
            pass
        return 0
    def _pixel_occlusion_ratio(self,frame,hwnd,rect):
        try:
            if user32 is None:
                self.last_occl_check=time.time()
                return 0.0
            hwnd_int=int(hwnd) if hwnd is not None else None
            if hwnd_int is None:
                self.last_occl_check=time.time()
                return 1.0
            occ_state=self._dwm_occlusion_state(hwnd_int)
            if occ_state==1:
                self.last_occl_check=time.time()
                return 1.0
            left,top,right,bottom=rect;w=max(1,right-left);h=max(1,bottom-top);area=float(w*h)
            overlaps=[];cur=user32.GetWindow(ctypes.wintypes.HWND(hwnd_int),ctypes.c_int(GW_HWNDPREV));seen=set()
            while cur:
                h=int(cur)
                if h in seen:
                    break
                seen.add(h)
                cur_hwnd=ctypes.wintypes.HWND(cur)
                if user32.IsWindow(cur_hwnd)!=0 and user32.IsWindowVisible(cur_hwnd) and user32.IsIconic(cur_hwnd)==0:
                    other=self._get_ext_frame_rect(h)
                    if other is not None:
                        inter=_rect_intersection(rect,other)
                        if inter and inter[2]>inter[0] and inter[3]>inter[1]:
                            if not self._is_window_transparent(h):
                                overlaps.append(inter)
                cur=user32.GetWindow(cur_hwnd,ctypes.c_int(GW_HWNDPREV))
            occ=float(_rect_union_area(overlaps));ratio=occ/area if area>0 else 1.0
            if frame is not None and ratio<=1.0:
                capture=self._print_window(hwnd_int,w,h)
                if capture is not None:
                    if capture.shape[0]!=frame.shape[0] or capture.shape[1]!=frame.shape[1]:
                        sample=cv2.resize(frame,(capture.shape[1],capture.shape[0]))
                    else:
                        sample=frame
                    diff=cv2.absdiff(cv2.cvtColor(sample,cv2.COLOR_BGR2GRAY),cv2.cvtColor(capture,cv2.COLOR_BGR2GRAY));mask=(diff>18).astype(np.uint8);ratio=max(ratio,float(np.count_nonzero(mask))/mask.size)
            if occ_state==2:
                ratio=min(ratio,0.05)
            self.last_occl_check=time.time()
            return min(max(ratio,0.0),1.0)
        except Exception:
            return None
    def _print_window(self,hwnd,width,height):
        if user32 is None or gdi32 is None or width<=0 or height<=0:
            return None
        hdc=None;memdc=None;bmp=None
        try:
            hdc=user32.GetWindowDC(ctypes.wintypes.HWND(hwnd))
            if not hdc:
                return None
            memdc=gdi32.CreateCompatibleDC(hdc)
            if not memdc:
                return None
            bmp=gdi32.CreateCompatibleBitmap(hdc,width,height)
            if not bmp:
                return None
            old=gdi32.SelectObject(memdc,bmp)
            res=user32.PrintWindow(ctypes.wintypes.HWND(hwnd),memdc,ctypes.c_uint(PW_RENDERFULLCONTENT))
            buf=ctypes.create_string_buffer(width*height*4)
            got=gdi32.GetBitmapBits(bmp,len(buf),buf)
            gdi32.SelectObject(memdc,old)
            if res==0 or got==0:
                return None
            arr=np.frombuffer(buf,dtype=np.uint8).reshape((height,width,4))
            return arr[:,:,:3].copy()
        except Exception:
            return None
        finally:
            try:
                if bmp:gdi32.DeleteObject(bmp)
            except Exception:
                pass
            try:
                if memdc:gdi32.DeleteDC(memdc)
            except Exception:
                pass
            try:
                if hdc:user32.ReleaseDC(ctypes.wintypes.HWND(hwnd),hdc)
            except Exception:
                pass
    def _is_window_transparent(self,hwnd):
        try:
            if user32 is None:
                return False
            h=ctypes.wintypes.HWND(hwnd)
            ex=user32.GetWindowLongW(h,ctypes.c_int(GWL_EXSTYLE))
            if ex & WS_EX_TRANSPARENT:
                return True
            func=getattr(user32,'GetLayeredWindowAttributes',None)
            if ex & WS_EX_LAYERED and func is not None:
                key=ctypes.c_uint()
                alpha=ctypes.c_ubyte()
                flags=ctypes.c_uint()
                if func(h,ctypes.byref(key),ctypes.byref(alpha),ctypes.byref(flags))!=0:
                    if flags.value & LWA_ALPHA and alpha.value<250:
                        return True
            return False
        except Exception:
            return False
    def _monitor_coverage(self,rect):
        area=_rect_area(rect)
        if area<=0:
            return 0.0
        covered=0
        for mx,my,mw,mh in self.monitor_bounds:
            inter=_rect_intersection(rect,(mx,my,mx+mw,my+mh))
            if inter:
                covered+=_rect_area(inter)
        return covered/area if area>0 else 0.0
    def _check_window_visible(self,hwnd,rect):
        try:
            if user32 is None or hwnd is None:
                return False,'系统不支持',0,0.0
            h=ctypes.wintypes.HWND(int(hwnd))
            if user32.IsWindow(h)==0:
                return False,'窗口句柄无效',0,0.0
            if user32.IsWindowVisible(h)==0:
                return False,'窗口不可见',0,0.0
            if user32.IsIconic(h)!=0:
                return False,'窗口最小化',0,self._monitor_coverage(rect) if rect else 0.0
            occ=self._dwm_occlusion_state(hwnd)
            if dwmapi is not None:
                cloaked=ctypes.c_int(0)
                try:
                    DWMWA_CLOAKED=14
                    dwmapi.DwmGetWindowAttribute(h,ctypes.c_int(DWMWA_CLOAKED),ctypes.byref(cloaked),ctypes.sizeof(cloaked))
                    if cloaked.value!=0:
                        return False,'窗口被系统隐藏',occ,self._monitor_coverage(rect) if rect else 0.0
                except Exception:
                    pass
            coverage=self._monitor_coverage(rect) if rect else 0.0
            if coverage<=0.01:
                return False,'窗口不在显示器范围',occ,coverage
            if occ==1:
                return False,'窗口被遮挡',occ,coverage
            if occ==2:
                return True,'完全可见',occ,coverage
            return True,'检测到窗口',occ,coverage
        except Exception:
            return False,'检测异常',0,0.0
    def _update_window_status(self,force=False):
        if self.window_obj is None:
            self.window_rect=None;self.window_hwnd=None;self.window_visible=False;self.window_full=False;self.window_occ_state=0;self.window_visible_reason='未选择窗口';self.window_full_reason='未选择窗口';self.schedule(lambda:self.visible_var.set('可见: 否(未选择窗口)'));self.schedule(lambda:self.full_var.set('完整: 否(未选择窗口)'));return
        try:
            hwnd=self._resolve_window_handle()
            self.window_hwnd=hwnd
            rect=self._get_ext_frame_rect(hwnd) if hwnd is not None else None
            self.window_rect=rect
            visible=False
            full=False
            vis_reason='无法检测'
            full_reason='无法检测'
            if rect:
                visible,vis_reason,occ,coverage=self._check_window_visible(hwnd,rect)
                self.window_occ_state=occ
                if visible:
                    if coverage>=0.99:
                        ratio=self._pixel_occlusion_ratio(None,hwnd,rect)
                        if ratio is not None:
                            full=ratio<=OCCLUSION_THR
                            full_reason=f"遮挡率:{ratio*100:.2f}%"
                        else:
                            full=False
                            full_reason='遮挡检测失败'
                    else:
                        full=False
                        full_reason=f"覆盖率:{coverage*100:.2f}%"
                else:
                    full=False
                    full_reason=vis_reason
            else:
                self.window_occ_state=0
                vis_reason='无法获取窗口区域'
                full_reason=vis_reason
            self.window_visible=visible
            self.window_full=full
            self.window_visible_reason=vis_reason
            self.window_full_reason=full_reason
            self.schedule(lambda v=visible,r=vis_reason:self.visible_var.set(f"可见: {'是' if v else '否'}({r})"))
            self.schedule(lambda f=full,t=full_reason:self.full_var.set(f"完整: {'是' if f else '否'}({t})"))
        except Exception:
            self.window_rect=None
            self.window_hwnd=None
            self.window_visible=False
            self.window_full=False
            self.window_occ_state=0
            self.window_visible_reason='检测异常'
            self.window_full_reason='检测异常'
            self.schedule(lambda:self.visible_var.set('可见: 否(检测异常)'))
            self.schedule(lambda:self.full_var.set('完整: 否(检测异常)'))
    def schedule(self,func):
        if self.running:self.ui_queue.put(func)
    def _gpu_metrics_nvml(self):
        try:
            if _gpu_count>0:
                handle=nvmlDeviceGetHandleByIndex(0)
                util=nvmlDeviceGetUtilizationRates(handle);mem=nvmlDeviceGetMemoryInfo(handle)
                gpu=float(util.gpu);vram=float(mem.used*100.0/max(mem.total,1));return gpu,vram,"NVML"
        except Exception:
            return None
        return None
    def _gpu_metrics_typeperf(self):
        try:
            p=subprocess.Popen(["typeperf","-sc","1","\\GPU Engine(*)\\Utilization Percentage","\\GPU Adapter Memory(*)\\Dedicated Usage","\\GPU Adapter Memory(*)\\Dedicated Limit"],stdout=subprocess.PIPE,stderr=subprocess.PIPE,shell=False)
            out,err=p.communicate(timeout=3)
            s=out.decode(errors="ignore").strip().splitlines()
            if len(s)<3:return None
            values=s[-1].split(",");nums=[]
            for v in values[1:]:
                v=v.strip().strip('"')
                try:nums.append(float(v))
                except Exception:nums.append(0.0)
            if len(nums)<3:return None
            util_vals=nums[:-2];used=nums[-2];limit=nums[-1] if nums[-1]>0 else 1.0
            gpu=max(0.0,min(100.0,float(max(util_vals) if util_vals else 0.0)));vram=max(0.0,min(100.0,float(used*100.0/limit)));return gpu,vram,"WMI"
        except Exception:
            return None
    def _gpu_metrics_gputil(self):
        try:
            gpus=GPUtil.getGPUs()
            if gpus:
                g=gpus[0];return float(g.load*100.0),float(g.memoryUtil*100.0),"GPUtil"
        except Exception:
            return None
        return None
    def _gpu_metrics_ext(self):
        got=self._gpu_metrics_nvml()
        if got is None:got=self._gpu_metrics_typeperf()
        if got is None:got=self._gpu_metrics_gputil()
        if got is None:
            self.gpu_source="不可用";return 0.0,0.0,0.0,0.0
        gpu,mem,src=got;temp=0.0;pwr_ratio=0.0
        if src=="NVML":
            try:
                h=nvmlDeviceGetHandleByIndex(0);temp=float(nvmlDeviceGetTemperature(h,0));p=float(nvmlDeviceGetPowerUsage(h))/1000.0;pl=float(nvmlDeviceGetEnforcedPowerLimit(h))/1000.0;pwr_ratio=p/max(pl,1e-6)
            except Exception:
                pass
        self.gpu_source=src;return gpu,mem,temp,pwr_ratio
    def _monitor_loop(self):
        last_threads=None
        while self.running:
            cpu=float(psutil.cpu_percent(interval=None));mem=float(psutil.virtual_memory().percent);gpu,mem_gpu,temp,pwr_ratio=self._gpu_metrics_ext()
            if self.gpu_source=="不可用":M=max(cpu,mem)/100.0
            else:M=max(cpu,mem,gpu,mem_gpu)/100.0
            freq=max(0.0,100.0*(1.0-M))
            self.metrics={"cpu":cpu,"mem":mem,"gpu":gpu,"vram":mem_gpu,"freq":freq,"temp":temp,"pwr":pwr_ratio}
            if freq>0:self.capture_interval=1.0/freq
            self.ai_interval=0.02+0.08*min(max(M,0.0),1.0)
            if temp>=80 or pwr_ratio>=0.95:self.ai_interval*=2.0
            self.schedule(lambda s=self.gpu_source:self.gpu_src_var.set(f"GPU指标来源: {s}"))
            if freq<=0:
                if not self.resource_paused:
                    self.resource_paused=True;self._stop_ai();self._mouse_up_safety();self.set_mode("learn");self.recording_enabled=False;self.sleep_btn.configure(state="disabled");self.getup_btn.configure(state="disabled");self.pause_var.set("已暂停且 10 秒切换规则失效")
            else:
                if self.resource_paused:
                    self.resource_paused=False;self.recording_enabled=True;self.pause_var.set("")
                    if self.mode=="init" and self.window_obj is not None:self.set_mode("learn")
                    if self.window_obj is not None:self.sleep_btn.configure(state="normal" if self.models_ready() else "disabled")
            try:
                if os.cpu_count():
                    target_threads=max(1,int(min(os.cpu_count(),max(1,int((1.0-M)*os.cpu_count())))))
                    if target_threads!=last_threads:
                        torch.set_num_threads(target_threads);last_threads=target_threads
            except Exception:
                pass
            time.sleep(0.5)
    def _enqueue_drop(self,q,item):
        try:
            q.put_nowait(item)
        except queue.Full:
            try:
                q.get_nowait()
            except Exception:
                pass
            try:
                q.put_nowait(item)
            except Exception:
                pass
    def _capture_loop(self):
        while self.running:
            start=time.time();f=self.metrics.get("freq",0.0)
            if f<=0.0 or not self.capture_enabled or self.resource_paused:time.sleep(0.05);continue
            if self.window_obj is not None:
                self._update_window_status()
                if self.window_visible and self.window_rect is not None:
                    left,top,right,bottom=self.window_rect;width=right-left;height=bottom-top
                    if width>0 and height>0:
                        try:
                            shot=self.mss_ctx.grab({"left":left,"top":top,"width":width,"height":height});frame=np.array(shot);frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
                            now=time.time()
                            if (now-self.last_occl_check>1.0 or not self.window_full) and self.window_visible:
                                ratio=self._pixel_occlusion_ratio(frame,self.window_hwnd,self.window_rect)
                                if ratio is not None:
                                    coverage=self._monitor_coverage(self.window_rect)
                                    self.window_full=bool(coverage>=0.99 and ratio<=OCCLUSION_THR)
                                else:
                                    self.window_full=False
                            self._enqueue_drop(self.cap_queue,(frame,now))
                        except Exception:
                            pass
                else:
                    with self.frame_lock:self.frame=None
            elapsed=time.time()-start;wait=max(self.capture_interval-elapsed,0.001) if self.metrics.get("freq",0.0)>0 else 0.05;time.sleep(wait)
    def _preprocess_loop(self):
        while self.running:
            try:
                item=self.cap_queue.get(timeout=0.1)
            except Exception:
                time.sleep(0.01);continue
            frame,_=item
            scale=1.0
            if self.metrics["temp"]>=80 or self.metrics["pwr"]>=0.95 or self.metrics["vram"]>=90.0:scale=0.75
            if self.metrics["vram"]>=96.0:scale=0.5
            if scale!=1.0:
                h,w=frame.shape[:2];nw=max(64,int(w*scale));nh=max(64,int(h*scale));frame=cv2.resize(frame,(nw,nh))
            with self.frame_lock:self.frame=frame
            self._append_history(frame)
            if self.mode=="learn" and not self.resource_paused and self.window_visible and self.window_full:self._analyze_ui_async(frame)
            self._enqueue_drop(self.prep_queue,(frame,time.time()))
            try:self.cap_queue.task_done()
            except Exception:pass
    def _writer_loop(self):
        while self.running:
            try:
                frame,_=self.prep_queue.get(timeout=0.1)
            except Exception:
                time.sleep(0.01);continue
            if self.recording_enabled and not self.resource_paused and self.capture_enabled and self.window_visible and self.window_full and self.window_rect is not None and self.mode in ("learn","train"):
                rect=self.window_rect;center=((rect[0]+rect[2])/2.0,(rect[1]+rect[3])/2.0)
                action=[0.0,0.0,0.0]
                pos=[(center[0]-rect[0])/(rect[2]-rect[0]),(center[1]-rect[1])/(rect[3]-rect[1])]
                source=1 if self.mode=="learn" else 2;title=self.window_obj.title if self.window_obj else ""
                try:self.writer.record(frame,action,pos,source,rect,title,self.mode,"frame");self.buffer.add(frame,action,source,0,0,-1,self.perception.last_text)
                except Exception:pass
            try:self.prep_queue.task_done()
            except Exception:pass
    def _append_history(self,frame):
        with self.history_lock:
            self.frame_history.append(frame)
            if len(self.frame_history)>8:self.frame_history.pop(0)
    def _save_click_patches(self,frame,nx,ny):
        try:
            clicks_dir=os.path.join(experience_dir,"clicks");os.makedirs(clicks_dir,exist_ok=True)
            H,W=frame.shape[:2];x=int(nx*W);y=int(ny*H);sz=max(16,min(H,W)//10);x1=max(0,x-sz);y1=max(0,y-sz);x2=min(W,x+sz);y2=min(H,y+sz)
            pos=frame[y1:y2,x1:x2]
            neg=None
            for _ in range(10):
                rx=random.randint(sz,W-sz-1);ry=random.randint(sz,H-sz-1)
                if abs(rx-x)>2*sz and abs(ry-y)>2*sz:
                    neg=frame[ry-sz:ry+sz,rx-sz:rx+sz];break
            ts=int(time.time()*1000)
            if pos is not None:cv2.imwrite(os.path.join(clicks_dir,f"pos_{ts}.png"),pos)
            if neg is not None:cv2.imwrite(os.path.join(clicks_dir,f"neg_{ts}.png"),neg)
        except Exception:
            pass
    def _on_mouse_move(self,x,y):
        self.last_user_input=time.time()
        if self.drag_active and self.window_rect is not None:
            if self.window_rect[0]<=x<=self.window_rect[2] and self.window_rect[1]<=y<=self.window_rect[3]:
                self.drag_points.append((x,y,time.time()))
        if self.mode=="learn" and self.window_visible and self.window_full and self.window_rect is not None and self.recording_enabled and not self.resource_paused and self.capture_enabled:
            rect=self.window_rect
            if rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]:
                width=max(rect[2]-rect[0],1);height=max(rect[3]-rect[1],1);norm_x=(x-rect[0])/width;norm_y=(y-rect[1])/height;action=[0.0,0.0,0.0]
                frame=self._current_frame_copy()
                if frame is not None:self.writer.record(frame,action,[norm_x,norm_y],1,rect,self.window_obj.title if self.window_obj else "",self.mode,"move");self.buffer.add(frame,action,1,1,0,-1,self.perception.last_text)
    def _on_mouse_click(self,x,y,button,pressed):
        self.last_user_input=time.time()
        inside=False;rect=self.window_rect
        if rect is not None:inside=(rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3])
        if pressed and self.mode=="learn" and self.window_visible and self.window_full and inside and self.recording_enabled and not self.resource_paused and self.capture_enabled:
            self.drag_active=True;self.drag_points=[(x,y,time.time())];self.drag_start=time.time()
            width=max(rect[2]-rect[0],1);height=max(rect[3]-rect[1],1);norm_x=(x-rect[0])/width;norm_y=(y-rect[1])/height;action=[0.0,0.0,1.0];frame=self._current_frame_copy()
            if frame is not None:
                pid=self.pool.add_click(frame,rect,norm_x,norm_y);self._save_click_patches(frame,norm_x,norm_y);self.writer.record(frame,action,[norm_x,norm_y],1,rect,self.window_obj.title if self.window_obj else "",self.mode,"press",{"proto":int(pid)});self.buffer.add(frame,action,1,2,1,int(pid),self.perception.last_text);self.goal_graph.add(_now_ms(),int(pid),self.perception.last_text,1,np.zeros((64,),np.float32))
        if (not pressed) and self.drag_active:
            self.drag_active=False
            if self.window_rect is not None:
                width=max(rect[2]-rect[0],1);height=max(rect[3]-rect[1],1)
                if len(self.drag_points)>=2:
                    simp=_poly_simplify([(px,py) for px,py,_ in self.drag_points],eps=3.0)
                    path=[((px-rect[0])/width,(py-rect[1])/height) for px,py in simp]
                    frame=self._current_frame_copy()
                    if frame is not None:
                        pid=self.pool.add_drag(frame,rect,path);self.writer.record(frame,[0.0,0.0,0.0],path[-1],1,rect,self.window_obj.title if self.window_obj else "",self.mode,"drag",{"path":path,"duration":time.time()-self.drag_start,"proto":int(pid)});self.buffer.add(frame,[0.0,0.0,0.0],1,5,2,int(pid),self.perception.last_text);self.goal_graph.add(_now_ms(),int(pid),self.perception.last_text,2,np.zeros((64,),np.float32))
    def _on_key_press(self,key):
        self.last_user_input=time.time()
        if key==keyboard.Key.esc:self.schedule(self.stop)
        if self.mode=="train":
            self._mouse_up_safety();self.ai_stop.set();self.root.after(0,self.set_mode,"learn")
    def _current_frame_copy(self):
        with self.frame_lock:
            if self.frame is None:return None
            return self.frame.copy()
    def _update_ui(self):
        while not self.ui_queue.empty():
            func=self.ui_queue.get()
            try:func()
            except Exception:pass
        with self.frame_lock:frame=self.frame.copy() if self.frame is not None else None
        if frame is not None:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB);image=Image.fromarray(rgb);w=min(640,image.width);h=int(image.height*float(w)/float(image.width));image=image.resize((w,h))
            self.photo=ImageTk.PhotoImage(image=image);self.frame_label.configure(image=self.photo)
        else:self.frame_label.configure(image="")
        self.cpu_var.set(f"CPU:{self.metrics['cpu']:.1f}%");self.mem_var.set(f"Memory:{self.metrics['mem']:.1f}%");self.gpu_var.set(f"GPU:{self.metrics['gpu']:.1f}%");self.vram_var.set(f"VRAM:{self.metrics['vram']:.1f}%");self.freq_var.set(f"Capture:{self.metrics['freq']:.1f} Hz");self.progress_text.set(f"{self.progress_var.get():.0f}%")
        self.root.after(50,self._update_ui)
    def _check_mode_switch(self):
        if not self.resource_paused and self.mode=="learn" and self.recording_enabled and self.window_visible and self.window_full and self.capture_enabled:
            if time.time()-self.last_user_input>=10.0:self.set_mode("train")
        self.root.after(200,self._check_mode_switch)
    def _mouse_up_safety(self):
        try:pyautogui.mouseUp()
        except Exception:pass
        self.hold_state=False
    def set_mode(self,mode):
        if not self.running:return
        if mode==self.mode:return
        if mode=="train":
            if not self.window_visible or not self.window_full or self.resource_paused or not self.capture_enabled:return
            self.mode="train";self.mode_var.set("Training");self._start_ai()
        elif mode=="learn":
            self._mouse_up_safety();self.mode="learn";self.mode_var.set("Learning");self._stop_ai()
        elif mode=="optimize":
            self._mouse_up_safety();self.mode="optimize";self.mode_var.set("Optimizing");self._stop_ai()
        elif mode=="init":
            self._mouse_up_safety();self.mode="init";self.mode_var.set("Init");self._stop_ai()
    def _start_ai(self):
        self.ai_stop.clear()
        if self.ai_thread is None or not self.ai_thread.is_alive():
            self.ai_thread=threading.Thread(target=self._ai_loop,daemon=True);self.ai_thread.start()
    def _stop_ai(self):
        self.ai_stop.set();self.hold_state=False
        try:pyautogui.mouseUp()
        except Exception:pass
    def _get_history_tensor(self):
        with self.history_lock:
            if len(self.frame_history)<4:return None
            frames=self.frame_history[-4:]
        arr=np.stack(frames,0).astype(np.float32)/255.0;arr=np.transpose(arr,(0,3,1,2));arr=np.expand_dims(arr,0);tensor=torch.from_numpy(arr).to(device,non_blocking=True);return tensor
    def _ai_loop(self):
        while self.running and not self.ai_stop.is_set():
            if self.mode!="train" or not self.window_visible or not self.window_full or self.window_rect is None or self.resource_paused or not self.capture_enabled:time.sleep(0.05);continue
            frames=self._get_history_tensor()
            if frames is None:time.sleep(0.05);continue
            with torch.no_grad():
                self.model.eval()
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device=="cuda")):
                    action,_,_,_,_=self.model(frames)
            action=action.squeeze(0).float().cpu().numpy();dx=float(action[0]);dy=float(action[1]);click_prob=float(action[2]);rect=self.window_rect;width=max(rect[2]-rect[0],1);height=max(rect[3]-rect[1],1)
            target_x=rect[0]+width*(0.5+dx*0.25);target_y=rect[1]+height*(0.5+dy*0.25)
            if click_prob>0.7 and not self.hold_state:self.hold_state=True;pyautogui.mouseDown()
            if click_prob<0.3 and self.hold_state:self.hold_state=False;pyautogui.mouseUp()
            pyautogui.moveTo(target_x,target_y,duration=0.01)
            if not self.hold_state and 0.5<click_prob<=0.7:pyautogui.click()
            frame=self._current_frame_copy()
            if frame is not None and self.recording_enabled and not self.resource_paused:
                norm_x=(target_x-rect[0])/width;norm_y=(target_y-rect[1])/height;self.writer.record(frame,[dx,dy,click_prob],[norm_x,norm_y],2,rect,self.window_obj.title if self.window_obj else "",self.mode,"ai");self.buffer.add(frame,[dx,dy,click_prob],2,3,0,-1,self.perception.last_text)
            time.sleep(self.ai_interval)
    def _analyze_ui_async(self,frame):
        def worker(img):
            try:
                dets,txt=self.perception.detect(img);self.pool.update_state(txt)
                if dets and self.window_rect is not None:
                    rect=self.window_rect
                    for cx,cy,label,score in dets[:8]:
                        self.writer.record(img,[0.0,0.0,0.0],[cx,cy],1,rect,self.window_obj.title if self.window_obj else "",self.mode,"ui",{"label":label,"score":float(score)});self.buffer.add(img,[0.0,0.0,0.0],1,4,0,-1,txt)
                if txt:self.rule_text=txt
            except Exception:
                pass
        threading.Thread(target=worker,args=(frame.copy(),),daemon=True).start()
    def on_sleep(self):
        if self.mode in ("learn","train") and self.recording_enabled and not self.resource_paused and self.models_ready():
            self._mouse_up_safety();self.recording_enabled=False;self.getup_btn.configure(state="normal");self.sleep_btn.configure(state="disabled");self.progress_var.set(0.0);self.progress_text.set("0%");self.set_mode("optimize");self.optimize_event=threading.Event();self.optimize_thread=threading.Thread(target=self._optimize_loop,daemon=True);self.optimize_thread.start()
        else:
            self.pause_var.set("模型未就绪或资源受限，已暂停且 10 秒切换规则失效")
    def _confirm_dialog(self):
        q=queue.Queue(maxsize=1)
        def dlg():
            top=tk.Toplevel(self.root);top.title("Optimization Done")
            ttk.Label(top,text="优化完成").grid(row=0,column=0,sticky="ew")
            def ok():
                if q.empty():q.put(True);top.destroy()
            ttk.Button(top,text="Confirm",command=ok).grid(row=1,column=0,sticky="ew")
            top.grab_set();top.protocol("WM_DELETE_WINDOW",ok)
        self.schedule(dlg);q.get()
    def on_getup(self):
        if self.mode=="optimize" and self.optimize_event is not None:
            self.optimize_event.set();self.progress_var.set(0.0);self.progress_text.set("0%");self.sleep_btn.configure(state="normal" if self.models_ready() else "disabled");self.getup_btn.configure(state="disabled");self.recording_enabled=True;self._mouse_up_safety();self.set_mode("learn")
    def _compute_reward(self,seq_frames,txt):
        try:
            a=seq_frames[-1].astype(np.float32);b=seq_frames[-2].astype(np.float32);diff=np.mean(np.abs(a-b))/255.0
        except Exception:
            diff=0.0
        bonus=0.0;s=(txt or "").lower()
        if any(k in s for k in ["score","win","victory","complete","success","level"]):bonus+=1.0
        if any(k in s for k in ["fail","lose","game over","defeat"]):bonus-=1.0
        m=re.findall(r"\d+",s)
        if len(m)>=2:
            try:
                if int(m[-1])>int(m[-2]):bonus+=0.5
            except Exception:
                pass
        return float(np.clip(diff*0.5+bonus,-1.0,2.0))
    def _optimize_loop(self):
        backup=None
        try:
            if os.path.exists(model_path):backup=torch.load(model_path,map_location="cpu")
        except Exception:backup=None
        hot=self.metrics["temp"]>=80 or self.metrics["pwr"]>=0.95
        seq_len=2 if hot else 4
        ds=DiskExperienceDataset(experience_dir,seq=seq_len,aug=True)
        M=max(self.metrics["cpu"],self.metrics["mem"],self.metrics["gpu"],self.metrics["vram"])/100.0
        num_workers=max(0,min(6,os.cpu_count() or 2)-1)
        bs=max(4,int(8+64*(1.0-M)));lr=1e-4*(0.5 if hot else 1.0)
        for g in self.optimizer.param_groups:g["lr"]=lr
        prefetch_factor=4
        dl=torch.utils.data.DataLoader(ds,batch_size=bs,shuffle=True,num_workers=num_workers,pin_memory=True,drop_last=True,persistent_workers=(num_workers>0),prefetch_factor=prefetch_factor if num_workers>0 else None)
        epochs=3;total=max(1,(len(ds)//max(1,bs))*epochs);done_steps=0
        self.model.train();best_loss=None;c_scores={}
        start_time=time.time();self.schedule(lambda:self.pause_var.set("优化进行中，10秒切换规则失效"))
        for ep in range(epochs):
            if self.optimize_event.is_set():break
            it=iter(dl)
            while True:
                try:item=next(it)
                except StopIteration:break
                if self.optimize_event.is_set():break
                if isinstance(item,tuple) and len(item)==7:
                    seq,act,atype_t,ctrl_t,path_vec,texts,success=item
                else:
                    seq,act=item;atype_t=torch.zeros((seq.shape[0],),dtype=torch.long);ctrl_t=torch.full((seq.shape[0],),-1,dtype=torch.long);path_vec=torch.zeros((seq.shape[0],64),dtype=torch.float32);texts=[""]*seq.shape[0];success=torch.zeros((seq.shape[0],),dtype=torch.long)
                sample=self.buffer.sample(batch=max(2,bs//2),seq=seq_len)
                if sample is not None:
                    frames_b,actions_b,_,events_b,atypes_b,ctrls_b,txts_b=sample
                    seq_np=frames_b.astype(np.float32)
                    seq_t=torch.from_numpy(np.transpose(seq_np,(0,1,4,2,3))/255.0).to(device,non_blocking=True)
                    act_t=torch.from_numpy(actions_b).to(device,non_blocking=True)
                    atype_t=torch.from_numpy(atypes_b).to(device,non_blocking=True)
                    ctrl_t=torch.from_numpy(ctrls_b).to(device,non_blocking=True)
                    path_vec=torch.zeros((seq_t.shape[0],64),device=device,dtype=torch.float32)
                    texts=list(txts_b);success=torch.tensor([1 if any(k in (t or "").lower() for k in ["win","victory","success","complete"]) else 0 for t in texts],device=device,dtype=torch.long)
                else:
                    seq_t=torch.from_numpy(np.transpose(seq.numpy(),(0,1,4,2,3))/255.0).to(device,non_blocking=True);act_t=act.to(device,non_blocking=True);atype_t=atype_t.to(device,non_blocking=True);ctrl_t=ctrl_t.to(device,non_blocking=True);path_vec=path_vec.to(device,non_blocking=True).float();success=success.to(device,non_blocking=True).long();texts=list(texts)
                with torch.autocast(device_type="cuda",dtype=torch.float16,enabled=(device=="cuda")):
                    logits,values,atype_logits,elem_emb,path_pred=self.model(seq_t)
                    loss_policy=F.mse_loss(logits,act_t)
                    with torch.no_grad():
                        seq_np=(seq_t.detach().float().cpu().numpy()*255.0).transpose(0,1,3,4,2)
                        rewards=np.array([self._compute_reward(s,(texts[i] if i<len(texts) else "")) for i,s in enumerate(seq_np)],dtype=np.float32)
                        returns=torch.from_numpy(rewards).to(device)
                    loss_value=F.mse_loss(values,returns)
                    at_mask=(atype_t>=0)&(atype_t<=2)
                    loss_atype=F.cross_entropy(atype_logits[at_mask],atype_t[at_mask]) if at_mask.any() else torch.zeros((),device=device)
                    protoM,id2idx=self.pool.matrix()
                    if protoM is not None and protoM.shape[0]>0:
                        if ctrl_t.dim()==0:ctrl_t=ctrl_t.unsqueeze(0)
                        map_idx=torch.full_like(ctrl_t,-1)
                        if id2idx:
                            for i in range(ctrl_t.shape[0]):
                                cid=int(ctrl_t[i].item())
                                if cid in id2idx:map_idx[i]=id2idx[cid]
                        c_mask=map_idx>=0
                        loss_ctrl=torch.zeros((),device=device)
                        if c_mask.any():
                            sim=torch.matmul(elem_emb[c_mask],protoM.T)
                            loss_ctrl=F.cross_entropy(sim,map_idx[c_mask])
                    else:
                        loss_ctrl=torch.zeros((),device=device)
                    loss_path=F.mse_loss(path_pred,path_vec) if path_vec is not None else torch.zeros((),device=device)
                    goal_weight=torch.ones_like(values)
                    try:
                        gscores=self.goal_graph.granger()
                        for i in range(ctrl_t.shape[0]):
                            pid=int(ctrl_t[i].item())
                            if pid in gscores:goal_weight[i]=goal_weight[i]*(1.0+max(0.0,min(1.0,gscores[pid])))
                        c_scores=gscores
                    except Exception:
                        pass
                    loss_value=(loss_value*goal_weight.mean()).mean()
                    loss=loss_policy+0.5*loss_value+0.2*loss_atype+0.2*loss_ctrl+0.1*loss_path
                self.optimizer.zero_grad(set_to_none=True)
                if scaler is not None:
                    scaler.scale(loss).step(self.optimizer);scaler.update()
                else:
                    loss.backward();self.optimizer.step()
                done_steps+=1;prog=100.0*min(1.0,done_steps/max(1,total));self.progress_var.set(prog)
                if best_loss is None or float(loss.item())<best_loss:best_loss=float(loss.item())
                if self.optimize_event.is_set():break
        interrupted=self.optimize_event.is_set() if self.optimize_event is not None else False
        if not interrupted:
            try:torch.save(self.model.state_dict(),model_path)
            except Exception:pass
            self.progress_var.set(100.0);self._confirm_dialog();self.progress_var.set(0.0);self.progress_text.set("0%");self.sleep_btn.configure(state="normal" if self.models_ready() else "disabled");self.getup_btn.configure(state="disabled");self.recording_enabled=True;self.set_mode("learn")
        else:
            self.progress_var.set(0.0);self.progress_text.set("0%")
        self.schedule(lambda:self.pause_var.set(""))
    def stop(self):
        self.running=False
        try:self._stop_ai()
        except Exception:pass
        try:self.root.destroy()
        except Exception:pass
    def run(self):
        self.root.mainloop()
def main():
    app=App();app.run()
if __name__=="__main__":
    main()
