import sys,subprocess,importlib,os,threading,time,random,contextlib
def _pip(x,base_dir):
    try:
        importlib.import_module(x)
    except:
        try:
            subprocess.check_call([sys.executable,"-m","pip","download","-d",base_dir,"--no-input","--timeout","30",x])
        except:
            pass
        try:
            subprocess.check_call([sys.executable,"-m","pip","install","--upgrade","--no-input","--timeout","30",x])
        except:
            pass
home=os.path.expanduser("~");desk=os.path.join(home,"Desktop");base_dir=os.path.join(desk,"GameAI");os.makedirs(base_dir,exist_ok=True)
for p in ["psutil","pillow","numpy","opencv-python","mss","pynput","pyautogui","torch","torchvision","GPUtil","pynvml","pygetwindow","screeninfo"]:
    _pip(p,base_dir)
import psutil,pyautogui,torch,torch.nn as nn,torch.nn.functional as F,torch.optim as optim,GPUtil,cv2,numpy as np,mss
from pynvml import nvmlInit,nvmlDeviceGetHandleByIndex,nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo,nvmlDeviceGetCount
from pynput import mouse,keyboard
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk
import pygetwindow as gw
from screeninfo import get_monitors
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
model_dir=base_dir;model_path=os.path.join(model_dir,"model.pt");exp_dir=os.path.join(model_dir,"exp");os.makedirs(exp_dir,exist_ok=True)
scaler=torch.amp.GradScaler("cuda") if device=="cuda" else None
pre_url="https://download.pytorch.org/models/resnet18-f37072fd.pth";pre_path=os.path.join(model_dir,"resnet18.pth")
try:
    if not os.path.exists(pre_path):
        state=torch.hub.load_state_dict_from_url(pre_url,model_dir=model_dir,progress=False);torch.save(state,pre_path)
except:
    pass
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
        self.cap=cap;self.seq=seq;self.buf=[];self.src=[]
    def push(self,frame,act,source):
        if len(self.buf)>=self.cap:
            self.buf.pop(0);self.src.pop(0)
        self.buf.append((frame,act,_now()));self.src.append(source)
    def can_sample(self,batch):
        return len(self.buf)>self.seq+batch
    def sample(self,batch):
        idx=[];mx=len(self.buf)-self.seq-1
        for _ in range(batch):
            idx.append(random.randint(0,mx))
        seqs=[];acts=[];rewards=[];masks=[]
        for k in idx:
            fr=[self.buf[k+i][0] for i in range(self.seq)]
            ac=self.buf[k+self.seq-1][1]
            pre=self.buf[k+self.seq-2][0];post=self.buf[k+self.seq-1][0]
            pre=cv2.Canny(pre,32,128);post=cv2.Canny(post,32,128)
            r=float(np.mean(np.abs(post.astype(np.float32)-pre.astype(np.float32)))/255.0)
            seqs.append(np.stack(fr,0));acts.append(np.array(ac,dtype=np.float32));rewards.append(r);masks.append(1.0)
        return np.stack(seqs,0),np.stack(acts,0),np.array(rewards,dtype=np.float32),np.array(masks,dtype=np.float32)
class BrainInspired(nn.Module):
    def __init__(self,dim):
        super().__init__();self.w=nn.Parameter(torch.randn(dim,dim)*0.02);self.register_buffer("trace",torch.zeros(dim,dim))
    def forward(self,x):
        y=x@self.w;self.last=(x.detach(),y.detach());return y
    def hebbian_loss(self,lam=1e-3):
        x,y=self.last;corr=(x.T@y)/(x.shape[0]+1e-6);self.trace=self.trace.mul(0.95).add_(0.05*corr);return lam*torch.mean(self.trace**2)
class Net(nn.Module):
    def __init__(self,seq=4):
        super().__init__();self.seq=seq;self.enc=nn.Sequential(nn.Conv2d(3*seq,32,5,2,2),nn.ReLU(),nn.BatchNorm2d(32),nn.Conv2d(32,64,3,2,1),nn.ReLU(),nn.Dropout2d(0.1),nn.Conv2d(64,128,3,2,1),nn.ReLU(),nn.Dropout2d(0.1));self.pool=nn.AdaptiveAvgPool2d((8,8));self.proj=nn.Linear(128*64,256);self.attn=nn.MultiheadAttention(256,4,batch_first=True);self.lstm=nn.LSTM(256,256,batch_first=True);self.brain=BrainInspired(256);self.pi=nn.Linear(256,3);self.v=nn.Linear(256,1)
    def forward(self,x):
        b,t,c,h,w=x.shape;z=self.enc(x.reshape(b,c*t,h,w));z=self.pool(z).flatten(1);z=self.proj(z);z=z.view(b,1,256).repeat(1,t,1);z,_=self.attn(z,z,z,need_weights=False);z,_=self.lstm(z);z=self.brain(z[:,-1]);p=self.pi(z);v=self.v(z).squeeze(-1);dx=torch.tanh(p[:,0]);dy=torch.tanh(p[:,1]);cl=torch.sigmoid(p[:,2]);return torch.stack([dx,dy,cl],1),v
def _to_tensor(frames):
    x=torch.from_numpy(frames.astype(np.float32)/255.0);x=x.permute(0,1,4,2,3).contiguous();return x
class Agent:
    def __init__(self,seq=4,lr=1e-3):
        self.seq=seq;self.net=Net(seq).to(device);self.opt=optim.AdamW(self.net.parameters(),lr=lr,weight_decay=1e-4);self.sched=optim.lr_scheduler.ReduceLROnPlateau(self.opt,patience=2,factor=0.5);self.amp=device=="cuda";self.last_losses=(0,0,0)
    def act(self,frames):
        self.net.eval();ctx=torch.amp.autocast("cuda") if self.amp else contextlib.nullcontext()
        with ctx:
            x=_to_tensor(frames[None]).to(device);a,_=self.net(x);a=a[0].detach().cpu().numpy()
        return a
    def train_step(self,replay,batches=50,batch=32,stop_flag=None,progress_sink=None):
        total=batches;done=0;policy_loss=0;value_loss=0;reg_loss=0
        for _ in range(batches):
            if stop_flag and stop_flag.is_set():
                break
            if not replay.can_sample(batch):
                time.sleep(0.05);continue
            s,a,r,m=replay.sample(batch);x=_to_tensor(s).to(device);t=torch.from_numpy(a).to(device);rew=torch.from_numpy(r).to(device);mask=torch.from_numpy(m).to(device);self.net.train();ctx=torch.amp.autocast("cuda") if self.amp else contextlib.nullcontext()
            with ctx:
                pi,v=self.net(x);bcl=F.smooth_l1_loss(pi,t);adv=(rew-v.detach());pl=-torch.mean(-0.5*((pi-t)**2).sum(1)*adv);vl=F.mse_loss(v,rew);hl=self.net.brain.hebbian_loss(1e-3);loss=bcl+pl+0.5*vl+hl
            self.opt.zero_grad()
            if scaler:
                scaler.scale(loss).backward();scaler.unscale_(self.opt);torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0);scaler.step(self.opt);scaler.update()
            else:
                loss.backward();torch.nn.utils.clip_grad_norm_(self.net.parameters(),1.0);self.opt.step()
            self.sched.step((bcl+vl).item());policy_loss+=pl.item();value_loss+=vl.item();reg_loss+=hl.item();done+=1
            if progress_sink:
                progress_sink(int(done*100/max(total,1)))
        self.last_losses=(policy_loss/max(done,1),value_loss/max(done,1),reg_loss/max(done,1))
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
        self.img_ref=None;self.selected=None;self.selected_title="";self.rect=None;self.visible=False;self.complete=False;self.stop_all=False;self.mode="idle";self.optimizing=False;self.ai_acting=False;self.last_user_time=_now();self.last_any_time=_now();self.inactive_seconds=10.0;self.frame_seq=[];self.seq=4;self.exp=Replay(30000,self.seq);self.agent=Agent(self.seq,1e-3);self.capture_hz=30;self.last_user_action=[0.0,0.0,0.0];self.action_ema=[0.0,0.0,0.0];self.action_lock=threading.Lock();self.progress_val=0.0;self.cpu=0.0;self.mem=0.0;self.gpu=0.0;self.vram=0.0
        try:
            if os.path.exists(model_path):
                self.agent.net.load_state_dict(torch.load(model_path,map_location=device))
        except:
            pass
        self.refresh_windows();threading.Thread(target=self.resource_loop,daemon=True).start();threading.Thread(target=self.capture_loop,daemon=True).start()
        self.k_listener=keyboard.Listener(on_press=self.on_key_press);self.k_listener.start();self.m_listener=mouse.Listener(on_move=self.on_mouse,on_click=self.on_mouse,on_scroll=self.on_mouse);self.m_listener.start();self.root.bind("<Escape>",lambda e:self.quit());self.ui_tick()
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
            c,m,g,vr=_sys_usage();self.cpu=c;self.mem=m;self.gpu=g;self.vram=vr;M=max(c,m,g,vr)/100.0;self.capture_hz=int(_clamp(round(100*(1.0-M)),1,100));time.sleep(0.5)
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
                cx=(x1+x2)/2.0;cy=(y1+y2)/2.0;hw=max(1,(x2-x1)/2.0);hh=max(1,(y2-y1)/2.0);dx=float(_clamp((x-cx)/hw,-1.0,1.0));dy=float(_clamp((y-cy)/hh,-1.0,1.0))
                with self.action_lock:
                    self.last_user_action=[dx,dy,clicked]
        except:
            pass
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
    def optimize_thread(self):
        steps=80
        def sink(p):
            self.progress_val=p
        self.agent.train_step(self.exp,batches=steps,batch=32,stop_flag=self.opt_stop,progress_sink=sink)
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
                t=1.0/max(getattr(self,"capture_hz",30),1)
                try:
                    img=self.grab_frame(rect);self.last_frame=img
                    if not self.optimizing and self.mode in ["learn","train"]:
                        if len(self.frame_seq)>=self.seq:
                            self.frame_seq.pop(0)
                        self.frame_seq.append(cv2.resize(img,(160,120)))
                        if len(self.frame_seq)==self.seq:
                            if self.mode=="learn":
                                act=self.read_user_action();self.exp.push(self.frame_seq[-1],act,"user")
                            elif self.mode=="train":
                                act=self.ai_action_and_apply();self.exp.push(self.frame_seq[-1],act,"ai")
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
            a=0.2;self.action_ema=[a*dx+(1-a)*self.action_ema[0],a*dy+(1-a)*self.action_ema[1],max(cl,self.action_ema[2]*0.8)]
            return [float(_clamp(self.action_ema[0],-1,1)),float(_clamp(self.action_ema[1],-1,1)),float(_clamp(self.action_ema[2],0,1))]
        except:
            return [0.0,0.0,0.0]
    def ai_action_and_apply(self):
        if len(self.frame_seq)<self.seq:
            return [0.0,0.0,0.0]
        a=self.agent.act(np.stack(self.frame_seq,0))
        if self.mode!="train":
            return a.tolist()
        if not (self.visible and self.complete):
            return a.tolist()
        x1,y1,x2,y2=self.rect;cx=(x1+x2)//2;cy=(y1+y2)//2;dx=int(a[0]*20);dy=int(a[1]*20);px=cx+dx;py=cy+dy
        try:
            pyautogui.moveTo(px,py,duration=0.01)
            if a[2]>0.5:
                pyautogui.click()
        except:
            pass
        return a.tolist()
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
                time.sleep(0.06)
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