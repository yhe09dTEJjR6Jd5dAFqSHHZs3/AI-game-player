import os,sys,threading,time,queue,math,random,ctypes,json
import psutil,pyautogui,torch,torch.nn as nn,torch.optim as optim,torch.nn.functional as F,GPUtil,cv2,numpy as np,mss,requests,pygetwindow as gw
from pynput import mouse,keyboard
from pynvml import nvmlInit,nvmlDeviceGetCount,nvmlDeviceGetHandleByIndex,nvmlDeviceGetUtilizationRates,nvmlDeviceGetMemoryInfo
from PIL import Image,ImageTk
import tkinter as tk
from tkinter import ttk,messagebox
from screeninfo import get_monitors
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
home=os.path.expanduser("~")
base_dir=os.path.join(home,"Desktop","GameAI")
models_dir=os.path.join(base_dir,"models")
experience_dir=os.path.join(base_dir,"experience")
os.makedirs(models_dir,exist_ok=True)
os.makedirs(experience_dir,exist_ok=True)
model_path=os.path.join(models_dir,"policy.pt")
resnet_path=os.path.join(models_dir,"resnet18.pth")
yolo_path=os.path.join(models_dir,"yolov8n.pt")
clip_path=os.path.join(models_dir,"clip_vitb32.pt")
sam_path=os.path.join(models_dir,"sam_vit_b.pth")
device="cuda" if torch.cuda.is_available() else "cpu"
scaler=torch.amp.GradScaler("cuda") if device=="cuda" else None
try:
    nvmlInit()
    _gpu_count=nvmlDeviceGetCount()
except Exception:
    _gpu_count=0
class ModelSpec:
    def __init__(self,name,path,url,generator):
        self.name=name
        self.path=path
        self.url=url
        self.generator=generator
class ExperienceWriter:
    def __init__(self,root_dir):
        self.root_dir=root_dir
        self.learn_dir=os.path.join(root_dir,"learn")
        self.train_dir=os.path.join(root_dir,"train")
        os.makedirs(self.learn_dir,exist_ok=True)
        os.makedirs(self.train_dir,exist_ok=True)
        self.lock=threading.Lock()
    def record(self,frame,action,pos,source,rect,title,mode):
        if frame is None or rect is None:
            return
        arr=np.asarray(frame,dtype=np.uint8)
        norm_action=[float(action[0]),float(action[1]),float(action[2])]
        norm_pos=[float(pos[0]),float(pos[1])]
        data={"time":int(time.time()*1000),"action":norm_action,"pos":norm_pos,"source":int(source),"rect":[int(rect[0]),int(rect[1]),int(rect[2]),int(rect[3])],"window":title or "","mode":mode}
        directory=self.learn_dir if int(source)==1 else self.train_dir
        os.makedirs(directory,exist_ok=True)
        name=os.path.join(directory,f"{data['time']}_{int(source)}.npz")
        with self.lock:
            try:
                np.savez_compressed(name,frame=arr,meta=json.dumps(data,ensure_ascii=False))
            except Exception:
                pass
class ReplayBuffer:
    def __init__(self,cap=50000):
        self.cap=cap
        self.data=[]
        self.lock=threading.Lock()
    def add(self,frame,action,source):
        with self.lock:
            if len(self.data)>=self.cap:
                self.data.pop(0)
            self.data.append((frame.astype(np.uint8),np.array(action,dtype=np.float32),int(source)))
    def size(self):
        with self.lock:
            return len(self.data)
    def sample(self,batch,seq=4):
        with self.lock:
            if len(self.data)<seq+batch:
                return None
            idx=[random.randint(0,len(self.data)-seq) for _ in range(batch)]
            frames=[]
            actions=[]
            sources=[]
            for i in idx:
                seq_frames=[self.data[i+j][0] for j in range(seq)]
                frames.append(np.stack(seq_frames,0))
                actions.append(self.data[i+seq-1][1])
                sources.append(self.data[i+seq-1][2])
            return np.stack(frames,0),np.stack(actions,0),np.array(sources)
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
    def forward(self,x):
        emb=self.encoder(x)
        logits=self.policy(emb)
        value=self.value(emb).squeeze(-1)
        dx=torch.tanh(logits[:,0])
        dy=torch.tanh(logits[:,1])
        click=torch.sigmoid(logits[:,2])
        return torch.stack([dx,dy,click],1),value
def generate_policy(path):
    net=PolicyNet()
    torch.save(net.state_dict(),path)
    return True
def generate_placeholder(path):
    torch.save({"weights":[]},path)
    return True
class ModelManager:
    def __init__(self,app):
        self.app=app
        self.specs=[ModelSpec("policy",model_path,"https://example.com/policy.pt",generate_policy),ModelSpec("resnet",resnet_path,"https://download.pytorch.org/models/resnet18-f37072fd.pth",generate_placeholder),ModelSpec("yolo",yolo_path,"https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt",generate_placeholder),ModelSpec("clip",clip_path,"https://huggingface.co/openai/clip-vit-base-patch32/resolve/main/pytorch_model.bin",generate_placeholder),ModelSpec("sam",sam_path,"https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",generate_placeholder)]
    def ensure(self):
        threading.Thread(target=self._worker,daemon=True).start()
    def _worker(self):
        for spec in self.specs:
            if os.path.exists(spec.path):
                continue
            done=False
            while not done and self.app.running:
                try:
                    if self._download(spec.url,spec.path):
                        done=True
                        break
                except Exception:
                    pass
                if not done:
                    choice=self.app.prompt_retry_or_local(spec.name)
                    if choice=="retry":
                        continue
                    if choice=="local":
                        if spec.generator(spec.path):
                            done=True
                        else:
                            self.app.schedule(lambda n=spec.name:messagebox.showerror("Error",f"Failed to generate {n} locally."))
                    else:
                        continue
    def _download(self,url,path):
        try:
            with requests.get(url,timeout=45,stream=True) as r:
                r.raise_for_status()
                tmp=path+".download"
                with open(tmp,"wb") as f:
                    for chunk in r.iter_content(65536):
                        if chunk:
                            f.write(chunk)
                os.replace(tmp,path)
            return True
        except Exception:
            if os.path.exists(path+".download"):
                try:
                    os.remove(path+".download")
                except Exception:
                    pass
            return False
class App:
    def __init__(self):
        self.root=tk.Tk()
        self.root.title("Game AI")
        self.ui_queue=queue.Queue()
        self.running=True
        self.writer=ExperienceWriter(experience_dir)
        self.buffer=ReplayBuffer()
        self.model=PolicyNet().to(device)
        if os.path.exists(model_path):
            try:
                self.model.load_state_dict(torch.load(model_path,map_location=device),strict=False)
            except Exception:
                pass
        self.optimizer=optim.Adam(self.model.parameters(),lr=1e-4)
        self.capture_interval=0.05
        self.metrics={"cpu":0.0,"mem":0.0,"gpu":0.0,"vram":0.0,"freq":20.0}
        self.frame=None
        self.photo=None
        self.frame_lock=threading.Lock()
        self.frame_history=[]
        self.history_lock=threading.Lock()
        self.mode="learn"
        self.recording_enabled=True
        self.optimize_event=None
        self.optimize_thread=None
        self.ai_thread=None
        self.ai_stop=threading.Event()
        self.selected_title=tk.StringVar()
        self.window_state=tk.StringVar(value="No window")
        self.visibility_state=tk.StringVar(value="Unknown")
        self.mode_var=tk.StringVar(value="Learning")
        self.cpu_var=tk.StringVar(value="CPU:0.0%")
        self.mem_var=tk.StringVar(value="Memory:0.0%")
        self.gpu_var=tk.StringVar(value="GPU:0.0%")
        self.vram_var=tk.StringVar(value="VRAM:0.0%")
        self.progress_var=tk.DoubleVar(value=0.0)
        self.last_user_input=time.time()
        self.window_obj=None
        self.window_rect=None
        self.window_visible=False
        self.window_full=False
        self.visibility_text="Unknown"
        self.status_lock=threading.Lock()
        self.monitor_bounds=[(m.x,m.y,m.width,m.height) for m in get_monitors()]
        self.mss_ctx=mss.mss()
        self.capture_thread=threading.Thread(target=self._capture_loop,daemon=True)
        self.monitor_thread=threading.Thread(target=self._monitor_loop,daemon=True)
        self.listener_mouse=mouse.Listener(on_move=self._on_mouse_move,on_click=self._on_mouse_click)
        self.listener_keyboard=keyboard.Listener(on_press=self._on_key_press)
        self._build_ui()
        self.model_manager=ModelManager(self)
        self.model_manager.ensure()
        self.capture_thread.start()
        self.monitor_thread.start()
        self.listener_mouse.start()
        self.listener_keyboard.start()
        self.root.protocol("WM_DELETE_WINDOW",self.stop)
        self.root.after(50,self._update_ui)
        self.root.after(200,self._check_mode_switch)
    def _build_ui(self):
        self.root.columnconfigure(0,weight=1)
        top=tk.Frame(self.root)
        top.grid(row=0,column=0,sticky="nsew")
        top.columnconfigure(1,weight=1)
        ttk.Label(top,text="Window:").grid(row=0,column=0,sticky="w")
        self.window_combo=ttk.Combobox(top,textvariable=self.selected_title,state="readonly",width=50)
        self.window_combo.grid(row=0,column=1,sticky="ew")
        ttk.Button(top,text="Refresh",command=self.refresh_windows).grid(row=0,column=2,sticky="ew")
        ttk.Button(top,text="Select",command=self.select_window).grid(row=0,column=3,sticky="ew")
        ttk.Label(top,textvariable=self.window_state).grid(row=1,column=0,columnspan=2,sticky="w")
        ttk.Label(top,textvariable=self.visibility_state).grid(row=1,column=2,columnspan=2,sticky="e")
        mid=tk.Frame(self.root)
        mid.grid(row=1,column=0,sticky="nsew")
        mid.rowconfigure(0,weight=1)
        mid.columnconfigure(0,weight=1)
        self.frame_label=tk.Label(mid)
        self.frame_label.grid(row=0,column=0,sticky="nsew")
        stats=tk.Frame(self.root)
        stats.grid(row=2,column=0,sticky="ew")
        for i in range(4):
            stats.columnconfigure(i,weight=1)
        ttk.Label(stats,textvariable=self.cpu_var).grid(row=0,column=0,sticky="w")
        ttk.Label(stats,textvariable=self.mem_var).grid(row=0,column=1,sticky="w")
        ttk.Label(stats,textvariable=self.gpu_var).grid(row=0,column=2,sticky="w")
        ttk.Label(stats,textvariable=self.vram_var).grid(row=0,column=3,sticky="w")
        control=tk.Frame(self.root)
        control.grid(row=3,column=0,sticky="ew")
        control.columnconfigure(0,weight=1)
        control.columnconfigure(1,weight=1)
        ttk.Label(control,textvariable=self.mode_var).grid(row=0,column=0,columnspan=2,sticky="w")
        self.sleep_btn=ttk.Button(control,text="Sleep",command=self.on_sleep)
        self.sleep_btn.grid(row=1,column=0,sticky="ew")
        self.getup_btn=ttk.Button(control,text="Get Up",command=self.on_getup,state="disabled")
        self.getup_btn.grid(row=1,column=1,sticky="ew")
        self.progress=ttk.Progressbar(control,variable=self.progress_var,maximum=100)
        self.progress.grid(row=2,column=0,columnspan=2,sticky="ew")
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
            self.window_state.set(f"Selected:{title}")
            self._update_window_rect()
    def _update_window_rect(self):
        if self.window_obj is None:
            self.window_rect=None
            self.window_visible=False
            self.window_full=False
            with self.status_lock:
                self.visibility_text="No window"
            return
        try:
            self.window_obj.refresh()
            rect=(self.window_obj.left,self.window_obj.top,self.window_obj.right,self.window_obj.bottom)
            self.window_rect=rect
            self.window_visible=self._check_visibility()
            self.window_full=self.window_visible
            status="Visible" if self.window_visible else "Hidden"
            with self.status_lock:
                self.visibility_text=status
        except Exception:
            self.window_rect=None
            self.window_visible=False
            self.window_full=False
            with self.status_lock:
                self.visibility_text="Unknown"
    def _check_visibility(self):
        if self.window_obj is None:
            return False
        try:
            if not self.window_obj.isVisible:
                return False
            if self.window_obj.isMinimized:
                return False
            rect=(self.window_obj.left,self.window_obj.top,self.window_obj.right,self.window_obj.bottom)
            if rect[2]<=rect[0] or rect[3]<=rect[1]:
                return False
            monitor_ok=False
            for mx,my,mw,mh in self.monitor_bounds:
                if rect[0]>=mx and rect[1]>=my and rect[2]<=mx+mw and rect[3]<=my+mh:
                    monitor_ok=True
                    break
            if not monitor_ok:
                return False
            import win32gui
            top=win32gui.GetForegroundWindow()
            if top!=self.window_obj._hWnd:
                return False
            return True
        except Exception:
            return False
    def schedule(self,func):
        if self.running:
            self.ui_queue.put(func)
    def prompt_retry_or_local(self,name):
        result=queue.Queue(maxsize=1)
        def dialog():
            top=tk.Toplevel(self.root)
            top.title(f"Model {name}")
            ttk.Label(top,text=f"Download {name} failed. Choose action:").grid(row=0,column=0,columnspan=2,sticky="w")
            def choose(val):
                if result.empty():
                    result.put(val)
                top.destroy()
            ttk.Button(top,text="Retry",command=lambda:choose("retry")).grid(row=1,column=0,sticky="ew")
            ttk.Button(top,text="Local",command=lambda:choose("local")).grid(row=1,column=1,sticky="ew")
            top.grab_set()
            top.protocol("WM_DELETE_WINDOW",lambda:choose("retry"))
        self.schedule(dialog)
        return result.get()
    def _monitor_loop(self):
        while self.running:
            cpu=float(psutil.cpu_percent(interval=None))
            mem=float(psutil.virtual_memory().percent)
            gpu,mem_gpu=self._gpu_metrics()
            M=max(cpu,mem,gpu,mem_gpu)/100.0
            freq=max(1.0,100.0*(1.0-M))
            interval=1.0/max(freq,1.0)
            self.metrics={"cpu":cpu,"mem":mem,"gpu":gpu,"vram":mem_gpu,"freq":freq}
            self.capture_interval=interval
            time.sleep(0.5)
    def _gpu_metrics(self):
        try:
            if _gpu_count>0:
                handle=nvmlDeviceGetHandleByIndex(0)
                util=nvmlDeviceGetUtilizationRates(handle)
                mem=nvmlDeviceGetMemoryInfo(handle)
                gpu=float(util.gpu)
                vram=float(mem.used*100.0/max(mem.total,1))
                return gpu,vram
        except Exception:
            pass
        try:
            gpus=GPUtil.getGPUs()
            if gpus:
                g=gpus[0]
                return float(g.load*100.0),float(g.memoryUtil*100.0)
        except Exception:
            pass
        return 0.0,0.0
    def _capture_loop(self):
        while self.running:
            start=time.time()
            if self.window_obj is not None:
                self._update_window_rect()
                if self.window_visible and self.window_rect is not None:
                    left,top,right,bottom=self.window_rect
                    width=right-left
                    height=bottom-top
                    if width>0 and height>0:
                        try:
                            shot=self.mss_ctx.grab({"left":left,"top":top,"width":width,"height":height})
                            frame=np.array(shot)
                            frame=cv2.cvtColor(frame,cv2.COLOR_BGRA2BGR)
                            with self.frame_lock:
                                self.frame=frame
                            self._append_history(frame)
                            self._auto_record_frame(frame)
                        except Exception:
                            pass
                else:
                    with self.frame_lock:
                        self.frame=None
            elapsed=time.time()-start
            wait=max(self.capture_interval-elapsed,0.005)
            time.sleep(wait)
    def _append_history(self,frame):
        with self.history_lock:
            self.frame_history.append(frame)
            if len(self.frame_history)>8:
                self.frame_history.pop(0)
    def _auto_record_frame(self,frame):
        if not self.recording_enabled:
            return
        if self.mode in ("learn","train") and self.window_visible and self.window_rect is not None:
            rect=self.window_rect
            center=((rect[0]+rect[2])/2.0,(rect[1]+rect[3])/2.0)
            action=[0.0,0.0,0.0]
            pos=[(center[0]-rect[0])/(rect[2]-rect[0]),(center[1]-rect[1])/(rect[3]-rect[1])]
            source=1 if self.mode=="learn" else 2
            title=self.window_obj.title if self.window_obj else ""
            self.writer.record(frame,action,pos,source,rect,title,self.mode)
            self.buffer.add(frame,action,source)
    def _on_mouse_move(self,x,y):
        self.last_user_input=time.time()
        if self.mode=="learn" and self.window_visible and self.window_rect is not None and self.recording_enabled:
            rect=self.window_rect
            if rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]:
                width=max(rect[2]-rect[0],1)
                height=max(rect[3]-rect[1],1)
                norm_x=(x-rect[0])/width
                norm_y=(y-rect[1])/height
                action=[0.0,0.0,0.0]
                frame=self._current_frame_copy()
                if frame is not None:
                    self.writer.record(frame,action,[norm_x,norm_y],1,rect,self.window_obj.title if self.window_obj else "",self.mode)
                    self.buffer.add(frame,action,1)
    def _on_mouse_click(self,x,y,button,pressed):
        self.last_user_input=time.time()
        if pressed and self.mode=="learn" and self.window_visible and self.window_rect is not None and self.recording_enabled:
            rect=self.window_rect
            if rect[0]<=x<=rect[2] and rect[1]<=y<=rect[3]:
                width=max(rect[2]-rect[0],1)
                height=max(rect[3]-rect[1],1)
                norm_x=(x-rect[0])/width
                norm_y=(y-rect[1])/height
                action=[0.0,0.0,1.0]
                frame=self._current_frame_copy()
                if frame is not None:
                    self.writer.record(frame,action,[norm_x,norm_y],1,rect,self.window_obj.title if self.window_obj else "",self.mode)
                    self.buffer.add(frame,action,1)
    def _on_key_press(self,key):
        self.last_user_input=time.time()
        if key==keyboard.Key.esc:
            self.schedule(self.stop)
        if self.mode=="train":
            self.schedule(lambda:self.set_mode("learn"))
    def _current_frame_copy(self):
        with self.frame_lock:
            if self.frame is None:
                return None
            return self.frame.copy()
    def _update_ui(self):
        while not self.ui_queue.empty():
            func=self.ui_queue.get()
            try:
                func()
            except Exception:
                pass
        with self.frame_lock:
            frame=self.frame.copy() if self.frame is not None else None
        if frame is not None:
            rgb=cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            image=Image.fromarray(rgb)
            image=image.resize((min(640,image.width),min(360,image.height)))
            self.photo=ImageTk.PhotoImage(image=image)
            self.frame_label.configure(image=self.photo)
        else:
            self.frame_label.configure(image="")
        with self.status_lock:
            vis=self.visibility_text
        self.visibility_state.set(vis)
        self.cpu_var.set(f"CPU:{self.metrics['cpu']:.1f}%")
        self.mem_var.set(f"Memory:{self.metrics['mem']:.1f}%")
        self.gpu_var.set(f"GPU:{self.metrics['gpu']:.1f}%")
        self.vram_var.set(f"VRAM:{self.metrics['vram']:.1f}%")
        self.root.after(50,self._update_ui)
    def _check_mode_switch(self):
        if self.mode=="learn" and self.recording_enabled and self.window_visible:
            if time.time()-self.last_user_input>=10.0:
                self.set_mode("train")
        self.root.after(200,self._check_mode_switch)
    def set_mode(self,mode):
        if not self.running:
            return
        if mode==self.mode:
            return
        if mode=="train":
            if not self.window_visible:
                return
            self.mode="train"
            self.mode_var.set("Training")
            self._start_ai()
        elif mode=="learn":
            self.mode="learn"
            self.mode_var.set("Learning")
            self._stop_ai()
        elif mode=="optimize":
            self.mode="optimize"
            self.mode_var.set("Optimizing")
            self._stop_ai()
    def _start_ai(self):
        self.ai_stop.clear()
        if self.ai_thread is None or not self.ai_thread.is_alive():
            self.ai_thread=threading.Thread(target=self._ai_loop,daemon=True)
            self.ai_thread.start()
    def _stop_ai(self):
        self.ai_stop.set()
    def _ai_loop(self):
        while self.running and not self.ai_stop.is_set():
            if self.mode!="train" or not self.window_visible or self.window_rect is None:
                time.sleep(0.05)
                continue
            frames=self._get_history_tensor()
            if frames is None:
                time.sleep(0.05)
                continue
            with torch.no_grad():
                self.model.eval()
                action,_=self.model(frames)
            action=action.squeeze(0).cpu().numpy()
            dx=float(action[0])
            dy=float(action[1])
            click_prob=float(action[2])
            rect=self.window_rect
            width=max(rect[2]-rect[0],1)
            height=max(rect[3]-rect[1],1)
            target_x=rect[0]+width*(0.5+dx*0.25)
            target_y=rect[1]+height*(0.5+dy*0.25)
            pyautogui.moveTo(target_x,target_y,duration=0.01)
            if click_prob>0.7:
                pyautogui.click()
            frame=self._current_frame_copy()
            if frame is not None and self.recording_enabled:
                norm_x=(target_x-rect[0])/width
                norm_y=(target_y-rect[1])/height
                self.writer.record(frame,[dx,dy,click_prob],[norm_x,norm_y],2,rect,self.window_obj.title if self.window_obj else "",self.mode)
                self.buffer.add(frame,[dx,dy,click_prob],2)
            time.sleep(0.05)
    def _get_history_tensor(self):
        with self.history_lock:
            if len(self.frame_history)<4:
                return None
            frames=self.frame_history[-4:]
        arr=np.stack(frames,0)
        arr=arr.astype(np.float32)/255.0
        arr=np.transpose(arr,(0,3,1,2))
        arr=np.expand_dims(arr,0)
        tensor=torch.from_numpy(arr).to(device)
        return tensor
    def on_sleep(self):
        if self.mode in ("learn","train") and self.recording_enabled:
            self.recording_enabled=False
            self.getup_btn.configure(state="normal")
            self.sleep_btn.configure(state="disabled")
            self.progress_var.set(0.0)
            self.set_mode("optimize")
            self.optimize_event=threading.Event()
            self.optimize_thread=threading.Thread(target=self._optimize_loop,daemon=True)
            self.optimize_thread.start()
    def on_getup(self):
        if self.mode=="optimize" and self.optimize_event is not None:
            self.optimize_event.set()
            self.progress_var.set(0.0)
            self.sleep_btn.configure(state="normal")
            self.getup_btn.configure(state="disabled")
            self.recording_enabled=True
            self.set_mode("learn")
    def _optimize_loop(self):
        epochs=3
        batch=8
        total_steps=epochs*max(self.buffer.size(),1)
        step=0
        for epoch in range(epochs):
            if self.optimize_event.is_set():
                break
            while not self.optimize_event.is_set():
                sample=self.buffer.sample(batch)
                if sample is None:
                    break
                frames,actions,sources=sample
                frames=frames.astype(np.float32)/255.0
                frames=np.transpose(frames,(0,1,4,2,3))
                frames=torch.from_numpy(frames).to(device)
                actions=torch.from_numpy(actions).to(device)
                self.model.train()
                self.optimizer.zero_grad()
                logits,values=self.model(frames)
                loss_policy=F.mse_loss(logits,actions)
                loss_value=values.pow(2).mean()
                loss=loss_policy+0.1*loss_value
                loss.backward()
                if scaler is not None:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(),1.0)
                self.optimizer.step()
                step+=1
                progress=min(100.0,(step/max(total_steps,1))*100.0)
                self.schedule(lambda v=progress:self.progress_var.set(v))
                if self.optimize_event.is_set():
                    break
            if sample is None:
                time.sleep(0.5)
        if not self.optimize_event.is_set():
            torch.save(self.model.state_dict(),model_path)
            self.optimize_thread=None
            self.schedule(self._show_optimize_done)
        else:
            self.optimize_thread=None
            self.schedule(self._reset_after_opt_cancel)
    def _show_optimize_done(self):
        dialog=tk.Toplevel(self.root)
        dialog.title("Optimization")
        ttk.Label(dialog,text="Optimization complete.").grid(row=0,column=0,sticky="ew")
        def confirm():
            dialog.destroy()
            self.progress_var.set(0.0)
            self.sleep_btn.configure(state="normal")
            self.getup_btn.configure(state="disabled")
            self.recording_enabled=True
            self.set_mode("learn")
            self.optimize_event=None
            self.optimize_thread=None
        ttk.Button(dialog,text="Confirm",command=confirm).grid(row=1,column=0,sticky="ew")
        dialog.grab_set()
    def _reset_after_opt_cancel(self):
        self.progress_var.set(0.0)
        self.sleep_btn.configure(state="normal")
        self.getup_btn.configure(state="disabled")
        self.recording_enabled=True
        self.set_mode("learn")
        self.optimize_event=None
        self.optimize_thread=None
    def stop(self):
        if not self.running:
            return
        self.running=False
        self._stop_ai()
        if self.optimize_event is not None:
            self.optimize_event.set()
        self.root.destroy()
    def run(self):
        self.root.mainloop()
if __name__=="__main__":
    App().run()
