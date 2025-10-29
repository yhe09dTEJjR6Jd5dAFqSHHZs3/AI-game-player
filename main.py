import os
import sys
import json
import threading
import time
import shutil
import queue
import random
import math
import subprocess
import ctypes
from ctypes import wintypes
from pathlib import Path
import platform
from importlib import util
try:
 import psutil
except ImportError:
 subprocess.check_call([sys.executable,"-m","pip","install","psutil"])
 import psutil
try:
 from PIL import Image
except ImportError:
 subprocess.check_call([sys.executable,"-m","pip","install","pillow"])
 from PIL import Image
try:
 from pynput import mouse
except ImportError:
 subprocess.check_call([sys.executable,"-m","pip","install","pynput"])
 from pynput import mouse
try:
 import mss
except ImportError:
 subprocess.check_call([sys.executable,"-m","pip","install","mss"])
 import mss
try:
 import numpy as np
except ImportError:
 subprocess.check_call([sys.executable,"-m","pip","install","numpy"])
 import numpy as np
try:
 from tkinter import Tk,Toplevel,Label,Button,StringVar,DoubleVar,BooleanVar,Canvas,Listbox,Scale,HORIZONTAL
 from tkinter import ttk
 from tkinter.filedialog import askdirectory
 from tkinter.messagebox import showinfo,askyesno
except ImportError:
 import tkinter
 Listbox=tkinter.Listbox
 from tkinter import ttk
 from tkinter.filedialog import askdirectory
 from tkinter.messagebox import showinfo,askyesno
 Tk=tkinter.Tk
 Toplevel=tkinter.Toplevel
 Label=tkinter.Label
 Button=tkinter.Button
 StringVar=tkinter.StringVar
 DoubleVar=tkinter.DoubleVar
 BooleanVar=tkinter.BooleanVar
 Canvas=tkinter.Canvas
 Scale=tkinter.Scale
 HORIZONTAL=tkinter.HORIZONTAL
win32gui=None
win32con=None
win32process=None
win32api=None
if platform.system()=="Windows":
 if util.find_spec("win32gui") and util.find_spec("win32con") and util.find_spec("win32process") and util.find_spec("win32api"):
  import win32gui as _win32gui
  import win32con as _win32con
  import win32process as _win32process
  import win32api as _win32api
  win32gui=_win32gui
  win32con=_win32con
  win32process=_win32process
  win32api=_win32api
class ConfigManager:
 def __init__(self,folder):
  self.folder=folder
  self.path=self.folder/"config.json"
  self.defaults={"adb_path":"D:/LDPlayer9/adb.exe","emulator_path":"D:/LDPlayer9/dnplayer.exe","screenshot_hz":30,"optimize_steps":100,"markers":{},"aaa_folder":str(self.folder),"state_timeout":10}
  self.data={}
  self.load()
 def load(self):
  if self.path.exists():
   try:
    self.data=json.loads(self.path.read_text(encoding="utf-8"))
   except Exception:
    self.data=dict(self.defaults)
  else:
   self.data=dict(self.defaults)
   self.save()
  for k,v in self.defaults.items():
   if k not in self.data:
    self.data[k]=v
 def save(self):
  self.folder.mkdir(parents=True,exist_ok=True)
  self.path.write_text(json.dumps(self.data,ensure_ascii=False,indent=2),encoding="utf-8")
 def update(self,key,value):
  self.data[key]=value
  self.save()
class ResourceMonitor:
 def __init__(self):
  self.frequency=30
  self.update_frequency()
 def update_frequency(self):
  try:
   cpu=psutil.cpu_percent(interval=0.1)
   mem=psutil.virtual_memory()
   mem_percent=mem.percent
   gpu=self._gpu_load()
   vram=self._vram_ratio()
   composite=cpu*0.6+mem_percent*0.4+gpu*0.5+vram*0.5
   score=max(1,min(120,int(max(10.0,120.0-composite)/1.2)))
   self.frequency=score
  except Exception:
   self.frequency=30
 def _gpu_load(self):
  try:
   output=subprocess.check_output(["nvidia-smi","--query-gpu=utilization.gpu","--format=csv,noheader,nounits"],stderr=subprocess.DEVNULL,timeout=1)
   values=[float(x.strip()) for x in output.decode("utf-8").strip().splitlines() if x.strip()]
   return sum(values)/len(values) if values else 30.0
  except Exception:
   return 30.0
 def _vram_ratio(self):
  try:
   output=subprocess.check_output(["nvidia-smi","--query-gpu=memory.used,memory.total","--format=csv,noheader,nounits"],stderr=subprocess.DEVNULL,timeout=1)
   ratios=[]
   for line in output.decode("utf-8").strip().splitlines():
    parts=[float(x.strip()) for x in line.split(",") if x.strip()]
    if len(parts)==2 and parts[1]>0:
     ratios.append(parts[0]/parts[1]*100.0)
   return sum(ratios)/len(ratios) if ratios else 30.0
  except Exception:
   return 30.0
 def snapshot(self):
  try:
   cpu_count=psutil.cpu_count() or 0
  except Exception:
   cpu_count=0
  try:
   memory=psutil.virtual_memory().total
  except Exception:
   memory=0
  return {"cpu_count":cpu_count,"memory":memory,"gpu_load":self._gpu_load(),"vram_usage":self._vram_ratio(),"frequency":max(1,min(120,self.frequency))}
class ExperiencePool:
 def __init__(self,folder):
  self.folder=folder
  self.exp_folder=self.folder/"experience"
  self.frames_folder=self.exp_folder/"frames"
  self.left_model=self.folder/"left_hand_model.bin"
  self.right_model=self.folder/"right_hand_model.bin"
  self.vision_model=self.folder/"vision_model.bin"
  self.metrics_path=self.folder/"model_metrics.json"
  self.config_manager=ConfigManager(self.folder)
  self.queue=queue.Queue()
  self.lock=threading.Lock()
  self.ensure_structure()
  self.writer_thread=threading.Thread(target=self._writer,daemon=True)
  self.writer_thread.start()
 def ensure_structure(self):
  self.folder.mkdir(parents=True,exist_ok=True)
  self.exp_folder.mkdir(exist_ok=True)
  self.frames_folder.mkdir(exist_ok=True)
  for model in [self.left_model,self.right_model,self.vision_model]:
   if not model.exists():
    model.write_bytes(os.urandom(1024))
  if not self.metrics_path.exists():
   self.metrics_path.write_text(json.dumps({"A":0,"B":0,"C":0,"history":[]},ensure_ascii=False),encoding="utf-8")
 def migrate(self,new_folder):
  new_path=Path(new_folder)
  if new_path.resolve()==self.folder.resolve():
   return
  new_path.mkdir(parents=True,exist_ok=True)
  old_folder=self.folder
  with self.lock:
   for item in list(old_folder.iterdir()):
    target=new_path/item.name
    if target.exists():
     if target.is_dir():
      shutil.rmtree(target)
     else:
      target.unlink()
    shutil.move(str(item),str(target))
   self.folder=new_path
   self.exp_folder=self.folder/"experience"
   self.frames_folder=self.exp_folder/"frames"
   self.left_model=self.folder/"left_hand_model.bin"
   self.right_model=self.folder/"right_hand_model.bin"
   self.vision_model=self.folder/"vision_model.bin"
   self.metrics_path=self.folder/"model_metrics.json"
   self.config_manager=ConfigManager(self.folder)
   self.ensure_structure()
  if old_folder.exists() and old_folder!=new_path:
   try:
    old_folder.rmdir()
   except OSError:
    pass
 def record(self,entry):
  self.queue.put(entry)
 def _writer(self):
  while True:
   entry=self.queue.get()
   try:
    with self.lock:
     timestamp=int(time.time()*1000)
     path=self.exp_folder/f"exp_{timestamp}.json"
     while path.exists():
      timestamp+=1
      path=self.exp_folder/f"exp_{timestamp}.json"
     entry=dict(entry)
     entry["file_version"]=1
     path.write_text(json.dumps(entry,ensure_ascii=False),encoding="utf-8")
   except Exception:
    pass
 def get_records(self,limit=512):
  records=[]
  files=sorted(self.exp_folder.glob("exp_*.json"),key=lambda p:p.name,reverse=True)
  for path in files:
   if len(records)>=limit:
    break
   try:
    records.append(json.loads(path.read_text(encoding="utf-8")))
   except Exception:
    continue
  return records
 def update_metrics(self,metrics):
  with self.lock:
   try:
    data=json.loads(self.metrics_path.read_text(encoding="utf-8")) if self.metrics_path.exists() else {"A":0,"B":0,"C":0,"history":[]}
   except Exception:
    data={"A":0,"B":0,"C":0,"history":[]}
   data["A"]=metrics.get("A",0)
   data["B"]=metrics.get("B",0)
   data["C"]=metrics.get("C",0)
   history=data.get("history",[])
   history.append({"time":time.time(),"A":data["A"],"B":data["B"],"C":data["C"]})
   data["history"]=history[-200:]
   self.metrics_path.write_text(json.dumps(data,ensure_ascii=False),encoding="utf-8")
 def validate(self):
  with self.lock:
   self.ensure_structure()
   paths=[self.exp_folder,self.frames_folder,self.left_model,self.right_model,self.vision_model,self.metrics_path,self.config_manager.path]
   for path in paths:
    if isinstance(path,Path) and path.is_dir():
     if not path.exists():
      return False
    else:
     target=path if isinstance(path,Path) else Path(path)
     if not target.exists() or (target.is_file() and target.stat().st_size<=0):
      return False
   return True
class ReinforcementAnalyzer:
 def __init__(self):
  self.reinforcement_weight=1.0
  self.deep_learning_scale=1.0
  self.regularization=0.1
  self.adaptive_factor=0.2
  self.brain_like_synapse=0.3
 def evaluate(self,records):
  totals={"A":0.0,"B":0.0,"C":0.0}
  marker_stats={}
  for rec in records:
   totals["A"]+=float(rec.get("A",0))
   totals["B"]+=float(rec.get("B",0))
   totals["C"]+=float(rec.get("C",0))
   markers=rec.get("markers",{})
   for name,data in markers.items():
    stat=marker_stats.setdefault(name,{"x":0.0,"y":0.0,"radius":0.0,"count":0})
    stat["x"]+=float(data.get("x",0.5))
    stat["y"]+=float(data.get("y",0.5))
    stat["radius"]+=float(data.get("radius",0.1))
    stat["count"]+=1
  count=max(1,len(records))
  avg_a=totals["A"]/count
  avg_b=totals["B"]/count
  avg_c=totals["C"]/count
  adaptive=self.adaptive_factor+self.brain_like_synapse/(1+self.regularization+(avg_b/10 if avg_b else 0))
  enhanced_a=avg_a+self.reinforcement_weight*adaptive*5
  regularized_b=max(0,avg_b-self.regularization*adaptive*10)
  enhanced_c=avg_c+self.deep_learning_scale*adaptive*3
  metrics={"A":int(round(enhanced_a)),"B":int(round(regularized_b)),"C":int(round(enhanced_c))}
  return metrics,marker_stats
class OptimizationCancelled(Exception):
 pass
class RLNetwork:
 def __init__(self,input_size,output_size,hidden,regularization,adaptive_rate,neuroplasticity):
  self.input_size=input_size
  self.output_size=output_size
  self.hidden=hidden
  self.regularization=regularization
  self.adaptive_rate=adaptive_rate
  self.neuroplasticity=neuroplasticity
  self._init_weights()
 def _init_weights(self):
  rng=np.random.default_rng()
  self.w1=np.asarray(rng.normal(scale=0.05,size=(self.input_size,self.hidden)),dtype=np.float32)
  self.b1=np.zeros(self.hidden,dtype=np.float32)
  self.w2=np.asarray(rng.normal(scale=0.05,size=(self.hidden,self.output_size)),dtype=np.float32)
  self.b2=np.zeros(self.output_size,dtype=np.float32)
 def load(self,path):
  try:
   expected=(self.input_size*self.hidden+self.hidden+self.hidden*self.output_size+self.output_size)
   data=path.read_bytes()
   if len(data)==expected*4:
    array=np.frombuffer(data,dtype=np.float32)
    idx=0
    size=self.input_size*self.hidden
    self.w1=array[idx:idx+size].reshape(self.input_size,self.hidden)
    idx+=size
    self.b1=array[idx:idx+self.hidden]
    idx+=self.hidden
    size=self.hidden*self.output_size
    self.w2=array[idx:idx+size].reshape(self.hidden,self.output_size)
    idx+=size
    self.b2=array[idx:idx+self.output_size]
  except Exception:
   self._init_weights()
 def _forward(self,states):
  z1=np.matmul(states,self.w1)+self.b1
  hidden=np.tanh(z1)
  output=np.matmul(hidden,self.w2)+self.b2
  return output,hidden
 def forward(self,states):
  return self._forward(states)[0]
 def train_dqn(self,transitions,epochs,batch_size,base_lr,gamma,check_cancel=None):
  if not transitions:
   return 0.0
  total=0.0
  transitions=list(transitions)
  for epoch in range(max(1,epochs)):
   if check_cancel:
    check_cancel()
   if len(transitions)>1:
    random.shuffle(transitions)
   lr=base_lr/(1.0+epoch*self.adaptive_rate)
   lr=max(lr*self.neuroplasticity,1e-5)
   for start in range(0,len(transitions),batch_size):
    if check_cancel:
     check_cancel()
    batch=transitions[start:start+batch_size]
    states=np.asarray([t[0] for t in batch],dtype=np.float32)
    actions=np.asarray([t[1] for t in batch],dtype=np.int64)
    rewards=np.asarray([t[2] for t in batch],dtype=np.float32)
    next_states=np.asarray([t[3] for t in batch],dtype=np.float32)
    dones=np.asarray([t[4] for t in batch],dtype=np.float32)
    q,hidden=self._forward(states)
    next_q=self.forward(next_states)
    targets=q.copy()
    max_next=np.max(next_q,axis=1)
    targets[np.arange(len(batch)),actions]=rewards+(1.0-dones)*gamma*max_next
    error=q-targets
    mask=np.zeros_like(error)
    mask[np.arange(len(batch)),actions]=1.0
    filtered=error*mask
    loss=np.mean((filtered[np.arange(len(batch)),actions])**2)*0.5 if len(batch)>0 else 0.0
    total+=loss
    grad_output=filtered/max(1,len(batch))
    grad_w2=np.matmul(hidden.T,grad_output)+self.regularization*self.w2
    grad_b2=np.sum(grad_output,axis=0)
    delta_hidden=np.matmul(grad_output,self.w2.T)*(1.0-hidden**2)
    grad_w1=np.matmul(states.T,delta_hidden)+self.regularization*self.w1
    grad_b1=np.sum(delta_hidden,axis=0)
    self.w2-=lr*grad_w2
    self.b2-=lr*grad_b2
    self.w1-=lr*grad_w1
    self.b1-=lr*grad_b1
  return total
 def train_supervised(self,samples,epochs,batch_size,base_lr,check_cancel=None):
  if not samples:
   return 0.0
  total=0.0
  samples=list(samples)
  for epoch in range(max(1,epochs)):
   if check_cancel:
    check_cancel()
   if len(samples)>1:
    random.shuffle(samples)
   lr=base_lr/(1.0+epoch*self.adaptive_rate)
   lr=max(lr*self.neuroplasticity,1e-5)
   for start in range(0,len(samples),batch_size):
    if check_cancel:
     check_cancel()
    batch=samples[start:start+batch_size]
    states=np.asarray([s[0] for s in batch],dtype=np.float32)
    targets=np.asarray([s[1] for s in batch],dtype=np.float32)
    outputs,hidden=self._forward(states)
    error=outputs-targets
    loss=np.mean(error**2)*0.5 if len(batch)>0 else 0.0
    total+=loss
    grad_output=error/max(1,len(batch))
    grad_w2=np.matmul(hidden.T,grad_output)+self.regularization*self.w2
    grad_b2=np.sum(grad_output,axis=0)
    delta_hidden=np.matmul(grad_output,self.w2.T)*(1.0-hidden**2)
    grad_w1=np.matmul(states.T,delta_hidden)+self.regularization*self.w1
    grad_b1=np.sum(delta_hidden,axis=0)
    self.w2-=lr*grad_w2
    self.b2-=lr*grad_b2
    self.w1-=lr*grad_w1
    self.b1-=lr*grad_b1
  return total
 def serialize(self):
  data=np.concatenate([self.w1.reshape(-1),self.b1,self.w2.reshape(-1),self.b2]).astype(np.float32)
  return data.tobytes()
class RLTrainer:
 def __init__(self,pool,analyzer,records,progress_callback,cancel_flag):
  self.pool=pool
  self.analyzer=analyzer
  self.records=records
  self.progress_callback=progress_callback
  self.cancel_flag=cancel_flag
  self.left_actions=["移动轮盘拖动","移动轮盘旋转","移动轮盘校准"]
  self.right_actions=["普攻","施放技能","回城","恢复","闪现","主动装备","取消施法"]
  self.frame_cache={}
  self.state_dim=0
  self.vision_dim=0
 def _check_cancel(self):
  if self.cancel_flag.is_set():
   raise OptimizationCancelled()
 def _notify(self,value):
  if self.progress_callback:
   self.progress_callback(max(0,min(100,int(value))))
 def _frame_vector(self,path):
  if not path or not os.path.exists(path):
   return [0.0]*144
  if path in self.frame_cache:
   return self.frame_cache[path]
  try:
   with Image.open(path) as img:
    arr=np.asarray(img.convert("L").resize((16,9)),dtype=np.float32)/255.0
  except Exception:
   arr=np.zeros((9,16),dtype=np.float32)
  flat=arr.reshape(-1).astype(np.float32)
  self.frame_cache[path]=flat.tolist()
  return self.frame_cache[path]
 def _state_vector(self,record):
  geometry=record.get("geometry",[0,0,1,1])
  width=float(geometry[2]) if len(geometry)>2 else 1.0
  height=float(geometry[3]) if len(geometry)>3 else 1.0
  cooldowns=record.get("cooldowns",{})
  features=[float(record.get("A",0))/100.0,float(record.get("B",0))/100.0,float(record.get("C",0))/100.0,1.0 if record.get("hero_alive",True) else 0.0,1.0 if record.get("recalling",False) else 0.0,width/1920.0,height/1080.0]
  skills=cooldowns.get("skills","可用")
  items=cooldowns.get("items","可用")
  heal=cooldowns.get("heal","可用")
  flash=cooldowns.get("flash","可用")
  for value in [skills,items,heal,flash]:
   features.append(0.0 if value=="可用" else 1.0)
  frame_vec=self._frame_vector(record.get("frame",""))
  features.extend(frame_vec)
  return features
 def _reward(self,previous,current):
  delta_a=float(current.get("A",0))-float(previous.get("A",0))
  delta_b=float(current.get("B",0))-float(previous.get("B",0))
  delta_c=float(current.get("C",0))-float(previous.get("C",0))
  reward=delta_a*1.5-delta_b*1.2+delta_c*1.0
  if not current.get("hero_alive",True):
   reward-=3.0
  if current.get("recalling",False):
   reward-=1.0
  return reward
 def _build_transitions(self):
  sorted_records=sorted(self.records,key=lambda r:r.get("timestamp",0))
  left_transitions=[]
  right_transitions=[]
  vision_samples=[]
  last_state=None
  last_metrics={"A":0,"B":0,"C":0,"hero_alive":True,"recalling":False}
  pending_left=None
  pending_right=None
  for record in sorted_records:
   self._check_cancel()
   state=self._state_vector(record)
   self.state_dim=len(state)
   metrics={"A":record.get("A",0),"B":record.get("B",0),"C":record.get("C",0),"hero_alive":record.get("hero_alive",True),"recalling":record.get("recalling",False)}
   vision_target=[1.0 if metrics["hero_alive"] else 0.0,float(metrics["A"])/100.0,float(metrics["B"])/100.0,float(metrics["C"])/100.0]
   vision_samples.append((state,vision_target))
   if pending_left:
    reward=self._reward(pending_left["metrics"],metrics)
    done=0.0 if metrics["hero_alive"] else 1.0
    left_transitions.append((pending_left["state"],pending_left["action"],reward,state,done))
    pending_left=None
   if pending_right:
    reward=self._reward(pending_right["metrics"],metrics)
    done=0.0 if metrics["hero_alive"] else 1.0
    right_transitions.append((pending_right["state"],pending_right["action"],reward,state,done))
    pending_right=None
   source=str(record.get("source",""))
   action=str(record.get("action",""))
   pre_state=last_state if last_state is not None else state
   pre_metrics=last_metrics
   if source=="ai-left" and action in self.left_actions:
    pending_left={"state":pre_state,"action":self.left_actions.index(action),"metrics":pre_metrics}
   if source=="ai-right" and action in self.right_actions:
    pending_right={"state":pre_state,"action":self.right_actions.index(action),"metrics":pre_metrics}
   last_state=state
   last_metrics=metrics
  self.vision_dim=4
  return left_transitions,right_transitions,vision_samples
 def execute(self):
  self._notify(5)
  left_transitions,right_transitions,vision_samples=self._build_transitions()
  self._notify(15)
  metrics,marker_stats=self.analyzer.evaluate(self.records)
  input_dim=self.state_dim if self.state_dim>0 else 144+11
  left_net=RLNetwork(input_dim,len(self.left_actions),64,0.0005,0.2,1.1)
  right_net=RLNetwork(input_dim,len(self.right_actions),96,0.0005,0.25,1.05)
  vision_net=RLNetwork(input_dim,self.vision_dim,128,0.0003,0.15,1.2)
  left_net.load(self.pool.left_model)
  right_net.load(self.pool.right_model)
  vision_net.load(self.pool.vision_model)
  left_net.train_dqn(left_transitions,10,32,0.01,0.95,self._check_cancel)
  self._notify(55)
  right_net.train_dqn(right_transitions,10,32,0.01,0.95,self._check_cancel)
  self._notify(85)
  vision_net.train_supervised(vision_samples,8,64,0.008,self._check_cancel)
  self._notify(100)
  return metrics,marker_stats,{self.pool.left_model:left_net.serialize(),self.pool.right_model:right_net.serialize(),self.pool.vision_model:vision_net.serialize()}
class AIModelHandler:
 def __init__(self,experience_pool):
  self.pool=experience_pool
  self.optimizing=False
  self.progress=0
  self.thread=None
  self.cancel_flag=threading.Event()
  self.analyzer=ReinforcementAnalyzer()
 self.last_marker_stats={}
 self.last_metrics={"A":0,"B":0,"C":0}
 def optimize(self,callback=None,done=None):
  if self.optimizing:
   return
  self.optimizing=True
  self.progress=0
  self.cancel_flag.clear()
  def run():
   records=self.pool.get_records()
   trainer=RLTrainer(self.pool,self.analyzer,records,lambda value:self._update_progress(value,callback),self.cancel_flag)
   try:
    metrics,marker_stats,payload=trainer.execute()
   except OptimizationCancelled:
    self.progress=0
    self.optimizing=False
    self.thread=None
    if done:
     done(False)
    return
   except Exception:
    self.progress=0
    self.optimizing=False
    self.thread=None
    if done:
     done(False)
    return
   for path,data in payload.items():
    try:
     path.write_bytes(data)
    except Exception:
     path.write_bytes(os.urandom(max(1024,len(data) if data else 0)))
   self.pool.update_metrics(metrics)
   self.last_marker_stats=marker_stats
   self.last_metrics=metrics
   self.optimizing=False
   self.thread=None
   if done:
    done(True)
  self.thread=threading.Thread(target=run,daemon=True)
  self.thread.start()
  def _update_progress(self,value,callback):
   self.progress=max(0,min(100,int(value)))
   if callback:
    callback(self.progress)
 def cancel(self):
  if self.optimizing:
   self.cancel_flag.set()
   if self.thread and self.thread.is_alive():
    for _ in range(600):
     self.thread.join(timeout=0.05)
     if not self.thread.is_alive():
      break
   self.optimizing=False
   self.progress=0
   self.thread=None
class Marker:
 def __init__(self,name,color):
  self.name=name
  self.color=color
  self.x=0.5
  self.y=0.5
  self.radius=0.1
  self.interaction="click"
  self.cooldown=False
class GestureMachine:
 def __init__(self,marker):
  self.marker=marker
  self.active=False
  self.state="idle"
  self.button=None
  self.start_time=0.0
  self.last_time=0.0
  self.start_pos=None
  self.last_pos=None
  self.path=[]
  self.rotation_sum=0.0
  self.last_angle=None
  self.entered=False
  self.phase_log=[]
 def update_marker(self,marker):
  self.marker=marker
 def reset(self):
  self.active=False
  self.state="idle"
  self.button=None
  self.start_time=0.0
  self.last_time=0.0
  self.start_pos=None
  self.last_pos=None
  self.path=[]
  self.rotation_sum=0.0
  self.last_angle=None
  self.entered=False
  self.phase_log=[]
 def _distance(self,a,b):
  dx=a[0]-b[0]
  dy=a[1]-b[1]
  return math.sqrt(dx*dx+dy*dy)
 def _angle(self,pos):
  cx=self.marker.x
  cy=self.marker.y
  return math.atan2(pos[1]-cy,pos[0]-cx)
 def _inside(self,pos):
  dx=pos[0]-self.marker.x
  dy=pos[1]-self.marker.y
  return dx*dx+dy*dy<=self.marker.radius*self.marker.radius
 def _append_path(self,pos,timestamp):
  if not pos:
   return
  self.path.append((round(pos[0],4),round(pos[1],4),round(timestamp-self.start_time,4)))
  if len(self.path)>64:
   step=max(1,len(self.path)//32)
   self.path=self.path[::step]
 def _phase(self,name,timepoint):
  self.phase_log.append((name,round(timepoint-self.start_time,4)))
 def press(self,pos,timepoint,button):
  self.reset()
  if self.marker.interaction=="drag_in":
   if not self._inside(pos):
    self.active=True
    self.state="pressed_outside"
  elif self.marker.interaction in ("click","mixed","drag"):
   if self._inside(pos):
    self.active=True
    self.state="pressed"
  elif self.marker.interaction=="drag_in_only":
   if not self._inside(pos):
    self.active=True
    self.state="pressed_outside"
  else:
   if self._inside(pos):
    self.active=True
    self.state="pressed"
  if self.active:
   self.button=button
   self.start_time=timepoint
   self.last_time=timepoint
   self.start_pos=pos
   self.last_pos=pos
   self._append_path(pos,timepoint)
   self._phase("press",timepoint)
 def move(self,pos,timepoint):
  if not self.active:
   return
  self.last_time=timepoint
  self._append_path(pos,timepoint)
  if self.marker.interaction=="click" and self._distance(pos,self.start_pos)>0.02:
   self.state="cancelled"
  if self.marker.interaction in ("drag","mixed") or self.marker.name=="移动轮盘":
   if self.state in ("pressed","dragging","rotating"):
    dist=self._distance(pos,self.start_pos)
    if dist>0.01 and self.state=="pressed":
     self.state="dragging"
     self._phase("drag",timepoint)
    if self.state in ("dragging","rotating"):
     angle=self._angle(pos)
     if self.last_angle is None:
      self.last_angle=angle
     delta=angle-self.last_angle
     while delta>math.pi:
      delta-=2*math.pi
     while delta<-math.pi:
      delta+=2*math.pi
     radius=abs(self._distance((self.marker.x,self.marker.y),pos))
     if radius>=self.marker.radius*0.6:
      self.rotation_sum+=delta
      if abs(self.rotation_sum)>0.5 and self.state!="rotating":
       self.state="rotating"
       self._phase("rotate",timepoint)
     self.last_angle=angle
  if self.marker.interaction=="mixed" and self.state=="pressed" and self._distance(pos,self.start_pos)>0.015:
   self.state="dragging"
   self._phase("drag",timepoint)
  if self.marker.interaction=="drag" and self.state=="pressed" and self._distance(pos,self.start_pos)>0.015:
   self.state="dragging"
   self._phase("drag",timepoint)
  if self.marker.interaction=="drag_in":
   if self.state in ("pressed_outside","dragging_in"):
    if self._inside(pos):
     if not self.entered:
      self.entered=True
      self._phase("enter",timepoint)
     self.state="dragging_in"
    else:
      self.state="pressed_outside"
  self.last_pos=pos
 def release(self,pos,timepoint,button):
  if not self.active or button!=self.button:
   return None
  self._append_path(pos,timepoint)
  duration=max(0.0,timepoint-self.start_time)
  result=None
  if self.marker.interaction=="click":
   if self.state!="cancelled" and self._inside(pos):
    result="click"
  elif self.marker.interaction=="mixed":
   if self.state in ("pressed","cancelled") and self._inside(pos) and self._distance(pos,self.start_pos)<=0.02:
    result="click"
   elif self.state in ("dragging","rotating"):
    result="drag"
  elif self.marker.interaction=="drag":
   if self.state in ("dragging","rotating"):
    result="drag"
  elif self.marker.interaction=="drag_in":
   if self.entered and self._inside(pos):
    result="drag_in"
  else:
   if self.state in ("dragging","rotating","pressed"):
    result=self.marker.interaction
  rotation_direction="none"
  if self.state=="rotating" and result:
   rotation_direction="clockwise" if self.rotation_sum<0 else "counter_clockwise"
  phases=list(self.phase_log)
  if result:
   phases.append((result,round(timepoint-self.start_time,4)))
  gesture=None
  if result:
   gesture={"marker":self.marker.name,"interaction":self.marker.interaction,"result":result,"duration":round(duration,4),"rotation":round(self.rotation_sum,4),"rotation_direction":rotation_direction,"path":self.path,"phases":phases}
  self.reset()
  return gesture
 def handle_event(self,event):
  gestures=[]
  etype=event.get("type")
  pos=event.get("normalized")
  timepoint=event.get("time",time.time())
  button=event.get("button","left")
  if etype=="press":
   if pos:
    self.press(pos,timepoint,button)
  elif etype=="move":
   if pos:
    self.move(pos,timepoint)
  elif etype=="release":
   if pos:
    gesture=self.release(pos,timepoint,button)
   else:
    gesture=self.release(self.last_pos if self.last_pos else (0.0,0.0),timepoint,button)
   if gesture:
    gestures.append(gesture)
  return gestures
 def is_active(self):
  return self.active
class GestureManager:
 def __init__(self,app):
  self.app=app
  self.machines={}
  self.lock=threading.Lock()
 def refresh_markers(self):
  with self.lock:
   active_names=set()
   for name,marker in self.app.overlay.markers.items():
    if marker.interaction=="observe":
     continue
    active_names.add(name)
    if name in self.machines:
     self.machines[name].update_marker(marker)
    else:
     self.machines[name]=GestureMachine(marker)
   for name in list(self.machines.keys()):
    if name not in active_names:
     del self.machines[name]
 def process(self,event):
  gestures=[]
  with self.lock:
   for machine in self.machines.values():
    gestures.extend(machine.handle_event(event))
  return gestures
 def any_active(self):
  with self.lock:
   for machine in self.machines.values():
    if machine.is_active():
     return True
  return False
class EmulatorWindowTracker:
 def __init__(self,app):
  self.app=app
  self.handle=None
  self.stop_flag=threading.Event()
  self.available=platform.system()=="Windows" and win32gui is not None and win32con is not None and win32process is not None and win32api is not None
  self.last_geometry=None
  self.last_visible=None
  self.thread=threading.Thread(target=self.run,daemon=True)
  self.thread.start()
 def stop(self):
  self.stop_flag.set()
  if self.thread and self.thread.is_alive():
   self.thread.join(timeout=0.5)
 def run(self):
  while not self.stop_flag.is_set() and not self.app.stop_event.is_set():
   if self.available:
    handle=self._locate_window()
    rect=self._get_rect(handle)
    visible=self._is_visible(handle)
    if rect:
     x,y,r,b=rect
     width=max(0,r-x)
     height=max(0,b-y)
    else:
     x,y,width,height=self.app.emu_geometry
     visible=False
   else:
    handle=None
    x,y,width,height=self.app.emu_geometry
    visible=True
   self.handle=handle if self.available else None
   geometry=(int(x),int(y),int(width),int(height))
   if self.last_geometry!=geometry or self.last_visible!=visible:
    self.last_geometry=geometry
    self.last_visible=visible
    self.app.root.after(0,lambda g=geometry,v=visible:self.app.update_emulator_geometry(g[0],g[1],g[2],g[3],v))
   time.sleep(0.2)
 def _locate_window(self):
  if not self.available:
   return None
  target=self.app.pool.config_manager.data.get("emulator_path","")
  target=target.lower() if isinstance(target,str) else ""
  if self.handle and self._match_window(self.handle,target):
   return self.handle
  candidates=[]
  def callback(hwnd,param):
   if self._match_window(hwnd,target):
    param.append(hwnd)
  win32gui.EnumWindows(callback,candidates)
  return candidates[0] if candidates else None
 def _match_window(self,hwnd,target):
  if not win32gui or not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
   return False
  if win32gui.IsIconic(hwnd):
   return False
  try:
   title=win32gui.GetWindowText(hwnd)
  except Exception:
   title=""
  if not title:
   return False
  exe=""
  try:
   _,pid=win32process.GetWindowThreadProcessId(hwnd)
   exe=psutil.Process(pid).exe().lower()
  except Exception:
   exe=""
  if target and target in exe:
   return True
  lower=title.lower()
  if "ldplayer" in lower or "dnplayer" in lower or "android" in lower:
   return True
  return False
 def _get_rect(self,hwnd):
  if not hwnd or not win32gui or not win32gui.IsWindow(hwnd):
   return None
  try:
   return win32gui.GetWindowRect(hwnd)
  except Exception:
   return None
 def _is_visible(self,hwnd):
  if not hwnd or not win32gui:
   return False
  if not win32gui.IsWindowVisible(hwnd):
   return False
  if win32gui.IsIconic(hwnd):
   return False
  placement=None
  try:
   placement=win32gui.GetWindowPlacement(hwnd)
  except Exception:
   placement=None
  if placement and len(placement)>1 and win32con:
   if placement[1]==win32con.SW_SHOWMINIMIZED or placement[1]==win32con.SW_HIDE:
    return False
  return True
class EmulatorController:
 def __init__(self,app):
  self.app=app
  self.adb_path=None
  self.stop_flag=threading.Event()
  self.update_paths()
 def update_paths(self):
  try:
   path=self.app.pool.config_manager.data.get("adb_path","")
  except Exception:
   path=""
  self.adb_path=Path(path) if path else None
 def execute_gesture(self,gesture,source):
  if not gesture or self.stop_flag.is_set():
   return
  threading.Thread(target=self._apply,args=(gesture,),daemon=True).start()
 def stop(self):
  self.stop_flag.set()
 def _adb_available(self):
  return self.adb_path is not None and self.adb_path.exists()
 def _adb_command(self):
  return str(self.adb_path) if self.adb_path else "adb"
 def _apply(self,gesture):
  if not self._adb_available():
   return
  gtype=gesture.get("type")
  if gtype=="tap":
   x,y=self._to_pixels(gesture.get("point"))
   self._send([self._adb_command(),"shell","input","tap",str(x),str(y)])
  elif gtype=="swipe":
   x1,y1=self._to_pixels(gesture.get("start"))
   x2,y2=self._to_pixels(gesture.get("end"))
   duration=int(max(1,int(gesture.get("duration",0.1)*1000)))
   self._send([self._adb_command(),"shell","input","swipe",str(x1),str(y1),str(x2),str(y2),str(duration)])
  elif gtype=="drag_in":
   x1,y1=self._to_pixels(gesture.get("start"))
   x2,y2=self._to_pixels(gesture.get("end"))
   duration=int(max(1,int(gesture.get("duration",0.1)*1000)))
   self._send([self._adb_command(),"shell","input","swipe",str(x1),str(y1),str(x2),str(y2),str(duration)])
  elif gtype=="arc":
   center=gesture.get("center",(0.5,0.5))
   radius=float(gesture.get("radius",0.1))
   base=float(gesture.get("angle",0.0))
   duration=float(gesture.get("duration",0.3))
   steps=max(2,int(duration/0.1))
   prev=center
   for step in range(1,steps+1):
    angle=base+step*(math.pi/steps)
    point=(self._clamp(center[0]+math.cos(angle)*radius),self._clamp(center[1]+math.sin(angle)*radius))
    x1,y1=self._to_pixels(prev)
    x2,y2=self._to_pixels(point)
    self._send([self._adb_command(),"shell","input","swipe",str(x1),str(y1),str(x2),str(y2),str(int(duration*1000/steps))])
    prev=point
 def _clamp(self,value):
  return max(0.02,min(0.98,float(value) if value is not None else 0.5))
 def _to_pixels(self,point):
  if point is None:
   point=(0.5,0.5)
  _,_,width,height=self.app.emu_geometry
  width=max(1,int(width))
  height=max(1,int(height))
  x=int(self._clamp(point[0])*width)
  y=int(self._clamp(point[1])*height)
  return x,y
 def _send(self,command):
  if self.stop_flag.is_set():
   return
  try:
   subprocess.Popen(command,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
  except Exception:
   pass
class OverlayManager:
 def __init__(self,app):
  self.app=app
  self.window=None
  self.canvas=None
  self.markers={}
  self.selected=None
  self.dragging=False
  self.resizing=False
  self.last_pos=None
  self.hwnd=None
  self.click_through=None
  self.cursor_job=None
 def open(self):
  if self.window:
   return
  self.window=Toplevel(self.app.root)
  self.window.overrideredirect(True)
  self.window.attributes("-topmost",True)
  x,y,width,height=self.app.emu_geometry
  width=max(1,int(width))
  height=max(1,int(height))
  self.window.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
  self.window.attributes("-alpha",0.01)
  self.canvas=Canvas(self.window,bg="",highlightthickness=0)
  self.canvas.pack(fill="both",expand=True)
  self.canvas.bind("<Button-1>",self.on_click)
  self.canvas.bind("<B1-Motion>",self.on_drag)
  self.canvas.bind("<ButtonRelease-1>",self.on_release)
  self.canvas.bind("<Motion>",self.on_motion)
  self.canvas.bind("<Leave>",self.on_leave)
  self._setup_window_styles()
  self.draw_markers()
  self.update_geometry(x,y,width,height,self.app.emu_visible)
 def _setup_window_styles(self):
  self.hwnd=None
  if not self.window:
   return
  if platform.system()=="Windows" and win32gui and win32con:
   try:
    self.window.update_idletasks()
    self.hwnd=self.window.winfo_id()
   except Exception:
    self.hwnd=None
   if self.hwnd:
    self._set_click_through(True)
    try:
     alpha=max(1,int(255*0.01))
     win32gui.SetLayeredWindowAttributes(self.hwnd,0,alpha,win32con.LWA_ALPHA)
    except Exception:
     pass
    self._start_cursor_monitor()
  else:
   self._set_click_through(True)
 def _start_cursor_monitor(self):
  if platform.system()!="Windows" or not self.window or not self.canvas or not win32gui or not win32con:
   return
  if self.cursor_job:
   try:
    self.window.after_cancel(self.cursor_job)
   except Exception:
    pass
   self.cursor_job=None
  self._cursor_monitor()
 def _cursor_monitor(self):
  if not self.window or not self.canvas:
   self.cursor_job=None
   return
  pos=self._get_cursor_pos()
  capture=False
  if pos:
   try:
    wx=self.window.winfo_rootx()
    wy=self.window.winfo_rooty()
    width=self.canvas.winfo_width() or self.window.winfo_width()
    height=self.canvas.winfo_height() or self.window.winfo_height()
    rel_x=pos[0]-wx
    rel_y=pos[1]-wy
    if 0<=rel_x<=width and 0<=rel_y<=height:
     capture=self._hit_test(rel_x,rel_y)
   except Exception:
    capture=False
  if self.dragging or self.resizing:
   capture=True
  self._set_click_through(not capture)
  if self.window:
   try:
    self.cursor_job=self.window.after(16,self._cursor_monitor)
   except Exception:
    self.cursor_job=None
 def _get_cursor_pos(self):
  if platform.system()!="Windows":
   return None
  try:
   point=wintypes.POINT()
   if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
    return point.x,point.y
  except Exception:
   return None
  return None
 def _set_click_through(self,enable):
  if self.click_through==enable:
   return
  self.click_through=enable
  if platform.system()=="Windows" and self.hwnd and win32gui and win32con:
   try:
    style=win32gui.GetWindowLong(self.hwnd,win32con.GWL_EXSTYLE)
    if enable:
     style|=win32con.WS_EX_LAYERED|win32con.WS_EX_TRANSPARENT
    else:
     style|=win32con.WS_EX_LAYERED
     style&=~win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(self.hwnd,win32con.GWL_EXSTYLE,style)
   except Exception:
    pass
 def _hit_test(self,x,y):
  if not self.canvas or not self.window:
   return False
  width=self.canvas.winfo_width() or self.window.winfo_width()
  height=self.canvas.winfo_height() or self.window.winfo_height()
  if width<=0 or height<=0:
   return False
  for marker in self.markers.values():
   mx=marker.x*width
   my=marker.y*height
   r=marker.radius*min(width,height)
   if r<=0:
    continue
   margin=max(4,r*0.1)
   if (x-mx)**2+(y-my)**2<=(r+margin)**2:
    return True
  return False
 def _update_click_through(self,x=None,y=None):
  capture=self.dragging or self.resizing
  if not capture and x is not None and y is not None:
   capture=self._hit_test(x,y)
  self._set_click_through(not capture)
 def on_motion(self,event):
  self._update_click_through(event.x,event.y)
 def on_leave(self,event):
  self._update_click_through()
 def close(self):
  if self.window:
   if self.cursor_job:
    try:
     self.window.after_cancel(self.cursor_job)
    except Exception:
     pass
    self.cursor_job=None
   self.window.destroy()
   self.window=None
   self.canvas=None
   self.selected=None
   self.hwnd=None
   self.click_through=None
 def load_markers(self,markers):
  self.markers={name:self._marker_from_data(name,data) for name,data in markers.items()}
  self.app.refresh_gesture_markers()
 def get_markers_data(self):
  result={}
  for name,marker in self.markers.items():
   result[name]={"color":marker.color,"x":marker.x,"y":marker.y,"radius":marker.radius,"interaction":marker.interaction,"cooldown":marker.cooldown}
  return result
 def select_marker(self,name):
  marker=self.markers.get(name)
  if marker:
   self.selected=marker
   self.draw_markers()
 def _marker_from_data(self,name,data):
  marker=Marker(name,data.get("color","white"))
  marker.x=data.get("x",0.5)
  marker.y=data.get("y",0.5)
  marker.radius=data.get("radius",0.1)
  marker.interaction=data.get("interaction","click")
  marker.cooldown=data.get("cooldown",False)
  return marker
 def ensure_marker(self,name,color,interaction,cooldown):
  if name not in self.markers:
   marker=Marker(name,color)
   marker.interaction=interaction
   marker.cooldown=cooldown
   self.markers[name]=marker
   self.app.refresh_gesture_markers()
 def draw_markers(self):
  if not self.canvas:
   return
  self.canvas.delete("all")
  width=self.canvas.winfo_width() or self.window.winfo_width()
  height=self.canvas.winfo_height() or self.window.winfo_height()
  for name,marker in self.markers.items():
   x=marker.x*width
   y=marker.y*height
   r=marker.radius*min(width,height)
   self.canvas.create_oval(x-r,y-r,x+r,y+r,outline=marker.color,width=3,fill=self._fill_color(marker))
   if self.selected==marker:
    self.canvas.create_oval(x-r-4,y-r-4,x+r+4,y+r+4,outline="yellow",width=2)
   self.canvas.create_text(x,y,text=name,fill="white")
 def update_geometry(self,x,y,width,height,visible):
  if not self.window:
   return
  w=max(1,int(width))
  h=max(1,int(height))
  if not visible or width<=0 or height<=0:
   try:
    self.window.withdraw()
   except Exception:
    pass
   return
  try:
   self.window.deiconify()
  except Exception:
   pass
  self.window.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
  if self.canvas:
   self.canvas.config(width=w,height=h)
  self.window.update_idletasks()
  self.draw_markers()
  self._update_click_through()
 def _fill_color(self,marker):
  return "#80ffffff"
 def on_click(self,event):
  width=self.canvas.winfo_width()
  height=self.canvas.winfo_height()
  click_x=event.x
  click_y=event.y
  self.resizing=False
  for marker in self.markers.values():
   x=marker.x*width
   y=marker.y*height
   r=marker.radius*min(width,height)
   if (click_x-x)**2+(click_y-y)**2<=r**2:
    self.selected=marker
    self.dragging=True
    self.last_pos=(click_x,click_y)
    boundary=abs((click_x-x)**2+(click_y-y)**2-r**2)
    if boundary<r:
     self.resizing=True
    self._update_click_through(click_x,click_y)
    return
  self.selected=None
  self.dragging=False
  self.resizing=False
  self.last_pos=None
  self._update_click_through()
 def on_drag(self,event):
  if not self.selected or not self.dragging:
   return
  width=self.canvas.winfo_width()
  height=self.canvas.winfo_height()
  dx=event.x-(self.last_pos[0] if self.last_pos else event.x)
  dy=event.y-(self.last_pos[1] if self.last_pos else event.y)
  if self.resizing:
   r=self.selected.radius*min(width,height)
   r=max(10,min(min(width,height)/2,r+dx))
   self.selected.radius=r/min(width,height)
  else:
   x=self.selected.x*width+dx
   y=self.selected.y*height+dy
   self.selected.x=max(0,min(1,x/width))
   self.selected.y=max(0,min(1,y/height))
  self._update_click_through(event.x,event.y)
  self.last_pos=(event.x,event.y)
  self.draw_markers()
 def on_release(self,event):
  self.dragging=False
  self.resizing=False
  self.last_pos=None
  if event is not None:
   self._update_click_through(event.x,event.y)
  else:
   self._update_click_through()
class Mode:
 INIT="初始化"
 LEARNING="学习模式"
 TRAINING="训练模式"
 OPTIMIZING="优化中"
 CONFIG="配置模式"
class HandController:
 def __init__(self,name,model_path,actions,hidden,regularization,adaptive,plasticity):
  self.name=name
  self.model_path=model_path
  self.actions=actions
  self.hidden=hidden
  self.regularization=regularization
  self.adaptive=adaptive
  self.plasticity=plasticity
  self.app=None
  self.network=None
  self.input_dim=155
  self.local_random=random.Random(name)
  self.lock=threading.Lock()
 def start(self,app):
  self.app=app
  self.reload_model()
 def reload_model(self):
  if not self.app:
   return
  vector=self.app.build_state_vector(None)
  self.input_dim=len(vector)
  self.network=RLNetwork(self.input_dim,len(self.actions),self.hidden,self.regularization,self.adaptive,self.plasticity)
  self.network.load(self.model_path)
 def stop(self):
  pass
 def _forward(self,state):
  if self.network is None:
   return None
  try:
   arr=np.asarray([state],dtype=np.float32)
   output=self.network.forward(arr)[0]
   return output
  except Exception:
   return None
 def decide(self,state):
  with self.lock:
   output=self._forward(state)
   if output is None or not np.isfinite(output).all():
    return self.local_random.randint(0,len(self.actions)-1),None
   idx=int(np.argmax(output))
   return idx,output
 def execute(self,state):
  idx,output=self.decide(state)
  idx=max(0,min(len(self.actions)-1,idx))
  action=self.actions[idx]
  gesture=self._gesture_for_action(action,output,state)
  if gesture and self.app and self.app.emulator_controller:
   self.app.emulator_controller.execute_gesture(gesture,self.name)
  payload={"action":action,"mode":"training","state_index":idx}
  if gesture:
   payload["gesture"]=gesture
  return payload
 def _marker(self,name):
  if not self.app:
   return None
  return self.app.overlay.markers.get(name)
 def _gesture_for_action(self,action,output,state):
  marker=None
  if action.startswith("移动轮盘") or action=="移动轮盘校准":
   marker=self._marker("移动轮盘")
   if not marker:
    return None
   angle=self._angle_from_output(output,0.0)
   strength=self._strength_from_state(state,0.35,marker.radius)
   start=(marker.x,marker.y)
   end=(self._clamp(start[0]+math.cos(angle)*strength),self._clamp(start[1]+math.sin(angle)*strength))
   duration=0.3 if action=="移动轮盘拖动" else 0.5
   if action=="移动轮盘旋转":
    return {"type":"arc","center":start,"radius":strength,"angle":angle,"duration":duration}
   return {"type":"swipe","start":start,"end":end,"duration":duration}
  if action=="普攻":
   marker=self._marker("普攻")
   if not marker:
    return None
   return {"type":"tap","point":(marker.x,marker.y),"duration":0.05}
  if action=="施放技能":
   skill=self._select_skill()
   marker=self._marker(skill)
   if not marker:
    return None
   return {"type":"tap","point":(marker.x,marker.y),"duration":0.05,"skill":skill}
  if action=="回城":
   marker=self._marker("回城")
   if not marker:
    return None
   return {"type":"tap","point":(marker.x,marker.y),"duration":0.05,"state":"recall"}
  if action=="恢复":
   marker=self._marker("恢复")
   if not marker:
    return None
   return {"type":"tap","point":(marker.x,marker.y),"duration":0.05}
  if action=="闪现":
   marker=self._marker("闪现")
   if not marker:
    return None
   angle=self._angle_from_output(output,0.75)
   strength=self._strength_from_state(state,marker.radius,marker.radius*1.2)
   start=(marker.x,marker.y)
   end=(self._clamp(start[0]+math.cos(angle)*strength),self._clamp(start[1]+math.sin(angle)*strength))
   return {"type":"swipe","start":start,"end":end,"duration":0.15}
  if action=="主动装备":
   marker=self._marker("主动装备")
   if not marker:
    return None
   return {"type":"tap","point":(marker.x,marker.y),"duration":0.05}
  if action=="取消施法":
   marker=self._marker("取消施法")
   if not marker:
    return None
   start=(self._clamp(marker.x-marker.radius*1.2),self._clamp(marker.y-marker.radius*1.2))
   end=(marker.x,marker.y)
   return {"type":"drag_in","start":start,"end":end,"duration":0.2}
  return None
 def _angle_from_output(self,output,offset):
  base=offset
  if output is not None and len(output)>0 and np.isfinite(output).all():
   base=float(np.mean(output))
  return (base%1.0)*2*math.pi
 def _strength_from_state(self,state,base,cap=None):
  total=sum(state) if state else 1.0
  scale=min(max(total/(len(state) if state else 1),0.1),3.0)
  value=base*scale
  if cap is not None:
   value=min(cap,value)
  return max(0.02,value)
 def _clamp(self,value):
  return max(0.02,min(0.98,value))
 def _select_skill(self):
  if not self.app:
   return "一技能"
  details=self.app.cooldown_state.get("skills",{})
  if isinstance(details,dict):
   available=[k for k,v in details.items() if v=="可用"]
   if available:
    return sorted(available)[0]
  return "一技能"
class GestureMachine:
 def __init__(self,marker):
  self.marker=marker
  self.active=False
  self.state="idle"
  self.button=None
  self.start_time=0.0
  self.last_time=0.0
  self.start_pos=None
  self.last_pos=None
  self.path=[]
  self.rotation_sum=0.0
  self.last_angle=None
  self.entered=False
  self.phase_log=[]
 def update_marker(self,marker):
  self.marker=marker
 def reset(self):
  self.active=False
  self.state="idle"
  self.button=None
  self.start_time=0.0
  self.last_time=0.0
  self.start_pos=None
  self.last_pos=None
  self.path=[]
  self.rotation_sum=0.0
  self.last_angle=None
  self.entered=False
  self.phase_log=[]
 def _distance(self,a,b):
  dx=a[0]-b[0]
  dy=a[1]-b[1]
  return math.sqrt(dx*dx+dy*dy)
 def _angle(self,pos):
  cx=self.marker.x
  cy=self.marker.y
  return math.atan2(pos[1]-cy,pos[0]-cx)
 def _inside(self,pos):
  dx=pos[0]-self.marker.x
  dy=pos[1]-self.marker.y
  return dx*dx+dy*dy<=self.marker.radius*self.marker.radius
 def _append_path(self,pos,timestamp):
  if not pos:
   return
  self.path.append((round(pos[0],4),round(pos[1],4),round(timestamp-self.start_time,4)))
  if len(self.path)>64:
   step=max(1,len(self.path)//32)
   self.path=self.path[::step]
 def _phase(self,name,timepoint):
  self.phase_log.append((name,round(timepoint-self.start_time,4)))
 def press(self,pos,timepoint,button):
  self.reset()
  if self.marker.interaction=="drag_in":
   if not self._inside(pos):
    self.active=True
    self.state="pressed_outside"
  elif self.marker.interaction in ("click","mixed","drag"):
   if self._inside(pos):
    self.active=True
    self.state="pressed"
  elif self.marker.interaction=="drag_in_only":
   if not self._inside(pos):
    self.active=True
    self.state="pressed_outside"
  else:
   if self._inside(pos):
    self.active=True
    self.state="pressed"
  if self.active:
   self.button=button
   self.start_time=timepoint
   self.last_time=timepoint
   self.start_pos=pos
   self.last_pos=pos
   self._append_path(pos,timepoint)
   self._phase("press",timepoint)
 def move(self,pos,timepoint):
  if not self.active:
   return
  self.last_time=timepoint
  self._append_path(pos,timepoint)
  if self.marker.interaction=="click" and self._distance(pos,self.start_pos)>0.02:
   self.state="cancelled"
  if self.marker.interaction in ("drag","mixed") or self.marker.name=="移动轮盘":
   if self.state in ("pressed","dragging","rotating"):
    dist=self._distance(pos,self.start_pos)
    if dist>0.01 and self.state=="pressed":
     self.state="dragging"
     self._phase("drag",timepoint)
    if self.state in ("dragging","rotating"):
     angle=self._angle(pos)
     if self.last_angle is None:
      self.last_angle=angle
     delta=angle-self.last_angle
     while delta>math.pi:
      delta-=2*math.pi
     while delta<-math.pi:
      delta+=2*math.pi
     radius=abs(self._distance((self.marker.x,self.marker.y),pos))
     if radius>=self.marker.radius*0.6:
      self.rotation_sum+=delta
      if abs(self.rotation_sum)>0.5 and self.state!="rotating":
       self.state="rotating"
       self._phase("rotate",timepoint)
     self.last_angle=angle
  if self.marker.interaction=="mixed" and self.state=="pressed" and self._distance(pos,self.start_pos)>0.015:
   self.state="dragging"
   self._phase("drag",timepoint)
  if self.marker.interaction=="drag" and self.state=="pressed" and self._distance(pos,self.start_pos)>0.015:
   self.state="dragging"
   self._phase("drag",timepoint)
  if self.marker.interaction=="drag_in":
   if self.state in ("pressed_outside","dragging_in"):
    if self._inside(pos):
     if not self.entered:
      self.entered=True
      self._phase("enter",timepoint)
     self.state="dragging_in"
    else:
      self.state="pressed_outside"
  self.last_pos=pos
 def release(self,pos,timepoint,button):
  if not self.active or button!=self.button:
   return None
  self._append_path(pos,timepoint)
  duration=max(0.0,timepoint-self.start_time)
  result=None
  if self.marker.interaction=="click":
   if self.state!="cancelled" and self._inside(pos):
    result="click"
  elif self.marker.interaction=="mixed":
   if self.state in ("pressed","cancelled") and self._inside(pos) and self._distance(pos,self.start_pos)<=0.02:
    result="click"
   elif self.state in ("dragging","rotating"):
    result="drag"
  elif self.marker.interaction=="drag":
   if self.state in ("dragging","rotating"):
    result="drag"
  elif self.marker.interaction=="drag_in":
   if self.entered and self._inside(pos):
    result="drag_in"
  else:
   if self.state in ("dragging","rotating","pressed"):
    result=self.marker.interaction
  rotation_direction="none"
  if self.state=="rotating" and result:
   rotation_direction="clockwise" if self.rotation_sum<0 else "counter_clockwise"
  phases=list(self.phase_log)
  if result:
   phases.append((result,round(timepoint-self.start_time,4)))
  gesture=None
  if result:
   gesture={"marker":self.marker.name,"interaction":self.marker.interaction,"result":result,"duration":round(duration,4),"rotation":round(self.rotation_sum,4),"rotation_direction":rotation_direction,"path":self.path,"phases":phases}
  self.reset()
  return gesture
 def handle_event(self,event):
  gestures=[]
  etype=event.get("type")
  pos=event.get("normalized")
  timepoint=event.get("time",time.time())
  button=event.get("button","left")
  if etype=="press":
   if pos:
    self.press(pos,timepoint,button)
  elif etype=="move":
   if pos:
    self.move(pos,timepoint)
  elif etype=="release":
   if pos:
    gesture=self.release(pos,timepoint,button)
   else:
    gesture=self.release(self.last_pos if self.last_pos else (0.0,0.0),timepoint,button)
   if gesture:
    gestures.append(gesture)
  return gestures
 def is_active(self):
  return self.active
class GestureManager:
 def __init__(self,app):
  self.app=app
  self.machines={}
  self.lock=threading.Lock()
 def refresh_markers(self):
  with self.lock:
   active_names=set()
   for name,marker in self.app.overlay.markers.items():
    if marker.interaction=="observe":
     continue
    active_names.add(name)
    if name in self.machines:
     self.machines[name].update_marker(marker)
    else:
     self.machines[name]=GestureMachine(marker)
   for name in list(self.machines.keys()):
    if name not in active_names:
     del self.machines[name]
 def process(self,event):
  gestures=[]
  with self.lock:
   for machine in self.machines.values():
    gestures.extend(machine.handle_event(event))
  return gestures
 def any_active(self):
  with self.lock:
   for machine in self.machines.values():
    if machine.is_active():
     return True
  return False
class EmulatorWindowTracker:
 def __init__(self,app):
  self.app=app
  self.handle=None
  self.stop_flag=threading.Event()
  self.available=platform.system()=="Windows" and win32gui is not None and win32con is not None and win32process is not None and win32api is not None
  self.last_geometry=None
  self.last_visible=None
  self.thread=threading.Thread(target=self.run,daemon=True)
  self.thread.start()
 def stop(self):
  self.stop_flag.set()
  if self.thread and self.thread.is_alive():
   self.thread.join(timeout=0.5)
 def run(self):
  while not self.stop_flag.is_set() and not self.app.stop_event.is_set():
   if self.available:
    handle=self._locate_window()
    rect=self._get_rect(handle)
    visible=self._is_visible(handle)
    if rect:
     x,y,r,b=rect
     width=max(0,r-x)
     height=max(0,b-y)
    else:
     x,y,width,height=self.app.emu_geometry
     visible=False
   else:
    handle=None
    x,y,width,height=self.app.emu_geometry
    visible=True
   self.handle=handle if self.available else None
   geometry=(int(x),int(y),int(width),int(height))
   if self.last_geometry!=geometry or self.last_visible!=visible:
    self.last_geometry=geometry
    self.last_visible=visible
    self.app.root.after(0,lambda g=geometry,v=visible:self.app.update_emulator_geometry(g[0],g[1],g[2],g[3],v))
   time.sleep(0.2)
 def _locate_window(self):
  if not self.available:
   return None
  target=self.app.pool.config_manager.data.get("emulator_path","")
  target=target.lower() if isinstance(target,str) else ""
  if self.handle and self._match_window(self.handle,target):
   return self.handle
  candidates=[]
  def callback(hwnd,param):
   if self._match_window(hwnd,target):
    param.append(hwnd)
  win32gui.EnumWindows(callback,candidates)
  return candidates[0] if candidates else None
 def _match_window(self,hwnd,target):
  if not win32gui or not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
   return False
  if win32gui.IsIconic(hwnd):
   return False
  try:
   title=win32gui.GetWindowText(hwnd)
  except Exception:
   title=""
  if not title:
   return False
  exe=""
  try:
   _,pid=win32process.GetWindowThreadProcessId(hwnd)
   exe=psutil.Process(pid).exe().lower()
  except Exception:
   exe=""
  if target and target in exe:
   return True
  lower=title.lower()
  if "ldplayer" in lower or "dnplayer" in lower or "android" in lower:
   return True
  return False
 def _get_rect(self,hwnd):
  if not hwnd or not win32gui or not win32gui.IsWindow(hwnd):
   return None
  try:
   return win32gui.GetWindowRect(hwnd)
  except Exception:
   return None
 def _is_visible(self,hwnd):
  if not hwnd or not win32gui:
   return False
  if not win32gui.IsWindowVisible(hwnd):
   return False
  if win32gui.IsIconic(hwnd):
   return False
  placement=None
  try:
   placement=win32gui.GetWindowPlacement(hwnd)
  except Exception:
   placement=None
  if placement and len(placement)>1 and win32con:
   if placement[1]==win32con.SW_SHOWMINIMIZED or placement[1]==win32con.SW_HIDE:
    return False
  return True
class OverlayManager:
 def __init__(self,app):
  self.app=app
  self.window=None
  self.canvas=None
  self.markers={}
  self.selected=None
  self.dragging=False
  self.resizing=False
  self.last_pos=None
  self.hwnd=None
  self.click_through=None
  self.cursor_job=None
 def open(self):
  if self.window:
   return
  self.window=Toplevel(self.app.root)
  self.window.overrideredirect(True)
  self.window.attributes("-topmost",True)
  x,y,width,height=self.app.emu_geometry
  width=max(1,int(width))
  height=max(1,int(height))
  self.window.geometry(f"{width}x{height}+{int(x)}+{int(y)}")
  self.window.attributes("-alpha",0.01)
  self.canvas=Canvas(self.window,bg="",highlightthickness=0)
  self.canvas.pack(fill="both",expand=True)
  self.canvas.bind("<Button-1>",self.on_click)
  self.canvas.bind("<B1-Motion>",self.on_drag)
  self.canvas.bind("<ButtonRelease-1>",self.on_release)
  self.canvas.bind("<Motion>",self.on_motion)
  self.canvas.bind("<Leave>",self.on_leave)
  self._setup_window_styles()
  self.draw_markers()
  self.update_geometry(x,y,width,height,self.app.emu_visible)
 def _setup_window_styles(self):
  self.hwnd=None
  if not self.window:
   return
  if platform.system()=="Windows" and win32gui and win32con:
   try:
    self.window.update_idletasks()
    self.hwnd=self.window.winfo_id()
   except Exception:
    self.hwnd=None
   if self.hwnd:
    self._set_click_through(True)
    try:
     alpha=max(1,int(255*0.01))
     win32gui.SetLayeredWindowAttributes(self.hwnd,0,alpha,win32con.LWA_ALPHA)
    except Exception:
     pass
    self._start_cursor_monitor()
  else:
   self._set_click_through(True)
 def _start_cursor_monitor(self):
  if platform.system()!="Windows" or not self.window or not self.canvas or not win32gui or not win32con:
   return
  if self.cursor_job:
   try:
    self.window.after_cancel(self.cursor_job)
   except Exception:
    pass
   self.cursor_job=None
  self._cursor_monitor()
 def _cursor_monitor(self):
  if not self.window or not self.canvas:
   self.cursor_job=None
   return
  pos=self._get_cursor_pos()
  capture=False
  if pos:
   try:
    wx=self.window.winfo_rootx()
    wy=self.window.winfo_rooty()
    width=self.canvas.winfo_width() or self.window.winfo_width()
    height=self.canvas.winfo_height() or self.window.winfo_height()
    rel_x=pos[0]-wx
    rel_y=pos[1]-wy
    if 0<=rel_x<=width and 0<=rel_y<=height:
     capture=self._hit_test(rel_x,rel_y)
   except Exception:
    capture=False
  if self.dragging or self.resizing:
   capture=True
  self._set_click_through(not capture)
  if self.window:
   try:
    self.cursor_job=self.window.after(16,self._cursor_monitor)
   except Exception:
    self.cursor_job=None
 def _get_cursor_pos(self):
  if platform.system()!="Windows":
   return None
  try:
   point=wintypes.POINT()
   if ctypes.windll.user32.GetCursorPos(ctypes.byref(point)):
    return point.x,point.y
  except Exception:
   return None
  return None
 def _set_click_through(self,enable):
  if self.click_through==enable:
   return
  self.click_through=enable
  if platform.system()=="Windows" and self.hwnd and win32gui and win32con:
   try:
    style=win32gui.GetWindowLong(self.hwnd,win32con.GWL_EXSTYLE)
    if enable:
     style|=win32con.WS_EX_LAYERED|win32con.WS_EX_TRANSPARENT
    else:
     style|=win32con.WS_EX_LAYERED
     style&=~win32con.WS_EX_TRANSPARENT
    win32gui.SetWindowLong(self.hwnd,win32con.GWL_EXSTYLE,style)
   except Exception:
    pass
 def _hit_test(self,x,y):
  if not self.canvas or not self.window:
   return False
  width=self.canvas.winfo_width() or self.window.winfo_width()
  height=self.canvas.winfo_height() or self.window.winfo_height()
  if width<=0 or height<=0:
   return False
  for marker in self.markers.values():
   mx=marker.x*width
   my=marker.y*height
   r=marker.radius*min(width,height)
   if r<=0:
    continue
   margin=max(4,r*0.1)
   if (x-mx)**2+(y-my)**2<=(r+margin)**2:
    return True
  return False
 def _update_click_through(self,x=None,y=None):
  capture=self.dragging or self.resizing
  if not capture and x is not None and y is not None:
   capture=self._hit_test(x,y)
  self._set_click_through(not capture)
 def on_motion(self,event):
  self._update_click_through(event.x,event.y)
 def on_leave(self,event):
  self._update_click_through()
 def close(self):
  if self.window:
   if self.cursor_job:
    try:
     self.window.after_cancel(self.cursor_job)
    except Exception:
     pass
    self.cursor_job=None
   self.window.destroy()
   self.window=None
   self.canvas=None
   self.selected=None
   self.hwnd=None
   self.click_through=None
 def load_markers(self,markers):
  self.markers={name:self._marker_from_data(name,data) for name,data in markers.items()}
  self.app.refresh_gesture_markers()
 def get_markers_data(self):
  result={}
  for name,marker in self.markers.items():
   result[name]={"color":marker.color,"x":marker.x,"y":marker.y,"radius":marker.radius,"interaction":marker.interaction,"cooldown":marker.cooldown}
  return result
 def select_marker(self,name):
  marker=self.markers.get(name)
  if marker:
   self.selected=marker
   self.draw_markers()
 def _marker_from_data(self,name,data):
  marker=Marker(name,data.get("color","white"))
  marker.x=data.get("x",0.5)
  marker.y=data.get("y",0.5)
  marker.radius=data.get("radius",0.1)
  marker.interaction=data.get("interaction","click")
  marker.cooldown=data.get("cooldown",False)
  return marker
 def ensure_marker(self,name,color,interaction,cooldown):
  if name not in self.markers:
   marker=Marker(name,color)
   marker.interaction=interaction
   marker.cooldown=cooldown
   self.markers[name]=marker
   self.app.refresh_gesture_markers()
 def draw_markers(self):
  if not self.canvas:
   return
  self.canvas.delete("all")
  width=self.canvas.winfo_width() or self.window.winfo_width()
  height=self.canvas.winfo_height() or self.window.winfo_height()
  for name,marker in self.markers.items():
   x=marker.x*width
   y=marker.y*height
   r=marker.radius*min(width,height)
   self.canvas.create_oval(x-r,y-r,x+r,y+r,outline=marker.color,width=3,fill=self._fill_color(marker))
   if self.selected==marker:
    self.canvas.create_oval(x-r-4,y-r-4,x+r+4,y+r+4,outline="yellow",width=2)
   self.canvas.create_text(x,y,text=name,fill="white")
 def update_geometry(self,x,y,width,height,visible):
  if not self.window:
   return
  w=max(1,int(width))
  h=max(1,int(height))
  if not visible or width<=0 or height<=0:
   try:
    self.window.withdraw()
   except Exception:
    pass
   return
  try:
   self.window.deiconify()
  except Exception:
   pass
  self.window.geometry(f"{w}x{h}+{int(x)}+{int(y)}")
  if self.canvas:
   self.canvas.config(width=w,height=h)
  self.window.update_idletasks()
  self.draw_markers()
  self._update_click_through()
 def _fill_color(self,marker):
  return "#80ffffff"
 def on_click(self,event):
  width=self.canvas.winfo_width()
  height=self.canvas.winfo_height()
  click_x=event.x
  click_y=event.y
  self.resizing=False
  for marker in self.markers.values():
   x=marker.x*width
   y=marker.y*height
   r=marker.radius*min(width,height)
   if (click_x-x)**2+(click_y-y)**2<=r**2:
    self.selected=marker
    self.dragging=True
    self.last_pos=(click_x,click_y)
    boundary=abs((click_x-x)**2+(click_y-y)**2-r**2)
    if boundary<r:
     self.resizing=True
    self._update_click_through(click_x,click_y)
    return
  self.selected=None
  self.dragging=False
  self.resizing=False
  self.last_pos=None
  self._update_click_through()
 def on_drag(self,event):
  if not self.selected or not self.dragging:
   return
  width=self.canvas.winfo_width()
  height=self.canvas.winfo_height()
  dx=event.x-(self.last_pos[0] if self.last_pos else event.x)
  dy=event.y-(self.last_pos[1] if self.last_pos else event.y)
  if self.resizing:
   r=self.selected.radius*min(width,height)
   r=max(10,min(min(width,height)/2,r+dx))
   self.selected.radius=r/min(width,height)
  else:
   x=self.selected.x*width+dx
   y=self.selected.y*height+dy
   self.selected.x=max(0,min(1,x/width))
   self.selected.y=max(0,min(1,y/height))
  self._update_click_through(event.x,event.y)
  self.last_pos=(event.x,event.y)
  self.draw_markers()
 def on_release(self,event):
  self.dragging=False
  self.resizing=False
  self.last_pos=None
  if event is not None:
   self._update_click_through(event.x,event.y)
  else:
   self._update_click_through()
class FrameCapture:
 def __init__(self,app):
  self.app=app
  self.stop_flag=threading.Event()
  self.fail_count=0
  self.last_error_time=0
  self.suspend_until=0
  self.notified=False
  self.thread=threading.Thread(target=self.run,daemon=True)
  self.thread.start()
 def run(self):
  while not self.stop_flag.is_set():
   freq=max(1,min(120,self.app.resource_monitor.frequency))
   if time.time()<self.suspend_until:
    time.sleep(min(0.5,1/max(1,freq)))
    continue
   if self.app.get_mode() in [Mode.LEARNING,Mode.TRAINING] and self.app.emu_visible:
    path=self._generate_frame()
    if path:
     self.app.process_frame(path)
    else:
     self.app.last_frame=""
   time.sleep(1/max(1,freq))
 def _generate_frame(self):
  try:
   width,height=self.app.get_emulator_geometry()
   width=max(1,int(width))
   height=max(1,int(height))
   for attempt in range(3):
    image=self._capture_image(width,height)
    if image:
     path=self._save_image(image)
     if path:
      self._handle_success()
      return path
    time.sleep(0.05)
   self._handle_failure()
   return None
  except Exception:
   self._handle_failure()
   return None
 def _capture_image(self,width,height):
  if platform.system()=="Windows" and width>0 and height>0:
   left=int(self.app.emu_geometry[0])
   top=int(self.app.emu_geometry[1])
   if hasattr(self.app,"emulator_tracker") and self.app.emulator_tracker.handle and win32gui:
    rect=self.app.emulator_tracker._get_rect(self.app.emulator_tracker.handle)
    if rect:
     left=int(rect[0])
     top=int(rect[1])
     width=max(1,int(rect[2]-rect[0]))
     height=max(1,int(rect[3]-rect[1]))
   try:
    with mss.mss() as sct:
     monitor={"left":left,"top":top,"width":width,"height":height}
     shot=sct.grab(monitor)
     if shot.width>0 and shot.height>0:
      return Image.frombytes("RGB",shot.size,shot.rgb)
   except Exception:
    return None
  return None
 def _save_image(self,image):
  try:
   timestamp=int(time.time()*1000)
   self.app.pool.frames_folder.mkdir(exist_ok=True)
   path=self.app.pool.frames_folder/f"frame_{timestamp}.png"
   while path.exists():
    timestamp+=1
    path=self.app.pool.frames_folder/f"frame_{timestamp}.png"
   image.save(path)
   files=sorted(self.app.pool.frames_folder.glob("frame_*.png"),key=lambda p:p.stat().st_mtime,reverse=True)
   for extra in files[200:]:
    try:
     extra.unlink()
    except Exception:
     continue
   return path
  except Exception:
   return None
 def _handle_failure(self):
  self.fail_count+=1
  if self.fail_count>=3:
   self.suspend_until=time.time()+2
   if not self.notified or time.time()-self.last_error_time>1:
    self.last_error_time=time.time()
    self.notified=True
    self.app.root.after(0,lambda:self.app.status_var.set("截屏失败，请检查窗口和权限"))
 def _handle_success(self):
  if self.fail_count>0:
   self.fail_count=0
  if self.suspend_until>0:
   self.suspend_until=0
  if self.notified:
   self.notified=False
   self.app.root.after(0,self._restore_status)
 def _restore_status(self):
  mode=self.app.get_mode()
  if mode==Mode.LEARNING:
   self.app.status_var.set("采集中")
  elif mode==Mode.TRAINING:
   self.app.status_var.set("AI执行中")
  elif mode==Mode.OPTIMIZING:
   self.app.status_var.set("优化中")
  elif mode==Mode.CONFIG:
   self.app.status_var.set("配置中")
 def stop(self):
  self.stop_flag.set()
class VisionAnalyzer:
 def __init__(self,app):
  self.app=app
  self.lock=threading.Lock()
  self.network=None
  self.reload()
 def reload(self):
  with self.lock:
   vector=self.app.build_state_vector(None)
   self.network=RLNetwork(len(vector),4,128,0.0003,0.15,1.2)
   self.network.load(self.app.pool.vision_model)
 def analyze(self,path):
  if not path:
   return
  try:
   with Image.open(path) as img:
    rgb=img.convert("RGB")
    gray=np.asarray(rgb.convert("L"),dtype=np.float32)/255.0
  except Exception:
   return
  small=Image.fromarray(np.clip(gray*255.0,0,255).astype(np.uint8)).resize((16,9))
  vector=self.app.build_state_vector(path,np.asarray(small,dtype=np.float32)/255.0)
  output=None
  with self.lock:
   if self.network:
    try:
     output=self.network.forward(np.asarray([vector],dtype=np.float32))[0]
    except Exception:
     output=None
  analysis=self._build_analysis(output,gray)
  snapshot=dict(analysis)
  if isinstance(analysis.get("cooldowns"),dict):
   snapshot["cooldowns"]=dict(analysis["cooldowns"])
  self.app.latest_analysis=snapshot
  if analysis.get("recall_active",1.0)>=0.5 and analysis.get("alive",True):
   with self.app.state_lock:
    if self.app.recalling and analysis["cooldowns"]["items"]=="可用":
     self.app.recalling=False
  self.app.update_state(snapshot["A"],snapshot["B"],snapshot["C"],snapshot["alive"],snapshot["cooldowns"]["skills"],snapshot["cooldowns"]["items"],snapshot["cooldowns"]["heal"],snapshot["cooldowns"]["flash"])
  mode=self.app.get_mode()
  if mode in (Mode.LEARNING,Mode.TRAINING):
   self.app.record_event("vision",{"mode":mode,"frame":str(path),"analysis":snapshot})
 def _build_analysis(self,output,gray):
  metrics=self._collect(gray)
  alive=self._alive(output,metrics)
  a=int(round((metrics["data"]["A"]+self._scale_metric(output,1))/2))
  b=int(round((metrics["data"]["B"]+self._scale_metric(output,2))/2))
  c=int(round((metrics["data"]["C"]+self._scale_metric(output,3))/2))
  cooldowns={"skills":metrics["skills"],"items":metrics["items"],"heal":metrics["heal"],"flash":metrics["flash"]}
  return {"alive":alive,"A":max(0,a),"B":max(0,b),"C":max(0,c),"cooldowns":cooldowns,"recall_active":metrics["recall"]}
 def _alive(self,output,metrics):
  if output is not None and len(output)>0:
   value=self._sigmoid(output[0])
   if value>=0.6:
    return True
   if value<=0.4:
    return False
  return metrics["recall"]>=0.3
 def _scale_metric(self,output,index):
  if output is None or len(output)<=index:
   return 0
  return int(max(0,min(400,round(self._sigmoid(output[index])*400))))
 def _sigmoid(self,value):
  try:
   v=float(value)
  except Exception:
   v=0.0
  return 1.0/(1.0+math.exp(-v))
 def _collect(self,gray):
  arr=np.asarray(gray,dtype=np.float32)
  if arr.max()>1.0:
   arr=arr/255.0
  height,width=arr.shape
  def sample(name):
   marker=self.app.overlay.markers.get(name)
   if not marker:
    return 0.0
   cx=int(marker.x*width)
   cy=int(marker.y*height)
   radius=int(max(1,marker.radius*min(width,height)))
   x0=max(0,cx-radius)
   x1=min(width,cx+radius)
   y0=max(0,cy-radius)
   y1=min(height,cy+radius)
   region=arr[y0:y1,x0:x1]
   if region.size==0:
    return 0.0
   return float(np.mean(region))
  skills={}
  for name in ["一技能","二技能","三技能","四技能"]:
   value=sample(name)
   skills[name]="可用" if value>=0.45 else "冷却"
  items_status="可用" if sample("主动装备")>=0.45 else "冷却"
  heal_status="可用" if sample("恢复")>=0.45 else "冷却"
  flash_status="可用" if sample("闪现")>=0.45 else "冷却"
  data={"A":int(round(sample("数据A")*400)),"B":int(round(sample("数据B")*400)),"C":int(round(sample("数据C")*400))}
  recall=sample("回城")
  return {"skills":skills,"items":items_status,"heal":heal_status,"flash":flash_status,"data":data,"recall":recall}
class MouseMonitor:
 def __init__(self,app):
  self.app=app
  self.listener=None
  self.queue=queue.Queue()
  self.thread=None
  self.stop_flag=threading.Event()
 def start(self):
  if self.thread and self.thread.is_alive():
   return
  self.stop_flag.clear()
  try:
   self.listener=mouse.Listener(on_move=self.on_move,on_click=self.on_click,on_scroll=self.on_scroll)
   self.listener.daemon=True
   self.listener.start()
  except Exception:
   self.listener=None
  self.thread=threading.Thread(target=self.process,daemon=True)
  self.thread.start()
 def on_move(self,x,y):
  self.queue.put({"type":"move","x":x,"y":y,"time":time.time()})
 def on_click(self,x,y,button,pressed):
  self.queue.put({"type":"click","x":x,"y":y,"button":str(button),"pressed":pressed,"time":time.time()})
 def on_scroll(self,x,y,dx,dy):
  self.queue.put({"type":"scroll","x":x,"y":y,"dx":dx,"dy":dy,"time":time.time()})
 def process(self):
  while not self.stop_flag.is_set() and not self.app.stop_event.is_set():
   try:
    event=self.queue.get(timeout=0.5)
   except queue.Empty:
    continue
   self.app.last_input=time.time()
   mode=self.app.get_mode()
   if mode==Mode.TRAINING:
    self.app.request_training_interrupt()
   x=event.get("x")
   y=event.get("y")
   normalized=self.app.normalize_position(x,y) if x is not None and y is not None else None
   button_name=str(event.get("button","Button.left")).split(".")[-1]
   gm_event=None
   if event.get("type")=="click":
    gm_event={"type":"press" if event.get("pressed",False) else "release","normalized":normalized,"time":event.get("time",time.time()),"button":button_name}
   elif event.get("type")=="move":
    gm_event={"type":"move","normalized":normalized,"time":event.get("time",time.time()),"button":button_name}
   manager=self.app.gesture_manager if hasattr(self.app,'gesture_manager') else None
   if gm_event is not None and manager:
    gestures=manager.process(gm_event)
   else:
    gestures=[]
   within=self.app.point_in_emulator(x,y) if x is not None and y is not None else False
   if not self.app.emu_visible:
    within=False
   active=manager.any_active() if manager else False
   active=active and self.app.emu_visible
   should_record=within or active
   if mode in [Mode.LEARNING,Mode.TRAINING] and should_record:
    payload={"mode":"learning" if mode==Mode.LEARNING else "training-user","event":event.get("type"),"position":[event.get("x",0),event.get("y",0)],"normalized":list(normalized) if normalized is not None else [],"event_time":event.get("time",time.time())}
    if event.get("type")=="click":
     payload["button"]=button_name
     payload["pressed"]=event.get("pressed",False)
    if event.get("type")=="scroll":
     payload["scroll"]=[event.get("dx",0),event.get("dy",0)]
    self.app.record_event("user-mouse",payload)
   for gesture in gestures:
    context="learning" if mode==Mode.LEARNING else "training-user"
    self.app.record_event("user-gesture",{"mode":context,"gesture":gesture})
   self.queue.task_done()
 def stop(self):
  self.stop_flag.set()
  if self.listener:
   try:
    self.listener.stop()
   except Exception:
    pass
   self.listener=None
class MainApp:
 def __init__(self):
  self.root=Tk()
  self.root.title("类脑智能自适应系统")
  self.state_var=StringVar(value=Mode.INIT)
  self.status_var=StringVar(value="待检测")
  self.mode_lock=threading.Lock()
  self.mode=Mode.INIT
  self.progress_var=DoubleVar(value=0)
  self.progress_text_var=StringVar(value="0%")
  self.hero_status_var=StringVar(value="存活")
  self.data_a_var=StringVar(value="0")
  self.data_b_var=StringVar(value="0")
  self.data_c_var=StringVar(value="0")
  self.cooldown_vars={"skills":StringVar(value="冷却状态"),"items":StringVar(value="冷却状态"),"heal":StringVar(value="冷却状态"),"flash":StringVar(value="冷却状态")}
  self.state_lock=threading.Lock()
  self.hero_alive=True
  self.recalling=False
  self.data_a=0
  self.data_b=0
  self.data_c=0
  self.cooldown_state={"skills":{},"items":"冷却状态","heal":"冷却状态","flash":"冷却状态"}
  self.optimize_button_state=BooleanVar(value=False)
  self.emu_geometry=(0,0,1280,720)
  self.emu_visible=True
  self.overlay=OverlayManager(self)
  self.gesture_manager=GestureManager(self)
  self.default_folder=Path(os.path.expanduser("~"))/"Desktop"/"AAA"
  self.pool=ExperiencePool(self.default_folder)
  config_data=self.pool.config_manager.data
  if "aaa_folder" in config_data:
   stored_folder=Path(config_data.get("aaa_folder",str(self.default_folder)))
   if stored_folder.resolve()!=self.pool.folder.resolve():
    self.pool.migrate(stored_folder)
    config_data=self.pool.config_manager.data
  else:
   self.pool.config_manager.update("aaa_folder",str(self.pool.folder))
   config_data=self.pool.config_manager.data
  self.overlay.load_markers(config_data.get("markers",{}))
  self.ensure_default_markers()
  self.stop_event=threading.Event()
  self.user_active=True
  self.last_input=time.time()
  self.training_thread=None
  self.learning_thread=None
  self.last_frame=""
  self.latest_analysis={"alive":True,"A":0,"B":0,"C":0,"cooldowns":{"skills":{},"items":"冷却状态","heal":"冷却状态","flash":"冷却状态"}}
  self.resource_monitor=ResourceMonitor()
  self.model_handler=AIModelHandler(self.pool)
  self.emulator_controller=EmulatorController(self)
  self.emulator_controller.update_paths()
  self.vision=VisionAnalyzer(self)
  self.frame_capture=FrameCapture(self)
  self.left_controller=HandController("ai-left",self.pool.left_model,["移动轮盘拖动","移动轮盘旋转","移动轮盘校准"],64,0.0005,0.2,1.1)
  self.right_controller=HandController("ai-right",self.pool.right_model,["普攻","施放技能","回城","恢复","闪现","主动装备","取消施法"],96,0.0005,0.25,1.05)
  self.mouse_monitor=MouseMonitor(self)
  self.window_tracker=EmulatorWindowTracker(self)
  self.create_ui()
  self.root.protocol("WM_DELETE_WINDOW",self.stop)
  self.root.bind("<Escape>",lambda e:self.stop())
  self.check_environment()
  self.left_controller.start(self)
  self.right_controller.start(self)
  self.mouse_monitor.start()
  self.input_monitor_thread=threading.Thread(target=self.monitor_input,daemon=True)
  self.input_monitor_thread.start()
  self.scheduler_thread=threading.Thread(target=self.scheduler,daemon=True)
  self.scheduler_thread.start()
 def create_ui(self):
  Label(self.root,textvariable=self.state_var,font=("Microsoft YaHei",20)).grid(row=0,column=0,columnspan=4,sticky="ew")
  Label(self.root,text="状态:").grid(row=1,column=0)
  Label(self.root,textvariable=self.status_var).grid(row=1,column=1,sticky="w")
  Button(self.root,text="优化",command=self.on_optimize).grid(row=2,column=0,sticky="ew")
  Button(self.root,text="取消优化",command=self.on_cancel_optimize).grid(row=2,column=1,sticky="ew")
  Button(self.root,text="配置",command=self.on_configure).grid(row=2,column=2,sticky="ew")
  Button(self.root,text="切换AAA文件夹",command=self.on_change_folder).grid(row=2,column=3,sticky="ew")
  ttk.Progressbar(self.root,variable=self.progress_var,maximum=100).grid(row=3,column=0,columnspan=4,sticky="ew")
  Label(self.root,textvariable=self.progress_text_var).grid(row=3,column=4,sticky="w")
  Label(self.root,text="英雄状态:").grid(row=4,column=0)
  Label(self.root,textvariable=self.hero_status_var).grid(row=4,column=1,sticky="w")
  Label(self.root,text="数据A:").grid(row=5,column=0)
  Label(self.root,textvariable=self.data_a_var).grid(row=5,column=1,sticky="w")
  Label(self.root,text="数据B:").grid(row=6,column=0)
  Label(self.root,textvariable=self.data_b_var).grid(row=6,column=1,sticky="w")
  Label(self.root,text="数据C:").grid(row=7,column=0)
  Label(self.root,textvariable=self.data_c_var).grid(row=7,column=1,sticky="w")
  Label(self.root,text="技能冷却:").grid(row=8,column=0)
  Label(self.root,textvariable=self.cooldown_vars["skills"]).grid(row=8,column=1,sticky="w")
  Label(self.root,text="主动装备冷却:").grid(row=9,column=0)
  Label(self.root,textvariable=self.cooldown_vars["items"]).grid(row=9,column=1,sticky="w")
  Label(self.root,text="恢复冷却:").grid(row=10,column=0)
  Label(self.root,textvariable=self.cooldown_vars["heal"]).grid(row=10,column=1,sticky="w")
  Label(self.root,text="闪现冷却:").grid(row=11,column=0)
  Label(self.root,textvariable=self.cooldown_vars["flash"]).grid(row=11,column=1,sticky="w")
  Label(self.root,text="截图频率(Hz):").grid(row=12,column=0)
  self.freq_var=StringVar(value=str(self.resource_monitor.frequency))
  Label(self.root,textvariable=self.freq_var).grid(row=12,column=1,sticky="w")
  Button(self.root,text="保存配置",command=self.save_config).grid(row=13,column=0,columnspan=2,sticky="ew")
  Button(self.root,text="加载配置",command=self.load_config).grid(row=13,column=2,columnspan=2,sticky="ew")
 def build_state_vector(self,frame_path=None,frame_array=None):
  with self.state_lock:
   a=float(self.data_a)
   b=float(self.data_b)
   c=float(self.data_c)
   alive=self.hero_alive
   recalling=self.recalling
   cooldowns=dict(self.cooldown_state)
   geometry=self.emu_geometry
  width=max(1.0,float(geometry[2]))
  height=max(1.0,float(geometry[3]))
  features=[a/100.0,b/100.0,c/100.0,1.0 if alive else 0.0,1.0 if recalling else 0.0,width/1920.0,height/1080.0]
  features.append(self._cooldown_scalar(cooldowns.get("skills")))
  features.append(self._cooldown_scalar(cooldowns.get("items")))
  features.append(self._cooldown_scalar(cooldowns.get("heal")))
  features.append(self._cooldown_scalar(cooldowns.get("flash")))
  features.extend(self._frame_features(frame_path,frame_array))
  return features
 def _cooldown_scalar(self,value):
  if isinstance(value,dict):
   total=len(value)
   if total<=0:
    return 0.0
   locked=len([v for v in value.values() if v not in ("可用","ready","Ready","AVAILABLE")])
   return min(1.0,max(0.0,locked/max(1,total)))
  if isinstance(value,str):
   return 0.0 if value in ("可用","ready","Ready","AVAILABLE") else 1.0
  if value in (0,0.0,False,None):
   return 0.0
  return 1.0
 def _frame_features(self,frame_path,frame_array):
  if frame_array is not None:
   try:
    arr=np.asarray(frame_array,dtype=np.float32)
    if arr.ndim==3:
     arr=np.mean(arr,axis=2)
    if arr.shape!=(9,16):
     img=Image.fromarray(np.clip(arr*255.0,0,255).astype(np.uint8))
     arr=np.asarray(img.resize((16,9)),dtype=np.float32)/255.0
    else:
     arr=np.clip(arr,0.0,1.0)
    return arr.reshape(-1).tolist()
   except Exception:
    pass
  path=frame_path or self.last_frame
  if path:
   try:
    with Image.open(path) as img:
     arr=np.asarray(img.convert("L").resize((16,9)),dtype=np.float32)/255.0
     return arr.reshape(-1).tolist()
   except Exception:
    pass
  return [0.0]*144
 def _skill_display(self,skills):
  if isinstance(skills,dict):
   return " ".join(f"{k}:{skills[k]}" for k in sorted(skills.keys()))
  return str(skills)
 def _normalize_skill_status(self,skills):
  names=["一技能","二技能","三技能","四技能"]
  if isinstance(skills,dict):
   normalized={}
   for name in names:
    normalized[name]=skills.get(name,skills.get(name.replace("技能",""),"冷却"))
   for key,value in skills.items():
    if key not in normalized:
     normalized[key]=value
   return normalized
  status=str(skills) if skills is not None else "冷却"
  return {name:status for name in names}
 def process_frame(self,path):
  self.last_frame=str(path)
  if hasattr(self,"vision") and self.vision:
   self.vision.analyze(path)
 def reload_models(self):
  self.left_controller.reload_model()
  self.right_controller.reload_model()
  if hasattr(self,"vision") and self.vision:
   self.vision.reload()
 def set_progress(self,value):
  self.progress_var.set(value)
  self.progress_text_var.set(f"{int(value)}%")
 def wait_thread(self,attr):
  thread=getattr(self,attr)
  if thread and thread.is_alive():
   for _ in range(200):
    thread.join(timeout=0.01)
    if not thread.is_alive():
     break
  if not thread or not thread.is_alive():
   setattr(self,attr,None)
 def set_mode(self,mode):
  with self.mode_lock:
   self.mode=mode
  self.state_var.set(mode)
 def get_mode(self):
  with self.mode_lock:
   return self.mode
 def request_training_interrupt(self):
  self.root.after(0,self.handle_training_interrupt)
 def handle_training_interrupt(self):
  if self.get_mode()==Mode.TRAINING:
   self.start_learning()
 def ensure_default_markers(self):
  specs=[("移动轮盘","red","drag",False),("回城","orange","click",False),("闪现","yellow","drag",True),("恢复","green","click",True),("普攻","blue","click",False),("一技能","indigo","mixed",True),("二技能","indigo","mixed",True),("三技能","indigo","mixed",True),("四技能","indigo","mixed",True),("取消施法","black","drag_in",False),("主动装备","purple","click",True),("数据A","white","observe",False),("数据B","white","observe",False),("数据C","white","observe",False)]
  for name,color,interaction,cooldown in specs:
   self.overlay.ensure_marker(name,color,interaction,cooldown)
  self.refresh_gesture_markers()
 def refresh_gesture_markers(self):
  if hasattr(self,'gesture_manager') and self.gesture_manager:
   self.gesture_manager.refresh_markers()
 def check_environment(self):
  adb=Path(self.pool.config_manager.data.get("adb_path"))
  emulator=Path(self.pool.config_manager.data.get("emulator_path"))
  models_ready=self.pool.validate()
  hardware_ready=self.hardware_ready()
  if adb.exists() and emulator.exists() and models_ready and hardware_ready:
   self.emulator_controller.update_paths()
   self.set_mode(Mode.LEARNING)
   self.start_learning()
  else:
   status=[]
   if not adb.exists():
    status.append("缺少ADB")
   if not emulator.exists():
    status.append("缺少模拟器")
   if not models_ready:
    status.append("AAA文件夹异常")
   if not hardware_ready:
    status.append("硬件资源不足")
   self.status_var.set("，".join(status) if status else "依赖缺失")
 def hardware_ready(self):
  snapshot=self.resource_monitor.snapshot()
  cpu_ok=snapshot["cpu_count"]>0
  memory_ok=snapshot["memory"]>0
  gpu_ok=snapshot["gpu_load"]>=0
  vram_ok=snapshot["vram_usage"]>=0
  freq_ok=1<=snapshot["frequency"]<=120
  return cpu_ok and memory_ok and gpu_ok and vram_ok and freq_ok
 def monitor_input(self):
  while not self.stop_event.is_set():
   time.sleep(0.5)
   if time.time()-self.last_input>self.pool.config_manager.data.get("state_timeout",10) and self.get_mode()==Mode.LEARNING:
    self.root.after(0,self.enter_training_mode)
 def scheduler(self):
  while not self.stop_event.is_set():
   time.sleep(1)
   self.resource_monitor.update_frequency()
   freq=str(self.resource_monitor.frequency)
   self.root.after(0,lambda value=freq:self.freq_var.set(value))
 def start_learning(self):
  self.set_mode(Mode.LEARNING)
  self.wait_thread('training_thread')
  if self.learning_thread and self.learning_thread.is_alive():
   return
  self.status_var.set("采集中")
  self.learning_thread=threading.Thread(target=self.learning_loop,daemon=True)
  self.learning_thread.start()
 def learning_loop(self):
  while not self.stop_event.is_set() and self.get_mode()==Mode.LEARNING:
   self.record_event("user",{"mode":"learning"})
   time.sleep(1/max(1,self.resource_monitor.frequency))
 def record_event(self,source,extra=None):
  with self.state_lock:
   hero_alive=self.hero_alive
   a=self.data_a
   b=self.data_b
   c=self.data_c
   cooldowns=dict(self.cooldown_state)
   recalling=self.recalling
  data={"timestamp":time.time(),"source":source,"hero_alive":hero_alive,"A":a,"B":b,"C":c,"cooldowns":cooldowns,"geometry":self.emu_geometry,"markers":self.overlay.get_markers_data(),"recalling":recalling,"frame":self.last_frame,"mode_context":self.get_mode(),"window_visible":self.emu_visible,"analysis":self.latest_analysis}
  if extra:
   data.update(extra)
  self.pool.record(data)
 def operation_allowed(self):
  with self.state_lock:
   return self.hero_alive and not self.recalling
 def enter_training_mode(self):
  if self.get_mode()!=Mode.LEARNING:
   return
  self.set_mode(Mode.TRAINING)
  self.status_var.set("AI执行中")
  self.wait_thread('learning_thread')
  if self.training_thread and self.training_thread.is_alive():
   return
  self.training_thread=threading.Thread(target=self.training_loop,daemon=True)
  self.training_thread.start()
 def training_loop(self):
  while not self.stop_event.is_set() and self.get_mode()==Mode.TRAINING:
   self.simulate_ai_action()
   self.record_event("ai",{"mode":"training"})
   time.sleep(1/max(1,self.resource_monitor.frequency))
  self.training_thread=None
 def update_state(self,a,b,c,alive,skills,items,heal,flash):
  def safe_int(value):
   try:
    return int(max(0,int(float(value))))
   except Exception:
    return 0
  safe_a=safe_int(a)
  safe_b=safe_int(b)
  safe_c=safe_int(c)
  alive_flag=bool(alive)
  skill_detail=self._normalize_skill_status(skills)
  item_text=str(items)
  heal_text=str(heal)
  flash_text=str(flash)
  def apply():
   with self.state_lock:
    self.data_a=safe_a
    self.data_b=safe_b
    self.data_c=safe_c
    self.hero_alive=alive_flag
    self.cooldown_state['skills']=skill_detail
    self.cooldown_state['items']=item_text
    self.cooldown_state['heal']=heal_text
    self.cooldown_state['flash']=flash_text
    if not alive_flag:
     self.recalling=False
   self.data_a_var.set(str(safe_a))
   self.data_b_var.set(str(safe_b))
   self.data_c_var.set(str(safe_c))
   self.hero_status_var.set('存活' if alive_flag else '阵亡')
   self.cooldown_vars['skills'].set(self._skill_display(skill_detail))
   self.cooldown_vars['items'].set(item_text)
   self.cooldown_vars['heal'].set(heal_text)
   self.cooldown_vars['flash'].set(flash_text)
  self.root.after(0,apply)
 def simulate_ai_action(self):
  with self.state_lock:
   hero_alive=self.hero_alive
   recalling=self.recalling
  if not hero_alive or recalling:
   time.sleep(0.2)
   return
  state=self.build_state_vector(None)
  results={}
  def left_task():
   results['left']=self.left_controller.execute(state)
  def right_task():
   results['right']=self.right_controller.execute(state)
  threads=[threading.Thread(target=left_task,daemon=True),threading.Thread(target=right_task,daemon=True)]
  for t in threads:
   t.start()
  for t in threads:
   t.join()
  timestamp=time.time()
  left_payload=results.get('left')
  right_payload=results.get('right')
  if left_payload:
   left_payload['timestamp']=timestamp
   self.record_event(self.left_controller.name,left_payload)
  if right_payload:
   right_payload['timestamp']=timestamp
   self.record_event(self.right_controller.name,right_payload)
   gesture=right_payload.get('gesture',{})
   if right_payload.get('action')=='回城' and gesture.get('state')=='recall':
    with self.state_lock:
     self.recalling=True
   if right_payload.get('action')=='取消施法':
    with self.state_lock:
     self.recalling=False
 def on_optimize(self):
  if self.get_mode()!=Mode.LEARNING:
   return
  self.set_mode(Mode.OPTIMIZING)
  self.wait_thread('learning_thread')
  self.wait_thread('training_thread')
  self.status_var.set('优化中')
  self.set_progress(0)
  def callback(progress):
   self.root.after(0,lambda value=progress:self.set_progress(value))
  def done(success):
   def finish():
    self.set_progress(0 if not success else 100)
    if success:
     self.reload_models()
     self.adjust_markers()
     self.apply_optimized_metrics()
     showinfo('提示','优化完成')
    self.set_mode(Mode.LEARNING)
    self.start_learning()
   self.root.after(0,finish)
  self.model_handler.optimize(callback,done)
 def adjust_markers(self):
  stats=self.model_handler.last_marker_stats if self.model_handler.last_marker_stats else {}
  updated=False
  for name,marker in self.overlay.markers.items():
   data=stats.get(name)
   if data and data.get('count'):
    count=max(1,data.get('count',1))
    avg_x=data.get('x',marker.x*count)/count
    avg_y=data.get('y',marker.y*count)/count
    avg_r=data.get('radius',marker.radius*count)/count
    marker.x=min(0.95,max(0.05,avg_x))
    marker.y=min(0.95,max(0.05,avg_y))
    marker.radius=min(0.4,max(0.05,avg_r))
    updated=True
   else:
    marker.x=min(0.95,max(0.05,marker.x+random.uniform(-0.02,0.02)))
    marker.y=min(0.95,max(0.05,marker.y+random.uniform(-0.02,0.02)))
    marker.radius=min(0.4,max(0.05,marker.radius+random.uniform(-0.01,0.01)))
    updated=True
  if updated:
   self.overlay.draw_markers()
   self.save_markers()
 def apply_optimized_metrics(self):
  metrics=self.model_handler.last_metrics
  with self.state_lock:
   alive=self.hero_alive
   skills=self.cooldown_state['skills']
   items=self.cooldown_state['items']
   heal=self.cooldown_state['heal']
   flash=self.cooldown_state['flash']
  self.update_state(metrics.get('A',self.data_a),metrics.get('B',self.data_b),metrics.get('C',self.data_c),alive,skills,items,heal,flash)
 def save_markers(self):
  data=self.overlay.get_markers_data()
  self.pool.config_manager.update('markers',data)
  self.pool.config_manager.save()
  self.refresh_gesture_markers()
 def on_cancel_optimize(self):
  self.model_handler.cancel()
  self.set_mode(Mode.LEARNING)
  self.status_var.set('采集中')
  self.set_progress(0)
  self.start_learning()
 def on_configure(self):
  if self.get_mode()!=Mode.LEARNING:
   return
  self.set_mode(Mode.CONFIG)
  self.wait_thread('learning_thread')
  self.status_var.set('配置中')
  self.overlay.open()
  config_window=Toplevel(self.root)
  config_window.title('标志管理')
  listbox=Listbox(config_window,exportselection=False)
  listbox.pack(fill='both',expand=True)
  name_var=StringVar(value='')
  entry=ttk.Entry(config_window,textvariable=name_var)
  entry.pack(fill='x')
  def current_selection():
   sel=listbox.curselection()
   if not sel:
    return None
   try:
    return listbox.get(sel[0])
   except Exception:
    return None
  def refresh_list(selected=None):
   listbox.delete(0,'end')
   for key in self.overlay.markers.keys():
    listbox.insert('end',key)
   if selected and selected in self.overlay.markers:
    keys=list(self.overlay.markers.keys())
    idx=keys.index(selected)
    listbox.selection_set(idx)
    listbox.activate(idx)
    name_var.set(selected)
   elif self.overlay.selected:
    for idx,key in enumerate(self.overlay.markers.keys()):
     if self.overlay.markers[key]==self.overlay.selected:
      listbox.selection_set(idx)
      listbox.activate(idx)
      name_var.set(key)
      break
   elif not self.overlay.markers:
    name_var.set('')
   else:
    name_var.set('')
  def on_select(event):
   name=current_selection()
   if not name:
    name_var.set('')
    self.overlay.selected=None
    self.overlay.draw_markers()
    return
   marker=self.overlay.markers.get(name)
   if marker:
    self.overlay.selected=marker
    name_var.set(name)
    self.overlay.draw_markers()
  listbox.bind('<<ListboxSelect>>',on_select)
  def add_marker():
   name=f'标志{len(self.overlay.markers)+1}'
   base=name
   counter=1
   while name in self.overlay.markers:
    counter+=1
    name=f'{base}_{counter}'
   marker=Marker(name,'white')
   marker.x=0.5
   marker.y=0.5
   marker.radius=0.1
   self.overlay.markers[name]=marker
   self.overlay.selected=marker
   self.overlay.draw_markers()
   refresh_list(name)
  def rename_marker():
   old=current_selection()
   new=name_var.get().strip()
   if not old or not new or new==old:
    return
   if new in self.overlay.markers:
    showinfo('提示','名称已存在')
    refresh_list(old)
    return
   items=list(self.overlay.markers.items())
   new_dict={}
   marker=None
   for key,value in items:
    if key==old:
     marker=value
     marker.name=new
     new_dict[new]=marker
    else:
     new_dict[key]=value
   if not marker:
    return
   self.overlay.markers=new_dict
   self.overlay.selected=marker
   self.overlay.draw_markers()
   refresh_list(new)
  def delete_marker():
   name=current_selection()
   if not name:
    return
   marker=self.overlay.markers.pop(name,None)
   if marker:
    if self.overlay.selected==marker:
     self.overlay.selected=None
    self.overlay.draw_markers()
    refresh_list()
  Button(config_window,text='添加标志',command=add_marker).pack(fill='x')
  Button(config_window,text='重命名',command=rename_marker).pack(fill='x')
  Button(config_window,text='删除标志',command=delete_marker).pack(fill='x')
  def save_and_close():
   self.save_markers()
   showinfo('提示','配置已保存')
   self.overlay.close()
   config_window.destroy()
   self.set_mode(Mode.LEARNING)
   self.start_learning()
  Button(config_window,text='保存',command=save_and_close).pack(fill='x')
  refresh_list()
  config_window.protocol('WM_DELETE_WINDOW',save_and_close)
 def on_change_folder(self):
  new_dir=askdirectory()
  if not new_dir:
   return
  try:
   self.pool.migrate(new_dir)
   self.left_controller.model_path=self.pool.left_model
   self.right_controller.model_path=self.pool.right_model
   self.overlay.load_markers(self.pool.config_manager.data.get('markers',{}))
   self.ensure_default_markers()
   self.pool.config_manager.update('aaa_folder',str(self.pool.folder))
   self.emulator_controller.update_paths()
   self.reload_models()
   self.save_markers()
   self.status_var.set('已迁移')
  except Exception as e:
   showinfo('错误',str(e))
 def save_config(self):
  data=self.pool.config_manager.data
  data.update({'screenshot_hz':self.resource_monitor.frequency,'markers':self.overlay.get_markers_data(),'aaa_folder':str(self.pool.folder)})
  self.pool.config_manager.save()
  showinfo('提示','配置已保存')
 def load_config(self):
  self.pool.config_manager.load()
  folder_path=Path(self.pool.config_manager.data.get('aaa_folder',str(self.pool.folder)))
  if folder_path.resolve()!=self.pool.folder.resolve():
   self.pool.migrate(folder_path)
   self.left_controller.model_path=self.pool.left_model
   self.right_controller.model_path=self.pool.right_model
  self.overlay.load_markers(self.pool.config_manager.data.get('markers',{}))
  self.ensure_default_markers()
  self.emulator_controller.update_paths()
  self.reload_models()
  self.status_var.set('配置已加载')
 def update_emulator_geometry(self,x,y,width,height,visible):
  geometry=(int(x),int(y),max(0,int(width)),max(0,int(height)))
  changed=geometry!=self.emu_geometry or visible!=self.emu_visible
  self.emu_geometry=geometry
  self.emu_visible=visible
  if changed:
   self.overlay.update_geometry(geometry[0],geometry[1],geometry[2],geometry[3],visible)
 def point_in_emulator(self,x,y):
  gx,gy,gw,gh=self.emu_geometry
  if gw<=0 or gh<=0 or x is None or y is None:
   return False
  return gx<=x<=gx+gw and gy<=y<=gy+gh
 def normalize_position(self,x,y):
  gx,gy,gw,gh=self.emu_geometry
  if gw<=0 or gh<=0 or x is None or y is None:
   return None
  return ((x-gx)/gw,(y-gy)/gh)
 def get_emulator_geometry(self):
  return self.emu_geometry[2],self.emu_geometry[3]
 def stop(self):
  self.stop_event.set()
  self.left_controller.stop()
  self.right_controller.stop()
  self.frame_capture.stop()
  self.mouse_monitor.stop()
  self.window_tracker.stop()
  if hasattr(self,'emulator_controller') and self.emulator_controller:
   self.emulator_controller.stop()
  self.root.quit()
 def monitor_user_action(self,event=None):
  self.last_input=time.time()
  if self.get_mode()==Mode.TRAINING:
   self.start_learning()
   return
  if self.get_mode()==Mode.LEARNING and event is not None and not hasattr(event,'keysym'):
   payload={'mode':'learning','event':'ui','position':[getattr(event,'x',0),getattr(event,'y',0)],'widget':str(getattr(event,'widget',''))}
   self.record_event('user-ui',payload)
 def scheduler_event_bindings(self):
  self.root.bind_all('<Key>',self.monitor_user_action)
  self.root.bind_all('<Button>',self.monitor_user_action)
  self.root.bind_all('<Motion>',self.monitor_user_action)
  self.root.bind_all('<MouseWheel>',self.monitor_user_action)
 def run(self):
  self.scheduler_event_bindings()
  self.root.mainloop()
app=None
if __name__=='__main__':
 app=MainApp()
 app.run()
