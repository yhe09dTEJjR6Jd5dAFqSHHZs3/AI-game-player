import os
import sys
import time
import json
import threading
import shutil
import math
import random
import ctypes
import logging
import importlib
import queue
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from enum import Enum
from ctypes import wintypes
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque,OrderedDict
logging.basicConfig(level=logging.INFO,format="%(asctime)s %(levelname)s %(message)s")
logger=logging.getLogger("aaa_runtime")
REQUIRED_MODULES=["psutil","pyautogui","pynput","PIL","numpy","torch","PyQt5"]
class ModuleMissingError(ImportError):
    def __init__(self,name,detail):
        super().__init__(detail)
        self.name=name
class DependencyManager:
    def __init__(self,names):
        self.names=list(names)
        self.lock=threading.Lock()
        self.modules={}
        self.failures={}
        self.preload_thread=None
        self.preload_started=False
    def register(self,name):
        with self.lock:
            if name not in self.names:
                self.names.append(name)
    def ensure_module(self,name):
        module=self.modules.get(name)
        if module is not None:
            return module
        failure=self.failures.get(name)
        if failure is not None:
            raise failure
        with self.lock:
            module=self.modules.get(name)
            if module is not None:
                return module
            failure=self.failures.get(name)
            if failure is not None:
                raise failure
            try:
                module=importlib.import_module(name)
                self.modules[name]=module
                return module
            except ImportError as e:
                err=ModuleMissingError(name,str(e))
                self.failures[name]=err
                raise err
    def verify(self):
        missing=[]
        for name in list(self.names):
            try:
                self.ensure_module(name)
            except ModuleMissingError as e:
                missing.append(e.name)
        return missing
    def start_preloading(self):
        with self.lock:
            if self.preload_started:
                return
            self.preload_started=True
        def worker():
            for name in list(self.names):
                try:
                    self.ensure_module(name)
                except ModuleMissingError:
                    pass
        thread=threading.Thread(target=worker)
        thread.daemon=True
        thread.start()
        with self.lock:
            self.preload_thread=thread
dependency_manager=DependencyManager(REQUIRED_MODULES)
class LazyModule:
    def __init__(self,name):
        self.name=name
        dependency_manager.register(name)
    def _load(self):
        return dependency_manager.ensure_module(self.name)
    def __getattr__(self,item):
        return getattr(self._load(),item)
    def __call__(self,*args,**kwargs):
        return self._load()(*args,**kwargs)
psutil=LazyModule("psutil")
pyautogui=LazyModule("pyautogui")
mouse=LazyModule("pynput.mouse")
keyboard=LazyModule("pynput.keyboard")
win32gui=LazyModule("win32gui")
win32con=LazyModule("win32con")
win32api=LazyModule("win32api")
ImageModule=LazyModule("PIL.Image")
ImageGrabModule=LazyModule("PIL.ImageGrab")
QtCore=LazyModule("PyQt5.QtCore")
QtGui=LazyModule("PyQt5.QtGui")
QtWidgets=LazyModule("PyQt5.QtWidgets")
class Mode(Enum):
    INIT=0
    LEARNING=1
    OPTIMIZING=2
    CONFIGURING=3
    TRAINING=4
def clamp(v,a,b):
    return max(a,min(b,v))
RIGHT_ACTION_LABELS=["回城","恢复","闪现","普攻","一技能","二技能","三技能","四技能","取消施法","主动装备","数据A","数据B","数据C"]
def build_right_mapping():
    mapping={}
    offset=0
    mapping["回城"]=(offset,1)
    offset+=1
    mapping["恢复"]=(offset,1)
    offset+=1
    mapping["闪现"]=(offset,4)
    offset+=4
    mapping["普攻"]=(offset,1)
    offset+=1
    mapping["一技能"]=(offset,4)
    offset+=4
    mapping["二技能"]=(offset,4)
    offset+=4
    mapping["三技能"]=(offset,4)
    offset+=4
    mapping["四技能"]=(offset,4)
    offset+=4
    mapping["取消施法"]=(offset,4)
    offset+=4
    mapping["主动装备"]=(offset,1)
    offset+=1
    mapping["数据A"]=(offset,1)
    offset+=1
    mapping["数据B"]=(offset,1)
    offset+=1
    mapping["数据C"]=(offset,1)
    offset+=1
    mapping["其他"]=(31,1)
    return mapping
def build_right_decode_table(mapping):
    table=[]
    size=32
    for idx in range(size):
        label="其他"
        count=1
        local=0
        base=mapping.get("其他",(31,1))[0]
        for name,(start,length) in mapping.items():
            if start<=idx<start+length and name!="其他":
                label=name
                count=length
                local=idx-start
                base=start
                break
        table.append((label,count,local,base))
    return table
RIGHT_ACTION_MAPPING=build_right_mapping()
RIGHT_ACTION_TABLE=build_right_decode_table(RIGHT_ACTION_MAPPING)
MARKER_SPEC={"移动轮盘":{"color":(255,0,0),"radius":0.14,"alpha":0.5,"required":True,"pos":(0.2,0.78)},"回城":{"color":(255,165,0),"radius":0.08,"alpha":0.5,"required":True,"pos":(0.88,0.82)},"恢复":{"color":(0,255,0),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.74,0.82)},"闪现":{"color":(255,255,0),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.62,0.82)},"普攻":{"color":(0,0,255),"radius":0.07,"alpha":0.5,"required":True,"pos":(0.86,0.68)},"一技能":{"color":(75,0,130),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.74,0.64)},"二技能":{"color":(75,0,130),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.88,0.56)},"三技能":{"color":(75,0,130),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.66,0.52)},"四技能":{"color":(75,0,130),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.94,0.48)},"取消施法":{"color":(0,0,0),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.5,0.5)},"主动装备":{"color":(128,0,128),"radius":0.06,"alpha":0.5,"required":True,"pos":(0.7,0.75)},"数据A":{"color":(200,200,0),"radius":0.05,"alpha":0.5,"required":True,"pos":(0.1,0.1)},"数据B":{"color":(0,200,200),"radius":0.05,"alpha":0.5,"required":True,"pos":(0.2,0.1)},"数据C":{"color":(200,0,200),"radius":0.05,"alpha":0.5,"required":True,"pos":(0.3,0.1)}}
def verify_dependencies():
    return dependency_manager.verify()
def get_desktop_path():
    return os.path.join(os.path.expanduser("~"),"Desktop")
class AAAFileManager:
    class MoveError(Exception):
        pass
    class PrecheckError(MoveError):
        def __init__(self,msg,level=logging.ERROR):
            super().__init__(msg)
            self.level=level
    class OperationError(MoveError):
        pass
    class RecoveryError(MoveError):
        pass
    def __init__(self):
        self.lock=threading.Lock()
        self.base_path=os.path.join(get_desktop_path(),"AAA")
        self.experience_dir=None
        self.config_path=None
        self.vision_model_path=None
        self.left_model_path=None
        self.right_model_path=None
        self.init_status_path=None
        self.option_model_path=None
        self.ensure_structure()
    def ensure_structure(self):
        with self.lock:
            if not os.path.exists(self.base_path):
                os.makedirs(self.base_path,exist_ok=True)
            self.experience_dir=os.path.join(self.base_path,"experience")
            os.makedirs(self.experience_dir,exist_ok=True)
            self.config_path=os.path.join(self.base_path,"config.json")
            self.vision_model_path=os.path.join(self.base_path,"vision_model.pt")
            self.left_model_path=os.path.join(self.base_path,"left_hand_model.pt")
            self.right_model_path=os.path.join(self.base_path,"right_hand_model.pt")
            self.neuro_model_path=os.path.join(self.base_path,"neuro_module.pt")
            self.option_model_path=os.path.join(self.base_path,"option_planner.pt")
            self.init_status_path=os.path.join(self.base_path,"init_status.json")
            if not os.path.exists(self.config_path):
                default_cfg={"markers":[],"aaa_path":self.base_path}
                with open(self.config_path,"w",encoding="utf-8") as f:
                    json.dump(default_cfg,f)
            if not os.path.exists(self.vision_model_path):
                torch.save({},self.vision_model_path)
            if not os.path.exists(self.left_model_path):
                torch.save({},self.left_model_path)
            if not os.path.exists(self.right_model_path):
                torch.save({},self.right_model_path)
            if not os.path.exists(self.neuro_model_path):
                torch.save({},self.neuro_model_path)
            if not os.path.exists(self.option_model_path):
                torch.save({},self.option_model_path)
    def _precheck_move(self,old_base,new_parent):
        if not new_parent:
            raise self.PrecheckError("目标路径为空",logging.INFO)
        requested=os.path.abspath(new_parent)
        if not requested:
            raise self.PrecheckError("目标路径无效")
        if os.path.basename(requested).lower()=="aaa":
            target_parent=os.path.dirname(requested)
            target_base=requested
        else:
            target_parent=requested
            target_base=os.path.join(target_parent,"AAA")
        if not target_parent:
            raise self.PrecheckError("缺少目标上级目录")
        try:
            common=os.path.commonpath([old_base,target_base])
        except ValueError:
            common=None
        if os.path.normcase(target_base)==os.path.normcase(old_base):
            raise self.PrecheckError("目标目录与当前目录相同",logging.INFO)
        if common==old_base:
            raise self.PrecheckError("目标目录位于当前目录内部")
        try:
            os.makedirs(target_parent,exist_ok=True)
        except OSError as e:
            raise self.PrecheckError("创建目标父目录失败:"+str(e))
        if os.path.exists(target_base):
            try:
                shutil.rmtree(target_base)
            except FileNotFoundError:
                pass
            except OSError as e:
                raise self.PrecheckError("清理旧目标目录失败:"+str(e))
        logger.info("AAA目录预检查通过:%s -> %s",old_base,target_base)
        return target_parent,target_base
    def _perform_move(self,old_base,target_base):
        logger.info("AAA目录开始移动:%s -> %s",old_base,target_base)
        try:
            shutil.move(old_base,target_base)
        except (shutil.Error,OSError) as e:
            raise self.OperationError("直接移动失败:"+str(e))
    def _recover_move(self,old_base,target_base,original_error):
        logger.info("AAA目录进入恢复流程:%s -> %s",old_base,target_base)
        logger.warning("AAA目录恢复触发原因:%s",original_error)
        if os.path.normcase(old_base)==os.path.normcase(target_base):
            return
        source_base=old_base if os.path.exists(old_base) else target_base
        if not os.path.exists(source_base):
            raise self.RecoveryError("源目录缺失")
        if os.path.normcase(source_base)==os.path.normcase(target_base):
            logger.info("AAA目录恢复检测:数据已在目标位置")
            return
        moved=[]
        try:
            os.makedirs(target_base,exist_ok=True)
            for name in os.listdir(source_base):
                src=os.path.join(source_base,name)
                dst=os.path.join(target_base,name)
                if os.path.normcase(src)==os.path.normcase(dst):
                    continue
                shutil.move(src,dst)
                moved.append((dst,src))
            if os.path.exists(old_base) and os.path.normcase(old_base)!=os.path.normcase(target_base):
                try:
                    shutil.rmtree(old_base)
                except FileNotFoundError:
                    pass
                except OSError as cleanup_error:
                    logger.warning("清理旧AAA残留失败:%s",cleanup_error)
            return
        except (OSError,shutil.Error) as e:
            for dst,src in reversed(moved):
                if os.path.exists(dst):
                    try:
                        shutil.move(dst,src)
                    except (OSError,shutil.Error) as rollback_error:
                        logger.error("AAA目录回滚失败:%s",rollback_error)
            raise self.RecoveryError("恢复流程失败:"+str(e))
    def move_dir(self,new_parent):
        with self.lock:
            old_base=os.path.abspath(self.base_path)
            try:
                target_parent,target_base=self._precheck_move(old_base,new_parent)
            except self.PrecheckError as e:
                level=getattr(e,"level",logging.ERROR)
                logger.log(level,"AAA目录预检查终止:%s",e)
                return old_base,old_base
            try:
                self._perform_move(old_base,target_base)
            except self.OperationError as move_error:
                logger.warning("AAA目录直接移动失败:%s",move_error)
                try:
                    self._recover_move(old_base,target_base,move_error)
                except self.RecoveryError as recovery_error:
                    logger.error("AAA目录迁移恢复失败:%s",recovery_error)
                    return old_base,old_base
            logger.info("AAA目录迁移完成:%s -> %s",old_base,target_base)
            self.base_path=target_base
            self.ensure_structure()
            self.update_config_path()
            return old_base,target_base
    def write_init_status(self,status):
        with self.lock:
            if self.init_status_path is None:
                self.init_status_path=os.path.join(self.base_path,"init_status.json")
            try:
                with open(self.init_status_path,"w",encoding="utf-8") as f:
                    json.dump(status,f)
            except OSError as e:
                logger.error("写入初始化状态失败:%s",e)
    def update_config_path(self):
        with self.lock:
            if os.path.exists(self.config_path):
                with open(self.config_path,"r",encoding="utf-8") as f:
                    cfg=json.load(f)
            else:
                cfg={"markers":[]}
            cfg["aaa_path"]=self.base_path
            with open(self.config_path,"w",encoding="utf-8") as f:
                json.dump(cfg,f)
    def save_markers(self,markers,window_rect):
        with self.lock:
            if os.path.exists(self.config_path):
                with open(self.config_path,"r",encoding="utf-8") as f:
                    cfg=json.load(f)
            else:
                cfg={"aaa_path":self.base_path}
            out=[]
            for m in markers:
                out.append({"label":m.label,"x_pct":m.x_pct,"y_pct":m.y_pct,"r_pct":m.r_pct,"color":m.color.name(),"alpha":m.alpha})
            cfg["markers"]=out
            with open(self.config_path,"w",encoding="utf-8") as f:
                json.dump(cfg,f)
    def load_markers(self):
        with self.lock:
            out=[]
            if os.path.exists(self.config_path):
                with open(self.config_path,"r",encoding="utf-8") as f:
                    cfg=json.load(f)
                    for m in cfg.get("markers",[]):
                        out.append(m)
            return out
    def save_models(self,vision,left,right,neuro,planner):
        with self.lock:
            torch.save(vision.state_dict(),self.vision_model_path)
            torch.save(left.state_dict(),self.left_model_path)
            torch.save(right.state_dict(),self.right_model_path)
            torch.save(neuro.state_dict(),self.neuro_model_path)
            torch.save(planner.state_dict(),self.option_model_path)
    def load_models(self,vision,left,right,neuro,planner):
        with self.lock:
            if os.path.exists(self.vision_model_path):
                try:
                    vision.load_state_dict(torch.load(self.vision_model_path,map_location="cpu"))
                except (OSError,RuntimeError,ValueError) as e:
                    logger.warning("加载视觉模型失败:%s",e)
            if os.path.exists(self.left_model_path):
                try:
                    left.load_state_dict(torch.load(self.left_model_path,map_location="cpu"))
                except (OSError,RuntimeError,ValueError) as e:
                    logger.warning("加载左手模型失败:%s",e)
            if os.path.exists(self.right_model_path):
                try:
                    right.load_state_dict(torch.load(self.right_model_path,map_location="cpu"))
                except (OSError,RuntimeError,ValueError) as e:
                    logger.warning("加载右手模型失败:%s",e)
            if os.path.exists(self.neuro_model_path):
                try:
                    neuro.load_state_dict(torch.load(self.neuro_model_path,map_location="cpu"))
                except (OSError,RuntimeError,ValueError) as e:
                    logger.warning("加载神经模块失败:%s",e)
            if os.path.exists(self.option_model_path):
                try:
                    planner.load_state_dict(torch.load(self.option_model_path,map_location="cpu"))
                except (OSError,RuntimeError,ValueError) as e:
                    logger.warning("加载意图规划器失败:%s",e)
    def to_relative(self,abs_path):
        with self.lock:
            base=self.base_path
        try:
            rel=os.path.relpath(abs_path,base)
        except ValueError as e:
            logger.warning("转换相对路径失败:%s",e)
            rel=abs_path
        if rel.startswith("..") and not abs_path.startswith(base):
            rel=abs_path
        return rel
    def to_absolute(self,path):
        with self.lock:
            base=self.base_path
        if not path:
            return path
        if os.path.isabs(path):
            return path
        return os.path.join(base,path)
class ExperienceBuffer:
    def __init__(self,manager,capacity=10000):
        self.manager=manager
        self.capacity=capacity
        self.lock=threading.Lock()
        self.data=[]
        self.meta_log=""
        self.cache_lock=threading.Lock()
        self.frame_cache=OrderedDict()
        self.cache_capacity=256
        self.refresh_paths()
    def refresh_paths(self):
        with self.lock:
            self._refresh_paths_locked()
    def _refresh_paths_locked(self):
        self.meta_log=os.path.join(self.manager.experience_dir,"exp.jsonl")
    def add(self,frame_img,action,source,metrics,hero_dead,cooldowns,window_rect):
        ts=time.time()
        fname=os.path.join(self.manager.experience_dir,str(int(ts*1000))+".png")
        rel_frame=None
        try:
            frame_img.save(fname)
            rel_frame=self.manager.to_relative(fname)
        except (OSError,ValueError) as e:
            logger.error("保存帧失败:%s",e)
        rec={"t":ts,"frame":rel_frame,"action":action,"source":source,"metrics":metrics,"hero_dead":hero_dead,"cooldowns":cooldowns,"rect":window_rect}
        with self.lock:
            self.data.append(rec)
            if len(self.data)>self.capacity:
                self.data.pop(0)
        try:
            with open(self.meta_log,"a",encoding="utf-8") as f:
                f.write(json.dumps(rec)+"\n")
        except OSError as e:
            logger.error("追加经验失败:%s",e)
    def _rewrite_locked(self):
        try:
            with open(self.meta_log,"w",encoding="utf-8") as f:
                for rec in self.data:
                    f.write(json.dumps(rec)+"\n")
        except OSError as e:
            logger.error("重写经验失败:%s",e)
    def on_aaa_moved(self,old_base,new_base):
        with self.lock:
            self._refresh_paths_locked()
            records=[]
            try:
                if os.path.exists(self.meta_log):
                    with open(self.meta_log,"r",encoding="utf-8") as f:
                        for line in f:
                            line=line.strip()
                            if not line:
                                continue
                            try:
                                rec=json.loads(line)
                                records.append(rec)
                            except json.JSONDecodeError as e:
                                logger.warning("忽略损坏经验记录:%s",e)
            except OSError as e:
                logger.error("读取经验失败:%s",e)
                records=[]
            if not records:
                records=list(self.data)
            normalized=[]
            for rec in records:
                frame=rec.get("frame")
                if frame:
                    abs_old=frame if os.path.isabs(frame) else os.path.join(old_base,frame)
                    try:
                        rel_tail=os.path.relpath(abs_old,old_base)
                    except ValueError as e:
                        logger.warning("经验路径转换失败:%s",e)
                        if abs_old.startswith(old_base):
                            rel_tail=abs_old[len(old_base):].lstrip(os.sep)
                        else:
                            rel_tail=os.path.basename(abs_old)
                    abs_new=os.path.join(new_base,rel_tail)
                    rec["frame"]=self.manager.to_relative(abs_new)
                normalized.append(rec)
            self.data=normalized
            if len(self.data)>self.capacity:
                self.data=self.data[-self.capacity:]
            self._rewrite_locked()
            with self.cache_lock:
                self.frame_cache=OrderedDict()
    def get_frame_arrays(self,records,size):
        w=int(size[0])
        h=int(size[1])
        results=[None]*len(records)
        missing=[]
        with self.cache_lock:
            for idx,rec in enumerate(records):
                frame=rec.get("frame")
                if not frame:
                    continue
                key=(frame,w,h)
                cached=self.frame_cache.get(key)
                if cached is not None:
                    results[idx]=cached
                    self.frame_cache.move_to_end(key)
                else:
                    missing.append((idx,key,frame))
        if not missing:
            return results
        load_queue=queue.Queue()
        def loader(entry):
            idx,key,frame=entry
            arr=None
            abs_path=self.manager.to_absolute(frame)
            try:
                with ImageModule.open(abs_path) as img:
                    img=img.convert("RGB")
                    img=img.resize((w,h))
                    arr=np.asarray(img,dtype=np.float32)/255.0
                arr=np.transpose(arr,(2,0,1))
            except (OSError,ValueError,TypeError) as e:
                logger.warning("加载经验帧失败:%s",e)
                arr=None
            load_queue.put((idx,key,arr))
        workers=min(len(missing),max(1,os.cpu_count() or 1))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for entry in missing:
                pool.submit(loader,entry)
        loaded=[]
        for _ in missing:
            loaded.append(load_queue.get())
        with self.cache_lock:
            for idx,key,arr in loaded:
                if arr is not None:
                    results[idx]=arr
                    self.frame_cache[key]=arr
            while len(self.frame_cache)>self.cache_capacity:
                self.frame_cache.popitem(last=False)
        return results
    def sample(self,batch_size=32):
        with self.lock:
            if len(self.data)==0:
                return []
            idx=[random.randint(0,len(self.data)-1) for _ in range(min(batch_size,len(self.data)))]
            return [self.data[i] for i in idx]
    def get_user_click_stats(self):
        with self.lock:
            acc={}
            for rec in self.data:
                if rec["source"]!="user":
                    continue
                act=rec["action"]
                if not act:
                    continue
                label=act.get("label")
                if label is None:
                    continue
                pos=act.get("pos")
                rect=rec["rect"]
                if rect and pos:
                    if label not in acc:
                        acc[label]={"xs":[],"ys":[]}
                    acc[label]["xs"].append((pos[0]-rect[0])/(max(rect[2]-rect[0],1)))
                    acc[label]["ys"].append((pos[1]-rect[1])/(max(rect[3]-rect[1],1)))
            out={}
            for k,v in acc.items():
                if len(v["xs"])>0:
                    mx=sum(v["xs"])/len(v["xs"])
                    my=sum(v["ys"])/len(v["ys"])
                    sx=(sum([(a-mx)**2 for a in v["xs"]])/len(v["xs"]))**0.5
                    sy=(sum([(a-my)**2 for a in v["ys"]])/len(v["ys"]))**0.5
                    r=(sx+sy)/2.0
                    out[k]={"x_pct":mx,"y_pct":my,"r_pct":r}
            return out
class VisionModel(nn.Module):
    def __init__(self):
        super(VisionModel,self).__init__()
        self.conv1=nn.Conv2d(3,16,3,2,1)
        self.conv2=nn.Conv2d(16,32,3,2,1)
        self.conv3=nn.Conv2d(32,64,3,2,1)
        self.attn=nn.MultiheadAttention(64,4,batch_first=True)
        self.attn_norm=nn.LayerNorm(64)
        self.pool=nn.AdaptiveAvgPool2d((30,30))
        self.fc_state=nn.Linear(64*30*30,32)
        self.fc_metrics=nn.Linear(32,3)
        self.fc_flags=nn.Linear(32,10)
    def forward(self,x):
        x=F.relu(self.conv1(x))
        x=F.relu(self.conv2(x))
        x=F.relu(self.conv3(x))
        b=x.size(0)
        c=x.size(1)
        h=x.size(2)
        w=x.size(3)
        seq=x.view(b,c,h*w).transpose(1,2)
        attn_out,_=self.attn(seq,seq,seq)
        seq=self.attn_norm(seq+attn_out)
        x=seq.transpose(1,2).view(b,c,h,w)
        x=self.pool(x)
        x=x.view(x.size(0),-1)
        h=F.relu(self.fc_state(x))
        metrics=F.relu(self.fc_metrics(h))
        flags=torch.sigmoid(self.fc_flags(h))
        return metrics,flags,h
class BrainInspiredNeuroModule(nn.Module):
    def __init__(self,dim):
        super(BrainInspiredNeuroModule,self).__init__()
        self.encoder=nn.Linear(dim,dim)
        self.gate=nn.Linear(dim,dim)
    def forward(self,x,state):
        sensory=torch.tanh(self.encoder(x))
        gating=torch.sigmoid(self.gate(x))
        updated=state*0.9+gating*sensory
        output=torch.tanh(updated)
        return output,updated
class RelativePositionalBias(nn.Module):
    def __init__(self,heads,max_len=256):
        super(RelativePositionalBias,self).__init__()
        self.heads=heads
        self.max_len=max_len
        self.bias=nn.Parameter(torch.zeros(2*max_len-1,heads))
        nn.init.trunc_normal_(self.bias,std=0.02)
    def forward(self,length):
        idx=torch.arange(length,device=self.bias.device)
        diff=idx[:,None]-idx[None,:]+self.max_len-1
        diff=diff.clamp(0,2*self.max_len-2)
        values=self.bias[diff]
        return values.permute(2,0,1).unsqueeze(0)
class LightweightTransformerLayer(nn.Module):
    def __init__(self,dim,heads,dropout=0.1):
        super(LightweightTransformerLayer,self).__init__()
        self.dim=dim
        self.heads=heads
        self.head_dim=dim//heads
        self.q_proj=nn.Linear(dim,dim)
        self.k_proj=nn.Linear(dim,dim)
        self.v_proj=nn.Linear(dim,dim)
        self.out_proj=nn.Linear(dim,dim)
        self.rel_bias=RelativePositionalBias(heads)
        self.norm1=nn.LayerNorm(dim)
        self.norm2=nn.LayerNorm(dim)
        self.dropout1=nn.Dropout(dropout)
        self.dropout2=nn.Dropout(dropout)
        self.ff=nn.Sequential(nn.Linear(dim,dim*2),nn.GELU(),nn.Linear(dim*2,dim))
    def forward(self,x):
        B,L,_=x.shape
        q=self.q_proj(x).view(B,L,self.heads,self.head_dim).transpose(1,2)
        k=self.k_proj(x).view(B,L,self.heads,self.head_dim).transpose(1,2)
        v=self.v_proj(x).view(B,L,self.heads,self.head_dim).transpose(1,2)
        attn=torch.matmul(q,k.transpose(-2,-1))/math.sqrt(self.head_dim)
        bias=self.rel_bias(L)
        attn=attn+bias
        weights=F.softmax(attn,dim=-1)
        context=torch.matmul(weights,v).transpose(1,2).contiguous().view(B,L,self.dim)
        x=self.norm1(x+self.dropout1(self.out_proj(context)))
        x=self.norm2(x+self.dropout2(self.ff(x)))
        return x
class TransformerHandPolicy(nn.Module):
    def __init__(self,in_dim,action_dim,depth=3,heads=4):
        super(TransformerHandPolicy,self).__init__()
        self.embed=nn.Linear(in_dim,128)
        self.layers=nn.ModuleList(LightweightTransformerLayer(128,heads,0.1) for _ in range(depth))
        self.norm=nn.LayerNorm(128)
        self.actor=nn.Linear(128,action_dim)
        self.critic=nn.Linear(128,1)
    def forward(self,x,option_vec=None):
        y=self.embed(x)
        if option_vec is not None:
            y=y+option_vec.unsqueeze(1)
        for layer in self.layers:
            y=layer(y)
        y=self.norm(y)
        pooled=y[:,-1]
        logits=self.actor(pooled)
        val=self.critic(pooled)
        return logits,val
class OptionPlanner(nn.Module):
    def __init__(self,dim,num_options):
        super(OptionPlanner,self).__init__()
        self.num_options=num_options
        self.state_proj=nn.Linear(dim,dim)
        self.policy_head=nn.Linear(dim,num_options)
        self.termination_head=nn.Linear(dim,num_options)
        self.value_head=nn.Linear(dim,1)
        self.option_embeddings=nn.Parameter(torch.randn(num_options,128))
        nn.init.xavier_uniform_(self.option_embeddings)
    def forward(self,state):
        z=torch.tanh(self.state_proj(state))
        logits=self.policy_head(z)
        termination=torch.sigmoid(self.termination_head(z))
        value=self.value_head(z)
        return logits,termination,self.option_embeddings,value
class AdaptiveRewardModel:
    def __init__(self,hardware):
        self.hardware=hardware
        self.alpha=0.06
        self.running=torch.zeros(3)
        self.variation=torch.ones(3)
        self.preference=torch.tensor([1.0,-1.0,0.5])
        self.temperature=1.0
        self.user_bias=0.5
        self.recent=deque(maxlen=128)
    def _resource_vector(self):
        return torch.tensor([max(0.0,1.0-self.hardware.cpu_load/100.0),max(0.0,self.hardware.mem_free),max(0.0,self.hardware.gpu_free)],dtype=torch.float32)
    def _observe(self,vec,source,hero_dead,is_ai):
        device=self.running.device
        vec=vec.to(device)
        self.running=self.running*(1.0-self.alpha)+vec*self.alpha
        diff=(vec-self.running).abs()+1e-4
        self.variation=self.variation*(1.0-self.alpha)+diff*self.alpha
        self.recent.append(vec.detach().cpu())
        resource=self._resource_vector().to(device)
        resource_score=clamp(resource.mean().item(),0.0,1.0)
        self.user_bias=(1.0-self.alpha)*self.user_bias+self.alpha*(0.7 if source=="user" else 0.4)
        hero_factor=0.6+0.4*float(hero_dead)
        density=len(self.recent)/float(self.recent.maxlen if self.recent.maxlen else 1)
        skill_focus=0.5+density*0.5
        emphasis=torch.stack([self.running[0]+resource_score,self.running[2]+skill_focus,self.running[1]+hero_factor])
        weights=F.softmax(emphasis,dim=0)
        adjust=torch.tensor([1.0+resource_score,hero_factor,skill_focus if is_ai else self.user_bias],dtype=torch.float32,device=device)
        pref=torch.stack([weights[0]*adjust[0],-(weights[2]*adjust[1]),weights[1]*adjust[2]],dim=0)
        self.preference=pref
        self.temperature=float(clamp(resource_score*1.8+0.4,0.5,2.5))
    def compute(self,metrics,source,hero_dead,is_ai):
        device=self.running.device
        vec=torch.tensor([metrics[0],metrics[1],metrics[2]],dtype=torch.float32,device=device)
        self._observe(vec,source,hero_dead,is_ai)
        normalized=(vec-self.running)/(self.variation+1e-3)
        reward=(normalized*self.preference).sum()
        scale=self.temperature*(1.1 if is_ai else self.user_bias+0.6)
        return float(reward.item()*scale)
class MetaLearner:
    def __init__(self,modules,step_size=0.05):
        self.modules=list(modules)
        self.step_size=step_size
    def capture(self):
        return [self._clone_state(m) for m in self.modules]
    def _clone_state(self,module):
        return {k:v.detach().clone() for k,v in module.state_dict().items()}
    def reptile_update(self,snapshots):
        for module,state in zip(self.modules,snapshots):
            current=module.state_dict()
            for name,param in current.items():
                target=state.get(name)
                if target is None:
                    continue
                param.data.add_((target-param.data)*self.step_size)
class RLAgent:
    def __init__(self,file_manager,hardware_manager):
        self.hardware=hardware_manager
        self.hardware.refresh(True)
        self.device=self.hardware.suggest_device()
        self.vision=VisionModel().to(self.device)
        self.left=TransformerHandPolicy(32,16).to(self.device)
        self.right=TransformerHandPolicy(32,32).to(self.device)
        self.neuro_module=BrainInspiredNeuroModule(32).to(self.device)
        self.option_planner=OptionPlanner(32,6).to(self.device)
        self.reward_model=AdaptiveRewardModel(self.hardware)
        self.file_manager=file_manager
        self.file_manager.load_models(self.vision,self.left,self.right,self.neuro_module,self.option_planner)
        self.left_opt=optim.Adam(self.left.parameters(),lr=1e-4,weight_decay=1e-5)
        self.right_opt=optim.Adam(self.right.parameters(),lr=1e-4,weight_decay=1e-5)
        self.vision_opt=optim.Adam(self.vision.parameters(),lr=1e-4,weight_decay=1e-5)
        self.neuro_opt=optim.Adam(self.neuro_module.parameters(),lr=1e-4,weight_decay=1e-5)
        self.option_opt=optim.Adam(self.option_planner.parameters(),lr=1e-4,weight_decay=1e-5)
        self.entropy_coef=0.01
        self.value_coef=0.5
        self.global_step=0
        self.neuro_state=torch.zeros(32,device=self.device)
        self.state_history=deque(maxlen=64)
        self.option_history=deque(maxlen=64)
        self.current_option=0
        self.meta_controller=MetaLearner([self.vision,self.left,self.right,self.neuro_module,self.option_planner],0.05)
        self.batch_buffer=None
        self.cpu_batch=None
        self.batch_capacity=0
        self.batch_shape=None
    def compute_reward(self,A,B,C,source,hero_dead):
        return self.reward_model.compute((A,B,C),source,hero_dead,source!="user")
    def ensure_device(self):
        self.hardware.refresh()
        target=self.hardware.suggest_device()
        if target==self.device:
            return
        self.vision.to(target)
        self.left.to(target)
        self.right.to(target)
        self.neuro_module.to(target)
        self.option_planner.to(target)
        for opt in [self.vision_opt,self.left_opt,self.right_opt,self.neuro_opt,self.option_opt]:
            for st in opt.state.values():
                for k,v in list(st.items()):
                    if torch.is_tensor(v):
                        st[k]=v.to(target)
        self.neuro_state=self.neuro_state.to(target)
        self.device=target
        self.batch_buffer=None
        self.cpu_batch=None
        self.batch_capacity=0
        self.batch_shape=None
    def preprocess_frame(self,img):
        self.ensure_device()
        w,h=self.hardware.suggest_visual_size()
        img=img.resize((w,h))
        arr=np.array(img).astype(np.float32)/255.0
        arr=np.transpose(arr,(2,0,1))
        t=torch.tensor(arr,dtype=torch.float32).unsqueeze(0).to(self.device)
        return t
    def infer_state(self,img):
        self.ensure_device()
        with torch.no_grad():
            t=self.preprocess_frame(img)
            metrics,flags,h=self.vision(t)
            brain_output,self.neuro_state=self.neuro_module(h[0],self.neuro_state)
            self.state_history.append(brain_output.detach().cpu())
            metrics=metrics[0].cpu().numpy()
            flags=flags[0].cpu().numpy()
            hero_dead=bool(flags[0]>0.5)
            in_recall=bool(flags[1]>0.5)
            cooldowns={"recall":False,"heal":bool(flags[2]>0.5),"flash":bool(flags[3]>0.5),"basic":False,"skill1":bool(flags[4]>0.5),"skill2":bool(flags[5]>0.5),"skill3":bool(flags[6]>0.5),"skill4":bool(flags[7]>0.5),"active_item":bool(flags[8]>0.5),"cancel":False}
            return {"A":int(max(metrics[0],0)),"B":int(max(metrics[1],0)),"C":int(max(metrics[2],0))},hero_dead,in_recall,cooldowns,brain_output.detach().cpu()
    def _build_sequence_tensor(self):
        if len(self.state_history)==0:
            return torch.zeros(1,1,32,device=self.device)
        window=min(len(self.state_history),16)
        states=list(self.state_history)[-window:]
        seq=torch.stack([s.to(self.device) for s in states],dim=0)
        if seq.dim()==1:
            seq=seq.unsqueeze(0)
        seq=seq.unsqueeze(0)
        return seq
    def _contrastive_loss(self,features):
        if features.size(0)<2:
            return torch.tensor(0.0,device=self.device)
        clean=F.normalize(features,dim=-1)
        augmented=F.normalize(clean+torch.randn_like(clean)*0.01,dim=-1)
        logits=clean@augmented.t()/0.1
        labels=torch.arange(features.size(0),device=self.device)
        return F.cross_entropy(logits,labels)
    def _map_action_to_option(self,action,A,B,C,hero_dead):
        if action and action.get("hand")=="left":
            return 0
        label=(action.get("label") if action else None) or ""
        if label in ["普攻","一技能","二技能","三技能","四技能"]:
            return 1
        if label=="闪现":
            return 2
        if label in ["恢复","回城"] or hero_dead:
            return 3
        if label in ["主动装备","数据A","数据B","数据C"]:
            return 4
        if label=="取消施法":
            return 5
        if action is None:
            if A>max(B,C):
                return 1
            if C>A:
                return 4
            if hero_dead or B>A:
                return 3
        return 5
    def select_actions(self,h_state,left_hidden,right_hidden):
        self.ensure_device()
        seq=self._build_sequence_tensor()
        planner_logits,planner_term,option_embeddings,_=self.option_planner(h_state.unsqueeze(0))
        option_probs=F.softmax(planner_logits,dim=-1)
        terminate=planner_term[0,self.current_option].item()
        if random.random()<terminate:
            new_option=torch.multinomial(option_probs[0],1).item()
        else:
            new_option=self.current_option
        self.current_option=new_option
        option_vec=option_embeddings[new_option].unsqueeze(0).to(self.device)
        self.option_history.append(torch.tensor(float(new_option)))
        llogits,lval=self.left(seq,option_vec)
        rlogits,rval=self.right(seq,option_vec)
        lprob=F.softmax(llogits,dim=-1)
        rprob=F.softmax(rlogits,dim=-1)
        laction=torch.multinomial(lprob,1).item()
        raction=torch.multinomial(rprob,1).item()
        return laction,raction,lprob[0,laction],rprob[0,raction],lval,rval,None,None
    def neuro_project(self,h_batch):
        self.ensure_device()
        outputs=[]
        state=torch.zeros(h_batch.size(1),device=self.device,dtype=h_batch.dtype)
        for i in range(h_batch.size(0)):
            out,state=self.neuro_module(h_batch[i],state)
            outputs.append(out.unsqueeze(0))
        return torch.cat(outputs,dim=0)
    def reset_neuro_state(self):
        self.ensure_device()
        self.neuro_state=torch.zeros(32,device=self.device)
        self.state_history.clear()
        self.option_history.clear()
        self.current_option=0
    def _ensure_batch_buffers(self,count,shape):
        c=shape[0]
        h=shape[1]
        w=shape[2]
        target=(c,h,w)
        if self.batch_shape!=target:
            self.batch_shape=target
            self.batch_capacity=0
            self.batch_buffer=None
            self.cpu_batch=None
        capacity=self.batch_capacity
        if self.batch_buffer is None or self.cpu_batch is None or capacity<count:
            new_cap=max(count,capacity*2 if capacity>0 else count)
            if self.device.type=='cuda':
                cpu_tensor=torch.empty((new_cap,c,h,w),dtype=torch.float32).pin_memory()
            else:
                cpu_tensor=torch.empty((new_cap,c,h,w),dtype=torch.float32)
            gpu_tensor=torch.empty((new_cap,c,h,w),dtype=torch.float32,device=self.device)
            self.cpu_batch=cpu_tensor
            self.batch_buffer=gpu_tensor
            self.batch_capacity=new_cap
    def _batch_from_arrays(self,arr_list):
        count=len(arr_list)
        shape=arr_list[0].shape
        self._ensure_batch_buffers(count,shape)
        for idx,arr in enumerate(arr_list):
            self.cpu_batch[idx].copy_(torch.from_numpy(arr))
        view=self.batch_buffer[:count]
        view.copy_(self.cpu_batch[:count],non_blocking=(self.device.type=='cuda'))
        return view
    def optimize_from_buffer(self,buffer,progress_get_cancel,progress_set,markers_updater,markers_persist,max_iters=1000):
        cancelled=False
        effective_steps=0
        for it in range(max_iters):
            self.hardware.refresh()
            self.entropy_coef=float(clamp(0.002+0.02*self.hardware.mem_free+0.01*self.reward_model.temperature,0.002,0.05))
            if progress_get_cancel():
                cancelled=True
                break
            self.ensure_device()
            inner_count=max(1,self.hardware.suggest_parallel())
            any_step=False
            for inner in range(inner_count):
                batch=buffer.sample(self.hardware.suggest_batch_size())
                if not batch:
                    continue
                size=self.hardware.suggest_visual_size()
                arrays=buffer.get_frame_arrays(batch,size)
                targets_left=[]
                targets_right=[]
                rewards=[]
                mask_left=[]
                mask_right=[]
                rl_mask_left=[]
                rl_mask_right=[]
                option_targets=[]
                option_supervision=[]
                option_rl_flags=[]
                arr_list=[]
                for rec,arr in zip(batch,arrays):
                    if arr is None:
                        continue
                    arr_list.append(arr)
                    A=float(rec["metrics"]["A"])
                    B=float(rec["metrics"]["B"])
                    C=float(rec["metrics"]["C"])
                    rew=self.compute_reward(A,B,C,rec.get("source","user"),rec.get("hero_dead",False))
                    rewards.append(rew)
                    act=rec["action"]
                    left_target=0
                    right_target=0
                    left_super=0
                    right_super=0
                    left_rl=0
                    right_rl=0
                    if act and act.get("action_id") is not None:
                        hand=act.get("hand")
                        aid=int(act.get("action_id"))
                        if hand=="left":
                            left_target=aid
                            left_rl=1
                            if rec["source"]=="user":
                                left_super=1
                        elif hand=="right":
                            right_target=aid
                            right_rl=1
                            if rec["source"]=="user":
                                right_super=1
                        option_targets.append(self._map_action_to_option(act,A,B,C,rec.get("hero_dead",False)))
                        option_supervision.append(1 if rec["source"]=="user" else 0)
                        option_rl_flags.append(1)
                    else:
                        option_targets.append(self._map_action_to_option(None,A,B,C,rec.get("hero_dead",False)))
                        option_supervision.append(0)
                        option_rl_flags.append(0)
                    targets_left.append(left_target)
                    targets_right.append(right_target)
                    mask_left.append(left_super)
                    mask_right.append(right_super)
                    rl_mask_left.append(left_rl)
                    rl_mask_right.append(right_rl)
                if not arr_list:
                    continue
                any_step=True
                effective_steps+=1
                t_batch=self._batch_from_arrays(arr_list)
                metrics,flags,h=self.vision(t_batch)
                h_brain=self.neuro_project(h)
                sequence=torch.stack([h_brain,torch.tanh(h_brain*0.5),torch.sin(h_brain)],dim=1)
                planner_logits,planner_term,option_embeddings,planner_value=self.option_planner(h_brain)
                with torch.no_grad():
                    R=torch.tensor(rewards,dtype=torch.float32,device=self.device).unsqueeze(1)
                option_indices=torch.tensor(option_targets,dtype=torch.long,device=self.device)
                option_super=torch.tensor(option_supervision,dtype=torch.float32,device=self.device).unsqueeze(1)
                option_rl=torch.tensor(option_rl_flags,dtype=torch.float32,device=self.device).unsqueeze(1)
                chosen_embeddings=option_embeddings.index_select(0,option_indices)
                left_logits,left_val=self.left(sequence,chosen_embeddings)
                right_logits,right_val=self.right(sequence,chosen_embeddings)
                left_logprob=F.log_softmax(left_logits,dim=-1)
                right_logprob=F.log_softmax(right_logits,dim=-1)
                left_ent=(-left_logprob.exp()*left_logprob).sum(dim=-1).mean()
                right_ent=(-right_logprob.exp()*right_logprob).sum(dim=-1).mean()
                tl=torch.tensor([min(max(t,0),15) for t in targets_left],dtype=torch.long,device=self.device)
                tr=torch.tensor([min(max(t,0),31) for t in targets_right],dtype=torch.long,device=self.device)
                ml=torch.tensor(mask_left,dtype=torch.float32,device=self.device)
                mr=torch.tensor(mask_right,dtype=torch.float32,device=self.device)
                ll_sup=(F.nll_loss(left_logprob,tl,reduction="none")*ml).mean() if ml.sum()>0 else torch.tensor(0.0,device=self.device)
                rl_sup=(F.nll_loss(right_logprob,tr,reduction="none")*mr).mean() if mr.sum()>0 else torch.tensor(0.0,device=self.device)
                adv_left=(R-left_val).detach()
                adv_right=(R-right_val).detach()
                rl_ml=torch.tensor(rl_mask_left,dtype=torch.float32,device=self.device).unsqueeze(1)
                rl_mr=torch.tensor(rl_mask_right,dtype=torch.float32,device=self.device).unsqueeze(1)
                if rl_ml.sum()>0:
                    rl_pg_left=-((left_logprob.gather(1,tl.unsqueeze(1))*adv_left)*rl_ml).sum()/rl_ml.sum()
                else:
                    rl_pg_left=torch.tensor(0.0,device=self.device)
                if rl_mr.sum()>0:
                    rl_pg_right=-((right_logprob.gather(1,tr.unsqueeze(1))*adv_right)*rl_mr).sum()/rl_mr.sum()
                else:
                    rl_pg_right=torch.tensor(0.0,device=self.device)
                v_loss=((left_val-R)**2+(right_val-R)**2).mean()
                planner_logprob=F.log_softmax(planner_logits,dim=-1)
                planner_entropy=(-planner_logprob.exp()*planner_logprob).sum(dim=-1).mean()
                if option_super.sum().item()>0:
                    planner_sup=(F.nll_loss(planner_logprob,option_indices,reduction="none")*option_super.squeeze(1)).mean()
                else:
                    planner_sup=torch.tensor(0.0,device=self.device)
                if option_rl.sum().item()>0:
                    planner_pg=-((planner_logprob.gather(1,option_indices.unsqueeze(1))*(R-planner_value).detach())*option_rl).sum()/option_rl.sum()
                else:
                    planner_pg=torch.tensor(0.0,device=self.device)
                termination_penalty=planner_term.gather(1,option_indices.unsqueeze(1)).mean()
                contrastive=self._contrastive_loss(h_brain)
                l2_reg=0.0
                for p in list(self.vision.parameters())+list(self.left.parameters())+list(self.right.parameters())+list(self.neuro_module.parameters())+list(self.option_planner.parameters()):
                    l2_reg=l2_reg+p.pow(2).sum()*1e-6
                loss=ll_sup+rl_sup+rl_pg_left+rl_pg_right+self.value_coef*v_loss-self.entropy_coef*(left_ent+right_ent+planner_entropy)+planner_sup+planner_pg+termination_penalty*0.1+contrastive*0.5+l2_reg
                meta_snapshot=self.meta_controller.capture()
                self.vision_opt.zero_grad()
                self.left_opt.zero_grad()
                self.right_opt.zero_grad()
                self.neuro_opt.zero_grad()
                self.option_opt.zero_grad()
                loss.backward()
                for opt in [self.vision_opt,self.left_opt,self.right_opt,self.neuro_opt,self.option_opt]:
                    for g in opt.param_groups:
                        base_lr=g["lr"]
                        scale=1.0/(1.0+0.0001*self.global_step)
                        g["lr"]=base_lr*scale
                    opt.step()
                self.meta_controller.reptile_update(meta_snapshot)
                self.global_step+=1
                if effective_steps%10==0:
                    markers_updater()
                if markers_persist and effective_steps%50==0:
                    markers_persist()
            progress_set(int(100.0*(it+1)/max_iters))
            if not any_step:
                break
        if cancelled:
            progress_set(0)
        else:
            progress_set(100)
            self.file_manager.save_models(self.vision,self.left,self.right,self.neuro_module,self.option_planner)
        if markers_persist:
            markers_persist()
        return not cancelled
MarkerWidget=None
OverlayWindow=None
WindowSelectorDialog=None
MainWindow=None
def ensure_qt_classes():
    global QtCore,QtGui,QtWidgets,MarkerWidget,OverlayWindow,WindowSelectorDialog,MainWindow
    if isinstance(QtCore,LazyModule):
        QtCore=QtCore._load()
    if isinstance(QtGui,LazyModule):
        QtGui=QtGui._load()
    if isinstance(QtWidgets,LazyModule):
        QtWidgets=QtWidgets._load()
    if MarkerWidget is not None and OverlayWindow is not None and WindowSelectorDialog is not None and MainWindow is not None:
        return
    class _MarkerWidget(QtWidgets.QWidget):
        def __init__(self,parent,label,color,alpha,x_pct,y_pct,r_pct):
            super(_MarkerWidget,self).__init__(parent)
            self.label=label
            self.color=QtGui.QColor(color)
            self.alpha=alpha
            self.x_pct=x_pct
            self.y_pct=y_pct
            self.r_pct=r_pct
            self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents,False)
            self.dragging=False
            self.resizing=False
            self.selected=False
            self.last_pos=None
        def update_geometry_from_parent(self):
            pr=self.parent()
            if pr is None:
                return
            w=pr.width()
            h=pr.height()
            r=int(self.r_pct*min(w,h))
            cx=int(self.x_pct*w)
            cy=int(self.y_pct*h)
            self.setGeometry(cx-r,cy-r,2*r,2*r)
            self.update()
        def paintEvent(self,e):
            qp=QtGui.QPainter(self)
            qp.setRenderHint(QtGui.QPainter.Antialiasing)
            c=QtGui.QColor(self.color)
            a=int(clamp(self.alpha,0.0,1.0)*255)
            c.setAlpha(a)
            qp.setBrush(QtGui.QBrush(c))
            qp.setPen(QtGui.QPen(QtGui.QColor(255,255,255,255 if self.selected else a),2))
            r=min(self.width(),self.height())/2
            qp.drawEllipse(QtCore.QPointF(self.width()/2,self.height()/2),r-2,r-2)
            qp.setPen(QtGui.QPen(QtGui.QColor(255,255,255,255),1))
            qp.drawText(self.rect(),QtCore.Qt.AlignCenter,self.label)
        def mousePressEvent(self,event):
            pos=event.pos()
            r=min(self.width(),self.height())/2
            center=QtCore.QPointF(self.width()/2,self.height()/2)
            dist=((pos.x()-center.x())**2+(pos.y()-center.y())**2)**0.5
            edge=abs(dist-r)<8
            if edge:
                self.resizing=True
            else:
                self.dragging=True
            self.selected=True
            p=self.parent()
            if hasattr(p,"selected_marker"):
                p.selected_marker=self
                for other in p.markers:
                    if other is not self:
                        other.selected=False
                        other.update()
            self.last_pos=event.globalPos()
            self.update()
        def mouseMoveEvent(self,event):
            if not self.last_pos:
                return
            delta=event.globalPos()-self.last_pos
            pr=self.parent()
            w=pr.width()
            h=pr.height()
            if self.dragging:
                cx=self.x_pct*w+delta.x()
                cy=self.y_pct*h+delta.y()
                self.x_pct=clamp(cx/float(w),0.0,1.0)
                self.y_pct=clamp(cy/float(h),0.0,1.0)
            elif self.resizing:
                r=int(self.r_pct*min(w,h))+max(delta.x(),delta.y())
                self.r_pct=clamp(r/float(min(w,h)),0.01,0.5)
            self.last_pos=event.globalPos()
            self.update_geometry_from_parent()
        def mouseReleaseEvent(self,event):
            self.dragging=False
            self.resizing=False
            self.last_pos=None
    class _OverlayWindow(QtWidgets.QWidget):
        def __init__(self,app_state):
            super(_OverlayWindow,self).__init__(None,QtCore.Qt.WindowStaysOnTopHint|QtCore.Qt.FramelessWindowHint|QtCore.Qt.Tool)
            self.setAttribute(QtCore.Qt.WA_TranslucentBackground,True)
            self.setWindowFlag(QtCore.Qt.WindowTransparentForInput,True)
            self.app_state=app_state
            self.markers=[]
            self.selected_marker=None
            self.config_mode=False
            self._marker_visible_cache={}
            self._overlay_visible=False
            self.sync_timer=QtCore.QTimer(self)
            self.sync_timer.timeout.connect(self.sync_with_window)
            self.sync_timer.start(200)
        def find_marker(self,label):
            for m in self.markers:
                if m.label==label:
                    return m
            return None
        def sizeHint(self):
            return QtCore.QSize(400,400)
        def set_overlay_visible(self,visible):
            self._overlay_visible=visible
            self.sync_with_window()
            self._update_marker_visibility(True)
            if visible:
                if not self.isVisible():
                    self.show()
            else:
                if self.isVisible():
                    self.hide()
        def _update_marker_visibility(self,force=False):
            for m in self.markers:
                desired=self.config_mode and self._overlay_visible
                if force or self._marker_visible_cache.get(m)!=desired:
                    if desired:
                        if not m.isVisible():
                            m.show()
                    else:
                        if m.isVisible():
                            m.hide()
                    self._marker_visible_cache[m]=desired
                m.update_geometry_from_parent()
        def set_config_mode(self,enabled):
            self.config_mode=enabled
            if enabled:
                self.setWindowFlag(QtCore.Qt.WindowTransparentForInput,False)
                self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents,True)
                for m in self.markers:
                    m.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents,False)
            else:
                self.setWindowFlag(QtCore.Qt.WindowTransparentForInput,True)
                self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents,True)
                for m in self.markers:
                    m.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents,True)
            self._update_marker_visibility(True)
            self.sync_with_window()
        def sync_with_window(self):
            with self.app_state.lock:
                hwnd=self.app_state.hwnd
                rect=self.app_state.window_rect
            if hwnd is None:
                if self.isVisible():
                    self.hide()
                return
            try:
                rect=get_window_rect(hwnd)
                with self.app_state.lock:
                    self.app_state.window_rect=rect
            except Exception as e:
                logger.debug("同步窗口失败:%s",e)
            x1,y1,x2,y2=self.app_state.window_rect
            if x2<=x1 or y2<=y1:
                if self.isVisible():
                    self.hide()
                return
            self.setGeometry(x1,y1,x2-x1,y2-y1)
            for m in self.markers:
                m.update_geometry_from_parent()
        def load_from_config(self,records,spec):
            for m in list(self.markers):
                m.setParent(None)
            self.markers=[]
            self.selected_marker=None
            self._marker_visible_cache={}
            seen=set()
            for rec in records:
                label=rec.get("label")
                if not label:
                    continue
                cfg=spec.get(label,spec.get("普攻"))
                base_color=cfg.get("color",(255,0,0))
                rec_color=rec.get("color")
                if rec_color:
                    color=QtGui.QColor(rec_color)
                else:
                    color=QtGui.QColor(base_color[0],base_color[1],base_color[2])
                alpha_val=rec.get("alpha")
                alpha=float(alpha_val) if alpha_val is not None else float(cfg.get("alpha",0.5))
                pos=cfg.get("pos",(0.5,0.5))
                rx=rec.get("x_pct")
                ry=rec.get("y_pct")
                rr=rec.get("r_pct")
                x_pct=clamp(float(rx) if rx is not None else float(pos[0]),0.0,1.0)
                y_pct=clamp(float(ry) if ry is not None else float(pos[1]),0.0,1.0)
                r_pct=clamp(float(rr) if rr is not None else float(cfg.get("radius",0.05)),0.01,0.5)
                marker=_MarkerWidget(self,label,color,alpha,x_pct,y_pct,r_pct)
                marker.update_geometry_from_parent()
                self.markers.append(marker)
                seen.add(label)
            self.ensure_required_markers(spec,seen)
            self._update_marker_visibility(True)
        def ensure_required_markers(self,spec,seen=None):
            existing=seen if seen is not None else set(m.label for m in self.markers)
            for label,cfg in spec.items():
                if not cfg.get("required"):
                    continue
                if label in existing:
                    continue
                color_tuple=cfg.get("color",(255,0,0))
                marker=self.add_marker(label,QtGui.QColor(color_tuple[0],color_tuple[1],color_tuple[2]),cfg.get("alpha",0.5),cfg.get("pos",(0.5,0.5))[0],cfg.get("pos",(0.5,0.5))[1],cfg.get("radius",0.05))
                if marker:
                    existing.add(label)
            self._update_marker_visibility(True)
        def add_marker(self,label,color=None,alpha=None,x_pct=None,y_pct=None,r_pct=None):
            if not label:
                return None
            cfg=MARKER_SPEC.get(label,MARKER_SPEC.get("普攻"))
            base_color=cfg.get("color",(255,0,0))
            target_color=color if color is not None else QtGui.QColor(base_color[0],base_color[1],base_color[2])
            target_alpha=alpha if alpha is not None else float(cfg.get("alpha",0.5))
            pos=cfg.get("pos",(0.5,0.5))
            xp=x_pct if x_pct is not None else 0.5
            yp=y_pct if y_pct is not None else 0.5
            rp=r_pct if r_pct is not None else float(cfg.get("radius",0.05))
            m=_MarkerWidget(self,label,target_color,target_alpha,clamp(float(xp),0.0,1.0),clamp(float(yp),0.0,1.0),clamp(float(rp),0.01,0.5))
            self.markers.append(m)
            m.update_geometry_from_parent()
            desired=self.config_mode and self._overlay_visible
            if desired:
                m.show()
            else:
                m.hide()
            return m
        def remove_selected_marker(self):
            if self.selected_marker in self.markers:
                target=self.selected_marker
                self.markers.remove(target)
                if target in self._marker_visible_cache:
                    self._marker_visible_cache.pop(target,None)
                target.setParent(None)
                self.selected_marker=None
        def get_marker_data(self):
            out=[]
            for m in self.markers:
                out.append(m)
            return out
    class _WindowSelectorDialog(QtWidgets.QDialog):
        def __init__(self,parent=None):
            super(_WindowSelectorDialog,self).__init__(parent)
            self.setWindowTitle("选择窗口")
            self.layout=QtWidgets.QVBoxLayout(self)
            self.listWidget=QtWidgets.QListWidget(self)
            self.layout.addWidget(self.listWidget)
            self.btnBox=QtWidgets.QDialogButtonBox(QtWidgets.QDialogButtonBox.Ok|QtWidgets.QDialogButtonBox.Cancel,self)
            self.layout.addWidget(self.btnBox)
            self.btnBox.accepted.connect(self.accept)
            self.btnBox.rejected.connect(self.reject)
            self.refresh_windows()
        def refresh_windows(self):
            self.listWidget.clear()
            self.windows=[]
            for hwnd,title,rect in enumerate_windows():
                item=QtWidgets.QListWidgetItem(title)
                item.setData(QtCore.Qt.UserRole,(hwnd,rect))
                self.listWidget.addItem(item)
        def selected(self):
            item=self.listWidget.currentItem()
            if not item:
                return None
            return item.data(QtCore.Qt.UserRole)
    class _MainWindow(QtWidgets.QMainWindow):
        def __init__(self,app_state,hardware):
            super(_MainWindow,self).__init__()
            self.app_state=app_state
            self.hardware=hardware
            self.setWindowTitle("AAA AI Game Player")
            self.rootWidget=QtWidgets.QWidget(self)
            self.setCentralWidget(self.rootWidget)
            self.layout=QtWidgets.QGridLayout(self.rootWidget)
            self.modeLabel=QtWidgets.QLabel("模式:初始化")
            self.metricsLabelA=QtWidgets.QLabel("A:0")
            self.metricsLabelB=QtWidgets.QLabel("B:0")
            self.metricsLabelC=QtWidgets.QLabel("C:0")
            self.heroDeadLabel=QtWidgets.QLabel("英雄存活:是")
            self.cooldownLabel=QtWidgets.QLabel("冷却信息")
            self.layout.addWidget(self.modeLabel,0,0,1,2)
            self.layout.addWidget(self.metricsLabelA,1,0)
            self.layout.addWidget(self.metricsLabelB,1,1)
            self.layout.addWidget(self.metricsLabelC,1,2)
            self.layout.addWidget(self.heroDeadLabel,2,0)
            self.layout.addWidget(self.cooldownLabel,2,1,1,2)
            self.progressBar=QtWidgets.QProgressBar()
            self.layout.addWidget(self.progressBar,3,0,1,3)
            self.chooseWindowBtn=QtWidgets.QPushButton("选择窗口")
            self.optimizeBtn=QtWidgets.QPushButton("优化")
            self.cancelOptimizeBtn=QtWidgets.QPushButton("取消优化")
            self.configBtn=QtWidgets.QPushButton("配置")
            self.saveConfigBtn=QtWidgets.QPushButton("保存配置")
            self.addMarkerBtn=QtWidgets.QPushButton("添加标志")
            self.delMarkerBtn=QtWidgets.QPushButton("删除标志")
            self.moveAAABtn=QtWidgets.QPushButton("修改AAA位置")
            self.layout.addWidget(self.chooseWindowBtn,4,0)
            self.layout.addWidget(self.optimizeBtn,4,1)
            self.layout.addWidget(self.cancelOptimizeBtn,4,2)
            self.layout.addWidget(self.configBtn,5,0)
            self.layout.addWidget(self.saveConfigBtn,5,1)
            self.layout.addWidget(self.addMarkerBtn,6,0)
            self.layout.addWidget(self.delMarkerBtn,6,1)
            self.layout.addWidget(self.moveAAABtn,6,2)
            self.cancelOptimizeBtn.setEnabled(False)
            self.saveConfigBtn.setEnabled(False)
            self.addMarkerBtn.setEnabled(False)
            self.delMarkerBtn.setEnabled(False)
            self.overlay=None
            self.chooseWindowBtn.clicked.connect(self.on_choose_window)
            self.optimizeBtn.clicked.connect(self.on_optimize)
            self.cancelOptimizeBtn.clicked.connect(self.on_cancel_optimize)
            self.configBtn.clicked.connect(self.on_configure)
            self.saveConfigBtn.clicked.connect(self.on_save_config)
            self.addMarkerBtn.clicked.connect(self.on_add_marker)
            self.delMarkerBtn.clicked.connect(self.on_delete_marker)
            self.moveAAABtn.clicked.connect(self.on_move_aaa)
            self.timer=QtCore.QTimer(self)
            self.timer.timeout.connect(self.on_tick)
            self.timer.start(200)
        def on_tick(self):
            if self.app_state.consume_window_prompt():
                QtWidgets.QMessageBox.information(self,"初始化完成","依赖与资源已就绪，请选择窗口")
            err=self.app_state.consume_init_error()
            if err:
                QtWidgets.QMessageBox.critical(self,"初始化失败",err)
            opt_status=self.app_state.consume_optimization_prompt()
            if opt_status:
                if opt_status=="completed":
                    QtWidgets.QMessageBox.information(self,"优化完成","经验优化已完成，已保存最新模型")
                else:
                    QtWidgets.QMessageBox.warning(self,"优化已取消","优化过程已取消，进度已重置")
                self.app_state.enter_learning()
                if opt_status!="completed":
                    self.app_state.set_progress(0)
            snap=self.app_state.get_state_snapshot()
            mode_map={Mode.INIT:"初始化",Mode.LEARNING:"学习",Mode.OPTIMIZING:"优化",Mode.CONFIGURING:"配置",Mode.TRAINING:"训练"}
            self.modeLabel.setText("模式:"+mode_map.get(snap["mode"],"未知"))
            self.metricsLabelA.setText("A:"+str(snap["metrics"].get("A",0)))
            self.metricsLabelB.setText("B:"+str(snap["metrics"].get("B",0)))
            self.metricsLabelC.setText("C:"+str(snap["metrics"].get("C",0)))
            self.heroDeadLabel.setText("英雄存活:"+("否" if snap["hero_dead"] else "是"))
            cd=snap["cooldowns"]
            cd_text="冷却:"+",".join(["%s:%s"%(k,"是" if v else "否") for k,v in cd.items()])
            self.cooldownLabel.setText(cd_text)
            self.progressBar.setValue(int(snap["progress"]))
            running_opt=self.app_state.optimization_running()
            rect=snap["window_rect"]
            has_window=(rect[2]-rect[0])>0 and (rect[3]-rect[1])>0
            self.chooseWindowBtn.setEnabled(snap["ready"])
            self.optimizeBtn.setEnabled(snap["mode"]==Mode.LEARNING and snap["ready"] and has_window and not running_opt)
            self.cancelOptimizeBtn.setEnabled(running_opt)
            self.configBtn.setEnabled(snap["mode"]==Mode.LEARNING and has_window and not running_opt)
            if self.app_state.mode==Mode.CONFIGURING:
                self.saveConfigBtn.setEnabled(True)
                self.addMarkerBtn.setEnabled(True)
                self.delMarkerBtn.setEnabled(True)
            else:
                self.saveConfigBtn.setEnabled(False)
                self.addMarkerBtn.setEnabled(False)
                self.delMarkerBtn.setEnabled(False)
            if self.app_state.overlay:
                self.app_state.overlay.sync_with_window()
        def on_choose_window(self):
            ensure_qt_classes()
            dlg=_WindowSelectorDialog(self)
            app=self.app_state
            if not app.ready:
                QtWidgets.QMessageBox.warning(self,"未就绪","初始化尚未完成")
                return
            if dlg.exec_()==QtWidgets.QDialog.Accepted:
                selected=dlg.selected()
                if selected:
                    hwnd,rect=selected
                    app.on_window_selected(hwnd,rect)
                    app.mark_user_input()
                    QtWidgets.QMessageBox.information(self,"学习模式","窗口已选择，已进入学习模式")
        def on_optimize(self):
            with self.app_state.lock:
                ready=self.app_state.ready
                mode=self.app_state.mode
                hwnd=self.app_state.hwnd
            if not ready or hwnd is None:
                QtWidgets.QMessageBox.warning(self,"无法优化","请先完成初始化并选择窗口")
                return
            if mode!=Mode.LEARNING:
                QtWidgets.QMessageBox.information(self,"切换模式","请先返回学习模式后再开始优化")
                return
            if self.app_state.optimization_running():
                QtWidgets.QMessageBox.information(self,"优化进行中","优化线程正在运行")
                return
            self.app_state.set_mode(Mode.OPTIMIZING)
            thread=OptimizationThread(self.app_state)
            self.app_state.register_optimization_thread(thread)
            thread.start()
            QtWidgets.QMessageBox.information(self,"开始优化","已开始基于经验池的优化，可随时取消")
        def on_cancel_optimize(self):
            if self.app_state.optimization_running():
                self.app_state.request_cancel_optimization()
        def on_configure(self):
            if self.app_state.mode!=Mode.LEARNING:
                QtWidgets.QMessageBox.information(self,"切换模式","请在学习模式下进行配置")
                return
            self.app_state.set_mode(Mode.CONFIGURING)
            overlay=self.app_state.ensure_overlay_initialized()
            overlay.set_overlay_visible(True)
            overlay.set_config_mode(True)
        def on_save_config(self):
            if self.app_state.mode!=Mode.CONFIGURING:
                return
            reply=QtWidgets.QMessageBox.question(self,"确认保存","确认保存当前配置？",QtWidgets.QMessageBox.Yes|QtWidgets.QMessageBox.No)
            if reply!=QtWidgets.QMessageBox.Yes:
                return
            self.save_markers_to_manager()
            QtWidgets.QMessageBox.information(self,"已保存","配置已保存")
            self.app_state.set_mode(Mode.LEARNING)
            if self.app_state.overlay:
                self.app_state.overlay.set_config_mode(False)
                self.app_state.overlay.set_overlay_visible(False)
        def save_markers_to_manager(self):
            overlay=self.app_state.overlay
            if not overlay:
                return
            rect=self.app_state.window_rect
            markers=[]
            for m in overlay.markers:
                markers.append(m)
            self.app_state.file_manager.save_markers(markers,rect)
        def on_add_marker(self):
            overlay=self.app_state.ensure_overlay_initialized()
            available=[label for label in MARKER_SPEC.keys() if overlay.find_marker(label) is None]
            if not available:
                QtWidgets.QMessageBox.information(self,"无需添加","所有标志均已存在")
                return
            label,ok=QtWidgets.QInputDialog.getItem(self,"选择标志","标志",available,0,False)
            if ok and label:
                marker=overlay.add_marker(label)
                overlay.selected_marker=marker
                if marker:
                    marker.selected=True
                    marker.update()
        def on_delete_marker(self):
            overlay=self.app_state.overlay
            if overlay:
                target=overlay.selected_marker
                if not target:
                    return
                if MARKER_SPEC.get(target.label,{}).get("required"):
                    QtWidgets.QMessageBox.warning(self,"不可删除","该标志为必需项，无法删除")
                    return
                overlay.remove_selected_marker()
        def on_move_aaa(self):
            d=QtWidgets.QFileDialog.getExistingDirectory(self,"选择新位置")
            if d:
                old_base,new_base=self.app_state.file_manager.move_dir(d)
                self.app_state.buffer.on_aaa_moved(old_base,new_base)
                self.app_state.ensure_overlay_initialized()
                QtWidgets.QMessageBox.information(self,"完成","AAA目录已迁移")
    MarkerWidget=_MarkerWidget
    OverlayWindow=_OverlayWindow
    WindowSelectorDialog=_WindowSelectorDialog
    MainWindow=_MainWindow
class HardwareAdaptiveRate:
    def __init__(self):
        self.batch_size=32
        self.parallel=1
        self.hand_interval=0.05
        self.device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.cpu_load=0.0
        self.mem_free=0.5
        self.gpu_free=0.5
        self.last_refresh=0.0
        self.visual_level=1
        self.lock=threading.Lock()
        self.snapshot_lock=threading.Lock()
        self.snapshot={"cpu":0.0,"mem":0.5,"gpu":0.5,"time":0.0}
        self.monitor_stop=False
        self.monitor_interval=0.75
        self._ema_cpu=None
        self._ema_mem=None
        self._ema_gpu=None
        self._ema_alpha=0.2
        self.cpu_target=0.85
        self.mem_target=0.15
        self.gpu_target=0.2
        self.adaptive_gain=0.12
        self.max_batch=160
        self.min_batch=8
        self.max_parallel=8
        self.min_interval=0.02
        self.max_interval=0.15
        self.monitor_thread=threading.Thread(target=self._monitor_loop,daemon=True)
        self.monitor_thread.start()
    def _collect_snapshot(self):
        try:
            cpu=psutil.cpu_percent(interval=None)
        except Exception as e:
            logger.error("获取CPU负载失败:%s",e)
            cpu=100.0
        try:
            vm=psutil.virtual_memory()
            mem_free=vm.available/float(vm.total if vm.total>0 else 1)
        except Exception as e:
            logger.error("获取内存信息失败:%s",e)
            mem_free=0.0
        gpu_free=0.5
        if torch.cuda.is_available():
            try:
                props=torch.cuda.get_device_properties(0)
                total=float(props.total_memory)
                allocated=float(torch.cuda.memory_allocated(0))
                reserved=float(torch.cuda.memory_reserved(0))
                used=max(allocated,reserved)
                gpu_free=max(0.0,min(1.0,(total-used)/total if total>0 else 0.5))
            except Exception as e:
                logger.warning("获取GPU信息失败:%s",e)
                gpu_free=0.5
        return {"cpu":cpu,"mem":mem_free,"gpu":gpu_free,"time":time.time()}
    def _apply_snapshot(self,snap):
        if self._ema_cpu is None:
            self._ema_cpu=snap["cpu"]
            self._ema_mem=snap["mem"]
            self._ema_gpu=snap["gpu"]
        else:
            alpha=self._ema_alpha
            self._ema_cpu=self._ema_cpu+(snap["cpu"]-self._ema_cpu)*alpha
            self._ema_mem=self._ema_mem+(snap["mem"]-self._ema_mem)*alpha
            self._ema_gpu=self._ema_gpu+(snap["gpu"]-self._ema_gpu)*alpha
        with self.snapshot_lock:
            self.snapshot["cpu"]=self._ema_cpu
            self.snapshot["mem"]=self._ema_mem
            self.snapshot["gpu"]=self._ema_gpu
            self.snapshot["time"]=snap["time"]
    def _monitor_loop(self):
        while not self.monitor_stop:
            snap=self._collect_snapshot()
            self._apply_snapshot(snap)
            time.sleep(self.monitor_interval)
    def refresh(self,force=False):
        now=time.time()
        with self.lock:
            if force:
                snap=self._collect_snapshot()
                self._apply_snapshot(snap)
            elif now-self.last_refresh<self.monitor_interval*0.5:
                return
            with self.snapshot_lock:
                cpu=self.snapshot["cpu"]
                mem=self.snapshot["mem"]
                gpu=self.snapshot["gpu"]
            self.cpu_load=cpu
            self.mem_free=mem
            self.gpu_free=gpu
            cpu_ratio=clamp(cpu/100.0,0.0,1.0)
            mem_ratio=clamp(mem,0.0,1.0)
            gpu_ratio=clamp(gpu,0.0,1.0)
            cpu_error=self.cpu_target-cpu_ratio
            mem_error=mem_ratio-self.mem_target
            gpu_error=gpu_ratio-self.gpu_target
            adaptive=cpu_error*0.6+mem_error*0.2+gpu_error*0.2
            batch_float=self.batch_size*(1.0+adaptive*self.adaptive_gain*10.0)
            if cpu_ratio>0.95 or mem_ratio<0.05:
                batch_float=self.batch_size*0.7
            self.batch_size=int(clamp(batch_float,self.min_batch,self.max_batch))
            parallel_float=self.parallel+adaptive*2.0
            if cpu_ratio>0.9:
                parallel_float=min(parallel_float,self.parallel)
            self.parallel=int(clamp(round(parallel_float),1,self.max_parallel))
            interval=self.hand_interval-adaptive*0.01
            if cpu_ratio>0.9 or gpu_ratio<0.05:
                interval+=0.01
            self.hand_interval=float(clamp(interval,self.min_interval,self.max_interval))
            if torch.cuda.is_available():
                if self.gpu_free>0.3 and cpu_ratio>0.7:
                    target=torch.device("cuda")
                elif self.gpu_free<0.08:
                    target=torch.device("cpu")
                else:
                    target=self.device
            else:
                target=torch.device("cpu")
            self.device=target
            resource_score=0.45*(1.0-cpu_ratio)+0.25*mem_ratio+0.3*gpu_ratio
            dynamic_level=int(clamp(math.floor(resource_score*4.0),0,3))
            self.visual_level=dynamic_level
            self.last_refresh=now
    def get_hz(self):
        with self.lock:
            score=(1.0-self.cpu_load/100.0)*0.4+self.mem_free*0.2+self.gpu_free*0.4
            score=clamp(score,0.0,1.0)
            hz=int(max(1,min(120,int(1.0+score*(120.0-1.0)))))
            return hz
    def suggest_batch_size(self):
        with self.lock:
            return self.batch_size
    def suggest_parallel(self):
        with self.lock:
            return self.parallel
    def suggest_hand_interval(self):
        with self.lock:
            return self.hand_interval
    def suggest_device(self):
        with self.lock:
            return self.device
    def suggest_visual_level(self):
        with self.lock:
            return self.visual_level
    def suggest_visual_size(self):
        level=self.suggest_visual_level()
        if level>=3:
            return 336,336
        if level==2:
            return 304,304
        if level==1:
            return 272,272
        return 240,240
def enum_windows():
    result=[]
    def callback(hwnd,extra):
        if win32gui.IsWindowVisible(hwnd):
            title=win32gui.GetWindowText(hwnd)
            if title and len(title.strip())>0:
                rect=win32gui.GetWindowRect(hwnd)
                w=rect[2]-rect[0]
                h=rect[3]-rect[1]
                if w>50 and h>50:
                    result.append((hwnd,title))
    win32gui.EnumWindows(callback,0)
    return result
def get_window_rect(hwnd):
    rect=win32gui.GetWindowRect(hwnd)
    return rect
def window_visible(hwnd):
    if hwnd is None:
        return False
    try:
        if not win32gui.IsWindow(hwnd):
            return False
    except Exception as e:
        logger.warning("窗口有效性检查失败:%s",e)
        return False
    if not win32gui.IsWindowVisible(hwnd):
        return False
    if win32gui.IsIconic(hwnd):
        return False
    try:
        rect=win32gui.GetWindowRect(hwnd)
    except Exception as e:
        logger.warning("窗口矩形获取失败:%s",e)
        return False
    if rect[2]<=rect[0] or rect[3]<=rect[1]:
        return False
    width=rect[2]-rect[0]
    height=rect[3]-rect[1]
    target_root=win32gui.GetAncestor(hwnd,win32con.GA_ROOT)
    samples=[(0.5,0.5),(0.2,0.2),(0.8,0.2),(0.2,0.8),(0.8,0.8)]
    for sx,sy in samples:
        px=int(rect[0]+width*sx)
        py=int(rect[1]+height*sy)
        try:
            top=win32gui.WindowFromPoint((px,py))
            top_root=win32gui.GetAncestor(top,win32con.GA_ROOT) if top else None
        except Exception as e:
            logger.warning("窗口点检测失败:%s",e)
            top_root=None
        if top_root not in [target_root,hwnd]:
            try:
                ex=win32gui.GetWindowLong(top_root,win32con.GWL_EXSTYLE)
            except Exception as e:
                logger.warning("窗口样式获取失败:%s",e)
                ex=0
            if (ex&win32con.WS_EX_TRANSPARENT)==0 or (ex&win32con.WS_EX_LAYERED)==0:
                return False
    try:
        cloaked=ctypes.c_int()
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd,14,ctypes.byref(cloaked),ctypes.sizeof(cloaked))==0 and cloaked.value!=0:
            return False
    except Exception as e:
        logger.info("窗口遮挡状态获取失败:%s",e)
    try:
        visible_rect=wintypes.RECT()
        if ctypes.windll.dwmapi.DwmGetWindowAttribute(hwnd,9,ctypes.byref(visible_rect),ctypes.sizeof(visible_rect))==0:
            if max(0,visible_rect.right-visible_rect.left)==0 or max(0,visible_rect.bottom-visible_rect.top)==0:
                return False
    except Exception as e:
        logger.info("窗口可见区域获取失败:%s",e)
    return True
class AppState:
    def __init__(self,file_manager,agent,buffer,hardware_manager):
        self.lock=threading.Lock()
        self.mode=Mode.INIT
        self.ready=False
        self.hwnd=None
        self.window_rect=(0,0,0,0)
        self.recording=False
        self.hero_dead=False
        self.in_recall=False
        self.metrics={"A":0,"B":0,"C":0}
        self.cooldowns={"recall":False,"heal":False,"flash":False,"basic":False,"skill1":False,"skill2":False,"skill3":False,"skill4":False,"active_item":False,"cancel":False}
        self.last_user_input=time.time()
        self.progress=0
        self.file_manager=file_manager
        self.agent=agent
        self.buffer=buffer
        self.hardware=hardware_manager
        self.overlay=None
        self.left_thread=None
        self.right_thread=None
        self.training_hidden_states={"left":None,"right":None}
        self.cancel_optimization=False
        self.idle_threshold=10.0
        self.paused_by_visibility=False
        self.current_hidden=torch.zeros(1,32)
        self.ai_action_queue=deque()
        self.marker_spec=MARKER_SPEC
        self.pending_window_prompt=False
        self.pending_init_error=None
        self.init_status={}
        self.optimization_thread=None
        self.pending_opt_prompt=False
        self.optimization_status=None
        self.init_thread=None
        self.mode_manager_thread=None
    def set_hwnd(self,hwnd):
        with self.lock:
            self.hwnd=hwnd
            self.window_rect=get_window_rect(hwnd)
            self.agent.reset_neuro_state()
    def update_window_rect(self):
        with self.lock:
            if self.hwnd is not None:
                self.window_rect=get_window_rect(self.hwnd)
    def set_mode(self,m):
        with self.lock:
            self.mode=m
    def can_record(self):
        with self.lock:
            return self.mode in [Mode.LEARNING,Mode.TRAINING] and self.hwnd is not None and window_visible(self.hwnd) and not self.paused_by_visibility
    def mark_user_input(self):
        with self.lock:
            self.last_user_input=time.time()
    def set_visibility_paused(self,paused):
        with self.lock:
            if self.paused_by_visibility!=paused:
                self.paused_by_visibility=paused
                self.last_user_input=time.time()
    def complete_initialization(self,success,status,message=None):
        with self.lock:
            self.ready=success
            self.init_status=status
            if success:
                self.pending_window_prompt=True
            elif message:
                self.pending_init_error=message
    def consume_window_prompt(self):
        with self.lock:
            if self.pending_window_prompt:
                self.pending_window_prompt=False
                return True
            return False
    def consume_init_error(self):
        with self.lock:
            msg=self.pending_init_error
            self.pending_init_error=None
            return msg
    def register_optimization_thread(self,thread):
        with self.lock:
            self.optimization_thread=thread
            self.cancel_optimization=False
            self.progress=0
    def optimization_running(self):
        with self.lock:
            thread=self.optimization_thread
        return thread is not None and thread.is_alive()
    def finish_optimization(self,status):
        with self.lock:
            self.optimization_status=status
            self.pending_opt_prompt=True
            self.optimization_thread=None
    def consume_optimization_prompt(self):
        with self.lock:
            if self.pending_opt_prompt:
                self.pending_opt_prompt=False
                status=self.optimization_status
                self.optimization_status=None
                return status
            return None
    def should_switch_to_training(self):
        with self.lock:
            return self.mode==Mode.LEARNING and (time.time()-self.last_user_input)>=self.idle_threshold and self.can_record()
    def must_back_to_learning(self):
        with self.lock:
            user_intervened=(time.time()-self.last_user_input)<0.2
            return self.mode==Mode.TRAINING and ((not self.can_record()) or user_intervened)
    def stop_training_threads(self):
        with self.lock:
            left=self.left_thread
            right=self.right_thread
            self.left_thread=None
            self.right_thread=None
        for t in [left,right]:
            if t:
                t.stop_flag=True
        for t in [left,right]:
            if t:
                t.join(timeout=1.0)
        with self.lock:
            self.training_hidden_states={"left":None,"right":None}
    def enter_learning(self):
        self.stop_training_threads()
        with self.lock:
            self.mode=Mode.LEARNING
            self.last_user_input=time.time()
    def enter_training(self):
        with self.lock:
            if self.mode==Mode.TRAINING:
                return
            self.mode=Mode.TRAINING
            self.last_user_input=time.time()
            self.training_hidden_states={"left":None,"right":None}
            need_left=self.left_thread is None or not self.left_thread.is_alive()
            need_right=self.right_thread is None or not self.right_thread.is_alive()
        if need_left:
            left=LeftHandThread(self)
            left.start()
            with self.lock:
                self.left_thread=left
        if need_right:
            right=RightHandThread(self)
            right.start()
            with self.lock:
                self.right_thread=right
    def update_state_from_frame(self,img):
        metrics,hero_dead,in_recall,cooldowns,hid=self.agent.infer_state(img)
        with self.lock:
            self.metrics=metrics
            self.hero_dead=hero_dead
            self.in_recall=in_recall
            self.cooldowns=cooldowns
            self.current_hidden=hid.clone().detach()
    def get_state_snapshot(self):
        with self.lock:
            return {"mode":self.mode,"metrics":self.metrics,"hero_dead":self.hero_dead,"in_recall":self.in_recall,"cooldowns":self.cooldowns,"progress":self.progress,"window_rect":self.window_rect,"ready":self.ready,"paused":self.paused_by_visibility}
    def set_progress(self,v):
        with self.lock:
            self.progress=v
    def request_cancel_optimization(self):
        with self.lock:
            self.cancel_optimization=True
    def consume_cancel_request(self):
        with self.lock:
            return self.cancel_optimization
    def clear_cancel_request(self):
        with self.lock:
            self.cancel_optimization=False
    def record_ai_action(self,action):
        with self.lock:
            if len(self.ai_action_queue)>64:
                self.ai_action_queue.popleft()
            self.ai_action_queue.append(action)
    def consume_ai_action(self):
        with self.lock:
            if self.ai_action_queue:
                return self.ai_action_queue.popleft()
            return None
    def get_marker_geometry(self,label):
        with self.lock:
            rect=self.window_rect
            overlay=self.overlay
        if overlay:
            marker=overlay.find_marker(label)
            if marker:
                cx=rect[0]+marker.x_pct*(rect[2]-rect[0])
                cy=rect[1]+marker.y_pct*(rect[3]-rect[1])
                r=marker.r_pct*min(rect[2]-rect[0],rect[3]-rect[1])
                return cx,cy,r
        cfg=self.marker_spec.get(label,self.marker_spec.get("普攻"))
        pos=cfg.get("pos",(0.5,0.5))
        radius=cfg.get("radius",0.05)
        cx=rect[0]+(rect[2]-rect[0])*pos[0]
        cy=rect[1]+(rect[3]-rect[1])*pos[1]
        r=min(rect[2]-rect[0],rect[3]-rect[1])*radius
        return cx,cy,r
    def ensure_overlay_initialized(self):
        ensure_qt_classes()
        with self.lock:
            overlay=self.overlay
        if overlay is None:
            overlay=_OverlayWindow(self)
            with self.lock:
                self.overlay=overlay
        markers=self.file_manager.load_markers()
        self.overlay.load_from_config(markers,self.marker_spec)
        self.overlay.ensure_required_markers(self.marker_spec)
        self.overlay.sync_with_window()
        return self.overlay
    def on_window_selected(self,hwnd,rect):
        with self.lock:
            self.hwnd=hwnd
            self.window_rect=rect
            self.agent.reset_neuro_state()
            self.last_user_input=time.time()
            self.mode=Mode.LEARNING
        self.set_visibility_paused(not window_visible(hwnd))
        overlay=self.ensure_overlay_initialized()
        overlay.set_overlay_visible(False)
        self.buffer.refresh_paths()
class InputTracker:
    def __init__(self,app_state):
        self.app_state=app_state
        self.action_queue=deque()
        self.lock=threading.Lock()
        self.left_labels=["移动轮盘"]
        self.right_mapping=dict(RIGHT_ACTION_MAPPING)
        self.button_paths={mouse.Button.left:[],mouse.Button.right:[],mouse.Button.middle:[]}
        self.button_state=set()
        self.listener_mouse=mouse.Listener(on_click=self.on_click,on_move=self.on_move,on_scroll=self.on_scroll)
        self.listener_keyboard=keyboard.Listener(on_press=self.on_key_press)
        self.listener_mouse.start()
        self.listener_keyboard.start()
    def summarize_path(self,path,max_points=6):
        if not path:
            return []
        pts=[]
        n=len(path)
        step=max(1,n//max(1,max_points-1))
        for idx in range(0,n,step):
            pts.append((path[idx][0],path[idx][1]))
        if pts[-1]!=(path[-1][0],path[-1][1]):
            pts.append((path[-1][0],path[-1][1]))
        return pts
    def quantize_angle(self,angle,count):
        if count<=1:
            return 0
        return int((angle%360.0)/(360.0/count))%count
    def encode_action(self,label,hand,action_type,path,start,end):
        summary=self.summarize_path(path)
        sx,sy=start[0],start[1]
        ex,ey=end[0],end[1]
        st=start[2] if len(start)>2 else end[2] if len(end)>2 else time.time()
        et=end[2] if len(end)>2 else st
        duration=max(0.0,et-st)
        angle=None
        if hand=="left":
            cx,cy,_=self.app_state.get_marker_geometry("移动轮盘")
            angle=(math.degrees(math.atan2(ey-cy,ex-cx))+360.0)%360.0
            aid=self.quantize_angle(angle,16)
            return aid,angle,summary,duration,st,et
        mapping=self.right_mapping
        base,count=mapping.get(label,mapping["其他"])
        if label:
            cx,cy,_=self.app_state.get_marker_geometry(label)
        else:
            cx,cy,_=self.app_state.get_marker_geometry("普攻")
        if count>1:
            angle=(math.degrees(math.atan2(ey-cy,ex-cx))+360.0)%360.0
            aid=base+self.quantize_angle(angle,count)
            return aid,angle,summary,duration,st,et
        if action_type=="drag" and (ex!=sx or ey!=sy):
            angle=(math.degrees(math.atan2(ey-sy,ex-sx))+360.0)%360.0
        return base,angle,summary,duration,st,et
    def in_window(self,x,y):
        with self.app_state.lock:
            rect=self.app_state.window_rect
            hwnd=self.app_state.hwnd
        if hwnd is None:
            return False
        return x>=rect[0] and x<=rect[2] and y>=rect[1] and y<=rect[3]
    def find_marker_by_pos(self,x,y):
        ov=self.app_state.overlay
        if ov is None:
            return None
        with self.app_state.lock:
            rect=self.app_state.window_rect
        if rect[2]-rect[0]==0 or rect[3]-rect[1]==0:
            return None
        for m in ov.markers:
            cx=rect[0]+m.x_pct*(rect[2]-rect[0])
            cy=rect[1]+m.y_pct*(rect[3]-rect[1])
            r=m.r_pct*min(rect[2]-rect[0],rect[3]-rect[1])
            if ((x-cx)**2+(y-cy)**2)<=r*r:
                return m
        return None
    def on_click(self,x,y,button,pressed):
        tracked=button in [mouse.Button.left,mouse.Button.right,mouse.Button.middle]
        if pressed:
            if self.in_window(x,y):
                self.app_state.mark_user_input()
                if tracked:
                    self.button_state.add(button)
                    self.button_paths[button]=[(x,y,time.time())]
        else:
            if tracked and button in self.button_state:
                self.app_state.mark_user_input()
                path=self.button_paths.get(button,[])
                t=time.time()
                if path:
                    if path[-1][0]!=x or path[-1][1]!=y:
                        path.append((x,y,t))
                else:
                    path=[(x,y,t)]
                start=path[0]
                end=path[-1]
                marker=self.find_marker_by_pos(x,y)
                if not marker and path:
                    marker=self.find_marker_by_pos(start[0],start[1])
                label=marker.label if marker else None
                action_type="drag" if len(path)>1 else "click"
                if button==mouse.Button.left and label in self.left_labels:
                    hand="left"
                else:
                    hand="right"
                aid,angle,points,duration,st,et=self.encode_action(label,hand,action_type,path,start,end)
                a={"type":action_type,"start":(start[0],start[1]),"end":(end[0],end[1]),"label":label,"action_id":aid,"hand":hand,"pos":(x,y),"angle":angle,"key_points":points,"duration":duration,"delta":(end[0]-start[0],end[1]-start[1]),"start_time":st,"end_time":et}
                with self.lock:
                    if len(self.action_queue)>256:
                        self.action_queue.popleft()
                    self.action_queue.append(a)
                self.button_state.discard(button)
                self.button_paths[button]=[]
            elif not pressed and self.in_window(x,y):
                self.app_state.mark_user_input()
    def on_move(self,x,y):
        inside=self.in_window(x,y)
        if inside:
            self.app_state.mark_user_input()
        if self.button_state:
            t=time.time()
            for btn in list(self.button_state):
                if btn not in self.button_paths:
                    self.button_paths[btn]=[]
                self.button_paths[btn].append((x,y,t))
    def on_scroll(self,x,y,dx,dy):
        if self.in_window(x,y):
            self.app_state.mark_user_input()
            a={"type":"scroll","delta":(dx,dy),"pos":(x,y),"label":None,"action_id":None,"hand":"right"}
            with self.lock:
                if len(self.action_queue)>256:
                    self.action_queue.popleft()
                self.action_queue.append(a)
    def on_key_press(self,key):
        try:
            if key==keyboard.Key.esc:
                os._exit(0)
            else:
                self.app_state.mark_user_input()
        except Exception as e:
            logger.error("键盘事件处理失败:%s",e)
    def pop_action(self):
        with self.lock:
            if self.action_queue:
                return self.action_queue.popleft()
            return None
class ScreenshotRecorder(threading.Thread):
    def __init__(self,app_state,buffer,rate_controller,input_tracker):
        super(ScreenshotRecorder,self).__init__()
        self.daemon=True
        self.app_state=app_state
        self.buffer=buffer
        self.rate_controller=rate_controller
        self.input_tracker=input_tracker
        self.stop_flag=False
    def run(self):
        while not self.stop_flag:
            self.rate_controller.refresh()
            hz=self.rate_controller.get_hz()
            dt=1.0/float(max(1,hz))
            with self.app_state.lock:
                hwnd=self.app_state.hwnd
                mode=self.app_state.mode
            if hwnd is None:
                time.sleep(dt)
                continue
            visible=window_visible(hwnd)
            self.app_state.set_visibility_paused(not visible)
            if not visible:
                time.sleep(dt)
                continue
            if not self.app_state.can_record():
                time.sleep(dt)
                continue
            rect=get_window_rect(hwnd)
            with self.app_state.lock:
                self.app_state.window_rect=rect
                mode=self.app_state.mode
            try:
                img=ImageGrabModule.grab(bbox=rect)
            except Exception as e:
                logger.warning("截图失败:%s",e)
                img=None
            if img is not None:
                self.app_state.update_state_from_frame(img)
                if mode==Mode.LEARNING:
                    actions=[]
                    while True:
                        act=self.input_tracker.pop_action()
                        if act is None:
                            break
                        actions.append(act)
                    src="user"
                else:
                    actions=[]
                    while True:
                        act=self.app_state.consume_ai_action()
                        if act is None:
                            break
                        actions.append(act)
                    src="ai"
                if not actions:
                    actions=[None]
                for act in actions:
                    self.buffer.add(img,act,src,self.app_state.metrics,self.app_state.hero_dead,self.app_state.cooldowns,rect)
            time.sleep(dt)
class LeftHandThread(threading.Thread):
    def __init__(self,app_state):
        super(LeftHandThread,self).__init__()
        self.daemon=True
        self.app_state=app_state
        self.stop_flag=False
    def run(self):
        while not self.stop_flag:
            with self.app_state.lock:
                cond=(self.app_state.mode==Mode.TRAINING and (not self.app_state.hero_dead) and (not self.app_state.in_recall) and window_visible(self.app_state.hwnd))
                rect=self.app_state.window_rect
                hidden_vec=self.app_state.current_hidden.clone().detach()
                lh=self.app_state.training_hidden_states["left"]
                rh=self.app_state.training_hidden_states["right"]
            if not cond:
                time.sleep(0.01)
                continue
            device=self.app_state.agent.device
            h_state=hidden_vec.to(device)
            if lh is not None:
                lh=lh.to(device)
            if rh is not None:
                rh=rh.to(device)
            laction,raction,lprob,rprob,lval,rval,lh2,rh2=self.app_state.agent.select_actions(h_state,lh,rh)
            lh2=lh2.detach() if lh2 is not None else None
            rh2=rh2.detach() if rh2 is not None else None
            with self.app_state.lock:
                self.app_state.training_hidden_states["left"]=lh2
                self.app_state.training_hidden_states["right"]=rh2
            center_x,center_y,r=self.app_state.get_marker_geometry("移动轮盘")
            angle=(laction/16.0)*2*math.pi
            dx=math.cos(angle)*r
            dy=math.sin(angle)*r
            end_x=center_x+dx
            end_y=center_y+dy
            path=[(center_x,center_y)]
            straight_x=end_x
            straight_y=end_y
            direction=1 if float(h_state.mean().item())>=0 else -1
            arc_span=(0.35+0.05*((laction%8)+1))*math.pi
            arc_steps=12
            self.app_state.hardware.refresh()
            hand_interval=self.app_state.hardware.suggest_hand_interval()
            drag_duration=max(0.02,hand_interval*1.5)
            arc_duration=max(0.01,hand_interval)
            try:
                pyautogui.moveTo(center_x,center_y)
                pyautogui.mouseDown()
                pyautogui.dragTo(straight_x,straight_y,duration=drag_duration,button='left',mouseDownUp=False)
                path.append((straight_x,straight_y))
                for step in range(1,arc_steps+1):
                    theta=angle+direction*arc_span*step/arc_steps
                    px=center_x+math.cos(theta)*r
                    py=center_y+math.sin(theta)*r
                    pyautogui.dragTo(px,py,duration=arc_duration,button='left',mouseDownUp=False)
                    path.append((px,py))
                pyautogui.mouseUp()
            except Exception as e:
                logger.warning("左手操作失败:%s",e)
            self.app_state.record_ai_action({"hand":"left","action_id":laction,"label":"移动轮盘","type":"drag","start":(center_x,center_y),"end":path[-1] if path else (center_x,center_y),"path":path,"timestamp":time.time(),"rotation_dir":"cw" if direction>0 else "ccw"})
            time.sleep(max(0.01,hand_interval))
class RightHandThread(threading.Thread):
    def __init__(self,app_state):
        super(RightHandThread,self).__init__()
        self.daemon=True
        self.app_state=app_state
        self.stop_flag=False
    def run(self):
        while not self.stop_flag:
            with self.app_state.lock:
                cond=(self.app_state.mode==Mode.TRAINING and (not self.app_state.hero_dead) and (not self.app_state.in_recall) and window_visible(self.app_state.hwnd))
                rect=self.app_state.window_rect
                hidden_vec=self.app_state.current_hidden.clone().detach()
                lh=self.app_state.training_hidden_states["left"]
                rh=self.app_state.training_hidden_states["right"]
            if not cond:
                time.sleep(0.01)
                continue
            device=self.app_state.agent.device
            h_state=hidden_vec.to(device)
            if lh is not None:
                lh=lh.to(device)
            if rh is not None:
                rh=rh.to(device)
            laction,raction,lprob,rprob,lval,rval,lh2,rh2=self.app_state.agent.select_actions(h_state,lh,rh)
            lh2=lh2.detach() if lh2 is not None else None
            rh2=rh2.detach() if rh2 is not None else None
            with self.app_state.lock:
                self.app_state.training_hidden_states["left"]=lh2
                self.app_state.training_hidden_states["right"]=rh2
            info=RIGHT_ACTION_TABLE[raction%len(RIGHT_ACTION_TABLE)]
            label=info[0]
            bins=info[1]
            local=info[2]
            if label not in RIGHT_ACTION_LABELS:
                label="普攻"
            cx,cy,r=self.app_state.get_marker_geometry(label)
            start_x=cx
            start_y=cy
            end_x=cx
            end_y=cy
            action_type="click"
            self.app_state.hardware.refresh()
            hand_interval=self.app_state.hardware.suggest_hand_interval()
            drag_duration=max(0.02,hand_interval*1.5)
            try:
                if bins>1:
                    angle=2*math.pi*((local+0.5)/float(bins))
                    if label=="取消施法":
                        start_x=cx+math.cos(angle)*r*2.0
                        start_y=cy+math.sin(angle)*r*2.0
                        end_x=cx
                        end_y=cy
                        pyautogui.moveTo(start_x,start_y)
                        pyautogui.mouseDown()
                        pyautogui.dragTo(end_x,end_y,duration=drag_duration,button='left')
                        pyautogui.mouseUp()
                    else:
                        end_x=cx+math.cos(angle)*r
                        end_y=cy+math.sin(angle)*r
                        pyautogui.moveTo(cx,cy)
                        pyautogui.mouseDown()
                        pyautogui.dragTo(end_x,end_y,duration=drag_duration,button='left')
                        pyautogui.mouseUp()
                    action_type="drag"
                elif label in ["回城","恢复","普攻","主动装备","数据A","数据B","数据C"]:
                    pyautogui.moveTo(cx,cy)
                    pyautogui.click()
                else:
                    pyautogui.moveTo(cx,cy)
                    pyautogui.click()
            except Exception as e:
                logger.warning("右手操作失败:%s",e)
            self.app_state.record_ai_action({"hand":"right","action_id":raction,"label":label,"type":action_type,"start":(start_x,start_y),"end":(end_x,end_y),"timestamp":time.time()})
            time.sleep(max(0.01,hand_interval))
class OptimizationThread(threading.Thread):
    def __init__(self,app_state):
        super(OptimizationThread,self).__init__()
        self.daemon=True
        self.app_state=app_state
        self.stop_flag=False
    def run(self):
        def get_cancel():
            return self.app_state.consume_cancel_request()
        def set_progress(v):
            self.app_state.set_progress(v)
        def update_markers():
            stats=self.app_state.buffer.get_user_click_stats()
            ov=self.app_state.overlay
            if ov:
                for m in ov.markers:
                    if m.label in stats:
                        st=stats[m.label]
                        m.x_pct=clamp(st["x_pct"],0.0,1.0)
                        m.y_pct=clamp(st["y_pct"],0.0,1.0)
                        m.r_pct=clamp(st["r_pct"],0.01,0.5)
                        m.update_geometry_from_parent()
        def persist_markers():
            ov=self.app_state.overlay
            if ov:
                self.app_state.file_manager.save_markers(ov.get_marker_data(),self.app_state.window_rect)
        completed=self.app_state.agent.optimize_from_buffer(self.app_state.buffer,get_cancel,set_progress,update_markers,persist_markers,1000)
        self.app_state.clear_cancel_request()
        if completed:
            self.app_state.set_progress(100)
        else:
            self.app_state.set_progress(0)
        status="completed" if completed else "cancelled"
        self.app_state.finish_optimization(status)
class InitializationThread(threading.Thread):
    def __init__(self,app_state):
        super(InitializationThread,self).__init__()
        self.daemon=True
        self.app_state=app_state
    def run(self):
        status={"timestamp":datetime.now().isoformat(),"dependencies":{},"aaa":{},"hardware":{}}
        missing=verify_dependencies()
        status["dependencies"]={"missing":missing,"ok":len(missing)==0}
        try:
            self.app_state.file_manager.ensure_structure()
            status["aaa"]={"ok":True,"path":self.app_state.file_manager.base_path}
        except Exception as e:
            status["aaa"]={"ok":False,"error":str(e)}
        self.app_state.hardware.refresh(True)
        status["hardware"]={"cpu":self.app_state.hardware.cpu_load,"mem":self.app_state.hardware.mem_free,"gpu":self.app_state.hardware.gpu_free}
        self.app_state.file_manager.write_init_status(status)
        error_msg=None
        if missing:
            error_msg="缺少依赖:"+",".join(missing)
        if not status["aaa"].get("ok",False):
            detail=status["aaa"].get("error","AAA目录初始化失败")
            error_msg=(error_msg+";"+detail) if error_msg else detail
        if error_msg:
            logger.error("初始化失败:%s",error_msg)
            self.app_state.complete_initialization(False,status,error_msg)
        else:
            logger.info("初始化完成,依赖就绪")
            self.app_state.complete_initialization(True,status,None)
class ModeManagerThread(threading.Thread):
    def __init__(self,app_state):
        super(ModeManagerThread,self).__init__()
        self.daemon=True
        self.app_state=app_state
        self.stop_flag=False
    def run(self):
        while not self.stop_flag:
            with self.app_state.lock:
                hwnd=self.app_state.hwnd
                mode=self.app_state.mode
            if hwnd:
                visible=window_visible(hwnd)
                self.app_state.set_visibility_paused(not visible)
                if not visible:
                    if mode==Mode.TRAINING:
                        self.app_state.enter_learning()
                else:
                    if self.app_state.should_switch_to_training():
                        self.app_state.enter_training()
                if self.app_state.must_back_to_learning():
                    self.app_state.enter_learning()
            time.sleep(0.1)
def main():
    dependency_manager.start_preloading()
    app=QtWidgets.QApplication(sys.argv)
    ensure_qt_classes()
    rate_controller=HardwareAdaptiveRate()
    rate_controller.refresh(True)
    file_manager=AAAFileManager()
    agent=RLAgent(file_manager,rate_controller)
    buffer=ExperienceBuffer(file_manager)
    app_state=AppState(file_manager,agent,buffer,rate_controller)
    input_tracker=InputTracker(app_state)
    recorder=ScreenshotRecorder(app_state,buffer,rate_controller,input_tracker)
    recorder.start()
    init_thread=InitializationThread(app_state)
    app_state.init_thread=init_thread
    init_thread.start()
    mode_manager=ModeManagerThread(app_state)
    app_state.mode_manager_thread=mode_manager
    mode_manager.start()
    w=MainWindow(app_state,rate_controller)
    w.show()
    sys.exit(app.exec_())
if __name__=="__main__":
    main()
