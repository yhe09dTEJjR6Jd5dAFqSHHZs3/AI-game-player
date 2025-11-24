import os
import time
import glob
import threading
import json
import math
import random
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from socketserver import ThreadingMixIn
import urllib.parse
import win32gui
import ctypes
import shutil
import psutil
from PIL import ImageGrab
from pynput import mouse, keyboard
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

home_dir = os.path.expanduser("~")
base_dir = os.path.join(home_dir, "Desktop", "AAA")
exp_root = os.path.join(base_dir, "experience")
model_root = os.path.join(base_dir, "models")
state_file = os.path.join(base_dir, "runtime_state.json")
active_profile = "default"
remembered_window_title = ""
os.makedirs(exp_root, exist_ok=True)
os.makedirs(model_root, exist_ok=True)

def sanitize_profile(name):
    txt = "" if name is None else str(name)
    cleaned = []
    for ch in txt:
        if ch.isalnum() or ch in (" ", "-", "_"):
            cleaned.append(ch)
        else:
            cleaned.append("_")
    clean = "".join(cleaned).strip()
    clean = "_".join([p for p in clean.split(" ") if p])
    if not clean:
        clean = "profile"
    return clean[:80]

def ensure_profile_dirs():
    os.makedirs(os.path.join(exp_root, active_profile), exist_ok=True)
    os.makedirs(os.path.join(model_root, active_profile), exist_ok=True)

def get_exp_dir():
    ensure_profile_dirs()
    return os.path.join(exp_root, active_profile)

def get_model_dir():
    ensure_profile_dirs()
    return os.path.join(model_root, active_profile)

def persist_runtime_state():
    try:
        data = {"window_title": remembered_window_title, "profile": active_profile}
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
    except Exception:
        pass

def load_runtime_state():
    global active_profile, remembered_window_title
    if os.path.exists(state_file):
        try:
            with open(state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            remembered_window_title = data.get("window_title", "") or ""
            prof = data.get("profile", "") or ""
            if prof:
                active_profile = prof
        except Exception:
            remembered_window_title = ""
    ensure_profile_dirs()

load_runtime_state()

state = {"exit": False, "optimize": False, "switch_to_training": False}
kbd_listener = None
windows_list = []
current_hwnd = None
worker_thread = None

ui_state = {
    "mode": "待机",
    "status": "等待启动",
    "exp_files": 0,
    "epoch": 0,
    "max_epoch": 0,
    "loss": 0.0,
    "hwnd_title": "",
    "device_str": "",
    "steps": 0,
    "total_steps": 0,
    "finished": False,
    "human_exp": 0,
    "ai_exp": 0,
    "recycled": 0,
    "brain_level": 0.0,
    "img_size": 128,
    "seq_len": 0,
    "path_len": 0,
    "embed_dim": 0,
    "heads": 0,
    "layers": 0,
    "screen": "",
    "dpi": "",
    "disk": "",
    "cpu": 0.0,
    "mem": 0.0,
    "gpu": 0.0,
    "vram": 0.0,
    "capture": "未绑定窗口A",
    "profile": active_profile,
    "remembered": remembered_window_title
}

def activate_profile(title):
    global active_profile, remembered_window_title
    remembered_window_title = title or ""
    active_profile = sanitize_profile(title) if title else "default"
    ui_state["profile"] = active_profile
    ui_state["remembered"] = remembered_window_title
    ensure_profile_dirs()
    persist_runtime_state()

activate_profile(remembered_window_title)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if device.type == "cuda":
    try:
        torch.backends.cuda.matmul.fp32_precision = "tf32"
        torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass

if device.type == "cuda":
    try:
        prop = torch.cuda.get_device_properties(0)
        vram_gb = float(prop.total_memory) / (1024.0 ** 3)
        gpu_name = prop.name
    except Exception:
        vram_gb = 4.0
        gpu_name = "Unknown CUDA"
else:
    vram_gb = 0.0
    gpu_name = "CPU"

if device.type == "cuda":
    ui_state["device_str"] = "CUDA · " + gpu_name + " · " + ("%.1f" % vram_gb) + " GB VRAM"
else:
    ui_state["device_str"] = "CPU 模式"

IMG_SIZE = 128
if vram_gb >= 10.0:
    SEQ_LEN = 10
    PATH_LEN = 32
    MODEL_EMBED_DIM = 448
    MODEL_HEADS = 8
    MODEL_LAYERS = 7
    MODEL_FF = 1536
    VISION_BASE = 40
elif vram_gb >= 6.0:
    SEQ_LEN = 8
    PATH_LEN = 24
    MODEL_EMBED_DIM = 384
    MODEL_HEADS = 8
    MODEL_LAYERS = 6
    MODEL_FF = 1280
    VISION_BASE = 32
elif vram_gb >= 4.0:
    SEQ_LEN = 8
    PATH_LEN = 20
    MODEL_EMBED_DIM = 352
    MODEL_HEADS = 8
    MODEL_LAYERS = 5
    MODEL_FF = 1152
    VISION_BASE = 32
else:
    SEQ_LEN = 6
    PATH_LEN = 16
    MODEL_EMBED_DIM = 288
    MODEL_HEADS = 4
    MODEL_LAYERS = 4
    MODEL_FF = 1024
    VISION_BASE = 24

ui_state["img_size"] = IMG_SIZE
ui_state["seq_len"] = SEQ_LEN
ui_state["path_len"] = PATH_LEN
ui_state["embed_dim"] = MODEL_EMBED_DIM
ui_state["heads"] = MODEL_HEADS
ui_state["layers"] = MODEL_LAYERS

def choose_hyperparams(num_samples, human_ratio):
    if device.type == "cuda":
        try:
            prop = torch.cuda.get_device_properties(0)
            vram_local = float(prop.total_memory) / (1024.0 ** 3)
        except Exception:
            vram_local = vram_gb
    else:
        vram_local = 0.0
    if vram_local >= 10.0:
        max_batch = 64
    elif vram_local >= 6.0:
        max_batch = 32
    elif vram_local >= 4.0:
        max_batch = 16
    elif vram_local > 0.0:
        max_batch = 8
    else:
        max_batch = 4
    if num_samples <= 0:
        batch_size = 1
    elif num_samples < 8:
        batch_size = min(4, num_samples)
    else:
        target = max(4, int(num_samples / 4))
        batch_size = min(max_batch, target)
        if batch_size > num_samples:
            batch_size = num_samples
    if num_samples < 200:
        min_epochs = 4
        max_epochs = 36
        base_lr = 4e-4
        wd = 6e-4
        grad_clip = 3.0
    elif num_samples < 800:
        min_epochs = 4
        max_epochs = 26
        base_lr = 3.5e-4
        wd = 4e-4
        grad_clip = 4.0
    elif num_samples < 3000:
        min_epochs = 3
        max_epochs = 20
        base_lr = 2.5e-4
        wd = 3e-4
        grad_clip = 5.0
    else:
        min_epochs = 2
        max_epochs = 16
        base_lr = 2.0e-4
        wd = 2e-4
        grad_clip = 5.0
    if human_ratio > 0.7:
        base_lr *= 0.85
        wd *= 1.1
    elif human_ratio < 0.3:
        base_lr *= 1.1
    if num_samples > 4000:
        base_lr *= 0.9
    if device.type != "cuda":
        batch_size = max(1, batch_size // 2)
    patience = 4 if num_samples < 800 else 3
    return batch_size, min_epochs, max_epochs, base_lr, wd, grad_clip, patience

def get_display_metrics():
    try:
        user32 = ctypes.windll.user32
        width = int(user32.GetSystemMetrics(0))
        height = int(user32.GetSystemMetrics(1))
    except Exception:
        width = 0
        height = 0
    dpi = 0.0
    try:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(1)
        except Exception:
            ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass
    try:
        dpi = float(ctypes.windll.shcore.GetDpiForSystem())
    except Exception:
        try:
            hwnd = ctypes.windll.user32.GetDesktopWindow()
            dpi = float(ctypes.windll.user32.GetDpiForWindow(hwnd))
        except Exception:
            dpi = 0.0
    return width, height, dpi

def get_disk_metrics():
    try:
        usage = shutil.disk_usage(base_dir)
        free_gb = usage.free / float(1024 ** 3)
        total_gb = usage.total / float(1024 ** 3)
        return free_gb, total_gb
    except Exception:
        return 0.0, 0.0

def get_resource_metrics():
    try:
        cpu = psutil.cpu_percent(interval=0.05)
    except Exception:
        cpu = 0.0
    try:
        mem = psutil.virtual_memory().percent
    except Exception:
        mem = 0.0
    gpu = 0.0
    vram = 0.0
    if device.type == "cuda":
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            used_mem = total_mem - free_mem
            if total_mem > 0:
                vram = (used_mem / float(total_mem)) * 100.0
        except Exception:
            try:
                total_mem = torch.cuda.get_device_properties(0).total_memory
                used_mem = torch.cuda.memory_allocated(0)
                if total_mem > 0:
                    vram = (used_mem / float(total_mem)) * 100.0
            except Exception:
                vram = 0.0
        try:
            gpu = float(torch.cuda.utilization(0))
        except Exception:
            gpu = vram
    return cpu, mem, gpu, vram

def capture_diagnostics():
    if current_hwnd is None:
        return "未绑定窗口A"
    try:
        if not win32gui.IsWindow(current_hwnd):
            return "窗口句柄失效"
        if win32gui.IsIconic(current_hwnd):
            return "窗口被最小化"
        if not win32gui.IsWindowVisible(current_hwnd):
            return "窗口不可见"
        rect = win32gui.GetWindowRect(current_hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        if width <= 40 or height <= 40:
            return "窗口尺寸异常"
        sw, sh, _ = get_display_metrics()
        if sw > 0 and sh > 0:
            if left < 0 or top < 0 or right > sw or bottom > sh:
                return "窗口未完全显示在屏幕内"
        fg = win32gui.GetForegroundWindow()
        if fg != current_hwnd:
            return "窗口未激活，可能被遮挡"
    except Exception:
        return "窗口状态未知"
    return "捕获就绪"

def update_realtime_metrics():
    width, height, dpi = get_display_metrics()
    if width > 0 and height > 0:
        ui_state["screen"] = str(width) + " × " + str(height)
    else:
        ui_state["screen"] = "未知"
    if dpi > 0:
        ui_state["dpi"] = ("%.0f" % dpi) + " DPI"
    else:
        ui_state["dpi"] = "DPI 未知"
    free_gb, total_gb = get_disk_metrics()
    if total_gb > 0:
        ui_state["disk"] = ("%.1f" % free_gb) + " GB 可用 / " + ("%.1f" % total_gb) + " GB"
    else:
        ui_state["disk"] = "磁盘信息未知"
    cpu, mem, gpu, vram = get_resource_metrics()
    ui_state["cpu"] = float(cpu)
    ui_state["mem"] = float(mem)
    ui_state["gpu"] = float(gpu)
    ui_state["vram"] = float(vram)
    ui_state["capture"] = capture_diagnostics()

def augment_sequence(imgs):
    imgs = imgs.astype(np.float32)
    if np.random.rand() < 0.8:
        scale = np.random.uniform(0.9, 1.1)
        imgs *= scale
    if np.random.rand() < 0.7:
        shift = np.random.uniform(-0.05, 0.05)
        imgs += shift
    if np.random.rand() < 0.5:
        noise = np.random.normal(0.0, 0.02, size=imgs.shape).astype(np.float32)
        imgs += noise
    imgs = np.clip(imgs, 0.0, 1.0)
    return imgs

class ExperienceDataset(Dataset):
    def __init__(self, files):
        self.files = files
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        path_file = self.files[idx]
        data = np.load(path_file)
        if "imgs" in data:
            imgs = data["imgs"]
            if imgs.ndim == 3:
                imgs = imgs[None, ...]
            if imgs.shape[0] >= SEQ_LEN:
                imgs = imgs[-SEQ_LEN:]
            else:
                first = imgs[0:1]
                pad = np.repeat(first, SEQ_LEN - imgs.shape[0], axis=0)
                imgs = np.concatenate([pad, imgs], axis=0)
        else:
            img = data["img"]
            if img.ndim == 3:
                img = img[None, ...]
            imgs = np.repeat(img, SEQ_LEN, axis=0)
        imgs = imgs.astype(np.float32) / 255.0
        imgs = augment_sequence(imgs)
        if "path" in data:
            path = data["path"].astype(np.float32)
            if path.shape[0] >= PATH_LEN:
                idxs = np.linspace(0, path.shape[0] - 1, PATH_LEN).astype(int)
                path = path[idxs]
            else:
                last = path[-1]
                pad = np.repeat(last[None, :], PATH_LEN - path.shape[0], axis=0)
                path = np.concatenate([path, pad], axis=0)
            target = path.reshape(PATH_LEN * 2)
        else:
            coord = data["coord"].astype(np.float32)
            target = np.tile(coord, PATH_LEN)
        if "src" in data:
            src = int(data["src"][0])
        else:
            name = os.path.basename(path_file)
            if name.startswith("A_"):
                src = 1
            else:
                src = 0
        img_tensor = torch.from_numpy(imgs)
        target_tensor = torch.from_numpy(target.astype(np.float32))
        return img_tensor, target_tensor, src

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, 1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(nn.Conv2d(in_channels, out_channels, 1), nn.BatchNorm2d(out_channels))
        else:
            self.shortcut = nn.Identity()
    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = out + identity
        out = self.relu(out)
        return out

class TensorLogic(nn.Module):
    def __init__(self, dim, heads):
        super().__init__()
        self.heads = heads
        self.scale = (dim // heads) ** -0.5
        self.query = nn.Linear(dim, dim)
        self.key = nn.Linear(dim, dim)
        self.value = nn.Linear(dim, dim)
        self.logic_gate = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
    def forward(self, tokens, seq):
        b, s, d = tokens.shape
        h = self.heads
        d_h = d // h
        q = self.query(tokens).view(b, s, h, d_h)
        k = self.key(seq).view(b, s, h, d_h)
        v = self.value(seq).view(b, s, h, d_h)
        attn = torch.einsum("bshd,bshd->bsh", q, k) * self.scale
        attn = torch.softmax(attn, dim=1).unsqueeze(-1)
        logic = (attn * v).sum(dim=1)
        gate = torch.sigmoid(self.logic_gate(tokens).mean(dim=1))
        fused = torch.tanh(self.out(logic.view(b, d) * gate))
        return fused

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        in_ch = 3
        base = VISION_BASE
        self.seq_len = SEQ_LEN
        self.embed_dim = MODEL_EMBED_DIM
        hidden_dim = max(512, self.embed_dim)
        logic_heads = max(1, MODEL_HEADS // 2)
        self.stem = nn.Sequential(
            nn.Conv2d(in_ch, base, 3, 2, 1),
            nn.BatchNorm2d(base),
            nn.ReLU(inplace=True),
            nn.Conv2d(base, base * 2, 3, 2, 1),
            nn.BatchNorm2d(base * 2),
            nn.ReLU(inplace=True)
        )
        self.block1 = ResidualBlock(base * 2, base * 2)
        self.down1 = nn.Sequential(
            nn.Conv2d(base * 2, base * 4, 3, 2, 1),
            nn.BatchNorm2d(base * 4),
            nn.ReLU(inplace=True)
        )
        self.block2 = ResidualBlock(base * 4, base * 4)
        self.down2 = nn.Sequential(
            nn.Conv2d(base * 4, base * 8, 3, 2, 1),
            nn.BatchNorm2d(base * 8),
            nn.ReLU(inplace=True)
        )
        self.block3 = ResidualBlock(base * 8, base * 8)
        self.proj = nn.Conv2d(base * 8, self.embed_dim, 1)
        num_tokens = (IMG_SIZE // 16) * (IMG_SIZE // 16)
        self.spatial_pos_embed = nn.Parameter(torch.zeros(1, num_tokens, self.embed_dim))
        self.temporal_pos_embed = nn.Parameter(torch.zeros(1, SEQ_LEN, self.embed_dim))
        encoder_layer = nn.TransformerEncoderLayer(d_model=self.embed_dim, nhead=MODEL_HEADS, dim_feedforward=MODEL_FF, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=MODEL_LAYERS)
        self.temporal_rnn = nn.GRU(input_size=self.embed_dim, hidden_size=self.embed_dim, batch_first=True)
        self.logic = TensorLogic(self.embed_dim, logic_heads)
        self.path_query = nn.Parameter(torch.randn(PATH_LEN, self.embed_dim))
        self.ctx_proj = nn.Linear(self.embed_dim * 3, hidden_dim)
        self.traj_gru = nn.GRU(input_size=self.embed_dim, hidden_size=hidden_dim, batch_first=True)
        self.traj_out = nn.Linear(hidden_dim, 2)
    def forward(self, x):
        b, s, c, h, w = x.shape
        x = x.view(b * s, c, h, w)
        x = self.stem(x)
        x = self.block1(x)
        x = self.down1(x)
        x = self.block2(x)
        x = self.down2(x)
        x = self.block3(x)
        x = self.proj(x)
        bs, c, h, w = x.shape
        tokens = h * w
        frame_feats = x.mean(dim=[2, 3])
        frame_feats = frame_feats.view(b, s, self.embed_dim)
        x = x.view(b, s, c, tokens)
        x = x.permute(0, 1, 3, 2)
        spe = self.spatial_pos_embed
        if spe.size(1) != tokens:
            if spe.size(1) > tokens:
                spe = spe[:, :tokens, :]
            else:
                repeat = (tokens + spe.size(1) - 1) // spe.size(1)
                spe = spe.repeat(1, repeat, 1)[:, :tokens, :]
        spe = spe.unsqueeze(1)
        tpe = self.temporal_pos_embed
        if tpe.size(1) != s:
            if tpe.size(1) > s:
                tpe = tpe[:, :s, :]
            else:
                repeat_t = (s + tpe.size(1) - 1) // tpe.size(1)
                tpe = tpe.repeat(1, repeat_t, 1)[:, :s, :]
        tpe = tpe.unsqueeze(2)
        x = x + spe + tpe
        x = x.reshape(b, s * tokens, self.embed_dim)
        x = self.transformer(x)
        context_tokens = x.mean(dim=1)
        temporal_out, temporal_hidden = self.temporal_rnn(frame_feats)
        context_seq = temporal_hidden[-1]
        logic_ctx = self.logic(frame_feats, temporal_out)
        context = torch.cat([context_tokens, context_seq, logic_ctx], dim=1)
        h0 = self.ctx_proj(context).unsqueeze(0)
        queries = self.path_query.unsqueeze(0).expand(b, -1, -1)
        traj, _ = self.traj_gru(queries, h0)
        traj = self.traj_out(traj)
        traj = torch.sigmoid(traj)
        traj = traj.view(b, PATH_LEN * 2)
        return traj

def capture_window(hwnd):
    try:
        if not win32gui.IsWindow(hwnd):
            return None, None
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return None, None
        img = ImageGrab.grab(bbox=(left, top, right, bottom))
        return img, rect
    except Exception:
        return None, None

def resample_path(path_points, length):
    pts = np.array(path_points, dtype=np.float32)
    if pts.shape[0] == 0:
        return None
    if pts.shape[0] >= length:
        idxs = np.linspace(0, pts.shape[0] - 1, length).astype(int)
        pts = pts[idxs]
    else:
        last = pts[-1]
        pad = np.repeat(last[None, :], length - pts.shape[0], axis=0)
        pts = np.concatenate([pts, pad], axis=0)
    return pts

def save_sequence_experience(hwnd, frames, path_points, source):
    try:
        if not path_points:
            return
        if not frames:
            img, _ = capture_window(hwnd)
            if img is None:
                return
            frames = [img]
        if len(frames) >= SEQ_LEN:
            use_frames = frames[-SEQ_LEN:]
        else:
            use_frames = [frames[0]] * (SEQ_LEN - len(frames)) + list(frames)
        imgs_proc = []
        for img in use_frames:
            img_r = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img_r)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = np.transpose(arr, (2, 0, 1))
            imgs_proc.append(arr)
        imgs_arr = np.stack(imgs_proc, axis=0)
        path = resample_path(path_points, PATH_LEN)
        if path is None:
            return
        ts = int(time.time() * 1000)
        prefix = "H_" if source == "human" else "A_"
        path_file = os.path.join(get_exp_dir(), prefix + str(ts) + ".npz")
        src_val = np.array([0 if source == "human" else 1], dtype=np.int8)
        np.savez_compressed(path_file, imgs=imgs_arr, path=path.astype(np.float32), src=src_val)
    except Exception:
        pass

def preprocess_sequence_for_model(frames):
    try:
        if not frames:
            return None
        if len(frames) >= SEQ_LEN:
            use_frames = frames[-SEQ_LEN:]
        else:
            use_frames = [frames[0]] * (SEQ_LEN - len(frames)) + list(frames)
        imgs_proc = []
        for img in use_frames:
            img_r = img.resize((IMG_SIZE, IMG_SIZE))
            arr = np.array(img_r)
            if arr.ndim == 2:
                arr = np.stack([arr] * 3, axis=-1)
            elif arr.shape[2] == 4:
                arr = arr[:, :, :3]
            arr = np.transpose(arr, (2, 0, 1))
            imgs_proc.append(arr)
        imgs_arr = np.stack(imgs_proc, axis=0).astype(np.float32) / 255.0
        tensor = torch.from_numpy(imgs_arr).unsqueeze(0).to(device)
        return tensor
    except Exception:
        return None

def load_model():
    model_path = os.path.join(get_model_dir(), "model_latest.pt")
    model = Net().to(device)
    if os.path.exists(model_path):
        try:
            state_dict = torch.load(model_path, map_location=device)
            model.load_state_dict(state_dict)
        except Exception:
            pass
    return model

def scan_experience_files():
    files = sorted(glob.glob(os.path.join(get_exp_dir(), "*.npz")))
    human_files = []
    ai_files = []
    for f in files:
        name = os.path.basename(f)
        if name.startswith("A_"):
            ai_files.append(f)
        elif name.startswith("H_"):
            human_files.append(f)
        else:
            human_files.append(f)
    ui_state["exp_files"] = len(files)
    ui_state["human_exp"] = len(human_files)
    ui_state["ai_exp"] = len(ai_files)
    return files, human_files, ai_files

def manage_experience_pool():
    files, human_files, ai_files = scan_experience_files()
    max_pool = 80000
    recycled = 0
    if len(files) > max_pool:
        excess = len(files) - max_pool
        candidates = []
        for lst, src_val in ((ai_files, 1), (human_files, 0)):
            for f in lst:
                try:
                    t = os.path.getmtime(f)
                except Exception:
                    t = 0.0
                candidates.append((t, src_val, f))
        candidates.sort()
        to_delete = [c[2] for c in candidates[:excess]]
        for f in to_delete:
            try:
                os.remove(f)
                recycled += 1
            except Exception:
                pass
    ui_state["recycled"] = recycled
    return scan_experience_files()

def offline_train():
    global device
    files, human_files, ai_files = manage_experience_pool()
    if not files:
        ui_state["status"] = "经验池为空，无法离线优化"
        return
    if human_files and ai_files:
        human_ratio = len(human_files) / float(len(files))
    elif human_files:
        human_ratio = 1.0
    elif ai_files:
        human_ratio = 0.0
    else:
        human_ratio = 0.5
    if human_files and ai_files:
        max_ai = int(len(human_files) * 2.0)
        if max_ai <= 0:
            max_ai = len(ai_files)
        if len(ai_files) > max_ai:
            idxs = np.linspace(0, len(ai_files) - 1, max_ai).astype(int)
            ai_selected = [ai_files[i] for i in idxs]
        else:
            ai_selected = ai_files
        files_selected = sorted(human_files + ai_selected)
    else:
        files_selected = files
    num_samples = len(files_selected)
    ui_state["exp_files"] = num_samples
    ui_state["human_exp"] = len(human_files)
    ui_state["ai_exp"] = len(ai_files)
    batch_size, min_epochs, max_epochs, lr, wd, grad_clip, patience = choose_hyperparams(num_samples, human_ratio)
    ui_state["max_epoch"] = min_epochs
    ui_state["epoch"] = 0
    ui_state["loss"] = 0.0
    dataset = ExperienceDataset(files_selected)
    use_cuda = device.type == "cuda"
    pin = use_cuda
    oom_retries = 0
    max_oom_retries = 3
    global_step = 0
    best_loss = None
    best_state = None
    while True:
        if state["exit"]:
            return
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=pin)
        model = load_model()
        model.train()
        use_amp = use_cuda
        scaler = torch.amp.GradScaler("cuda") if use_amp else None
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
        criterion = nn.SmoothL1Loss()
        max_possible_steps = max_epochs * max(len(dataloader), 1)
        ui_state["total_steps"] = max_possible_steps
        ui_state["steps"] = 0
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_possible_steps) if max_possible_steps > 0 else None
        epoch_losses = []
        no_improve = 0
        target_epochs = min_epochs
        ui_state["max_epoch"] = target_epochs
        try:
            for epoch in range(max_epochs):
                if state["exit"]:
                    break
                ui_state["epoch"] = epoch + 1
                ui_state["status"] = "离线优化中: 第 " + str(epoch + 1) + " 轮"
                epoch_loss = 0.0
                steps = 0
                for imgs, paths, src in dataloader:
                    if state["exit"]:
                        break
                    imgs = imgs.to(device, non_blocking=use_cuda)
                    paths = paths.to(device, non_blocking=use_cuda)
                    optimizer.zero_grad(set_to_none=True)
                    if use_amp:
                        with torch.amp.autocast("cuda"):
                            preds = model(imgs)
                            loss = criterion(preds, paths)
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        preds = model(imgs)
                        loss = criterion(preds, paths)
                        loss.backward()
                        nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    epoch_loss += float(loss.detach().cpu().item())
                    steps += 1
                    global_step += 1
                    ui_state["steps"] = global_step
                if steps > 0:
                    avg_loss = epoch_loss / steps
                else:
                    avg_loss = 0.0
                ui_state["loss"] = avg_loss
                epoch_losses.append(avg_loss)
                improved = False
                if best_loss is None or avg_loss < best_loss - 1e-4 * (abs(best_loss) + 1.0):
                    best_loss = avg_loss
                    try:
                        best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
                    except Exception:
                        best_state = None
                    no_improve = 0
                    improved = True
                else:
                    no_improve += 1
                if epoch + 1 >= min_epochs:
                    recent_improve = False
                    if len(epoch_losses) >= 2:
                        diff = epoch_losses[-2] - epoch_losses[-1]
                        thresh = 0.01 * (abs(epoch_losses[-2]) + 1.0)
                        if diff > thresh:
                            recent_improve = True
                    if no_improve >= patience:
                        ui_state["status"] = "提前停止: 损失已趋于收敛"
                        break
                    if epoch + 1 >= target_epochs:
                        if recent_improve and target_epochs < max_epochs:
                            target_epochs = min(max_epochs, target_epochs + 1)
                            ui_state["max_epoch"] = target_epochs
                        else:
                            break
            break
        except RuntimeError as e:
            msg = str(e).lower()
            if "out of memory" in msg or "cuda error" in msg:
                if device.type == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                oom_retries += 1
                ui_state["status"] = "显存自救中: 第 " + str(oom_retries) + " 次"
                if oom_retries >= max_oom_retries:
                    ui_state["status"] = "显存不足，自救失败，尝试转为CPU训练"
                    if device.type == "cuda":
                        try:
                            torch.cuda.empty_cache()
                        except Exception:
                            pass
                    cpu_device = torch.device("cpu")
                    model_cpu = Net().to(cpu_device)
                    try:
                        state_dict = model.state_dict()
                        model_cpu.load_state_dict(state_dict)
                    except Exception:
                        pass
                    device = cpu_device
                    ui_state["device_str"] = "CPU 模式 · 显存自救"
                    return offline_train()
                else:
                    if batch_size > 1:
                        batch_size = max(1, batch_size // 2)
                    else:
                        use_cuda = False
                        pin = False
                    continue
            else:
                raise
    model.eval()
    if best_state is not None:
        try:
            model.load_state_dict(best_state)
        except Exception:
            pass
    model_path = os.path.join(get_model_dir(), "model_latest.pt")
    torch.save(model.state_dict(), model_path)
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass
    brain_level = 0.0
    try:
        brain_level = min(100.0, math.log10(num_samples + 10.0) * 18.0)
    except Exception:
        brain_level = 0.0
    ui_state["brain_level"] = brain_level
    ui_state["steps"] = ui_state.get("total_steps", 0)
    ui_state["status"] = "离线优化完成，新模型已保存"

def on_key_press(key):
    try:
        if key == keyboard.Key.esc:
            state["exit"] = True
            ui_state["status"] = "收到 ESC，准备结束运行"
            return False
        if key == keyboard.Key.enter:
            state["optimize"] = True
        if key == keyboard.Key.space:
            state["switch_to_training"] = True
    except Exception:
        pass

def start_keyboard_listener():
    global kbd_listener
    if kbd_listener is not None:
        return
    kbd_listener = keyboard.Listener(on_press=on_key_press)
    kbd_listener.start()

def learning_mode(hwnd):
    frame_buffer = []
    frame_lock = threading.Lock()
    capture_running = True
    drag_active = False
    drag_points = []
    ui_state["mode"] = "学习模式"
    ui_state["status"] = "采集窗口A画面与人类鼠标轨迹"
    def capture_loop():
        nonlocal capture_running
        while capture_running and not state["exit"] and not state["optimize"] and not state["switch_to_training"]:
            img, _ = capture_window(hwnd)
            if img is not None:
                with frame_lock:
                    frame_buffer.append(img)
                    if len(frame_buffer) > SEQ_LEN * 4:
                        del frame_buffer[0:len(frame_buffer) - SEQ_LEN * 4]
            time.sleep(0.08)
    def inside_window(x, y):
        if not win32gui.IsWindow(hwnd):
            return False
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        if right <= left or bottom <= top:
            return False
        return left <= x <= right and top <= y <= bottom
    def pos_to_rel(x, y):
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return None, None
        x_rel = (x - left) / float(width)
        y_rel = (y - top) / float(height)
        if x_rel < 0.0:
            x_rel = 0.0
        if x_rel > 1.0:
            x_rel = 1.0
        if y_rel < 0.0:
            y_rel = 0.0
        if y_rel > 1.0:
            y_rel = 1.0
        return x_rel, y_rel
    def on_click(x, y, button, pressed):
        nonlocal drag_active, drag_points
        try:
            if button != mouse.Button.left:
                return
            if pressed:
                if not inside_window(x, y):
                    return
                xr, yr = pos_to_rel(x, y)
                if xr is None:
                    return
                drag_active = True
                drag_points = [(xr, yr)]
            else:
                if not drag_active:
                    return
                if inside_window(x, y):
                    xr, yr = pos_to_rel(x, y)
                    if xr is not None:
                        drag_points.append((xr, yr))
                with frame_lock:
                    frames_copy = list(frame_buffer)
                pts_copy = list(drag_points)
                drag_active = False
                drag_points = []
                if frames_copy and pts_copy:
                    save_sequence_experience(hwnd, frames_copy, pts_copy, "human")
        except Exception:
            pass
    def on_move(x, y):
        nonlocal drag_active, drag_points
        try:
            if not drag_active:
                return
            if not inside_window(x, y):
                return
            xr, yr = pos_to_rel(x, y)
            if xr is None:
                return
            if drag_points:
                last_x, last_y = drag_points[-1]
                if abs(xr - last_x) < 0.004 and abs(yr - last_y) < 0.004:
                    return
            drag_points.append((xr, yr))
        except Exception:
            pass
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()
    listener = mouse.Listener(on_click=on_click, on_move=on_move)
    listener.start()
    while True:
        if state["exit"] or state["optimize"] or state["switch_to_training"]:
            break
        time.sleep(0.05)
    capture_running = False
    listener.stop()
    try:
        cap_thread.join(timeout=1.0)
    except Exception:
        pass
    if state["exit"]:
        ui_state["status"] = "学习模式已结束"
        return
    if state["optimize"]:
        state["optimize"] = False
        ui_state["mode"] = "离线优化"
        ui_state["status"] = "停止记录，开始离线优化"
        offline_train()
        return
    if state["switch_to_training"]:
        state["switch_to_training"] = False
        ui_state["mode"] = "训练模式"
        ui_state["status"] = "进入训练模式，AI 控制鼠标"
        training_mode(hwnd)

def execute_mouse_path(hwnd, mctl, path_points):
    try:
        if not win32gui.IsWindow(hwnd):
            return
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            return
        screen_points = []
        for x_rel, y_rel in path_points:
            x = int(left + x_rel * width)
            y = int(top + y_rel * height)
            screen_points.append((x, y))
        if not screen_points:
            return
        first = screen_points[0]
        mctl.position = first
        mctl.press(mouse.Button.left)
        for pos in screen_points[1:]:
            mctl.position = pos
            time.sleep(0.01)
        mctl.release(mouse.Button.left)
    except Exception:
        pass

def training_mode(hwnd):
    ui_state["mode"] = "训练模式"
    ui_state["status"] = "AI 正在根据窗口A画面输出鼠标轨迹"
    model_path = os.path.join(get_model_dir(), "model_latest.pt")
    if not os.path.exists(model_path):
        ui_state["status"] = "未找到模型文件，无法进入训练模式"
        return
    model = load_model()
    model.eval()
    frame_buffer = []
    frame_lock = threading.Lock()
    capture_running = True
    def capture_loop():
        nonlocal capture_running
        while capture_running and not state["exit"]:
            img, _ = capture_window(hwnd)
            if img is not None:
                with frame_lock:
                    frame_buffer.append(img)
                    if len(frame_buffer) > SEQ_LEN * 4:
                        del frame_buffer[0:len(frame_buffer) - SEQ_LEN * 4]
            time.sleep(0.08)
    cap_thread = threading.Thread(target=capture_loop, daemon=True)
    cap_thread.start()
    mctl = mouse.Controller()
    while True:
        if state["exit"]:
            break
        with frame_lock:
            frames_copy = list(frame_buffer)
        tensor = preprocess_sequence_for_model(frames_copy)
        if tensor is None:
            time.sleep(0.05)
            continue
        with torch.no_grad():
            out = model(tensor)[0].detach().cpu().numpy()
        path_points = []
        for i in range(PATH_LEN):
            x_rel = float(out[2 * i])
            y_rel = float(out[2 * i + 1])
            if x_rel < 0.0:
                x_rel = 0.0
            if x_rel > 1.0:
                x_rel = 1.0
            if y_rel < 0.0:
                y_rel = 0.0
            if y_rel > 1.0:
                y_rel = 1.0
            path_points.append((x_rel, y_rel))
        execute_mouse_path(hwnd, mctl, path_points)
        with frame_lock:
            frames_for_save = list(frame_buffer)
        save_sequence_experience(hwnd, frames_for_save, path_points, "ai")
        time.sleep(0.4)
    capture_running = False
    try:
        cap_thread.join(timeout=1.0)
    except Exception:
        pass
    ui_state["status"] = "训练模式结束"

def refresh_window_list():
    global windows_list
    windows = []
    def enum_handler(hwnd, _):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                t = title.strip()
                if t:
                    windows.append((hwnd, t))
    try:
        win32gui.EnumWindows(enum_handler, None)
    except Exception:
        windows = []
    windows_list = windows

def bind_window_by_index(idx):
    global current_hwnd
    refresh_window_list()
    if not windows_list:
        ui_state["status"] = "没有可用窗口"
        return
    if idx < 0 or idx >= len(windows_list):
        idx = 0
    hwnd, title = windows_list[idx]
    current_hwnd = hwnd
    ui_state["hwnd_title"] = title
    ui_state["status"] = "窗口A 已绑定: " + title
    activate_profile(title)
    scan_experience_files()
    try:
        win32gui.SetForegroundWindow(hwnd)
    except Exception:
        pass

def run_pipeline():
    start_keyboard_listener()
    if current_hwnd is None:
        ui_state["status"] = "未绑定窗口A"
        return
    learning_mode(current_hwnd)
    state["exit"] = True
    ui_state["mode"] = "已结束"
    ui_state["status"] = "系统已结束"
    os._exit(0)

HTML_PAGE = '''<!DOCTYPE html>
<html lang="zh-CN">
<head>
<meta charset="UTF-8">
<title>NEURO DESKTOP · SYNAPTIC WINDOW AI</title>
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<style>
* {
  box-sizing: border-box;
}
html, body {
  margin: 0;
  padding: 0;
  width: 100%;
}
body {
  margin: 0;
  padding: clamp(10px, 2vw, 18px);
  background: radial-gradient(circle at 12% 10%, rgba(56,189,248,0.12), transparent 26%), radial-gradient(circle at 88% 8%, rgba(168,85,247,0.12), transparent 24%), radial-gradient(circle at 40% 60%, rgba(34,197,94,0.12), transparent 30%), #020617;
  font-family: Consolas, "JetBrains Mono", monospace;
  color: #e5e5e5;
  min-height: 100vh;
  overflow-x: hidden;
  overflow-y: auto;
}
#aurora {
  position: fixed;
  inset: 0;
  background: radial-gradient(circle at 20% 30%, rgba(56,189,248,0.16), transparent 32%), radial-gradient(circle at 80% 24%, rgba(14,165,233,0.12), transparent 28%), radial-gradient(circle at 18% 78%, rgba(34,197,94,0.18), transparent 32%), radial-gradient(circle at 82% 82%, rgba(168,85,247,0.18), transparent 30%);
  filter: blur(20px) saturate(130%);
  opacity: 0.8;
  mix-blend-mode: screen;
  pointer-events: none;
  animation: pulse 14s ease-in-out infinite alternate;
}
#grid {
  position: fixed;
  inset: 0;
  background-image:
    linear-gradient(rgba(59,130,246,0.22) 1px, transparent 1px),
    linear-gradient(90deg, rgba(45,212,191,0.18) 1px, transparent 1px);
  background-size: 28px 28px, 28px 28px;
  opacity: 0.6;
  pointer-events: none;
  animation: drift 12s linear infinite;
}
#frame {
  position: fixed;
  inset: clamp(8px, 1.6vw, 18px);
  border: 1px solid rgba(56,189,248,0.3);
  box-shadow: 0 0 24px rgba(56,189,248,0.22), inset 0 0 18px rgba(94,234,212,0.18);
  border-radius: 18px;
  pointer-events: none;
  mix-blend-mode: screen;
  backdrop-filter: blur(2px);
}
#beam {
  position: fixed;
  inset: 0;
  background: conic-gradient(from 120deg at 20% 20%, rgba(56,189,248,0.12), rgba(94,234,212,0.08), rgba(168,85,247,0.12), rgba(56,189,248,0.12));
  opacity: 0.26;
  pointer-events: none;
  animation: rotate 22s linear infinite;
  filter: blur(28px);
}
#glow {
  position: fixed;
  inset: -40%;
  background:
    radial-gradient(circle at 14% 18%, rgba(56,189,248,0.2), transparent 58%),
    radial-gradient(circle at 82% 74%, rgba(34,197,94,0.16), transparent 55%),
    radial-gradient(circle at 50% 60%, rgba(168,85,247,0.12), transparent 60%);
  mix-blend-mode: screen;
  pointer-events: none;
}
#scanlines {
  position: fixed;
  inset: 0;
  background: repeating-linear-gradient( to bottom, rgba(255,255,255,0.04) 0, rgba(255,255,255,0.04) 1px, transparent 1px, transparent 3px);
  opacity: 0.45;
  mix-blend-mode: soft-light;
  pointer-events: none;
  animation: scan 8s linear infinite;
}
#stars {
  position: fixed;
  inset: 0;
  overflow: hidden;
  pointer-events: none;
  z-index: 1;
}
#holo {
  position: fixed;
  width: 360px;
  height: 360px;
  border-radius: 50%;
  top: -120px;
  right: -60px;
  background: conic-gradient(from 45deg, rgba(56,189,248,0.12), rgba(45,212,191,0.26), rgba(168,85,247,0.18), rgba(56,189,248,0.12));
  filter: blur(44px);
  opacity: 0.8;
  animation: rotate 16s linear infinite;
  pointer-events: none;
}
#app {
  position: relative;
  z-index: 2;
  padding: clamp(16px, 2vw, 26px);
  width: min(1320px, calc(100vw - 18px));
  margin: 0 auto;
}
.row {
  display: flex;
  gap: clamp(12px, 2vw, 18px);
  flex-wrap: wrap;
}
.col {
  background: rgba(8,15,30,0.92);
  border-radius: 14px;
  border: 1px solid rgba(148,163,184,0.38);
  box-shadow: 0 0 24px rgba(15,23,42,0.9), 0 0 42px rgba(56,189,248,0.22);
  padding: 14px 16px;
  position: relative;
  overflow: hidden;
}
.col::before {
  content: "";
  position: absolute;
  inset: -60% -40% auto auto;
  width: 240px;
  height: 240px;
  background: radial-gradient(circle at center, rgba(56,189,248,0.16), transparent 58%);
  opacity: 0.6;
  mix-blend-mode: screen;
  animation: floaty 8s ease-in-out infinite;
  pointer-events: none;
}
.col::after {
  content: "";
  position: absolute;
  inset: auto auto -55% -35%;
  width: 220px;
  height: 220px;
  background: radial-gradient(circle at center, rgba(168,85,247,0.16), transparent 58%);
  opacity: 0.5;
  mix-blend-mode: screen;
  animation: floaty 9s ease-in-out infinite reverse;
  pointer-events: none;
}
.col::after {
  transform: rotate(12deg);
}
.col-left {
  flex: 1.4;
  min-width: 320px;
}
.col-right {
  width: 290px;
}
.header {
  display: flex;
  align-items: center;
  justify-content: space-between;
  margin-bottom: 10px;
}
.title {
  font-size: clamp(15px, 2vw, 18px);
  letter-spacing: 0.16em;
  color: #a5b4fc;
  background: linear-gradient(120deg, #38bdf8, #22c55e, #a855f7);
  -webkit-background-clip: text;
  -webkit-text-fill-color: transparent;
  text-shadow: 0 0 18px rgba(56,189,248,0.45);
}
.badge {
  display: inline-flex;
  align-items: center;
  gap: 6px;
  background: linear-gradient(120deg, rgba(56,189,248,0.22), rgba(34,197,94,0.14));
  border: 1px solid rgba(56,189,248,0.5);
  color: #c7f9cc;
  padding: 6px 12px;
  font-size: 12px;
  border-radius: 14px;
  box-shadow: 0 8px 30px rgba(59,130,246,0.35);
}
.badge.pulse {
  background: linear-gradient(120deg, rgba(56,189,248,0.32), rgba(94,234,212,0.18));
  box-shadow: 0 0 24px rgba(34,211,238,0.25);
  animation: breathe 3.6s ease-in-out infinite;
}
.chip-row {
  display: flex;
  gap: 8px;
  align-items: center;
  flex-wrap: wrap;
}
.chip {
  background: rgba(30,41,59,0.8);
  border: 1px solid rgba(94,234,212,0.5);
  color: #e2f3ff;
  padding: 6px 10px;
  border-radius: 999px;
  font-size: 11px;
  box-shadow: 0 8px 24px rgba(8,47,73,0.6);
  white-space: nowrap;
}
.section-title {
  font-size: 13px;
  color: #a5f3fc;
  margin-bottom: 8px;
  letter-spacing: 1px;
}
.card {
  border-radius: 12px;
  border: 1px solid rgba(51,65,85,0.9);
  background: radial-gradient(circle at top left, rgba(34,197,94,0.09), rgba(15,23,42,0.96));
  padding: 10px 12px 12px;
  margin-bottom: 10px;
  position: relative;
  overflow: hidden;
}
.label {
  font-size: 12px;
  color: #94a3b8;
  margin-bottom: 4px;
}
select, button, .btn {
  font-family: inherit;
  font-size: 12px;
  width: 100%;
}
select {
  padding: 6px 8px;
  border-radius: 8px;
  border: 1px solid rgba(148,163,184,0.8);
  background: linear-gradient(120deg, rgba(15,23,42,0.9), rgba(15,23,42,0.86));
  color: #e5e5e5;
}
select:focus {
  outline: none;
  box-shadow: 0 0 0 1px #22c55e;
}
.control-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(220px, 1fr));
  gap: 10px;
  align-items: center;
  margin-top: 6px;
}
.btn {
  flex: 1;
  padding: 7px 10px;
  border-radius: 8px;
  border: 0;
  cursor: pointer;
  text-align: center;
  letter-spacing: 0.06em;
  text-transform: uppercase;
  transition: transform 0.08s ease-out, box-shadow 0.08s ease-out, background 0.15s ease-out;
}
.btn-ghost {
  background: rgba(15,23,42,0.95);
  color: #e5e5e5;
  box-shadow: 0 0 0 1px rgba(148,163,184,0.7);
}
.btn-ghost:hover {
  transform: translateY(-1px);
  box-shadow: 0 0 0 1px #38bdf8, 0 6px 14px rgba(15,23,42,0.9);
}
.btn-primary {
  background: linear-gradient(90deg, #22c55e, #22c55e, #2dd4bf);
  color: #020617;
  box-shadow: 0 6px 18px rgba(16,185,129,0.55);
}
.btn-primary:hover {
  transform: translateY(-1px);
  box-shadow: 0 8px 22px rgba(16,185,129,0.75);
}
.status-line {
  font-size: 12px;
  margin-top: 4px;
  color: #e5e5e5;
}
.status-mode {
  color: #38bdf8;
}
.status-text {
  color: #e5e5e5;
}
.info-grid {
  display: grid;
  grid-template-columns: repeat(2, minmax(0,1fr));
  gap: 6px 12px;
  margin-top: 6px;
}
.info-label {
  font-size: 11px;
  color: #64748b;
}
.info-value {
  font-size: 12px;
  color: #e5e5e5;
}
.info-highlight {
  color: #22c55e;
}
.info-alert {
  color: #fbbf24;
}
.progress-shell {
  margin-top: 10px;
  position: relative;
  height: 20px;
  border-radius: 999px;
  background: radial-gradient(circle at top, #020617, #020617);
  box-shadow: inset 0 0 0 1px rgba(30,64,175,0.9), 0 0 16px rgba(15,23,42,0.9);
  overflow: hidden;
}
.progress-fill {
  position: absolute;
  inset: 1px;
  width: 0%;
  border-radius: 999px;
  background: linear-gradient(90deg, #22c55e, #2dd4bf);
  box-shadow: 0 0 18px rgba(34,197,94,0.9);
  transition: width 0.12s linear, background 0.12s linear, box-shadow 0.12s linear;
}
.progress-shell.flash {
  box-shadow: 0 0 24px rgba(56,189,248,0.9), 0 0 54px rgba(248,250,252,0.45);
}
.progress-shell::before,
.progress-shell::after {
  content: "";
  position: absolute;
  inset: -80%;
  background:
    radial-gradient(circle at 10% 0, rgba(148,163,184,0.0), transparent 40%),
    radial-gradient(circle at 30% 50%, rgba(96,165,250,0.09), transparent 52%),
    radial-gradient(circle at 90% 50%, rgba(45,212,191,0.11), transparent 55%);
  opacity: 0.0;
  mix-blend-mode: screen;
  transition: opacity 0.18s ease-out;
}
.progress-shell.discharge::before,
.progress-shell.discharge::after {
  opacity: 1;
}
.wave-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 10px;
  margin-top: 6px;
}
.wave-box {
  padding: 8px;
  border-radius: 12px;
  background: linear-gradient(120deg, rgba(15,23,42,0.9), rgba(30,41,59,0.8));
  box-shadow: inset 0 0 0 1px rgba(94,234,212,0.35), 0 10px 30px rgba(15,23,42,0.4);
  position: relative;
  overflow: hidden;
}
.wave-box::after {
  content: "";
  position: absolute;
  inset: 0;
  background: radial-gradient(circle at 10% 10%, rgba(94,234,212,0.16), transparent 35%), radial-gradient(circle at 80% 0%, rgba(59,130,246,0.18), transparent 40%);
  opacity: 0.65;
  pointer-events: none;
}
.wave-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  font-size: 12px;
  color: #cbd5f5;
  letter-spacing: 0.5px;
}
.wave-header span:last-child {
  color: #38fbd4;
}
.wave-canvas {
  margin-top: 6px;
  border-radius: 10px;
  width: 100%;
  height: 70px;
  background: radial-gradient(circle at 20% 30%, rgba(56,189,248,0.12), transparent 40%), rgba(2,6,23,0.85);
  box-shadow: inset 0 0 0 1px rgba(148,163,184,0.12), inset 0 0 12px rgba(34,211,238,0.18);
}
@media (max-width: 1180px) {
  .row {
    flex-direction: column;
  }
  .col-left,
  .col-right {
    width: 100%;
    flex: 1 1 auto;
  }
  .chip-row {
    flex-wrap: wrap;
  }
}
@media (max-width: 760px) {
  #app {
    width: 100%;
  }
  .wave-grid {
    grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
  }
  .badge, .chip, .info-value, .info-label {
    font-size: 12px;
  }
  .title {
    font-size: 15px;
  }
}
.footer {
  margin-top: 6px;
  font-size: 11px;
  color: #64748b;
}
.footer span {
  color: #22c55e;
}
.code {
  font-size: 11px;
  color: #94a3b8;
  line-height: 1.8;
}
.badge.neon {
  background: linear-gradient(90deg, rgba(56,189,248,0.3), rgba(94,234,212,0.35));
  color: #e0f2fe;
  box-shadow: 0 0 20px rgba(59,130,246,0.35);
}
.card.holo {
  background: radial-gradient(circle at 20% 10%, rgba(56,189,248,0.12), transparent 30%), radial-gradient(circle at 80% 0%, rgba(34,197,94,0.16), transparent 26%), rgba(15,23,42,0.82);
  border: 1px solid rgba(59,130,246,0.25);
  box-shadow: inset 0 0 0 1px rgba(94,234,212,0.2), 0 10px 30px rgba(2,6,23,0.65);
}
.card.holo::after {
  content: "";
  position: absolute;
  inset: 0;
  border-radius: inherit;
  background: linear-gradient(120deg, rgba(56,189,248,0.12), rgba(34,211,238,0.05));
  opacity: 0;
  transition: opacity 0.3s ease-out;
  pointer-events: none;
}
.card.holo:hover::after {
  opacity: 1;
}
.ionic-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
  gap: 10px;
}
.hyper {
  letter-spacing: 0.8px;
  color: #67e8f9;
  text-shadow: 0 0 14px rgba(103,232,249,0.7);
}
.star {
  position: absolute;
  width: 2px;
  height: 2px;
  background: rgba(148,163,184,0.9);
  border-radius: 50%;
  box-shadow: 0 0 8px rgba(56,189,248,0.8);
  opacity: 0.75;
  animation: twinkle 3s ease-in-out infinite;
}
@keyframes drift { from { background-position: 0 0, 0 0; } to { background-position: 28px 28px, 28px 28px; } }
@keyframes scan { from { transform: translateY(0); } to { transform: translateY(-100%); } }
@keyframes pulse { 0% { opacity: 0.7; transform: scale(1); } 100% { opacity: 1; transform: scale(1.04); } }
@keyframes rotate { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes floaty { 0% { transform: translateY(0); } 50% { transform: translateY(-8px); } 100% { transform: translateY(0); } }
@keyframes breathe { 0% { transform: translateZ(0); box-shadow: 0 0 14px rgba(56,189,248,0.25); } 50% { box-shadow: 0 0 28px rgba(94,234,212,0.32); } 100% { transform: translateZ(0); box-shadow: 0 0 14px rgba(56,189,248,0.25); } }
@keyframes twinkle { 0%, 100% { opacity: 0.35; transform: scale(1); } 50% { opacity: 1; transform: scale(1.8); } }
</style>
</head>
<body>
<div id="aurora"></div>
<div id="grid"></div>
<div id="frame"></div>
<div id="beam"></div>
<div id="glow"></div>
<div id="scanlines"></div>
<div id="stars"></div>
<div id="holo"></div>
<div id="app">
  <div class="header">
    <div>
      <div class="title">NEURO DESKTOP · SYNAPTIC WINDOW AI</div>
    </div>
    <div class="chip-row">
      <div class="chip" id="captureTag">捕获诊断 · 未绑定</div>
      <div class="chip" id="screenTag">屏幕 -</div>
      <div class="chip" id="diskTag">磁盘 -</div>
      <div class="chip pulse" id="profileTag">窗口记忆 · -</div>
    </div>
  </div>
  <div class="row">
    <div class="col col-left">
      <div class="section-title">窗口A 绑定与控制</div>
      <div class="card holo">
        <div class="label hyper">选择作为窗口A的应用窗口</div>
        <div class="control-grid">
          <select id="windowSelect">
            <option value="">正在扫描可见窗口...</option>
          </select>
          <button class="btn btn-ghost" id="btnRefresh">刷新窗口列表</button>
          <button class="btn btn-primary" id="btnStart">绑定窗口A并启动学习模式</button>
        </div>
        <div class="ionic-grid" style="margin-top:8px;">
          <div>
            <div class="info-label">记忆窗口</div>
            <div class="info-value" id="rememberedText">-</div>
          </div>
          <div>
            <div class="info-label">模型分组</div>
            <div class="info-value info-highlight" id="profileText">default</div>
          </div>
        </div>
        <div class="status-line">
          <span class="status-mode" id="modeText">模式: 待机</span><br>
          <span class="status-text" id="statusText">状态: 等待启动</span>
        </div>
      </div>
      <div class="section-title">训练进程 · Loss 感应 · 放电闪动</div>
      <div class="card holo">
        <div class="info-grid">
          <div>
            <div class="info-label">窗口A</div>
            <div class="info-value" id="windowLabel">未绑定</div>
          </div>
          <div>
            <div class="info-label">经验样本</div>
            <div class="info-value"><span id="expCount">0</span></div>
          </div>
          <div>
            <div class="info-label">人类经验</div>
            <div class="info-value info-highlight" id="humanCount">0</div>
          </div>
          <div>
            <div class="info-label">AI 经验</div>
            <div class="info-value" id="aiCount">0</div>
          </div>
          <div>
            <div class="info-label">训练轮次</div>
            <div class="info-value" id="epochText">-</div>
          </div>
          <div>
            <div class="info-label">最近 Loss</div>
            <div class="info-value" id="lossText">-</div>
          </div>
        </div>
        <div class="progress-shell" id="progressShell">
          <div class="progress-fill" id="progressFill"></div>
        </div>
        <div class="info-grid" style="margin-top:10px;">
          <div>
            <div class="info-label">捕获状态</div>
            <div class="info-value" id="captureStatus">未绑定窗口A</div>
          </div>
          <div>
            <div class="info-label">显示器</div>
            <div class="info-value" id="screenInfo">-</div>
          </div>
          <div>
            <div class="info-label">DPI</div>
            <div class="info-value" id="dpiInfo">-</div>
          </div>
          <div>
            <div class="info-label">磁盘空间</div>
            <div class="info-value" id="diskInfo">-</div>
          </div>
        </div>
        <div class="footer">
          键盘控制：<span>Enter</span> = 停止记录并离线优化 · <span>Space</span> = 训练模式 · <span>Esc</span> = 结束运行
        </div>
      </div>
    </div>
    <div class="col col-right">
      <div class="section-title">系统画像</div>
      <div class="card holo">
        <div class="info-label">设备</div>
        <div class="info-value" id="deviceText"></div>
        <div class="info-label" style="margin-top:6px;">视觉理解</div>
        <div class="info-value" id="visionText"></div>
        <div class="info-label" style="margin-top:6px;">时间与轨迹</div>
        <div class="info-value" id="seqText"></div>
        <div class="info-label" style="margin-top:6px;">大脑结构</div>
        <div class="info-value" id="netText"></div>
        <div class="info-label" style="margin-top:6px;">经验池管理</div>
        <div class="info-value code" id="recycleText"></div>
        <div class="info-label" style="margin-top:10px;">实时资源</div>
        <div class="wave-grid">
          <div class="wave-box">
            <div class="wave-header"><span>CPU</span><span id="cpuText">0%</span></div>
            <canvas class="wave-canvas" id="cpuWave" width="320" height="70"></canvas>
          </div>
          <div class="wave-box">
            <div class="wave-header"><span>内存</span><span id="memText">0%</span></div>
            <canvas class="wave-canvas" id="memWave" width="320" height="70"></canvas>
          </div>
          <div class="wave-box">
            <div class="wave-header"><span>GPU</span><span id="gpuText">0%</span></div>
            <canvas class="wave-canvas" id="gpuWave" width="320" height="70"></canvas>
          </div>
          <div class="wave-box">
            <div class="wave-header"><span>显存</span><span id="vramText">0%</span></div>
            <canvas class="wave-canvas" id="vramWave" width="320" height="70"></canvas>
          </div>
        </div>
      </div>
    </div>
  </div>
</div>
<script>
let lastState = null;
let typingModeTimer = null;
let typingStatusTimer = null;
function seedStars() {
  const container = document.getElementById("stars");
  if (!container) return;
  container.innerHTML = "";
  const total = 86;
  for (let i = 0; i < total; i++) {
    const dot = document.createElement("div");
    dot.className = "star";
    dot.style.left = Math.random() * 100 + "%";
    dot.style.top = Math.random() * 100 + "%";
    dot.style.opacity = 0.35 + Math.random() * 0.6;
    dot.style.animationDuration = 2.2 + Math.random() * 2.5 + "s";
    dot.style.animationDelay = Math.random() * 3 + "s";
    container.appendChild(dot);
  }
}
function typeWriter(element, fullText, prefix) {
  if (!element) return;
  if (typingModeTimer && element.id === "modeText") clearInterval(typingModeTimer);
  if (typingStatusTimer && element.id === "statusText") clearInterval(typingStatusTimer);
  let text = "";
  let i = 0;
  let target = prefix ? prefix + fullText : fullText;
  let timer = setInterval(() => {
    i++;
    text = target.slice(0, i);
    element.textContent = text;
    if (i >= target.length) {
      clearInterval(timer);
    }
  }, 18 + Math.random() * 12);
  if (element.id === "modeText") typingModeTimer = timer;
  if (element.id === "statusText") typingStatusTimer = timer;
}
function lossToColor(loss) {
  if (!isFinite(loss) || loss <= 0) return "linear-gradient(90deg,#22c55e,#2dd4bf)";
  let n = Math.tanh(loss);
  if (n < 0) n = 0;
  if (n > 1) n = 1;
  let h = 120 * (1 - n);
  return "linear-gradient(90deg, hsl(" + h + ",90%,52%), hsl(" + (h+30) + ",90%,58%))";
}
const metricHistory = {cpu: [], mem: [], gpu: [], vram: []};
const waveColors = {cpu: "#60a5fa", mem: "#a855f7", gpu: "#22d3ee", vram: "#f59e0b"};
function resizeWaves() {
  const ratio = window.devicePixelRatio || 1;
  ["cpu", "mem", "gpu", "vram"].forEach(name => {
    let canvas = document.getElementById(name + "Wave");
    if (!canvas) return;
    let width = canvas.clientWidth || 320;
    let height = 70;
    canvas.width = Math.max(140, width * ratio);
    canvas.height = height * ratio;
    canvas.style.height = height + "px";
  });
}
function pushWave(name, value) {
  if (!metricHistory[name]) metricHistory[name] = [];
  let v = isFinite(value) ? value : 0;
  if (v < 0) v = 0;
  if (v > 100) v = 100;
  metricHistory[name].push(v);
  if (metricHistory[name].length > 140) metricHistory[name].shift();
  drawWave(name);
}
function drawWave(name) {
  let canvas = document.getElementById(name + "Wave");
  if (!canvas) return;
  let ctx = canvas.getContext("2d");
  const w = canvas.width;
  const h = canvas.height;
  ctx.clearRect(0, 0, w, h);
  ctx.fillStyle = "rgba(2,6,23,0.82)";
  ctx.fillRect(0, 0, w, h);
  ctx.strokeStyle = "rgba(148,163,184,0.25)";
  ctx.lineWidth = 1;
  for (let i = 0; i < 5; i++) {
    let y = (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(0, y);
    ctx.lineTo(w, y);
    ctx.stroke();
  }
  const arr = metricHistory[name] || [];
  if (arr.length < 2) return;
  ctx.strokeStyle = waveColors[name] || "#22d3ee";
  ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    let x = (i / (arr.length - 1)) * w;
    let y = h - (arr[i] / 100) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.strokeStyle = "rgba(255,255,255,0.12)";
  ctx.lineWidth = 1;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    let x = (i / (arr.length - 1)) * w;
    let y = h - (arr[i] / 100) * h;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  let sparkX = Math.random() * w;
  let sparkY = Math.random() * h;
  ctx.fillStyle = "rgba(56,189,248,0.35)";
  ctx.fillRect(sparkX - 1, sparkY - 1, 2, 2);
}
function updateUI(state) {
  let modeEl = document.getElementById("modeText");
  let statusEl = document.getElementById("statusText");
  let winEl = document.getElementById("windowLabel");
  let expEl = document.getElementById("expCount");
  let humanEl = document.getElementById("humanCount");
  let aiEl = document.getElementById("aiCount");
  let epochEl = document.getElementById("epochText");
  let lossEl = document.getElementById("lossText");
  let deviceEl = document.getElementById("deviceText");
  let visionEl = document.getElementById("visionText");
  let seqEl = document.getElementById("seqText");
  let netEl = document.getElementById("netText");
  let recycleEl = document.getElementById("recycleText");
  let captureEl = document.getElementById("captureStatus");
  let screenEl = document.getElementById("screenInfo");
  let dpiEl = document.getElementById("dpiInfo");
  let diskEl = document.getElementById("diskInfo");
  let captureTag = document.getElementById("captureTag");
  let screenTag = document.getElementById("screenTag");
  let diskTag = document.getElementById("diskTag");
  let cpuText = document.getElementById("cpuText");
  let memText = document.getElementById("memText");
  let gpuText = document.getElementById("gpuText");
  let vramText = document.getElementById("vramText");
  let profileEl = document.getElementById("profileText");
  let rememberedEl = document.getElementById("rememberedText");
  let profileTag = document.getElementById("profileTag");
  let shell = document.getElementById("progressShell");
  let fill = document.getElementById("progressFill");
  if (!state) return;
  if (!lastState || lastState.mode !== state.mode) {
    typeWriter(modeEl, "模式: " + state.mode, "");
  } else {
    modeEl.textContent = "模式: " + state.mode;
  }
  if (!lastState || lastState.status !== state.status) {
    typeWriter(statusEl, "状态: " + state.status, "");
  } else {
    statusEl.textContent = "状态: " + state.status;
  }
  winEl.textContent = state.hwnd_title || "未绑定";
  expEl.textContent = state.exp_files || 0;
  humanEl.textContent = state.human_exp || 0;
  aiEl.textContent = state.ai_exp || 0;
  if (state.max_epoch > 0 && state.epoch > 0) {
    epochEl.textContent = state.epoch + " / " + state.max_epoch;
  } else {
    epochEl.textContent = "-";
  }
  if (state.epoch > 0) {
    lossEl.textContent = state.loss.toFixed(6);
  } else {
    lossEl.textContent = "-";
  }
  deviceEl.textContent = state.device_str || "";
  visionEl.textContent = "分辨率 " + state.img_size + " × " + state.img_size;
  seqEl.textContent = "时间步 " + state.seq_len + " · 轨迹点 " + state.path_len;
  netEl.textContent = "Embed " + state.embed_dim + " · Heads " + state.heads + " · 层数 " + state.layers;
  let recycleLine = "经验池文件 " + (state.exp_files || 0) + " 条";
  if (state.recycled > 0) recycleLine += " · 最近回收 " + state.recycled + " 条";
  recycleEl.textContent = recycleLine;
  let captureInfo = state.capture || "未知";
  if (captureEl) {
    captureEl.textContent = captureInfo;
    if (captureInfo.indexOf("就绪") >= 0) captureEl.className = "info-value info-highlight"; else captureEl.className = "info-value info-alert";
  }
  if (captureTag) captureTag.textContent = "捕获诊断 · " + captureInfo;
  if (screenEl) screenEl.textContent = state.screen || "-";
  if (screenTag) screenTag.textContent = "屏幕 " + (state.screen || "-");
  if (dpiEl) dpiEl.textContent = state.dpi || "-";
  if (diskEl) diskEl.textContent = state.disk || "-";
  if (diskTag) diskTag.textContent = "磁盘 " + (state.disk || "-");
  if (profileEl) profileEl.textContent = state.profile || "default";
  if (rememberedEl) rememberedEl.textContent = state.remembered || "-";
  if (profileTag) profileTag.textContent = "窗口记忆 · " + (state.remembered || "-");
  function setWave(val, key, textEl) {
    let pct = isFinite(val) ? val : 0;
    if (pct < 0) pct = 0;
    if (pct > 100) pct = 100;
    if (textEl) textEl.textContent = pct.toFixed(1) + "%";
    pushWave(key, pct);
  }
  setWave(state.cpu, "cpu", cpuText);
  setWave(state.mem, "mem", memText);
  setWave(state.gpu, "gpu", gpuText);
  setWave(state.vram, "vram", vramText);
  let total = state.total_steps || 0;
  let steps = state.steps || 0;
  let pct = 0;
  if (total > 0) {
    pct = Math.max(0, Math.min(100, (steps / total) * 100));
  }
  fill.style.width = pct + "%";
  if (state.epoch > 0 && isFinite(state.loss)) {
    fill.style.background = lossToColor(state.loss);
    fill.style.boxShadow = "0 0 18px rgba(34,197,94,0.8)";
  } else {
    fill.style.background = "linear-gradient(90deg,#22c55e,#2dd4bf)";
  }
  if (lastState && steps > (lastState.steps || 0)) {
    shell.classList.add("flash");
    shell.classList.add("discharge");
    setTimeout(() => {
      shell.classList.remove("flash");
    }, 120);
    setTimeout(() => {
      shell.classList.remove("discharge");
    }, 240);
  }
  lastState = state;
}
function refreshWindows() {
  fetch("/windows").then(r => r.json()).then(data => {
    let sel = document.getElementById("windowSelect");
    sel.innerHTML = "";
    if (!data || !data.windows || data.windows.length === 0) {
      let opt = document.createElement("option");
      opt.value = "";
      opt.textContent = "未找到可用窗口";
      sel.appendChild(opt);
      return;
    }
    let remembered = data.remembered || "";
    let profileTag = document.getElementById("profileTag");
    let rememberedText = document.getElementById("rememberedText");
    let profileText = document.getElementById("profileText");
    if (rememberedText) rememberedText.textContent = remembered || "-";
    if (profileTag) profileTag.textContent = "窗口记忆 · " + (remembered || "-");
    if (profileText && data.profile) profileText.textContent = data.profile;
    let matched = false;
    data.windows.forEach((w, idx) => {
      let opt = document.createElement("option");
      opt.value = idx;
      opt.textContent = idx + ": " + w.title;
      if (!matched && remembered && w.title === remembered) {
        opt.selected = true;
        matched = true;
      }
      sel.appendChild(opt);
    });
  }).catch(()=>{});
}
function bindAndStart() {
  let sel = document.getElementById("windowSelect");
  let idx = sel.value;
  if (idx === "") return;
  fetch("/start", {
    method: "POST",
    headers: {"Content-Type":"application/json"},
    body: JSON.stringify({index: parseInt(idx)})
  });
}
function tickState() {
  fetch("/state").then(r => r.json()).then(state => {
    updateUI(state);
  }).catch(()=>{});
}
function randomDischargePulse() {
  let shell = document.getElementById("progressShell");
  if (!shell) return;
  shell.classList.add("discharge");
  setTimeout(() => shell.classList.remove("discharge"), 180 + Math.random() * 180);
}
window.addEventListener("load", () => {
  document.getElementById("btnRefresh").addEventListener("click", refreshWindows);
  document.getElementById("btnStart").addEventListener("click", bindAndStart);
  resizeWaves();
  seedStars();
  refreshWindows();
  tickState();
  setInterval(tickState, 200);
  setInterval(() => { if (Math.random() > 0.6) randomDischargePulse(); }, 2200);
});
window.addEventListener("resize", () => {
  resizeWaves();
  ["cpu", "mem", "gpu", "vram"].forEach(drawWave);
});
</script>
</body>
</html>'''

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

class Handler(BaseHTTPRequestHandler):
    def _send_json(self, obj, code=200):
        data = json.dumps(obj).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            data = HTML_PAGE.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)
        elif parsed.path == "/state":
            files, human_files, ai_files = scan_experience_files()
            update_realtime_metrics()
            payload = dict(ui_state)
            payload["exp_files"] = len(files)
            payload["human_exp"] = len(human_files)
            payload["ai_exp"] = len(ai_files)
            self._send_json(payload)
        elif parsed.path == "/windows":
            refresh_window_list()
            ws = [{"title": t, "hwnd": int(hwnd)} for hwnd, t in windows_list]
            self._send_json({"windows": ws, "remembered": remembered_window_title, "profile": active_profile})
        else:
            self._send_json({"error": "not_found"}, code=404)
    def do_POST(self):
        parsed = urllib.parse.urlparse(self.path)
        length = int(self.headers.get("Content-Length", "0") or "0")
        body = b""
        if length > 0:
            body = self.rfile.read(length)
        try:
            data = json.loads(body.decode("utf-8")) if body else {}
        except Exception:
            data = {}
        if parsed.path == "/start":
            idx = int(data.get("index", 0))
            bind_window_by_index(idx)
            ui_state["mode"] = "学习模式"
            ui_state["status"] = "采集窗口A与人类鼠标, Enter=离线优化, Space=训练, Esc=结束"
            global worker_thread
            if worker_thread is None or not worker_thread.is_alive():
                state["exit"] = False
                state["optimize"] = False
                state["switch_to_training"] = False
                worker_thread = threading.Thread(target=run_pipeline, daemon=True)
                worker_thread.start()
            self._send_json({"ok": True})
        else:
            self._send_json({"error": "not_found"}, code=404)

def main():
    refresh_window_list()
    port = 8923
    server = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    url = "http://127.0.0.1:" + str(port) + "/"
    try:
        webbrowser.open(url)
    except Exception:
        pass
    while True:
        time.sleep(1)

if __name__ == "__main__":
    main()
