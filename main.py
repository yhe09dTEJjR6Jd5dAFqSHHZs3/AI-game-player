import sys
import os
import time
import json
import random
import threading
import queue
import concurrent.futures
import subprocess
import platform
import psutil
import cv2
import numpy as np
import mss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from pynput import mouse, keyboard
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QProgressBar,
                             QFrame, QMessageBox, QGraphicsDropShadowEffect, QDialog, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint, QRect, QSize, QEvent
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QBrush, QImage, QPixmap
import pyqtgraph as pg
from collections import deque
from paddleocr import PaddleOCR
import importlib.util
import logging

logging.getLogger("ppocr").setLevel(logging.ERROR)
logging.getLogger("ppocr.utils.logging").setLevel(logging.ERROR)
logging.getLogger("ppocr.utility").setLevel(logging.ERROR)


try:
    if torch.cuda.is_available():
        torch.backends.cudnn.conv.fp32_precision = "tf32"
        torch.backends.cuda.matmul.fp32_precision = "ieee"
except Exception:
    pass

BASE_DIR = os.path.join(os.path.expanduser("~"), "Desktop", "AAA")
DATA_DIR = os.path.join(BASE_DIR, "data")
IMG_DIR = os.path.join(DATA_DIR, "images")
MODEL_DIR = os.path.join(BASE_DIR, "models")
LOG_FILE = os.path.join(DATA_DIR, "data.csv")
REGION_FILE = os.path.join(BASE_DIR, "regions.json")

for d in [BASE_DIR, DATA_DIR, IMG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'w', encoding='utf-8') as f:
        f.write("timestamp,img_path,mx,my,click,source,ocr,ocr_age\n")

if not os.path.exists(REGION_FILE):
    with open(REGION_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_W, INPUT_H = 320, 180
SEQ_LEN = 4
MAX_OCR = 16
OCR_CONF_THRESHOLD = 0.6
OCR_AGE_MAX_MS = 5000.0
POOL_MAX_BYTES = 10 * 1024 * 1024 * 1024
CACHE_MAX_BYTES = 256 * 1024 * 1024
pynvml = None
POOL_DATA_CACHE = []
POOL_TOTAL_BYTES_CACHE = 0
POOL_LAST_READ_POS = 0
POOL_PREV_VALS_CACHE = []
POOL_REGION_SIGNATURE = None
POOL_LOCK = threading.Lock()
FORCE_CPU_OCR = False
if importlib.util.find_spec("pynvml") is not None:
    import pynvml

def letterbox(img, new_w, new_h):
    h, w = img.shape[:2]
    scale = min(new_w / w, new_h / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh))
    canvas = np.zeros((new_h, new_w, 3), dtype=np.uint8)
    top = (new_h - nh) // 2
    left = (new_w - nw) // 2
    canvas[top:top+nh, left:left+nw] = resized
    return canvas

def compute_letterbox_params(src_w, src_h, dst_w, dst_h):
    scale = min(dst_w / src_w, dst_h / src_h)
    resized_w, resized_h = int(src_w * scale), int(src_h * scale)
    left = (dst_w - resized_w) // 2
    top = (dst_h - resized_h) // 2
    return scale, left, top, resized_w, resized_h

def get_screen_size():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return monitor['width'], monitor['height'], monitor['left'], monitor['top']

def get_free_vram_bytes():
    if not torch.cuda.is_available():
        return float('inf')
    try:
        free, _ = torch.cuda.mem_get_info()
        return float(free)
    except Exception:
        pass
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                return float(info.free)
        except Exception:
            return 0.0
    return 0.0

SCREEN_W, SCREEN_H, SCREEN_LEFT, SCREEN_TOP = get_screen_size()
LB_SCALE, LB_LEFT, LB_TOP, LB_W, LB_H = compute_letterbox_params(SCREEN_W, SCREEN_H, INPUT_W, INPUT_H)
SCALE_PERCENT = 100
if platform.system() == 'Windows':
    try:
        import ctypes
        SCALE_PERCENT = ctypes.windll.shcore.GetScaleFactorForDevice(0)
    except Exception:
        SCALE_PERCENT = 100

def clamp01(v):
    return max(0.0, min(1.0, v))

def screen_to_letter_norm(x, y):
    lx = (x - SCREEN_LEFT) * LB_SCALE + LB_LEFT
    ly = (y - SCREEN_TOP) * LB_SCALE + LB_TOP
    return clamp01(lx / INPUT_W), clamp01(ly / INPUT_H)

def letter_norm_to_screen(nx, ny):
    lx = clamp01(nx) * INPUT_W
    ly = clamp01(ny) * INPUT_H
    sx = (lx - LB_LEFT) / LB_SCALE + SCREEN_LEFT
    sy = (ly - LB_TOP) / LB_SCALE + SCREEN_TOP
    if SCALE_PERCENT != 100:
        sx = SCREEN_LEFT + (sx - SCREEN_LEFT) * 100.0 / SCALE_PERCENT
        sy = SCREEN_TOP + (sy - SCREEN_TOP) * 100.0 / SCALE_PERCENT
    sx = max(SCREEN_LEFT, min(SCREEN_LEFT + SCREEN_W - 1, sx))
    sy = max(SCREEN_TOP, min(SCREEN_TOP + SCREEN_H - 1, sy))
    return sx, sy

def pack_ocr(vals, deltas=None, ages=None):
    buf = [0]*MAX_OCR
    for i, v in enumerate(vals[:MAX_OCR]):
        buf[i] = float(v)
    delta_buf = [0]*MAX_OCR
    if deltas is not None:
        for i, v in enumerate(deltas[:MAX_OCR]):
            delta_buf[i] = float(v)
    age_buf = [0]*MAX_OCR
    if ages is not None:
        for i, v in enumerate(ages[:MAX_OCR]):
            try:
                norm = clamp01(float(v) / OCR_AGE_MAX_MS)
            except Exception:
                norm = 0.0
            age_buf[i] = norm
    return buf + delta_buf + age_buf

def preprocess_frame(img):
    img = letterbox(img, INPUT_W, INPUT_H)
    detail = cv2.Laplacian(img, cv2.CV_8U)
    stacked = np.concatenate([img, detail[..., :1]], axis=2)
    stacked = stacked.astype(np.float32) / 255.0
    stacked = np.transpose(stacked, (2,0,1))
    return stacked

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.SiLU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_c)
        self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_c)
        self.se = SEBlock(out_c)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_c != out_c:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_c, out_c, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_c)
            )

    def forward(self, x):
        out = F.silu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        out += self.shortcut(x)
        out = F.silu(out)
        return out

class Brain(nn.Module):
    def __init__(self):
        super(Brain, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(4, 32, 5, 2, 2, bias=False),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.MaxPool2d(3, 2, 1)
        )
        self.layer1 = self._make_layer(32, 64, 2, stride=2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.ocr_enc = nn.Sequential(nn.Linear(MAX_OCR*3, 128), nn.SiLU(), nn.Linear(128, 64), nn.SiLU())
        self.ocr_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.gru = nn.GRU(256 + 64, 256, batch_first=True)
        self.dir_head = nn.Linear(256, 9)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )
        self.delta_head = nn.Sequential(nn.Linear(256, 128), nn.SiLU(), nn.Linear(128, 2), nn.Tanh())

    def _make_layer(self, in_c, out_c, blocks, stride):
        layers = [ResBlock(in_c, out_c, stride)]
        for _ in range(1, blocks):
            layers.append(ResBlock(out_c, out_c))
        return nn.Sequential(*layers)

    def forward(self, x, ocr):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        x = self.initial(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avg(x)
        x = torch.flatten(x, 1).view(b, t, -1)
        ocr_feat = self.ocr_enc(ocr)
        attn, _ = self.ocr_attn(ocr_feat, ocr_feat, ocr_feat)
        ocr_feat = 0.5 * (ocr_feat + attn)
        seq = torch.cat([x, ocr_feat], dim=2)
        out, _ = self.gru(seq)
        feat = out[:, -1, :]
        dirs = self.dir_head(feat)
        return self.fc(feat), dirs, self.delta_head(feat)

class ExperiencePool(Dataset):
    def __init__(self):
        self.cache = {}
        self.cache_order = deque()
        self.cache_bytes = 0
        self.cache_limit = CACHE_MAX_BYTES
        self.region_types = self.load_region_types()
        self.data, self.total_bytes = self.load_data()
        self.priorities = np.ones(len(self.data), dtype=np.float32)
        self.min_ts = min([d['ts'] for d in self.data], default=0)
        self.max_ts = max([d['ts'] for d in self.data], default=0)

    def load_region_types(self):
        try:
            with open(REGION_FILE, 'r', encoding='utf-8') as f:
                regions = json.load(f)
                return [r.get('type', 'red') for r in regions]
        except:
            return []

    def load_data(self):
        try:
            entries_by_ts = {}
            if os.path.exists(LOG_FILE):
                with open(LOG_FILE, 'r', encoding='utf-8') as f:
                    f.readline()
                    for line in f:
                        parts = line.strip().split(',')
                        if len(parts) < 6:
                            continue
                        try:
                            ts = int(parts[0])
                            img_path = parts[1]
                            mx, my, click = float(parts[2]), float(parts[3]), float(parts[4])
                            ocr_vals = [float(x) for x in parts[6].split('|') if x != ''] if len(parts) >= 7 else []
                            ocr_age = [float(x) for x in parts[7].split('|') if x != ''] if len(parts) >= 8 else [0.0] * len(ocr_vals)
                            entries_by_ts[ts] = {
                                'ts': ts,
                                'img': img_path,
                                'mx': mx,
                                'my': my,
                                'click': click,
                                'ocr_vals': ocr_vals,
                                'ocr_ages': ocr_age
                            }
                        except Exception:
                            continue

            entries = sorted(entries_by_ts.values(), key=lambda x: x['ts'])
            data = []
            total_bytes = 0
            prev_vals = [0] * len(self.region_types)
            for entry in entries:
                cur_vals = entry['ocr_vals'][:len(self.region_types)] + [0]*max(0, len(self.region_types) - len(entry['ocr_vals']))
                deltas = [cur_vals[i] - prev_vals[i] for i in range(len(self.region_types))]
                prev_vals = cur_vals
                feat = pack_ocr(entry['ocr_vals'], deltas, entry.get('ocr_ages'))
                weight = self.calc_weight(entry['ocr_vals'], deltas)
                size = (os.path.getsize(entry['img']) if os.path.exists(entry['img']) else 0) + 512
                data.append({'ts': entry['ts'], 'img': entry['img'], 'mx': entry['mx'], 'my': entry['my'], 'click': entry['click'], 'ocr': feat, 'weight': weight, 'size': size})
                total_bytes += size
                while total_bytes > POOL_MAX_BYTES and data:
                    dropped = data.pop(0)
                    total_bytes -= dropped.get('size', 0)
            return data, total_bytes
        except Exception:
            return [], 0

    def get_priority_weights(self):
        if not self.data:
            return []
        base = np.array([d.get('weight', 1.0) for d in self.data], dtype=np.float32)
        scaled = (base * (self.priorities + 1e-3)) ** 1.2
        scaled = np.clip(scaled, 1e-3, None)
        max_v = float(np.max(scaled)) if scaled.size > 0 else 1.0
        if max_v > 0:
            scaled = scaled / max_v
        return scaled.tolist()

    def update_priorities(self, indices, losses):
        if len(self.priorities) == 0:
            return
        for idx, loss_v in zip(indices, losses):
            i = int(idx)
            if 0 <= i < len(self.priorities):
                lv = float(max(loss_v, 0.0))
                boosted = 1.0 + lv
                self.priorities[i] = 0.9 * self.priorities[i] + 0.1 * boosted

    def calc_weight(self, ocr, deltas):
        if not self.region_types or not ocr:
            return 1.0
        score = 0.0
        total = 0.0
        for idx, t in enumerate(self.region_types):
            if idx < len(deltas):
                d = deltas[idx]
                total += abs(d)
                if t == 'blue':
                    score += d
                else:
                    score -= d
        norm = total + 1e-3
        direction = score / norm
        base = 1.0 + 0.5 * direction
        magnitude = 1.0 + float(np.log1p(total))
        if score < 0:
            base *= 0.5
        return max(0.05, base * magnitude)

    def cache_image(self, path):
        if path in self.cache:
            return self.cache[path]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        img = preprocess_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        size = img.nbytes
        self.cache[path] = img
        self.cache_order.append((path, size))
        self.cache_bytes += size
        while self.cache_bytes > self.cache_limit and self.cache_order:
            old, s = self.cache_order.popleft()
            if old in self.cache:
                del self.cache[old]
            self.cache_bytes -= s
        return img

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq_imgs = []
        seq_ocr = []
        start = max(0, idx - SEQ_LEN + 1)
        pad = SEQ_LEN - (idx - start + 1)
        for _ in range(pad):
            seq_imgs.append(np.zeros((4, INPUT_H, INPUT_W), dtype=np.float32))
            seq_ocr.append(pack_ocr([]))
        for j in range(start, idx + 1):
            item = self.data[j]
            img = self.cache_image(item['img'])
            seq_imgs.append(img)
            seq_ocr.append(item['ocr'])
        mx, my, click = self.data[idx]['mx'], self.data[idx]['my'], self.data[idx]['click']
        prev = self.data[idx-1] if idx > 0 else self.data[idx]
        dx, dy = mx - prev['mx'], my - prev['my']
        mag = (dx**2 + dy**2) ** 0.5
        if mag < 1e-3:
            direction = 8
        else:
            angle = (np.arctan2(dy, dx) + np.pi) / (2 * np.pi)
            direction = int(angle * 8) % 8
        next1 = self.data[min(len(self.data)-1, idx+1)]
        next2 = self.data[min(len(self.data)-1, idx+2)]
        next3 = self.data[min(len(self.data)-1, idx+3)]
        d1 = torch.tensor([next1['mx']-mx, next1['my']-my], dtype=torch.float32)
        d2 = torch.tensor([next2['mx']-mx, next2['my']-my], dtype=torch.float32)
        d3 = torch.tensor([next3['mx']-mx, next3['my']-my], dtype=torch.float32)
        delta_target = torch.cat([d1, d2, d3])
        recency_ratio = 0.0
        if self.max_ts > self.min_ts:
            recency_ratio = (self.data[idx]['ts'] - self.min_ts) / (self.max_ts - self.min_ts)
        else:
            recency_ratio = (idx + 1) / max(1, len(self.data))
        w = torch.tensor(self.data[idx]['weight'] * (0.5 + 0.5 * recency_ratio), dtype=torch.float32)
        return torch.tensor(np.stack(seq_imgs), dtype=torch.float32), torch.tensor(np.stack(seq_ocr), dtype=torch.float32), torch.tensor([mx, my, click], dtype=torch.float32), torch.tensor(direction, dtype=torch.long), delta_target, w, torch.tensor(idx, dtype=torch.long)

class OptimizerThread(QThread):
    finished_sig = pyqtSignal()
    progress_sig = pyqtSignal(int)
    status_sig = pyqtSignal(str)

    def __init__(self, batch_size=16):
        super().__init__()
        self.batch_size = batch_size

    def run(self):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        ds = ExperiencePool()
        if len(ds) < 10:
            self.finished_sig.emit()
            return

        model = Brain().to(DEVICE)
        weight_path = os.path.join(MODEL_DIR, "brain.pth")
        if os.path.exists(weight_path):
            try:
                model.load_state_dict(torch.load(weight_path, map_location=DEVICE))
            except:
                pass

        opt = optim.AdamW(model.parameters(), lr=8e-4)
        reg_loss = nn.SmoothL1Loss(reduction='none')
        bce = nn.BCELoss(reduction='none')

        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None

        epochs = 3 if os.path.exists(weight_path) else 5
        model.train()

        for ep in range(epochs):
            weights = ds.get_priority_weights()
            sampler = WeightedRandomSampler(weights=weights, num_samples=len(ds), replacement=True) if weights else None
            dl = DataLoader(ds, batch_size=self.batch_size, sampler=sampler, shuffle=sampler is None, num_workers=0)
            self.status_sig.emit(f"正在优化模型: 第 {ep+1}/{epochs} 轮")
            for i, (imgs, ocrs, targs, dirs, delta_t, ws, idxs) in enumerate(dl):
                imgs, ocrs, targs, dirs, delta_t, ws = imgs.to(DEVICE), ocrs.to(DEVICE), targs.to(DEVICE), dirs.to(DEVICE), delta_t.to(DEVICE), ws.to(DEVICE)
                opt.zero_grad(set_to_none=True)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        out, logits, delta_out = model(imgs, ocrs)
                        l_pos = reg_loss(out[:, :2], targs[:, :2]).mean(dim=1)
                        l_click = bce(out[:, 2], targs[:, 2])
                        l_dir = F.cross_entropy(logits, dirs, reduction='none')
                        l_delta = reg_loss(delta_out, delta_t).mean(dim=1)
                        comb = l_pos + l_click + 0.5 * l_dir + 0.3 * l_delta
                        loss = (comb * ws).sum() / (ws.sum() + 1e-6)
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    out, logits, delta_out = model(imgs, ocrs)
                    l_pos = reg_loss(out[:, :2], targs[:, :2]).mean(dim=1)
                    l_click = bce(out[:, 2], targs[:, 2])
                    l_dir = F.cross_entropy(logits, dirs, reduction='none')
                    l_delta = reg_loss(delta_out, delta_t).mean(dim=1)
                    comb = l_pos + l_click + 0.5 * l_dir + 0.3 * l_delta
                    loss = (comb * ws).sum() / (ws.sum() + 1e-6)
                    loss.backward()
                    opt.step()

                with torch.no_grad():
                    ds.update_priorities(idxs.cpu().numpy(), (comb.detach().cpu().numpy()))

            self.progress_sig.emit(int((ep + 1) / epochs * 100))

        torch.save(model.state_dict(), weight_path)
        self.status_sig.emit("优化完成")
        self.finished_sig.emit()

class SaveWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=256)
        self.running = True

    def enqueue(self, path, img, log_line):
        if self.q.full():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                return
        try:
            self.q.put_nowait((path, img, log_line))
        except queue.Full:
            pass

    def run(self):
        while self.running or not self.q.empty():
            try:
                path, img, log_line = self.q.get(timeout=0.1)
                try:
                    cv2.imwrite(path, img)
                except Exception:
                    pass
                try:
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        f.write(log_line)
                except Exception:
                    pass
                self.q.task_done()
            except queue.Empty:
                continue

    def stop(self):
        self.running = False
        while not self.q.empty():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break

class NvidiaSmiThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.result = (0.0, 0.0, 1.0)
        self.startup_info = None

    def run(self):
        while self.running:
            try:
                if self.startup_info is None and platform.system() == 'Windows':
                    si = subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    self.startup_info = si
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"], startupinfo=self.startup_info)
                self.result = tuple(map(float, o.decode('utf-8').strip().split(',')))
            except Exception:
                pass
            time.sleep(1.0)

    def stop(self):
        self.running = False

class DataWorker(QThread):
    ocr_result = pyqtSignal(object)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True
        self.save_worker = SaveWorker()
        self.save_worker.start()
        self.regions = []
        self.prev_rois = []
        self.prev_vals = []
        self.prev_change = []
        self.histories = []
        self.last_read_tick = []
        self.stable_counts = []
        self.last_update_time = []
        self.tick = 0
        self.force_cpu = FORCE_CPU_OCR
        use_gpu = torch.cuda.is_available() and not self.force_cpu
        self.ocr = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, gpu_mem=500 if use_gpu else 0)
        self.ocr_gpu_enabled = use_gpu
        self.ocr_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.ocr_lock = threading.Lock()
        self.ocr_tasks = []
        self.reload_regions()

    def release_ocr_resources(self):
        try:
            for i, t in enumerate(self.ocr_tasks):
                if t is not None and t.done():
                    continue
                self.ocr_tasks[i] = None
            with self.ocr_lock:
                if self.ocr is not None:
                    try:
                        del self.ocr
                    except Exception:
                        pass
                self.ocr = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def restore_ocr(self):
        try:
            use_gpu = torch.cuda.is_available() and not self.force_cpu
            with self.ocr_lock:
                self.ocr = PaddleOCR(use_angle_cls=False, use_gpu=use_gpu, gpu_mem=500 if use_gpu else 0)
                self.ocr_gpu_enabled = use_gpu
        except Exception:
            with self.ocr_lock:
                self.ocr = None
            self.ocr_gpu_enabled = False

    def set_force_cpu(self, force_cpu):
        self.force_cpu = force_cpu
        self.release_ocr_resources()
        self.restore_ocr()

    def reload_regions(self):
        try:
            with open(REGION_FILE, 'r', encoding='utf-8') as f:
                self.regions = json.load(f)
        except:
            self.regions = []
        self.prev_rois = [None] * len(self.regions)
        self.prev_vals = [None] * len(self.regions)
        self.prev_change = [0.0] * len(self.regions)
        self.histories = [deque(maxlen=5) for _ in self.regions]
        self.last_read_tick = [0] * len(self.regions)
        self.stable_counts = [0] * len(self.regions)
        now_ts = time.time()
        self.last_update_time = [now_ts] * len(self.regions)
        self.ocr_tasks = [None] * len(self.regions)

    def read_ocr(self, roi):
        try:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
            min_side = min(gray.shape[:2])
            scale = 3 if min_side < 40 else 2 if min_side < 80 else 1
            proc = cv2.resize(gray, None, fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)
            _, proc = cv2.threshold(proc, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            proc = cv2.cvtColor(proc, cv2.COLOR_GRAY2BGR)
            with self.ocr_lock:
                if self.ocr is None:
                    return None
                result = self.ocr.ocr(proc, cls=False)
            digits = ''
            best_conf = 0.0
            if result and len(result) > 0:
                texts = []
                for item in result[0]:
                    if len(item) <= 1 or len(item[1]) < 2:
                        continue
                    text, conf = item[1][0], float(item[1][1])
                    if conf >= OCR_CONF_THRESHOLD:
                        texts.append(text)
                        best_conf = max(best_conf, conf)
                digits = ''.join([c for c in ''.join(texts) if c.isdigit()])
            if digits and best_conf >= OCR_CONF_THRESHOLD:
                return int(digits), best_conf
            return None
        except:
            return None

    def run(self):
        while self.running:
            try:
                data = self.queue.get(timeout=0.1)
                full_img, small_img, mx, my, click, source, save = data
                self.tick += 1

                ocr_vals = []
                ocr_ages = []
                if self.regions:
                    for idx, r in enumerate(self.regions):
                        try:
                            roi = full_img[r['y']:r['y']+r['h'], r['x']:r['x']+r['w']]
                            if roi.size == 0:
                                raise ValueError
                            if idx >= len(self.prev_rois):
                                self.prev_rois.append(None)
                                self.prev_vals.append(None)
                                self.prev_change.append(0.0)
                                self.histories.append(deque(maxlen=5))
                                self.last_read_tick.append(0)
                                self.stable_counts.append(0)
                                self.last_update_time.append(time.time())
                            raw = 1.0 if self.prev_rois[idx] is None else float(np.mean(cv2.absdiff(roi, self.prev_rois[idx]))) / 255.0
                            diff = 0.3 * self.prev_change[idx] + 0.7 * raw
                            self.prev_change[idx] = diff
                            self.prev_rois[idx] = roi
                            prev_v = self.prev_vals[idx]
                            need_read = diff >= 0.002 or prev_v is None or self.tick - self.last_read_tick[idx] >= 3
                            candidate = prev_v
                            candidate_new = False
                            if idx < len(self.ocr_tasks) and self.ocr_tasks[idx] is not None and self.ocr_tasks[idx].done():
                                try:
                                    res = self.ocr_tasks[idx].result()
                                except Exception:
                                    res = None
                                if res is not None:
                                    candidate = res[0]
                                    candidate_new = True
                                    self.last_update_time[idx] = time.time()
                                    self.last_read_tick[idx] = self.tick
                                self.ocr_tasks[idx] = None
                            if need_read and (idx >= len(self.ocr_tasks) or self.ocr_tasks[idx] is None):
                                if idx >= len(self.ocr_tasks):
                                    self.ocr_tasks.append(None)
                                self.ocr_tasks[idx] = self.ocr_pool.submit(self.read_ocr, roi.copy())
                                self.last_read_tick[idx] = self.tick
                            if candidate is None:
                                candidate = 0
                            elif candidate_new:
                                self.last_update_time[idx] = time.time()
                            self.histories[idx].append(candidate)
                            smoothed = int(round(float(np.median(self.histories[idx])))) if self.histories[idx] else candidate
                            stable = self.stable_counts[idx] if idx < len(self.stable_counts) else 0
                            if prev_v is not None and abs(smoothed - prev_v) > 1 and diff < 0.01 and stable < 5:
                                smoothed = prev_v
                            final_v = max(0, smoothed)
                            self.prev_vals[idx] = final_v
                            if final_v == prev_v:
                                self.stable_counts[idx] = stable + 1
                            else:
                                self.stable_counts[idx] = 0
                            ocr_vals.append(final_v)
                            age_ms = int((time.time() - self.last_update_time[idx]) * 1000) if idx < len(self.last_update_time) else 0
                            ocr_ages.append(max(0, age_ms))
                        except:
                            ocr_vals.append(self.prev_vals[idx] if idx < len(self.prev_vals) else 0)
                            age_ms = int((time.time() - self.last_update_time[idx]) * 1000) if idx < len(self.last_update_time) else 0
                            ocr_ages.append(max(0, age_ms))
                    self.ocr_result.emit({'values': ocr_vals, 'ages': ocr_ages, 'ts': int(time.time() * 1000)})
                else:
                    self.ocr_result.emit({'values': [], 'ages': [], 'ts': int(time.time() * 1000)})

                if save:
                    ts = str(int(time.time() * 1000))
                    name = f"{ts}.jpg"
                    path = os.path.join(IMG_DIR, name)
                    ocr_txt = "|".join([str(int(v)) for v in ocr_vals]) if ocr_vals else ""
                    age_txt = "|".join([str(int(a)) for a in ocr_ages]) if ocr_ages else ""
                    log_line = f"{ts},{path},{mx:.5f},{my:.5f},{click:.2f},{source},{ocr_txt},{age_txt}\n"
                    self.save_worker.enqueue(path, small_img, log_line)

                self.queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                pass

    def stop(self):
        self.running = False
        self.save_worker.stop()
        self.save_worker.join()
        self.ocr_pool.shutdown(wait=False)
        self.wait()

class InferenceThread(QThread):
    def __init__(self, owner):
        super().__init__()
        self.owner = owner
        self.running = True

    def stop(self):
        self.running = False

    def run(self):
        while self.running:
            self.owner.loop()
            time.sleep(0.001)

class SciFiPlot(pg.PlotWidget):
    def __init__(self, title, color):
        super().__init__()
        self.setBackground('#050a0f')
        self.setTitle(title, color=color, size='10pt')
        self.showGrid(x=True, y=True, alpha=0.3)
        self.setYRange(0, 100)
        self.getPlotItem().hideAxis('bottom')
        self.getPlotItem().hideAxis('left')
        
class Overlay(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool | Qt.WindowTransparentForInput)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(SCREEN_LEFT, SCREEN_TOP, SCREEN_W, SCREEN_H)
        self.regions = []
        self.values = []
        self.colors = {
            'red': QColor(255, 50, 50),
            'blue': QColor(50, 50, 255)
        }
        self.reload()

    def reload(self):
        try:
            with open(REGION_FILE, 'r', encoding='utf-8') as f:
                self.regions = json.load(f)
            self.values = [0] * len(self.regions)
        except:
            self.regions = []
        self.update()

    def update_vals(self, vals):
        payload = vals.get('values', []) if isinstance(vals, dict) else vals
        if len(payload) == len(self.regions):
            self.values = payload
            self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        font = QFont("SimHei", 10, QFont.Bold)
        p.setFont(font)
        
        for i, r in enumerate(self.regions):
            c = self.colors.get(r['type'], QColor(255,255,255))
            p.setPen(QPen(c, 2))
            p.setBrush(Qt.NoBrush)
            p.drawRect(r['x'], r['y'], r['w'], r['h'])
            
            val = self.values[i] if i < len(self.values) else 0
            p.setPen(c)
            p.drawText(r['x'], r['y'] - 5, str(val))

class RegionEditor(QDialog):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint | Qt.Tool)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setGeometry(SCREEN_LEFT, SCREEN_TOP, SCREEN_W, SCREEN_H)
        self.setCursor(Qt.CrossCursor)
        self.regions = []
        self.current_type = 'red'
        self.start_p = None
        self.temp_rect = None
        try:
            with open(REGION_FILE, 'r', encoding='utf-8') as f:
                self.regions = json.load(f)
        except: pass
        
    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))
        
        colors = {
            'red': (QColor(255, 50, 50), "红框(越小越好)"),
            'blue': (QColor(50, 50, 255), "蓝框(越大越好)")
        }
        
        for r in self.regions:
            c, _ = colors.get(r['type'], (QColor(255,255,255), ""))
            p.setPen(QPen(c, 2))
            p.setBrush(QColor(c.red(), c.green(), c.blue(), 50))
            p.drawRect(r['x'], r['y'], r['w'], r['h'])
            
        if self.temp_rect:
            c, _ = colors[self.current_type]
            p.setPen(QPen(c, 2, Qt.DashLine))
            p.setBrush(Qt.NoBrush)
            p.drawRect(self.temp_rect)

        p.setPen(QColor(0, 240, 255))
        p.setFont(QFont("SimHei", 14, QFont.Bold))
        cur_c, cur_txt = colors[self.current_type]
        info = f"当前模式: {cur_txt} | [1-2]切换类型 | 鼠标拖拽创建 | 右键删除 | 回车保存 | ESC退出"
        detail = "红框=数值越小越好 | 蓝框=数值越大越好 | 请精确框选需要识别的数字或监测区域"
        p.drawText(20, 40, info)
        p.drawText(20, 70, detail)

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.start_p = e.pos()
        elif e.button() == Qt.RightButton:
            for i in range(len(self.regions)-1, -1, -1):
                r = self.regions[i]
                if QRect(r['x'], r['y'], r['w'], r['h']).contains(e.pos()):
                    del self.regions[i]
                    self.update()
                    break

    def mouseMoveEvent(self, e):
        if self.start_p:
            self.temp_rect = QRect(self.start_p, e.pos()).normalized()
            self.update()

    def mouseReleaseEvent(self, e):
        if self.start_p and self.temp_rect:
            if self.temp_rect.width() > 5 and self.temp_rect.height() > 5:
                self.regions.append({
                    'type': self.current_type,
                    'x': self.temp_rect.x(), 'y': self.temp_rect.y(),
                    'w': self.temp_rect.width(), 'h': self.temp_rect.height()
                })
            self.start_p = None
            self.temp_rect = None
            self.update()

    def keyPressEvent(self, e):
        if e.key() == Qt.Key_1: self.current_type = 'red'
        elif e.key() == Qt.Key_2: self.current_type = 'blue'
        elif e.key() in [Qt.Key_Return, Qt.Key_Enter]:
            with open(REGION_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.regions, f)
            self.accept()
        elif e.key() == Qt.Key_Escape:
            self.reject()
        self.update()

class MainWin(QMainWindow):
    key_signal = pyqtSignal(object)
    click_signal = pyqtSignal(object, bool)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("神经接口系统")
        self.resize(1000, 600)
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.Window)
        self.setStyleSheet("QMainWindow {background-color: #050a0f; color: #00f0ff; font-family: 'SimHei';}")
        
        self.brain = Brain().to(DEVICE)
        self.load_model()
        
        self.mouse = mouse.Controller()

        self.queue = queue.Queue(maxsize=1)
        self.worker = DataWorker(self.queue)
        self.worker.start()

        self.overlay = Overlay()
        self.overlay.show()
        self.worker.ocr_result.connect(self.overlay.update_vals)
        self.worker.ocr_result.connect(self.update_ocr)

        self.mode = "LEARNING"
        self.smooth_x, self.smooth_y = 0.5, 0.5
        self.dragging = False
        self.last_record = 0
        self.mouse_pressed = False
        self.latest_ocr = pack_ocr([])
        self.prev_ocr_vals = [0]*MAX_OCR
        self.seq_frames = deque(maxlen=SEQ_LEN)
        self.seq_ocr = deque(maxlen=SEQ_LEN)
        self.press_thresh = 0.7
        self.release_thresh = 0.3
        self.min_press = 0.12
        self.last_press_time = 0.0
        self.release_frames = 0
        self.press_frames = 0
        self.nvml_checked = False
        self.nvml_handle = None
        self.stop_flag = threading.Event()
        self.ocr_forced_cpu = False
        self.nvidia_thread = None
        self.window_dragging = False
        self.window_drag_pos = QPoint()
        self.vram_auto_cpu = False

        self.key_signal.connect(self.handle_key_event)
        self.click_signal.connect(self.handle_click_event)

        self.init_ui()
        self.setup_listeners()

        self.infer_thread = InferenceThread(self)
        self.infer_thread.start()

        self.stat_timer = QTimer()
        self.stat_timer.timeout.connect(self.update_stats)
        self.stat_timer.start(500)

    def load_model(self):
        path = os.path.join(MODEL_DIR, "brain.pth")
        if os.path.exists(path):
            try:
                self.brain.load_state_dict(torch.load(path, map_location=DEVICE))
                self.brain.eval()
            except: pass

    def init_ui(self):
        c = QWidget()
        self.setCentralWidget(c)
        main_layout = QVBoxLayout(c)
        
        frame = QFrame()
        frame.setStyleSheet("background: rgba(10, 20, 30, 200); border: 1px solid #00f0ff; border-radius: 8px;")
        eff = QGraphicsDropShadowEffect()
        eff.setBlurRadius(15)
        eff.setColor(QColor(0, 240, 255, 80))
        frame.setGraphicsEffect(eff)

        fl = QVBoxLayout(frame)

        header_widget = QWidget()
        header_widget.setObjectName("header_widget")
        header_widget.installEventFilter(self)
        self.header_widget = header_widget
        header_layout = QHBoxLayout(header_widget)
        header_layout.setContentsMargins(10, 10, 10, 10)
        self.mode_lbl = QLabel("模式: 学习模式")
        self.mode_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #00ff00;")
        header_layout.addWidget(self.mode_lbl)
        header_layout.addStretch()

        btn_style = """
            QPushButton {background: #001a26; color: #00f0ff; border: 1px solid #00f0ff; border-radius: 6px; padding: 6px 12px; font-weight: bold;}
            QPushButton:hover {background: #00f0ff; color: #001a26;}
        """
        self.btn_min = QPushButton("-")
        self.btn_min.setFixedWidth(32)
        self.btn_min.setStyleSheet(btn_style)
        self.btn_min.clicked.connect(self.showMinimized)
        self.btn_close = QPushButton("×")
        self.btn_close.setFixedWidth(32)
        self.btn_close.setStyleSheet(btn_style)
        self.btn_close.clicked.connect(self.close_app)
        header_layout.addWidget(self.btn_min)
        header_layout.addWidget(self.btn_close)
        fl.addWidget(header_widget)
        
        grid = QGridLayout()
        self.lbl_cpu = QLabel("处理器: 0%")
        self.lbl_mem = QLabel("内存: 0%")
        self.lbl_gpu = QLabel("显卡: 0%")
        self.lbl_vram = QLabel("显存: 0GB")
        self.lbl_disk = QLabel("磁盘: 0TB")
        self.lbl_res = QLabel("分辨率: 0x0")
        
        for i, l in enumerate([self.lbl_cpu, self.lbl_mem, self.lbl_gpu, self.lbl_vram, self.lbl_disk, self.lbl_res]):
            l.setStyleSheet("color: #00f0ff; font-size: 14px; padding: 5px;")
            grid.addWidget(l, i//3, i%3)
        fl.addLayout(grid)
        
        gl = QGridLayout()
        self.p_cpu = SciFiPlot("处理器", '#00ff00')
        self.p_mem = SciFiPlot("内存", '#00ffff')
        self.p_gpu = SciFiPlot("显卡", '#ff0055')
        self.p_vram = SciFiPlot("显存", '#ffaa00')
        
        self.d_cpu, self.d_mem, self.d_gpu, self.d_vram = [], [], [], []
        for x in [self.d_cpu, self.d_mem, self.d_gpu, self.d_vram]:
            x.extend([0]*100)
            
        gl.addWidget(self.p_cpu, 0, 0)
        gl.addWidget(self.p_mem, 0, 1)
        gl.addWidget(self.p_gpu, 1, 0)
        gl.addWidget(self.p_vram, 1, 1)
        fl.addLayout(gl, stretch=1)
        
        btns = QHBoxLayout()
        b_rec = QPushButton("区域识别")
        b_opt = QPushButton("优化模型")
        
        for b in [b_rec, b_opt]:
            b.setStyleSheet("""
                QPushButton {background: #002233; color: #00f0ff; border: 1px solid #00f0ff; padding: 10px; font-weight: bold; font-size: 14px;}
                QPushButton:hover {background: #004455; color: white;}
            """)
        
        b_rec.clicked.connect(self.do_select)
        b_opt.clicked.connect(self.do_opt)
        btns.addWidget(b_rec)
        btns.addWidget(b_opt)
        fl.addLayout(btns)

        self.bar = QProgressBar()
        self.bar.setStyleSheet("QProgressBar {border: 1px solid #00f0ff; text-align: center; color: white;} QProgressBar::chunk {background-color: #00f0ff;}")
        self.bar.hide()
        fl.addWidget(self.bar)
        
        fl.addWidget(QLabel("[空格] 切换训练 | [ESC] 退出 | 其他键中断训练", alignment=Qt.AlignCenter))
        main_layout.addWidget(frame)

    def setup_listeners(self):
        self.k_listen = keyboard.Listener(on_press=self.on_key)
        self.k_listen.start()
        self.m_listen = mouse.Listener(on_click=self.on_click)
        self.m_listen.start()

    def on_click(self, x, y, b, p):
        self.click_signal.emit(b, p)

    def on_key(self, key):
        self.key_signal.emit(key)

    def handle_click_event(self, b, p):
        if p and b == mouse.Button.left:
            self.mouse_pressed = True
        elif not p and b == mouse.Button.left:
            self.mouse_pressed = False

    def handle_key_event(self, key):
        if key == keyboard.Key.esc:
            self.stop_flag.set()
            self.close_app()
        elif key == keyboard.Key.space:
            if self.mode == "LEARNING":
                self.mode = "TRAINING"
                self.stop_flag.clear()
            elif self.mode == "TRAINING":
                self.mode = "LEARNING"
                self.stop_flag.set()
            self.update_mode()
        else:
            if self.mode == "TRAINING":
                self.mode = "LEARNING"
                self.stop_flag.set()
                self.update_mode()

    def do_select(self):
        if self.mode != "LEARNING": return
        self.mode = "SELECT"
        self.overlay.hide()
        dlg = RegionEditor()
        if dlg.exec_() == QDialog.Accepted:
            self.worker.reload_regions()
            self.overlay.reload()
        self.overlay.show()
        self.mode = "LEARNING"

    def do_opt(self):
        if self.mode != "LEARNING": return
        self.worker.release_ocr_resources()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        batch_size = 16
        free_vram = get_free_vram_bytes()
        if torch.cuda.is_available() and free_vram < 1024**3:
            batch_size = 8
            self.ocr_forced_cpu = True
            self.worker.set_force_cpu(True)
        else:
            self.ocr_forced_cpu = False
        self.mode = "OPTIMIZING"
        self.update_mode()
        self.bar.show()
        self.bar.setValue(0)
        self.bar.setFormat("正在优化: %p%")
        self.opt_thread = OptimizerThread(batch_size=batch_size)
        self.opt_thread.progress_sig.connect(self.bar.setValue)
        self.opt_thread.status_sig.connect(self.update_opt_status)
        self.opt_thread.finished_sig.connect(self.opt_done)
        self.opt_thread.start()

    def opt_done(self):
        if self.ocr_forced_cpu and get_free_vram_bytes() >= 1024**3:
            self.worker.set_force_cpu(False)
            self.ocr_forced_cpu = False
        self.worker.restore_ocr()
        self.load_model()
        self.bar.hide()
        QMessageBox.information(self, "完成", "神经网络优化完成。")
        self.mode = "LEARNING"
        self.update_mode()

    def update_mode(self):
        txt = "学习模式"
        c = "#00ff00"
        if self.mode == "TRAINING":
            txt = "训练模式"
            c = "#ff0000"
        elif self.mode == "OPTIMIZING":
            txt = "正在优化..."
            c = "#ffff00"
        self.mode_lbl.setText(f"模式: {txt}")
        self.mode_lbl.setStyleSheet(f"font-size: 22px; font-weight: bold; color: {c};")

    def update_opt_status(self, text):
        self.bar.setFormat(f"{text} - %p%")

    def update_ocr(self, vals):
        raw_vals = []
        ages = []
        if isinstance(vals, dict):
            raw_vals = vals.get('values', [])
            ages = vals.get('ages', [])
        else:
            raw_vals = vals
        full_vals = [0]*MAX_OCR
        deltas = [0]*MAX_OCR
        age_buf = [0]*MAX_OCR
        for i in range(MAX_OCR):
            v = raw_vals[i] if i < len(raw_vals) else 0
            a = ages[i] if i < len(ages) else 0
            full_vals[i] = v
            deltas[i] = v - self.prev_ocr_vals[i]
            self.prev_ocr_vals[i] = v
            age_buf[i] = a
        self.latest_ocr = pack_ocr(full_vals, deltas, age_buf)
        self.seq_ocr.append(self.latest_ocr)

    def update_stats(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent

        gpu_display = "N/A"
        vram_display = "N/A"
        gpu_u, vram_u, vram_t = 0, 0, 1
        if torch.cuda.is_available():
            if not self.nvml_checked and pynvml is not None:
                try:
                    pynvml.nvmlInit()
                    if pynvml.nvmlDeviceGetCount() > 0:
                        self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                except Exception:
                    self.nvml_handle = None
                self.nvml_checked = True
            if self.nvml_handle is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                    mem_info = pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                    gpu_u = float(util.gpu)
                    vram_u = mem_info.used / (1024 * 1024)
                    vram_t = max(mem_info.total / (1024 * 1024), 1)
                    gpu_display = f"{gpu_u}%"
                    vram_display = f"{vram_u/1024:.1f}/{vram_t/1024:.1f}GB"
                except Exception:
                    pass
            elif pynvml is not None and platform.system() == 'Windows':
                if self.nvidia_thread is None:
                    self.nvidia_thread = NvidiaSmiThread()
                    self.nvidia_thread.start()
                stats = self.nvidia_thread.result
                gpu_u, vram_u, vram_t = stats if len(stats) == 3 else (0, 0, 1)
                gpu_display = f"{gpu_u}%"
                vram_total = max(vram_t, 1)
                vram_display = f"{vram_u/1024:.1f}/{vram_total/1024:.1f}GB"

        if torch.cuda.is_available():
            vram_limit_mb = 3500
            vram_recover_mb = 2800
            if vram_u >= vram_limit_mb and not self.worker.force_cpu:
                self.worker.set_force_cpu(True)
                self.vram_auto_cpu = True
                self.ocr_forced_cpu = True
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
            elif self.vram_auto_cpu and vram_u < vram_recover_mb:
                self.worker.set_force_cpu(False)
                self.vram_auto_cpu = False
                self.ocr_forced_cpu = False

        disk = psutil.disk_usage(os.path.abspath(os.sep))
        scale = SCALE_PERCENT
        if platform.system() == 'Windows':
            try:
                import ctypes
                scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
            except Exception:
                scale = SCALE_PERCENT

        self.lbl_cpu.setText(f"处理器: {cpu}%")
        self.lbl_mem.setText(f"内存: {mem}%")
        self.lbl_gpu.setText(f"显卡: {gpu_display}")
        vram_hint = " (OCR已切换CPU)" if self.worker.force_cpu else ""
        self.lbl_vram.setText(f"显存: {vram_display}{vram_hint}")
        self.lbl_disk.setText(f"磁盘可用: {disk.free/(1024**4):.2f}TB")
        self.lbl_res.setText(f"分辨率: {SCREEN_W}x{SCREEN_H} ({scale}%)")

        for l, v in zip([self.d_cpu, self.d_mem, self.d_gpu, self.d_vram], [cpu, mem, gpu_u, (vram_u/vram_t)*100 if vram_t else 0]):
            l.pop(0)
            l.append(v)

        self.p_cpu.plot(self.d_cpu, pen=pg.mkPen('#00ff00', width=2), clear=True)
        self.p_mem.plot(self.d_mem, pen=pg.mkPen('#00ffff', width=2), clear=True)
        self.p_gpu.plot(self.d_gpu, pen=pg.mkPen('#ff0055', width=2), clear=True)
        self.p_vram.plot(self.d_vram, pen=pg.mkPen('#ffaa00', width=2), clear=True)

    def loop(self):
        if self.mode in ["OPTIMIZING", "SELECT"]: return

        if self.stop_flag.is_set() and self.mode == "TRAINING":
            self.mode = "LEARNING"
            self.update_mode()
            if self.dragging:
                self.mouse.release(mouse.Button.left)
                self.dragging = False
            self.press_frames = 0
            self.release_frames = 0
            return

        try:
            with mss.mss() as sct:
                monitor = sct.monitors[1]
                img = np.array(sct.grab(monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = preprocess_frame(rgb)
            self.seq_frames.append(proc)
            self.seq_ocr.append(self.latest_ocr)

            mx, my = self.mouse.position
            nx, ny = screen_to_letter_norm(mx, my)
            click = 1.0 if self.mouse_pressed else 0.0
            source = "USER"

            if self.mode == "TRAINING":
                seq_imgs = list(self.seq_frames)
                seq_ocr = list(self.seq_ocr)
                if self.stop_flag.is_set():
                    return
                while len(seq_imgs) < SEQ_LEN:
                    seq_imgs.insert(0, np.zeros((4, INPUT_H, INPUT_W), dtype=np.float32))
                    seq_ocr.insert(0, pack_ocr([]))
                seq_imgs = torch.tensor(np.stack(seq_imgs)[None, ...], dtype=torch.float32, device=DEVICE)
                seq_ocr = torch.tensor(np.stack(seq_ocr)[None, ...], dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            out, logits, delta_out = self.brain(seq_imgs, seq_ocr)
                    else:
                        out, logits, delta_out = self.brain(seq_imgs, seq_ocr)

                px, py, pc = out[0].cpu().numpy()
                delta_pred = delta_out[0].cpu().numpy()
                dx, dy = float(delta_pred[0]), float(delta_pred[1])
                base_x, base_y = clamp01(float(px)), clamp01(float(py))
                delta_x = clamp01(self.smooth_x + dx)
                delta_y = clamp01(self.smooth_y + dy)
                target_x = 0.5 * (base_x + delta_x)
                target_y = 0.5 * (base_y + delta_y)
                conf = float(torch.softmax(logits, dim=1)[0].max().cpu())
                blend = 0.3 + 0.7 * conf
                self.smooth_x = (1 - blend) * self.smooth_x + blend * target_x
                self.smooth_y = (1 - blend) * self.smooth_y + blend * target_y

                if self.stop_flag.is_set() or self.mode != "TRAINING":
                    return

                sx, sy = letter_norm_to_screen(self.smooth_x, self.smooth_y)
                self.mouse.position = (int(sx), int(sy))
                now = time.time()
                if pc >= self.press_thresh:
                    self.press_frames += 1
                else:
                    self.press_frames = 0
                if pc <= self.release_thresh:
                    self.release_frames += 1
                else:
                    self.release_frames = 0
                if not self.dragging and self.press_frames >= 2:
                    self.mouse.press(mouse.Button.left)
                    self.dragging = True
                    self.last_press_time = now
                    self.release_frames = 0
                elif self.dragging and (pc <= 0.1 or (self.release_frames >= 3 and now - self.last_press_time >= self.min_press)):
                    self.mouse.release(mouse.Button.left)
                    self.dragging = False
                    self.press_frames = 0
                    self.release_frames = 0

                nx, ny, click = self.smooth_x, self.smooth_y, (1.0 if self.dragging else 0.0)
                source = "AI"
            else:
                self.smooth_x, self.smooth_y = nx, ny
                if self.dragging:
                    self.mouse.release(mouse.Button.left)
                    self.dragging = False
                self.press_frames = 0
                self.release_frames = 0

            save = False
            if (self.mode == "LEARNING" and source == "USER") or (self.mode == "TRAINING" and source == "AI"):
                if time.time() - self.last_record > 0.033:
                    save = True
                    self.last_record = time.time()

            small = letterbox(frame, INPUT_W, INPUT_H)
            if self.queue.full():
                try:
                    self.queue.get_nowait()
                    self.queue.task_done()
                except queue.Empty:
                    pass
            try:
                self.queue.put_nowait((frame, small, nx, ny, click, source, save))
            except queue.Full:
                pass

        except Exception: pass

    def close_app(self):
        self.infer_thread.stop()
        self.infer_thread.wait()
        self.worker.stop()
        if self.nvidia_thread is not None:
            self.nvidia_thread.stop()
            self.nvidia_thread.join()
        if self.dragging:
            self.mouse.release(mouse.Button.left)
        QApplication.quit()
        sys.exit()

    def eventFilter(self, obj, event):
        if obj.objectName() == "header_widget":
            if event.type() == QEvent.MouseButtonPress and event.button() == Qt.LeftButton:
                self.window_dragging = True
                self.window_drag_pos = event.globalPos() - self.frameGeometry().topLeft()
                return True
            elif event.type() == QEvent.MouseMove and self.window_dragging:
                self.move(event.globalPos() - self.window_drag_pos)
                return True
            elif event.type() == QEvent.MouseButtonRelease and event.button() == Qt.LeftButton:
                self.window_dragging = False
                return True
        return super().eventFilter(obj, event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec_())
