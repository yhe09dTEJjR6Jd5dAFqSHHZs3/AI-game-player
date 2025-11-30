import sys
import os
import time
import json
import lmdb
import math
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
from torch.utils.data import Dataset
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
    import win32pdh
except Exception:
    win32pdh = None


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
LOG_DB = os.path.join(DATA_DIR, "data.lmdb")
LOG_INDEX_FILE = os.path.join(DATA_DIR, "index.meta")
REGION_FILE = os.path.join(BASE_DIR, "regions.json")
PRIORITY_FILE = os.path.join(DATA_DIR, "priorities.npy")
INPUT_W, INPUT_H = 320, 180
SEQ_LEN = 4
MAX_OCR = 16
OCR_CONF_THRESHOLD = 0.6
OCR_JUMP_THRESHOLD = 200.0
OCR_AGE_MAX_MS = 5000.0
POOL_MAX_BYTES = 10 * 1024 * 1024 * 1024
CACHE_MAX_BYTES = 256 * 1024 * 1024
GPU_VRAM_LOW = 600 * 1024 * 1024
GPU_VRAM_HIGH = 1200 * 1024 * 1024
GPU_HYSTERESIS_SEC = 2.5
FORCE_CPU_OCR = False
pynvml = None

class WindowsDiskStats:
    def __init__(self):
        self.available = False
        self.query = None
        if platform.system() != 'Windows' or win32pdh is None:
            return
        try:
            self.query = win32pdh.OpenQuery()
            self.counter_qd = win32pdh.AddCounter(self.query, r'\\PhysicalDisk(_Total)\\Current Disk Queue Length')
            self.counter_lat = win32pdh.AddCounter(self.query, r'\\PhysicalDisk(_Total)\\Avg. Disk sec/Transfer')
            self.counter_bps = win32pdh.AddCounter(self.query, r'\\PhysicalDisk(_Total)\\Disk Bytes/sec')
            self.available = True
            win32pdh.CollectQueryData(self.query)
        except Exception:
            self.available = False

    def read_value(self, counter):
        try:
            return float(win32pdh.GetFormattedCounterValue(counter, win32pdh.PDH_FMT_DOUBLE)[1])
        except Exception:
            return None

    def collect(self):
        if not self.available:
            return None
        try:
            win32pdh.CollectQueryData(self.query)
            qd = self.read_value(self.counter_qd)
            lat = self.read_value(self.counter_lat)
            bps = self.read_value(self.counter_bps)
            if qd is None or lat is None or bps is None:
                return None
            return {'qd': max(qd, 0.0), 'lat_ms': max(lat, 0.0) * 1000.0, 'throughput_mb': max(bps, 0.0) / (1024 * 1024)}
        except Exception:
            return None


WINDOWS_DISK = WindowsDiskStats()

for d in [BASE_DIR, DATA_DIR, IMG_DIR, MODEL_DIR]:
    os.makedirs(d, exist_ok=True)

if not os.path.exists(LOG_DB):
    env = lmdb.open(LOG_DB, map_size=POOL_MAX_BYTES * 2, subdir=True, max_dbs=1, lock=False)
    env.close()

if not os.path.exists(REGION_FILE):
    with open(REGION_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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

def get_total_vram_bytes():
    if not torch.cuda.is_available():
        return 0.0
    try:
        props = torch.cuda.get_device_properties(0)
        return float(getattr(props, 'total_memory', 0.0))
    except Exception:
        pass
    if pynvml is not None:
        try:
            pynvml.nvmlInit()
            if pynvml.nvmlDeviceGetCount() > 0:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceMemoryInfo(handle)
                return float(info.total)
        except Exception:
            return 0.0
    return 0.0

def encode_key(v):
    return int(v).to_bytes(8, byteorder='big', signed=False)

def decode_key(b):
    return int.from_bytes(b, byteorder='big', signed=False)

def pack_entry(meta, img_bytes):
    meta_bytes = json.dumps(meta, ensure_ascii=False).encode('utf-8')
    header = len(meta_bytes).to_bytes(4, byteorder='big', signed=False)
    return header + meta_bytes + (img_bytes if img_bytes is not None else b'')

def unpack_entry(buf):
    if buf is None or len(buf) < 4:
        return {}, b''
    meta_len = int.from_bytes(buf[:4], byteorder='big', signed=False)
    meta_bytes = buf[4:4+meta_len]
    try:
        meta = json.loads(meta_bytes.decode('utf-8')) if meta_bytes else {}
    except Exception:
        meta = {}
    img_bytes = buf[4+meta_len:] if len(buf) > 4 + meta_len else b''
    return meta, img_bytes

class StdSilencer:
    def __enter__(self):
        self.stdout = sys.stdout
        self.stderr = sys.stderr
        self.null = open(os.devnull, 'w')
        sys.stdout = self.null
        sys.stderr = self.null
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        try:
            self.null.close()
        except Exception:
            pass

class PoolCacheManager:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance.lock = threading.Lock()
            cls._instance.data = []
            cls._instance.total_bytes = 0
            cls._instance.last_key = -1
            cls._instance.prev_vals = []
            cls._instance.region_signature = None
        return cls._instance

    def reset_if_region_changed(self, region_types):
        sig = tuple(region_types)
        with self.lock:
            if self.region_signature != sig:
                self.data = []
                self.total_bytes = 0
                self.last_key = -1
                self.prev_vals = [0] * len(region_types)
                self.region_signature = sig

    def _commit_batch(self, batch):
        with self.lock:
            for entry in batch:
                self.data.append(entry)
                self.total_bytes += entry.get('size', 0)
                while self.total_bytes > POOL_MAX_BYTES and self.data:
                    dropped = self.data.pop(0)
                    self.total_bytes -= dropped.get('size', 0)

    def get_snapshot(self):
        with self.lock:
            return self.data, self.total_bytes

    def get_total_bytes(self):
        with self.lock:
            return self.total_bytes

    def load_from_db(self, region_types, chunk_size=256):
        self.reset_if_region_changed(region_types)
        if not os.path.exists(LOG_DB):
            return self.get_snapshot()

        env = lmdb.open(LOG_DB, map_size=POOL_MAX_BYTES * 2, subdir=True, max_dbs=1, readonly=True, lock=False, readahead=False)
        batch = []
        try:
            with env.begin(write=False) as txn:
                cur = txn.cursor()
                start_key = encode_key(self.last_key + 1) if self.last_key >= 0 else None
                has = cur.set_range(start_key) if start_key is not None else cur.first()
                while has:
                    try:
                        meta, img_bytes = unpack_entry(cur.value())
                    except Exception:
                        has = cur.next()
                        continue
                    try:
                        ts = int(meta.get('ts', 0))
                        mx, my, click = float(meta.get('mx', 0.0)), float(meta.get('my', 0.0)), float(meta.get('click', 0.0))
                        ocr_vals = [float(x) for x in meta.get('ocr', [])]
                        ocr_age = [float(x) for x in meta.get('ocr_age', [])]
                        novelty = float(meta.get('novelty', 0.0))
                        complexity = float(meta.get('complexity', 0.0))
                        pred_err = float(meta.get('prediction_error', 0.0))
                    except Exception:
                        has = cur.next()
                        continue
                    cur_vals = ocr_vals[:len(region_types)] + [0] * max(0, len(region_types) - len(ocr_vals))
                    with self.lock:
                        if len(self.prev_vals) != len(region_types):
                            self.prev_vals = [0] * len(region_types)
                        deltas = [cur_vals[i] - (self.prev_vals[i] if i < len(self.prev_vals) else 0) for i in range(len(region_types))]
                        self.prev_vals = cur_vals
                    feat = pack_ocr(ocr_vals, deltas, ocr_age)
                    weight = ExperiencePool.calc_weight_static(region_types, ocr_vals, deltas, click, novelty, complexity, pred_err)
                    size = 512 + (len(img_bytes) if img_bytes is not None else 0)
                    entry = {'ts': ts, 'img': '', 'img_bytes': img_bytes, 'mx': mx, 'my': my, 'click': click, 'ocr': feat, 'weight': weight, 'size': size, 'prediction_error': pred_err}
                    batch.append(entry)
                    self.last_key = decode_key(cur.key())
                    if len(batch) >= chunk_size:
                        self._commit_batch(batch)
                        batch = []
                    has = cur.next()
        finally:
            env.close()

        if batch:
            self._commit_batch(batch)
        return self.get_snapshot()

POOL_CACHE = PoolCacheManager()

def atomic_write_text(path, text):
    tmp = path + ".tmp"
    with open(tmp, 'w', encoding='utf-8') as f:
        f.write(str(text))
    os.replace(tmp, path)

def read_index_value(env):
    if os.path.exists(LOG_INDEX_FILE):
        try:
            with open(LOG_INDEX_FILE, 'r', encoding='utf-8') as f:
                return int(f.read().strip() or '0')
        except Exception:
            pass
    try:
        with env.begin(write=False) as txn:
            cur = txn.cursor()
            if cur.last():
                return decode_key(cur.key()) + 1
    except Exception:
        pass
    return 0

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

class PathPlanner:
    def __init__(self):
        self.path = deque()
        self.current = (0.5, 0.5)
        self.last_plan_time = 0.0
        self.last_target = (0.5, 0.5)
        self.lock_window = 0.2
        self.target_tolerance = 0.02
        self.deviation_tolerance = 0.01

    def reset(self, pos=None):
        self.path.clear()
        if pos is not None:
            self.current = (clamp01(pos[0]), clamp01(pos[1]))

    def plan(self, start, target, velocity):
        sx, sy = clamp01(start[0]), clamp01(start[1])
        tx, ty = clamp01(target[0]), clamp01(target[1])
        vx, vy = clamp01(0.5 + 0.5 * velocity[0]) - 0.5, clamp01(0.5 + 0.5 * velocity[1]) - 0.5
        c1 = (sx + vx * 0.4, sy + vy * 0.4)
        c2 = (tx - vx * 0.3, ty - vy * 0.3)
        self.path.clear()
        for t in np.linspace(0.0, 1.0, 8):
            a = 1 - t
            px = a * a * a * sx + 3 * a * a * t * c1[0] + 3 * a * t * t * c2[0] + t * t * t * tx
            py = a * a * a * sy + 3 * a * a * t * c1[1] + 3 * a * t * t * c2[1] + t * t * t * ty
            self.path.append((clamp01(px), clamp01(py)))
        self.last_plan_time = time.time()
        self.last_target = (tx, ty)

    def next_point(self):
        if self.path:
            self.current = self.path.popleft()
        return self.current

    def idle(self):
        return len(self.path) == 0

    def should_replan(self, current, target):
        if self.idle():
            return True
        now = time.time()
        if now - self.last_plan_time < self.lock_window:
            dist_target = ((target[0] - self.last_target[0]) ** 2 + (target[1] - self.last_target[1]) ** 2) ** 0.5
            head = self.path[0] if self.path else self.current
            deviation = ((current[0] - head[0]) ** 2 + (current[1] - head[1]) ** 2) ** 0.5
            if dist_target <= self.target_tolerance and deviation <= self.deviation_tolerance:
                return False
        return True

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

class SumTree:
    def __init__(self, capacity):
        cap = 1
        while cap < max(1, capacity):
            cap <<= 1
        self.capacity = cap
        self.size = capacity
        self.tree = np.zeros(2 * self.capacity, dtype=np.float32)
        self.epsilon = 1e-6

    def build(self, values):
        n = min(len(values), self.capacity)
        self.tree[self.capacity:self.capacity + n] = np.asarray(values[:n], dtype=np.float32) + self.epsilon
        for i in range(self.capacity - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    @property
    def total(self):
        return float(self.tree[1]) if len(self.tree) > 1 else 0.0

    def update(self, idx, value):
        if idx < 0 or idx >= self.size:
            return
        pos = self.capacity + idx
        self.tree[pos] = float(value) + self.epsilon
        pos //= 2
        while pos >= 1:
            self.tree[pos] = self.tree[2 * pos] + self.tree[2 * pos + 1]
            pos //= 2

    def recalibrate(self):
        for i in range(self.capacity - 1, 0, -1):
            self.tree[i] = self.tree[2 * i] + self.tree[2 * i + 1]

    def sample(self, batch_size):
        res = []
        weights = []
        total = max(self.total, self.epsilon)
        if total <= 0:
            return res, weights
        segment = total / batch_size
        for i in range(batch_size):
            target = random.random() * segment + i * segment
            res.append(self._find(target))
            weights.append(self.tree[self.capacity + res[-1]] if res[-1] is not None else 0.0)
        return res, weights

    def _find(self, value):
        idx = 1
        while idx < self.capacity:
            left = 2 * idx
            if self.tree[left] == 0 and self.tree[left + 1] == 0:
                idx = left
            elif value <= self.tree[left]:
                idx = left
            else:
                value -= self.tree[left]
                idx = left + 1
        res_idx = idx - self.capacity
        return res_idx if res_idx < self.size else self.size - 1

class ExperiencePool(Dataset):
    def __init__(self):
        self.cache = {}
        self.cache_order = deque()
        self.cache_bytes = 0
        self.cache_limit = CACHE_MAX_BYTES
        self.region_types = self.load_region_types()
        self.data, self.total_bytes = self.load_data()
        self.priorities = self.load_priorities()
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
            return POOL_CACHE.load_from_db(self.region_types)
        except Exception:
            return [], 0

    def load_priorities(self):
        arr = None
        if os.path.exists(PRIORITY_FILE):
            try:
                arr = np.load(PRIORITY_FILE)
            except Exception:
                arr = None
        if arr is None:
            arr = np.ones(len(self.data), dtype=np.float32)
        else:
            if arr.dtype != np.float32:
                arr = arr.astype(np.float32)
            if len(arr) < len(self.data):
                pad = np.ones(len(self.data) - len(arr), dtype=np.float32)
                arr = np.concatenate([arr, pad])
            elif len(arr) > len(self.data):
                arr = arr[:len(self.data)]
        try:
            np.save(PRIORITY_FILE, arr)
        except Exception:
            pass
        return arr

    def persist_priorities(self):
        try:
            np.save(PRIORITY_FILE, self.priorities)
        except Exception:
            pass

    def get_priority_weights(self):
        if not self.data:
            return []
        vals = [self.compute_priority_value(i) for i in range(len(self.data))]
        return vals

    def update_priorities(self, indices, losses):
        if len(self.priorities) == 0:
            return
        changed = False
        for idx, loss_v in zip(indices, losses):
            i = int(idx)
            if 0 <= i < len(self.priorities):
                lv = float(max(loss_v, 0.0))
                boosted = 1.0 + lv
                self.priorities[i] = 0.9 * self.priorities[i] + 0.1 * boosted
                changed = True
        if changed:
            self.persist_priorities()

    def compute_priority_value(self, idx):
        if idx < 0 or idx >= len(self.data):
            return 0.0
        base = float(self.data[idx].get('weight', 1.0))
        scaled = (base * (float(self.priorities[idx]) + 1e-3)) ** 1.2
        return float(max(scaled, 1e-3))

    def calc_weight(self, ocr, deltas, click=0.0, novelty=0.0, complexity=0.0, prediction_error=0.0):
        return ExperiencePool.calc_weight_static(self.region_types, ocr, deltas, click, novelty, complexity, prediction_error)

    @staticmethod
    def calc_weight_static(region_types, ocr, deltas, click=0.0, novelty=0.0, complexity=0.0, prediction_error=0.0):
        action_boost = 1.0 + 1.5 * abs(click)
        novelty = float(clamp01(novelty))
        complexity = float(clamp01(complexity))
        novelty_boost = 1.0 + 2.0 * novelty
        complexity_boost = 1.0 + 2.0 * complexity
        err = float(max(prediction_error, 0.0))
        err_boost = 1.0 + min(err, 5.0)
        if not region_types or not ocr:
            return max(0.05, action_boost * novelty_boost * complexity_boost * err_boost)
        score = 0.0
        total = 0.0
        for idx, t in enumerate(region_types):
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
        reward_boost = base * magnitude
        return max(0.05, reward_boost * action_boost * novelty_boost * complexity_boost * err_boost)

    def cache_image(self, item):
        key = str(item.get('id', id(item)))
        if key in self.cache:
            return self.cache[key]
        img_bytes = item.get('img_bytes')
        img = None
        if img_bytes:
            try:
                arr = np.frombuffer(img_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            except Exception:
                img = None
        if img is None and item.get('img'):
            img = cv2.imread(item.get('img'))
        if img is None:
            img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
        img = preprocess_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        size = img.nbytes
        self.cache[key] = img
        self.cache_order.append((key, size))
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
            img = self.cache_image(item)
            img = self.augment_image(img)
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

    def augment_image(self, img):
        if random.random() < 0.5:
            return img
        h, w = img.shape[1], img.shape[2]
        arr = np.transpose(img, (1, 2, 0)).copy()
        if random.random() < 0.5:
            gain = 0.8 + 0.4 * random.random()
            bias = (random.random() - 0.5) * 0.1
            arr[..., :3] = np.clip(arr[..., :3] * gain + bias, 0.0, 1.0)
        if random.random() < 0.3:
            noise = np.random.normal(0, 0.02, size=(h, w, 3)).astype(np.float32)
            arr[..., :3] = np.clip(arr[..., :3] + noise, 0.0, 1.0)
        if random.random() < 0.3:
            cut_ratio = 0.2 * random.random()
            ch = int(h * cut_ratio)
            cw = int(w * cut_ratio)
            cy = random.randint(0, max(0, h - ch))
            cx = random.randint(0, max(0, w - cw))
            arr[cy:cy + ch, cx:cx + cw, :3] = 0
        return np.transpose(arr.astype(np.float32), (2, 0, 1))

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
            tree = SumTree(len(weights))
            tree.build(weights)
            steps = math.ceil(len(ds) / self.batch_size)
            self.status_sig.emit(f"正在优化模型: 第 {ep+1}/{epochs} 轮")
            for step in range(steps):
                idxs, _ = tree.sample(self.batch_size)
                if not idxs:
                    break
                fixed = []
                for i in idxs:
                    if i is None or i < 0 or i >= len(ds):
                        fixed.append(random.randint(0, len(ds) - 1))
                    else:
                        fixed.append(i)
                batch = [ds[i] for i in fixed]
                imgs = torch.stack([b[0] for b in batch]).to(DEVICE)
                ocrs = torch.stack([b[1] for b in batch]).to(DEVICE)
                targs = torch.stack([b[2] for b in batch]).to(DEVICE)
                dirs = torch.stack([b[3] for b in batch]).to(DEVICE)
                delta_t = torch.stack([b[4] for b in batch]).to(DEVICE)
                ws = torch.stack([b[5] for b in batch]).to(DEVICE)
                idx_tensor = torch.stack([b[6] for b in batch])
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
                    loss_arr = comb.detach().cpu().numpy()
                    idx_arr = idx_tensor.cpu().numpy()
                    ds.update_priorities(idx_arr, loss_arr)
                    for j in idx_arr:
                        tree.update(int(j), ds.compute_priority_value(int(j)))

                if (step + 1) % 128 == 0:
                    tree.recalibrate()

                self.progress_sig.emit(int(((ep * steps) + step + 1) / (epochs * steps) * 100))

            self.progress_sig.emit(int((ep + 1) / epochs * 100))

        tmp_path = weight_path + '.tmp'
        torch.save(model.state_dict(), tmp_path)
        os.replace(tmp_path, weight_path)
        self.status_sig.emit("优化完成")
        self.finished_sig.emit()

class SaveWorker(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.q = queue.Queue(maxsize=256)
        self.running = True
        self.buffer = []
        self.buffer_bytes = 0
        self.flush_threshold = 4096
        self.last_flush = time.time()
        self.env = lmdb.open(LOG_DB, map_size=POOL_MAX_BYTES * 2, subdir=True, max_dbs=1, lock=True)
        self.next_id = read_index_value(self.env)

    def enqueue(self, img, meta):
        if self.q.full():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                return
        try:
            self.q.put_nowait((img, meta))
        except queue.Full:
            pass

    def flush_buffer(self):
        if not self.buffer:
            return
        try:
            with self.env.begin(write=True) as txn:
                for img, meta in self.buffer:
                    try:
                        ok, enc = cv2.imencode('.jpg', img, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
                        if not ok:
                            continue
                        meta['img'] = ''
                    except Exception:
                        continue
                    meta['id'] = self.next_id
                    data = pack_entry(meta, enc.tobytes())
                    txn.put(encode_key(self.next_id), data)
                    self.next_id += 1
            atomic_write_text(LOG_INDEX_FILE, str(self.next_id))
        except Exception:
            pass
        self.buffer.clear()
        self.buffer_bytes = 0
        self.last_flush = time.time()

    def run(self):
        while self.running or not self.q.empty():
            try:
                img, meta = self.q.get(timeout=0.1)
                payload = json.dumps(meta, ensure_ascii=False)
                self.buffer.append((img, meta))
                self.buffer_bytes += len(payload.encode('utf-8')) + img.nbytes
                now = time.time()
                if self.buffer_bytes >= self.flush_threshold or now - self.last_flush >= 1.0:
                    self.flush_buffer()
                self.q.task_done()
            except queue.Empty:
                if time.time() - self.last_flush >= 1.0:
                    self.flush_buffer()
                continue

    def stop(self):
        self.running = False
        while not self.q.empty():
            try:
                self.q.get_nowait()
                self.q.task_done()
            except queue.Empty:
                break
        self.flush_buffer()
        try:
            self.env.sync()
            self.env.close()
        except Exception:
            pass

class NvidiaSmiThread(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True)
        self.running = True
        self.result = (0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        self.startup_info = None

    def run(self):
        while self.running:
            try:
                if self.startup_info is None and platform.system() == 'Windows':
                    si = subprocess.STARTUPINFO()
                    si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    self.startup_info = si
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,fan.speed", "--format=csv,noheader,nounits"], startupinfo=self.startup_info)
                self.result = tuple(map(float, o.decode('utf-8').strip().split(',')))
            except Exception:
                pass
            time.sleep(1.0)

    def stop(self):
        self.running = False

class DataWorker(QThread):
    ocr_result = pyqtSignal(object)
    perf_signal = pyqtSignal(object)
    status_signal = pyqtSignal(str)

    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True
        self.paused = False
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
        self.kalman_states = []
        self.kalman_time = []
        self.tick = 0
        self.force_cpu = FORCE_CPU_OCR
        self.ocr_gpu_enabled = False
        self.ocr_gpu = None
        self.ocr_cpu = None
        self.ocr = None
        self.ocr_pool = concurrent.futures.ThreadPoolExecutor(max_workers=2)
        self.ocr_lock = threading.Lock()
        self.ocr_tasks = []
        self.prev_small = None
        self.prev_pos = None
        self.prev_vel = (0.0, 0.0)
        self.prev_time = None
        self.vram_low_since = None
        self.vram_high_since = None
        self.last_vram_check = 0.0
        self.gpu_forced_off = False
        self.reload_regions()

    def apply_kalman(self, idx, measurement, now, measurement_valid):
        if idx >= len(self.kalman_states):
            self.kalman_states.append({'x': 0.0, 'v': 0.0, 'p': 25.0})
            self.kalman_time.append(now)
        state = self.kalman_states[idx]
        last_t = self.kalman_time[idx] if idx < len(self.kalman_time) else now
        dt = max(now - last_t, 1e-3)
        pred_x = state['x'] + state['v'] * dt
        pred_p = state['p'] + 5.0 * dt
        if measurement_valid and measurement is not None:
            meas_val, meas_conf = measurement if isinstance(measurement, (list, tuple)) and len(measurement) >= 2 else (measurement, 1.0)
            prev_v = self.prev_vals[idx] if idx < len(self.prev_vals) else None
            if prev_v is not None and abs(float(meas_val) - float(prev_v)) > OCR_JUMP_THRESHOLD and meas_conf < 0.9:
                meas_val = None
            else:
                meas_val = float(meas_val)
            measurement = meas_val
            allowed = max(1.5, 80.0 * dt)
            fresh = idx >= len(self.prev_vals) or self.prev_vals[idx] is None
            if measurement is not None and (fresh or abs(measurement - pred_x) <= allowed):
                k = pred_p / (pred_p + 16.0)
                state['x'] = pred_x + k * (measurement - pred_x)
                state['v'] = (state['x'] - pred_x) / dt
                state['p'] = (1.0 - k) * pred_p
                self.last_update_time[idx] = now
            else:
                state['x'] = pred_x
                state['p'] = pred_p
        else:
            state['x'] = pred_x
            state['p'] = pred_p
        self.kalman_time[idx] = now
        return max(0, int(round(state['x'])))

    def init_ocr_instances(self):
        total_vram = get_total_vram_bytes()
        if total_vram > 0 and total_vram <= 4 * 1024 * 1024 * 1024:
            self.force_cpu = True
            self.gpu_forced_off = True
            self.status_signal.emit("显存<=4GB，默认使用CPU OCR")
        self.init_gpu_ocr()
        self.init_cpu_ocr()
        self.select_active_ocr()

    def init_gpu_ocr(self):
        if self.ocr_gpu is not None or self.force_cpu or self.gpu_forced_off or not torch.cuda.is_available():
            return
        try:
            with StdSilencer():
                self.ocr_gpu = PaddleOCR(use_angle_cls=False, use_gpu=True, gpu_mem=500)
            self.ocr_gpu_enabled = True
        except Exception as e:
            self.status_signal.emit(f"GPU OCR 初始化失败: {e}")
            self.ocr_gpu = None
            self.ocr_gpu_enabled = False

    def init_cpu_ocr(self):
        if self.ocr_cpu is not None:
            return
        try:
            with StdSilencer():
                self.ocr_cpu = PaddleOCR(use_angle_cls=False, use_gpu=False)
        except Exception as e:
            self.status_signal.emit(f"CPU OCR 初始化失败: {e}")
            self.ocr_cpu = None

    def drop_gpu_ocr(self):
        try:
            if self.ocr_gpu is not None:
                del self.ocr_gpu
        except Exception:
            pass
        self.ocr_gpu = None
        self.ocr_gpu_enabled = False
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

    def release_ocr_resources(self):
        try:
            for i, t in enumerate(self.ocr_tasks):
                if t is not None and t.done():
                    continue
                self.ocr_tasks[i] = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:
            pass

    def select_active_ocr(self):
        with self.ocr_lock:
            if not self.force_cpu and self.ocr_gpu is not None and self.ocr_gpu_enabled:
                self.ocr = self.ocr_gpu
            elif self.ocr_cpu is not None:
                self.ocr = self.ocr_cpu
            else:
                self.ocr = None

    def restore_ocr(self):
        self.select_active_ocr()
        if self.ocr is None:
            self.status_signal.emit("未找到可用的OCR实例")

    def set_force_cpu(self, force_cpu, drop_gpu=False):
        if self.force_cpu == force_cpu and not drop_gpu:
            return
        self.force_cpu = force_cpu
        if drop_gpu:
            self.drop_gpu_ocr()
        self.release_ocr_resources()
        if not self.force_cpu:
            self.init_gpu_ocr()
        self.restore_ocr()
        mode_txt = "CPU" if self.force_cpu or self.ocr_gpu is None else "GPU"
        self.status_signal.emit(f"OCR 已切换至 {mode_txt} 模式")

    def manage_ocr_mode(self):
        if not torch.cuda.is_available() or self.gpu_forced_off:
            return
        now = time.time()
        if now - self.last_vram_check < 0.5:
            return
        self.last_vram_check = now
        free = get_free_vram_bytes()
        if not self.force_cpu and free < GPU_VRAM_LOW:
            if self.vram_low_since is None:
                self.vram_low_since = now
            if now - self.vram_low_since >= GPU_HYSTERESIS_SEC:
                self.ocr_gpu_enabled = False
                self.set_force_cpu(True, drop_gpu=True)
                self.vram_high_since = None
        else:
            self.vram_low_since = None
        if self.force_cpu and free >= GPU_VRAM_HIGH and not self.gpu_forced_off:
            if self.vram_high_since is None:
                self.vram_high_since = now
            if now - self.vram_high_since >= GPU_HYSTERESIS_SEC:
                self.ocr_gpu_enabled = True
                self.set_force_cpu(False)
                self.vram_low_since = None
        elif self.force_cpu:
            self.vram_high_since = None

    def emit_status_snapshot(self):
        mode_txt = "未初始化"
        if self.ocr is not None:
            mode_txt = "GPU" if self.ocr is self.ocr_gpu else "CPU"
        self.status_signal.emit(f"OCR 已切换至 {mode_txt} 模式")

    def set_paused(self, paused):
        self.paused = paused

    def get_backlog(self):
        try:
            return self.save_worker.q.qsize()
        except Exception:
            return 0

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
        self.kalman_states = [{'x': 0.0, 'v': 0.0, 'p': 25.0} for _ in self.regions]
        self.kalman_time = [now_ts] * len(self.regions)
        self.ocr_tasks = [None] * len(self.regions)

    def read_ocr(self, roi):
        st = time.time()
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
        finally:
            self.perf_signal.emit({'ocr_time_ms': (time.time() - st) * 1000.0})

    def run(self):
        self.init_ocr_instances()
        self.emit_status_snapshot()
        while self.running:
            if self.paused:
                time.sleep(0.05)
                continue
            self.manage_ocr_mode()
            try:
                data = self.queue.get(timeout=0.1)
                full_img, small_img, mx, my, click, source, save, prediction_error = data
                self.tick += 1
                now = time.time()
                if self.prev_time is None:
                    self.prev_time = now
                dt = max(now - self.prev_time, 1e-3)
                vel_x = (mx - (self.prev_pos[0] if self.prev_pos else mx)) / dt
                vel_y = (my - (self.prev_pos[1] if self.prev_pos else my)) / dt
                acc_x = (vel_x - self.prev_vel[0]) / dt
                acc_y = (vel_y - self.prev_vel[1]) / dt
                vel_mag = (vel_x**2 + vel_y**2) ** 0.5
                prev_vel_mag = (self.prev_vel[0]**2 + self.prev_vel[1]**2) ** 0.5
                dot = vel_x * self.prev_vel[0] + vel_y * self.prev_vel[1]
                denom = max(vel_mag * prev_vel_mag, 1e-6)
                angle = np.arccos(np.clip(dot / denom, -1.0, 1.0)) if denom > 1e-6 else 0.0
                accel_mag = (acc_x**2 + acc_y**2) ** 0.5
                complexity = clamp01(0.5 * min(accel_mag, 5.0) / 5.0 + 0.5 * (angle / np.pi))
                novelty = 0.0
                if self.prev_small is not None:
                    diff = small_img.astype(np.float32) - self.prev_small.astype(np.float32)
                    novelty = clamp01(float(np.mean(diff * diff)) / (255.0 * 255.0) * 3.0)
                self.prev_small = small_img.copy()
                self.prev_pos = (mx, my)
                self.prev_vel = (vel_x, vel_y)
                self.prev_time = now

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
                                ts_now = time.time()
                                self.last_update_time.append(ts_now)
                                self.kalman_states.append({'x': 0.0, 'v': 0.0, 'p': 25.0})
                                self.kalman_time.append(ts_now)
                            raw = 1.0 if self.prev_rois[idx] is None else float(np.mean(cv2.absdiff(roi, self.prev_rois[idx]))) / 255.0
                            diff = 0.3 * self.prev_change[idx] + 0.7 * raw
                            self.prev_change[idx] = diff
                            self.prev_rois[idx] = roi
                            prev_v = self.prev_vals[idx]
                            need_read = diff >= 0.002 or prev_v is None or self.tick - self.last_read_tick[idx] >= 3
                            measurement = None
                            candidate_new = False
                            if idx < len(self.ocr_tasks) and self.ocr_tasks[idx] is not None and self.ocr_tasks[idx].done():
                                try:
                                    res = self.ocr_tasks[idx].result()
                                except Exception:
                                    res = None
                                if res is not None:
                                    measurement = res
                                    candidate_new = True
                                    self.last_read_tick[idx] = self.tick
                                self.ocr_tasks[idx] = None
                            if need_read and (idx >= len(self.ocr_tasks) or self.ocr_tasks[idx] is None):
                                if idx >= len(self.ocr_tasks):
                                    self.ocr_tasks.append(None)
                                self.ocr_tasks[idx] = self.ocr_pool.submit(self.read_ocr, roi.copy())
                                self.last_read_tick[idx] = self.tick
                            filtered = self.apply_kalman(idx, measurement, now, candidate_new)
                            self.histories[idx].append(filtered)
                            smoothed = int(round(float(np.median(self.histories[idx])))) if self.histories[idx] else filtered
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
                    ts = int(time.time() * 1000)
                    meta = {
                        'ts': ts,
                        'img': '',
                        'mx': mx,
                        'my': my,
                        'click': click,
                        'source': source,
                        'ocr': [float(v) for v in ocr_vals],
                        'ocr_age': [float(a) for a in ocr_ages],
                        'novelty': float(novelty),
                        'complexity': float(complexity),
                        'prediction_error': float(prediction_error)
                    }
                    self.save_worker.enqueue(full_img, meta)

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
            time.sleep(self.owner.loop_sleep)

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

        self.sct = None
        self.monitor = None
        try:
            self.sct = mss.mss()
            self.monitor = self.sct.monitors[1]
        except Exception as e:
            self.update_status_text(f"截屏初始化失败: {e}")

        self.queue = queue.Queue(maxsize=1)
        self.worker = DataWorker(self.queue)
        self.worker.start()

        self.overlay = Overlay()
        self.overlay.show()
        self.worker.ocr_result.connect(self.overlay.update_vals)
        self.worker.ocr_result.connect(self.update_ocr)
        self.worker.perf_signal.connect(self.update_perf)
        self.worker.status_signal.connect(self.update_status_text)
        self.worker.emit_status_snapshot()

        self.mode = "LEARNING"
        self.smooth_x, self.smooth_y = 0.5, 0.5
        self.path_planner = PathPlanner()
        self.last_target = (0.5, 0.5)
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
        self.capture_latency_ms = 0.0
        self.ocr_latency_ms = 0.0
        self.infer_latency_ms = 0.0
        self.loop_latency_ms = 0.0
        self.frame_interval_ms = 0.0
        self.prev_capture_ts = None
        self.prev_disk_io = psutil.disk_io_counters()
        self.prev_disk_ts = time.time()
        self.process = psutil.Process(os.getpid())
        self.loop_sleep = 0.001
        self.plot_pause_until = 0.0

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
                self.update_status_text("模型加载成功")
            except Exception as e:
                self.update_status_text(f"模型加载失败: {e}")

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
        self.lbl_disk = QLabel("经验池: 0%")
        self.lbl_res = QLabel("分辨率: 0x0")
        self.lbl_gpu_temp = QLabel("显卡温度: 0°C")
        self.lbl_gpu_power = QLabel("显卡功耗: 0W")
        self.lbl_gpu_fan = QLabel("风扇: 0%")
        self.lbl_disk_io = QLabel("系统磁盘IO: 0MB/s")
        self.lbl_disk_lat = QLabel("系统磁盘延迟: 0ms")
        self.lbl_disk_qd = QLabel("系统磁盘队列: 0")
        self.lbl_latency = QLabel("延迟: 截屏0ms | 推理0ms | OCR0ms | 循环0ms")
        self.lbl_fps = QLabel("帧率: 0fps | 截屏间隔0ms")

        info_labels = [self.lbl_cpu, self.lbl_mem, self.lbl_gpu, self.lbl_vram, self.lbl_disk, self.lbl_res, self.lbl_gpu_temp, self.lbl_gpu_power, self.lbl_gpu_fan, self.lbl_disk_io, self.lbl_disk_lat, self.lbl_disk_qd, self.lbl_latency, self.lbl_fps]
        for i, l in enumerate(info_labels):
            l.setStyleSheet("color: #00f0ff; font-size: 14px; padding: 5px;")
            grid.addWidget(l, i//3, i%3)
        fl.addLayout(grid)
        
        gl = QGridLayout()
        self.p_cpu = SciFiPlot("处理器", '#00ff00')
        self.p_mem = SciFiPlot("内存", '#00ffff')
        self.p_gpu = SciFiPlot("显卡", '#ff0055')
        self.p_vram = SciFiPlot("显存", '#ffaa00')
        self.p_pool = SciFiPlot("经验池", '#66aaff')

        self.d_cpu, self.d_mem, self.d_gpu, self.d_vram, self.d_pool = [], [], [], [], []
        for x in [self.d_cpu, self.d_mem, self.d_gpu, self.d_vram, self.d_pool]:
            x.extend([0]*100)

        gl.addWidget(self.p_cpu, 0, 0)
        gl.addWidget(self.p_mem, 0, 1)
        gl.addWidget(self.p_gpu, 1, 0)
        gl.addWidget(self.p_vram, 1, 1)
        gl.addWidget(self.p_pool, 2, 0, 1, 2)
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

        self.status_lbl = QLabel("状态: 就绪")
        self.status_lbl.setStyleSheet("color: #00f0ff; font-size: 14px; padding: 6px;")
        fl.addWidget(self.status_lbl)

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
            try:
                self.mouse.release(mouse.Button.left)
            except Exception:
                pass
            self.stop_flag.set()
            self.close_app()
        elif key == keyboard.Key.space:
            if self.mode == "LEARNING":
                self.mode = "TRAINING"
                self.stop_flag.clear()
            elif self.mode == "TRAINING":
                try:
                    self.mouse.release(mouse.Button.left)
                except Exception:
                    pass
                self.dragging = False
                self.mode = "LEARNING"
                self.stop_flag.set()
            self.update_mode()
        else:
            if self.mode == "TRAINING":
                try:
                    self.mouse.release(mouse.Button.left)
                except Exception:
                    pass
                self.dragging = False
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
            self.worker.set_force_cpu(True, drop_gpu=True)
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

    def update_status_text(self, text):
        if hasattr(self, 'status_lbl'):
            self.status_lbl.setText(f"状态: {text}")

    def update_perf(self, payload):
        if isinstance(payload, dict) and 'ocr_time_ms' in payload:
            try:
                self.ocr_latency_ms = float(payload.get('ocr_time_ms', self.ocr_latency_ms))
            except Exception:
                pass

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
        cpu = self.process.cpu_percent()
        mem = self.process.memory_info().rss / max(psutil.virtual_memory().total, 1) * 100

        gpu_display = "N/A"
        vram_display = "N/A"
        gpu_u, vram_u, vram_t = 0, 0, 1
        gpu_temp = 0.0
        gpu_power = 0.0
        gpu_fan = 0.0
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
                    gpu_temp = float(pynvml.nvmlDeviceGetTemperature(self.nvml_handle, pynvml.NVML_TEMPERATURE_GPU))
                    gpu_power = float(pynvml.nvmlDeviceGetPowerUsage(self.nvml_handle)) / 1000.0
                    gpu_fan = float(pynvml.nvmlDeviceGetFanSpeed(self.nvml_handle))
                    gpu_display = f"{gpu_u}%"
                    vram_display = f"{vram_u/1024:.1f}/{vram_t/1024:.1f}GB"
                except Exception:
                    pass
            elif pynvml is not None and platform.system() == 'Windows':
                if self.nvidia_thread is None:
                    self.nvidia_thread = NvidiaSmiThread()
                    self.nvidia_thread.start()
                stats = self.nvidia_thread.result
                gpu_u, vram_u, vram_t, gpu_temp, gpu_power, gpu_fan = stats if len(stats) == 6 else (0, 0, 1, 0, 0, 0)
                gpu_display = f"{gpu_u}%"
                vram_total = max(vram_t, 1)
                vram_display = f"{vram_u/1024:.1f}/{vram_total/1024:.1f}GB"

        if torch.cuda.is_available():
            vram_limit_mb = 3500
            vram_recover_mb = 2800
            if vram_u >= vram_limit_mb and not self.worker.force_cpu:
                self.worker.set_force_cpu(True, drop_gpu=True)
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
        now_io = psutil.disk_io_counters()
        now_ts = time.time()
        dt = max(now_ts - self.prev_disk_ts, 1e-6)
        rb = max(now_io.read_bytes - self.prev_disk_io.read_bytes, 0)
        wb = max(now_io.write_bytes - self.prev_disk_io.write_bytes, 0)
        io_throughput = (rb + wb) / dt / (1024 * 1024)
        ops = max((now_io.read_count - self.prev_disk_io.read_count) + (now_io.write_count - self.prev_disk_io.write_count), 1)
        lat_ms = max((now_io.read_time - self.prev_disk_io.read_time) + (now_io.write_time - self.prev_disk_io.write_time), 0) / ops
        qd = 0.0
        win_disk = WINDOWS_DISK.collect()
        if win_disk is not None:
            io_throughput = win_disk['throughput_mb']
            lat_ms = win_disk['lat_ms']
            qd = win_disk['qd']
        elif hasattr(now_io, 'busy_time'):
            busy_ms = max(now_io.busy_time - getattr(self.prev_disk_io, 'busy_time', 0), 0)
            if platform.system() != 'Windows' or busy_ms > 0:
                qd = busy_ms / (dt * 1000.0)
            else:
                iops = ops / dt if dt > 0 else 0.0
                qd = iops * (lat_ms / 1000.0)
        else:
            iops = ops / dt if dt > 0 else 0.0
            qd = iops * (lat_ms / 1000.0)
        self.prev_disk_io = now_io
        self.prev_disk_ts = now_ts
        scale = SCALE_PERCENT
        if platform.system() == 'Windows':
            try:
                import ctypes
                scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
            except Exception:
                scale = SCALE_PERCENT

        pool_bytes = POOL_CACHE.get_total_bytes()
        pool_percent = clamp01(pool_bytes / max(POOL_MAX_BYTES, 1)) * 100

        self.lbl_cpu.setText(f"处理器: {cpu}%")
        self.lbl_mem.setText(f"内存: {mem}%")
        self.lbl_gpu.setText(f"显卡: {gpu_display}")
        vram_hint = " (OCR已切换CPU)" if self.worker.force_cpu else ""
        self.lbl_vram.setText(f"显存: {vram_display}{vram_hint}")
        self.lbl_disk.setText(f"经验池: {pool_percent:.1f}% ({pool_bytes/(1024**3):.2f}/{POOL_MAX_BYTES/(1024**3):.2f}GB) | 磁盘可用: {disk.free/(1024**4):.2f}TB")
        self.lbl_res.setText(f"分辨率: {SCREEN_W}x{SCREEN_H} ({scale}%)")
        self.lbl_gpu_temp.setText(f"显卡温度: {gpu_temp:.0f}°C")
        self.lbl_gpu_power.setText(f"显卡功耗: {gpu_power:.1f}W")
        self.lbl_gpu_fan.setText(f"风扇: {gpu_fan:.0f}%")
        self.lbl_disk_io.setText(f"系统磁盘IO: {io_throughput:.2f}MB/s")
        self.lbl_disk_lat.setText(f"系统磁盘延迟: {lat_ms:.2f}ms")
        self.lbl_disk_qd.setText(f"系统磁盘队列: {qd:.2f}")
        fps = 1000.0 / self.frame_interval_ms if self.frame_interval_ms > 0 else 0.0
        self.lbl_latency.setText(f"延迟: 截屏{self.capture_latency_ms:.1f}ms | 推理{self.infer_latency_ms:.1f}ms | OCR{self.ocr_latency_ms:.1f}ms | 循环{self.loop_latency_ms:.1f}ms")
        self.lbl_fps.setText(f"帧率: {fps:.1f}fps | 截屏间隔{self.frame_interval_ms:.1f}ms")

        load = max(cpu, gpu_u)
        if load > 95 or (torch.cuda.is_available() and (vram_u / vram_t) * 100 >= 95):
            self.loop_sleep = 0.02
            self.plot_pause_until = time.time() + 2.0
        elif load > 80:
            self.loop_sleep = 0.008
        else:
            self.loop_sleep = 0.001

        backlog = self.worker.get_backlog() if hasattr(self, 'worker') and self.worker is not None else 0
        pause_worker = load > 95 or (torch.cuda.is_available() and (vram_u / vram_t) * 100 >= 95) or qd > 3.0 or backlog > 200
        self.worker.set_paused(pause_worker)

        for l, v in zip([self.d_cpu, self.d_mem, self.d_gpu, self.d_vram, self.d_pool], [cpu, mem, gpu_u, (vram_u/vram_t)*100 if vram_t else 0, pool_percent]):
            l.pop(0)
            l.append(v)

        if time.time() >= self.plot_pause_until:
            self.p_cpu.plot(self.d_cpu, pen=pg.mkPen('#00ff00', width=2), clear=True)
            self.p_mem.plot(self.d_mem, pen=pg.mkPen('#00ffff', width=2), clear=True)
            self.p_gpu.plot(self.d_gpu, pen=pg.mkPen('#ff0055', width=2), clear=True)
            self.p_vram.plot(self.d_vram, pen=pg.mkPen('#ffaa00', width=2), clear=True)
            self.p_pool.plot(self.d_pool, pen=pg.mkPen('#66aaff', width=2), clear=True)

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

        loop_start = time.time()
        try:
            cap_start = time.time()
            if self.sct is None or self.monitor is None:
                try:
                    self.sct = mss.mss()
                    self.monitor = self.sct.monitors[1]
                except Exception as e:
                    self.update_status_text(f"截屏失败: {e}")
                    return
            shot = self.sct.grab(self.monitor)
            img = np.frombuffer(shot.raw, dtype=np.uint8).reshape((shot.height, shot.width, 4))
            cap_end = time.time()
            self.capture_latency_ms = (cap_end - cap_start) * 1000.0
            if self.prev_capture_ts is not None:
                self.frame_interval_ms = (cap_start - self.prev_capture_ts) * 1000.0
            self.prev_capture_ts = cap_start
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = preprocess_frame(rgb)
            self.seq_frames.append(proc)
            self.seq_ocr.append(self.latest_ocr)

            mx, my = self.mouse.position
            nx, ny = screen_to_letter_norm(mx, my)
            click = 1.0 if self.mouse_pressed else 0.0
            source = "USER"
            prediction_error = 0.0

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

                infer_st = time.time()
                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            out, logits, delta_out = self.brain(seq_imgs, seq_ocr)
                    else:
                        out, logits, delta_out = self.brain(seq_imgs, seq_ocr)
                self.infer_latency_ms = (time.time() - infer_st) * 1000.0

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
                planned_target_x = (1 - blend) * self.smooth_x + blend * target_x
                planned_target_y = (1 - blend) * self.smooth_y + blend * target_y
                target_pair = (planned_target_x, planned_target_y)
                dist = ((target_pair[0] - self.last_target[0]) ** 2 + (target_pair[1] - self.last_target[1]) ** 2) ** 0.5
                if self.path_planner.should_replan((self.smooth_x, self.smooth_y), target_pair) or dist >= 0.05:
                    self.path_planner.plan((self.smooth_x, self.smooth_y), target_pair, (dx, dy))
                    self.last_target = target_pair
                nx, ny = self.path_planner.next_point()
                self.smooth_x, self.smooth_y = nx, ny
                prediction_error = float(-math.log(max(conf, 1e-6)))

                if self.stop_flag.is_set() or self.mode != "TRAINING":
                    return

                sx, sy = letter_norm_to_screen(self.smooth_x, self.smooth_y)
                if self.stop_flag.is_set() or self.mode != "TRAINING":
                    return
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
                if self.stop_flag.is_set() or self.mode != "TRAINING":
                    return
                if not self.dragging and self.press_frames >= 2:
                    self.mouse.press(mouse.Button.left)
                    self.dragging = True
                    self.last_press_time = now
                    self.release_frames = 0
                elif self.dragging and (pc <= 0.1 or (self.release_frames >= 3 and now - self.last_press_time >= self.min_press)):
                    if self.stop_flag.is_set() or self.mode != "TRAINING":
                        return
                    self.mouse.release(mouse.Button.left)
                    self.dragging = False
                    self.press_frames = 0
                    self.release_frames = 0

                nx, ny, click = self.smooth_x, self.smooth_y, (1.0 if self.dragging else 0.0)
                source = "AI"
            else:
                self.smooth_x, self.smooth_y = nx, ny
                self.path_planner.reset((self.smooth_x, self.smooth_y))
                self.last_target = (self.smooth_x, self.smooth_y)
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
                self.queue.put_nowait((frame, small, nx, ny, click, source, save, prediction_error))
            except queue.Full:
                pass
            self.loop_latency_ms = (time.time() - loop_start) * 1000.0

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
        if self.sct is not None:
            try:
                self.sct.close()
            except Exception:
                pass
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
