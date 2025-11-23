import os
import time
import random
import math
import collections
import threading
import subprocess
import ctypes
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.cuda.amp import autocast
from torch.amp import GradScaler
from ctypes import windll, wintypes, Structure, c_long, byref
import tkinter as tk
from tkinter import ttk

try:
    windll.user32.SetProcessDPIAware()
except Exception:
    pass

USER_HOME = os.path.expanduser("~")
BASE_DIR = os.path.join(USER_HOME, "Desktop", "GameAI")
MODEL_DIR = os.path.join(BASE_DIR, "AI模型")
POOL_DIR = os.path.join(BASE_DIR, "经验池")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(POOL_DIR, exist_ok=True)
MODEL_PATH = os.path.join(MODEL_DIR, "clash_multitask_vit.pth")

EMU_PATH = r"D:\LDPlayer9\dnplayer.exe"
ADB_PATH = r"D:\LDPlayer9\adb.exe"
WIN_TITLE = "LDPlayer"

BATCH_SIZE = 32
LR = 0.00025
GAMMA = 0.995
N_STEP = 3
PER_ALPHA = 0.6
PER_BETA_START = 0.4
PER_BETA_END = 1.0
PER_BETA_FRAMES = 200000
EPS_START = 1.0
EPS_END = 0.02
EPS_DECAY = 200000
MEMORY_SIZE = 12000
DEMO_SIZE = 6000
DEMO_SAVE_CHUNK = 256
IMG_H, IMG_W = 192, 160
STACK = 4

UI_PATCH_BUFFER_SIZE = 4000
UI_PATCH_BATCH = 32

DECK_CARDS = 8
HAND_CARDS = 4
CARD_CLASSES = 4
UI_LATENT_DIM = 8
CARD_VEC_PER_CARD = CARD_CLASSES + UI_LATENT_DIM
CARD_TYPE_VEC_DIM = DECK_CARDS * CARD_VEC_PER_CARD
BASE_EXTRA_DIM = 19
EXTRA_DIM = BASE_EXTRA_DIM + CARD_TYPE_VEC_DIM

VIT_PATCH = 16
VIT_DIM = 128
UI_SEG_CLASSES = 3
UI_REG_DIM = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
if torch.cuda.is_available():
    try:
        if hasattr(torch.backends, "cuda") and hasattr(torch.backends.cuda, "matmul") and hasattr(torch.backends.cuda.matmul, "fp32_precision"):
            torch.backends.cuda.matmul.fp32_precision = "ieee"
    except Exception:
        pass
    try:
        if hasattr(torch.backends, "cudnn") and hasattr(torch.backends.cudnn, "conv") and hasattr(torch.backends.cudnn.conv, "fp32_precision"):
            torch.backends.cudnn.conv.fp32_precision = "tf32"
    except Exception:
        pass

CARD_BAR_Y_MIN = 0.86
FIELD_Y_MIN = 0.16
FIELD_Y_MAX = 0.82
FIELD_X_MIN = 0.12
FIELD_X_MAX = 0.88
CARD_X_START = 0.33
CARD_X_STEP = 0.12

def build_drop_points():
    rows = [0.72, 0.62, 0.52, 0.42, 0.32]
    lanes = [0.23, 0.35, 0.65, 0.77]
    pts = []
    for y in rows:
        for x in lanes:
            pts.append((x, y))
    return pts

DROP_POINTS = build_drop_points()
N_PLACES = len(DROP_POINTS)
ACTION_SPACE = HAND_CARDS * N_PLACES + 1
MATCH_TIME_EST = 180.0

ui_lock = threading.Lock()

class POINT(Structure):
    _fields_ = [("x", c_long), ("y", c_long)]

class WinInput:
    def __init__(self, hwnd):
        self.hwnd = hwnd
        self.sw = windll.user32.GetSystemMetrics(0)
        self.sh = windll.user32.GetSystemMetrics(1)
        self.x = 0
        self.y = 0
        self.w = 1
        self.h = 1
        self.update_rect()
    def update_rect(self):
        if not self.hwnd:
            self.hwnd = windll.user32.FindWindowW(None, WIN_TITLE)
            if not self.hwnd:
                return
        rect = wintypes.RECT()
        windll.user32.GetClientRect(self.hwnd, byref(rect))
        pt = POINT()
        windll.user32.ClientToScreen(self.hwnd, byref(pt))
        self.x, self.y = pt.x, pt.y
        self.w, self.h = max(1, rect.right), max(1, rect.bottom)
    def to_screen(self, rx, ry):
        self.update_rect()
        sx = int(self.x + rx * self.w)
        sy = int(self.y + ry * self.h)
        return sx, sy
    def to_relative(self, sx, sy):
        self.update_rect()
        if self.w <= 0 or self.h <= 0:
            return None
        rx = (sx - self.x) / float(self.w)
        ry = (sy - self.y) / float(self.h)
        if rx < 0 or rx > 1 or ry < 0 or ry > 1:
            return None
        return rx, ry
    def _mouse_event(self, flags, x, y):
        nx = int(x * 65535 // max(1, self.sw - 1))
        ny = int(y * 65535 // max(1, self.sh - 1))
        windll.user32.mouse_event(flags, nx, ny, 0, 0)
    def drag(self, rx1, ry1, rx2, ry2):
        sx, sy = self.to_screen(rx1, ry1)
        ex, ey = self.to_screen(rx2, ry2)
        self._mouse_event(0x8001, sx, sy)
        self._mouse_event(0x0002, 0, 0)
        steps = 6
        for i in range(steps):
            t = (i + 1) / steps
            ease = t * t * (3 - 2 * t)
            cx = int(sx + (ex - sx) * ease)
            cy = int(sy + (ey - sy) * ease)
            self._mouse_event(0x8001, cx, cy)
            time.sleep(0.005)
        self._mouse_event(0x0004, 0, 0)
    def click(self, rx, ry):
        sx, sy = self.to_screen(rx, ry)
        self._mouse_event(0x8001, sx, sy)
        self._mouse_event(0x0002, 0, 0)
        time.sleep(0.03)
        self._mouse_event(0x0004, 0, 0)

class ScreenCap:
    def __init__(self, title):
        self.sct = mss.mss()
        self.title = title
        self.hwnd = windll.user32.FindWindowW(None, title)
        self.monitor = {}
    def grab(self):
        if not self.hwnd:
            self.hwnd = windll.user32.FindWindowW(None, self.title)
            if not self.hwnd:
                return None
        rect = wintypes.RECT()
        windll.user32.GetClientRect(self.hwnd, byref(rect))
        pt = POINT()
        windll.user32.ClientToScreen(self.hwnd, byref(pt))
        if rect.right < 100 or rect.bottom < 100:
            return None
        self.monitor = {"top": pt.y, "left": pt.x, "width": rect.right, "height": rect.bottom}
        try:
            img = np.array(self.sct.grab(self.monitor))
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2RGB)
            return cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        except Exception:
            return None

class DigitOCR:
    def __init__(self):
        self.templates = {}
        self.h_t = 32
        self.w_t = 24
        font = cv2.FONT_HERSHEY_SIMPLEX
        for d in range(10):
            img = np.zeros((self.h_t, self.w_t), np.uint8)
            cv2.putText(img, str(d), (2, self.h_t - 4), font, 1.0, 255, 2, cv2.LINE_AA)
            _, th = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            self.templates[d] = th
    def _preprocess(self, roi):
        if roi is None or roi.size == 0:
            return None
        g = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
        g = cv2.GaussianBlur(g, (3, 3), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return th
    def read_number(self, roi, max_digits=2):
        th = self._preprocess(roi)
        if th is None:
            return None, 0.0
        contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = []
        h, w = th.shape
        for c in contours:
            x, y, cw, ch = cv2.boundingRect(c)
            if ch < h * 0.3 or cw < 2:
                continue
            boxes.append((x, y, cw, ch))
        if not boxes:
            return None, 0.0
        boxes.sort(key=lambda b: b[0])
        digits = []
        scores = []
        for x, y, cw, ch in boxes[:max_digits]:
            crop = th[y:y + ch, x:x + cw]
            if crop.size == 0:
                continue
            crop = cv2.resize(crop, (self.w_t, self.h_t), interpolation=cv2.INTER_AREA)
            best_d = None
            best_score = -1e9
            for d, tpl in self.templates.items():
                if tpl.shape != crop.shape:
                    continue
                score = float((crop == tpl).sum())
                if score > best_score:
                    best_score = score
                    best_d = d
            if best_d is not None:
                digits.append(best_d)
                scores.append(best_score / float(crop.size + 1e-6))
        if not digits:
            return None, 0.0
        value = 0
        for d in digits:
            value = value * 10 + d
        conf = float(np.mean(scores)) if scores else 0.0
        return value, conf

class SEBlock(nn.Module):
    def __init__(self, c, r=16):
        super().__init__()
        self.sq = nn.AdaptiveAvgPool2d(1)
        self.ex = nn.Sequential(
            nn.Linear(c, c // r, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(c // r, c, bias=False),
            nn.Sigmoid()
        )
    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.sq(x).view(b, c)
        y = self.ex(y).view(b, c, 1, 1)
        return x * y

class ResBlock(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(),
            nn.Conv2d(c, c, 3, padding=1, bias=False),
            nn.BatchNorm2d(c),
            SEBlock(c)
        )
    def forward(self, x):
        return F.relu(x + self.conv(x))

class ViTBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4.0, attn_drop=0.0, drop=0.0):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, batch_first=True, dropout=attn_drop)
        self.drop1 = nn.Dropout(drop)
        self.ln2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, dim),
            nn.Dropout(drop)
        )
    def forward(self, x):
        y = self.ln1(x)
        y, _ = self.attn(y, y, y, need_weights=False)
        x = x + self.drop1(y)
        y = self.ln2(x)
        y = self.mlp(y)
        x = x + y
        return x

class ViTEncoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans, embed_dim, depth, num_heads):
        super().__init__()
        self.img_h, self.img_w = img_size
        self.patch = patch_size
        self.grid_h = self.img_h // self.patch
        self.grid_w = self.img_w // self.patch
        self.num_patches = self.grid_h * self.grid_w
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=self.patch, stride=self.patch)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.blocks = nn.ModuleList([ViTBlock(embed_dim, num_heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        b, n, c = x.shape
        cls = self.cls_token.expand(b, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = x + self.pos_embed[:, : x.size(1)]
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        return x

def build_seg_label_map(grid_h, grid_w):
    arr = np.zeros((grid_h, grid_w), dtype=np.int64)
    for iy in range(grid_h):
        y_norm = (iy + 0.5) / float(grid_h)
        if y_norm >= CARD_BAR_Y_MIN:
            cls = 2
        elif FIELD_Y_MIN <= y_norm <= FIELD_Y_MAX:
            cls = 1
        else:
            cls = 0
        arr[iy, :] = cls
    return torch.from_numpy(arr)

class CardRecognizer:
    def __init__(self, ocr):
        self.features = [None] * DECK_CARDS
        self.types = np.zeros((DECK_CARDS, CARD_CLASSES), dtype=np.float32)
        self.embeds = np.zeros((DECK_CARDS, UI_LATENT_DIM), dtype=np.float32)
        self.levels = np.zeros(DECK_CARDS, dtype=np.float32)
        self.hand_map = [-1] * HAND_CARDS
        self.cost_template = np.array([3.0, 2.5, 4.5, 4.0], dtype=np.float32)
        self.ocr = ocr
    def encode_patch(self, patch, level_norm=0.0):
        if patch is None or patch.size == 0:
            v = np.zeros(UI_LATENT_DIM, dtype=np.float32)
            v[0] = level_norm
            return v
        h, w, c = patch.shape
        if h <= 0 or w <= 0 or c != 3:
            v = np.zeros(UI_LATENT_DIM, dtype=np.float32)
            v[0] = level_norm
            return v
        g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        g = cv2.resize(g, (16, 16), interpolation=cv2.INTER_AREA)
        v = g.astype(np.float32).reshape(-1) / 255.0
        if v.shape[0] > UI_LATENT_DIM:
            v = v[:UI_LATENT_DIM]
        elif v.shape[0] < UI_LATENT_DIM:
            pad = np.zeros(UI_LATENT_DIM, dtype=np.float32)
            pad[:v.shape[0]] = v
            v = pad
        v[0] = level_norm
        return v
    def find_slot(self, feat):
        best = -1
        best_d = 1e9
        for i in range(DECK_CARDS):
            f = self.features[i]
            if f is None:
                continue
            d = float(np.mean((f - feat) ** 2))
            if d < best_d:
                best_d = d
                best = i
        if best_d > 0.015:
            for i in range(DECK_CARDS):
                if self.features[i] is None:
                    self.features[i] = feat
                    self.embeds[i] = feat
                    return i
        if best < 0:
            self.features[0] = feat
            self.embeds[0] = feat
            return 0
        self.features[best] = self.features[best] * 0.8 + feat * 0.2
        self.embeds[best] = self.embeds[best] * 0.8 + feat * 0.2
        return best
    def classify(self, patch):
        hsv = cv2.cvtColor(patch, cv2.COLOR_RGB2HSV)
        h, s, v = cv2.split(hsv)
        hmean = float(np.mean(h))
        smean = float(np.mean(s))
        vmean = float(np.mean(v))
        g = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(g, 50, 150)
        ed = float(np.count_nonzero(edges)) / float(edges.size if edges.size > 0 else 1)
        idx = 0
        if vmean < 80:
            idx = 3
        elif smean < 60:
            idx = 1
        elif hmean < 20 or hmean > 150:
            idx = 2
        else:
            if ed > 0.15:
                idx = 0
            else:
                idx = 1
        vec = np.zeros(CARD_CLASSES, dtype=np.float32)
        vec[idx] = 1.0
        return vec
    def update_from_image(self, img, card_slots):
        for idx, (x1, y1, x2, y2) in enumerate(card_slots):
            if x2 <= x1 or y2 <= y1:
                continue
            patch = img[y1:y2, x1:x2]
            if patch.size == 0:
                continue
            level_norm = 0.0
            if self.ocr is not None:
                ph, pw, _ = patch.shape
                ly1 = int(0.05 * ph)
                ly2 = int(0.45 * ph)
                lx1 = int(0.2 * pw)
                lx2 = int(0.8 * pw)
                lvl_roi = patch[ly1:ly2, lx1:lx2]
                val, conf = self.ocr.read_number(lvl_roi, max_digits=2)
                if val is not None and conf > 0.25 and val > 0:
                    level_norm = min(float(val), 15.0) / 15.0
            feat = self.encode_patch(patch, level_norm)
            slot = self.find_slot(feat)
            typ = self.classify(patch)
            self.types[slot] = self.types[slot] * 0.8 + typ * 0.2
            self.embeds[slot] = self.embeds[slot] * 0.9 + feat * 0.1
            self.levels[slot] = self.levels[slot] * 0.8 + level_norm * 0.2
            if 0 <= idx < HAND_CARDS:
                self.hand_map[idx] = slot
    def vector(self):
        self.embeds[:, 0] = np.clip(self.levels, 0.0, 1.0)
        type_flat = self.types.reshape(-1)
        emb_norm = np.tanh(self.embeds)
        emb_flat = emb_norm.reshape(-1)
        v = np.concatenate([type_flat, emb_flat], axis=0).astype(np.float32)
        if v.shape[0] < CARD_TYPE_VEC_DIM:
            pad = np.zeros(CARD_TYPE_VEC_DIM, dtype=np.float32)
            pad[:v.shape[0]] = v
            v = pad
        elif v.shape[0] > CARD_TYPE_VEC_DIM:
            v = v[:CARD_TYPE_VEC_DIM]
        return v
    def hand_cost(self, hand_idx):
        if hand_idx < 0 or hand_idx >= HAND_CARDS:
            return 3.0
        slot = self.hand_map[hand_idx]
        if slot < 0 or slot >= DECK_CARDS:
            return 3.0
        t = self.types[slot]
        if not np.any(t):
            return 3.0
        c = float(np.dot(t, self.cost_template))
        if c < 1.0:
            c = 1.0
        if c > 7.0:
            c = 7.0
        return c

class ProNet(nn.Module):
    def __init__(self, in_ch, n_act, extra_dim):
        super().__init__()
        self.embed_dim = VIT_DIM
        self.vit = ViTEncoder((IMG_H, IMG_W), VIT_PATCH, in_ch, self.embed_dim, depth=4, num_heads=4)
        self.grid_h = self.vit.grid_h
        self.grid_w = self.vit.grid_w
        self.seg_classes = UI_SEG_CLASSES
        self.reg_dim = UI_REG_DIM
        self.lstm = nn.LSTMCell(self.embed_dim + extra_dim, 512)
        self.adv = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, n_act))
        self.val = nn.Sequential(nn.Linear(512, 256), nn.ReLU(), nn.Linear(256, 1))
        self.reg_head = nn.Sequential(nn.LayerNorm(self.embed_dim), nn.Linear(self.embed_dim, self.embed_dim), nn.GELU(), nn.Linear(self.embed_dim, self.reg_dim))
        self.seg_head = nn.Linear(self.embed_dim, self.seg_classes)
    def encode_tokens(self, x):
        tokens = self.vit(x)
        cls = tokens[:, 0]
        patches = tokens[:, 1:]
        return cls, patches
    def forward(self, x, extra, h, c):
        cls, patches = self.encode_tokens(x)
        fc = torch.cat([cls, extra], dim=1)
        nh, nc = self.lstm(fc, (h, c))
        a = self.adv(nh)
        v = self.val(nh)
        q = v + a - a.mean(1, keepdim=True)
        return q, nh, nc
    def ui_heads(self, x):
        cls, patches = self.encode_tokens(x)
        reg = self.reg_head(cls)
        b, n, d = patches.shape
        seg = self.seg_head(patches)
        seg = seg.transpose(1, 2).reshape(b, self.seg_classes, self.grid_h, self.grid_w)
        return seg, reg

class PrioritizedReplayBuffer:
    def __init__(self, capacity, alpha):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.lock = threading.Lock()
    def push(self, transition, priority=None):
        with self.lock:
            max_prio = self.priorities.max() if self.buffer else 1.0
            if priority is None:
                priority = max_prio
            if len(self.buffer) < self.capacity:
                self.buffer.append(transition)
            else:
                self.buffer[self.pos] = transition
            self.priorities[self.pos] = float(priority)
            self.pos = (self.pos + 1) % self.capacity
    def sample(self, batch_size, beta):
        with self.lock:
            if len(self.buffer) == 0:
                return None, None, None
            if len(self.buffer) == self.capacity:
                prios = self.priorities
            else:
                prios = self.priorities[:len(self.buffer)]
            prios = prios.clip(min=1e-6)
            probs = prios ** self.alpha
            probs /= probs.sum()
            indices = np.random.choice(len(self.buffer), batch_size, p=probs)
            samples = [self.buffer[i] for i in indices]
            total = len(self.buffer)
            weights = (total * probs[indices]) ** (-beta)
            weights /= weights.max()
            return samples, indices, weights.astype(np.float32)
    def update_priorities(self, indices, prios):
        with self.lock:
            for idx, p in zip(indices, prios):
                if 0 <= idx < len(self.priorities):
                    self.priorities[idx] = float(abs(p) + 1e-6)
    def __len__(self):
        return len(self.buffer)

class SimpleBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)
        self.lock = threading.Lock()
    def push(self, item):
        with self.lock:
            self.buffer.append(item)
    def sample(self, batch_size):
        with self.lock:
            return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

patch_buffer = SimpleBuffer(UI_PATCH_BUFFER_SIZE)

class Env:
    def __init__(self):
        self.cap = ScreenCap(WIN_TITLE)
        self.inp = WinInput(self.cap.hwnd)
        self.ocr = DigitOCR()
        self.elixir = 5.0
        self.t_last = time.time()
        self.frames = collections.deque(maxlen=STACK)
        self.prev_enemy_hp = None
        self.prev_self_hp = None
        self.prev_enemy_lane = None
        self.prev_self_lane = None
        self.prev_enemy_crowns = 0
        self.prev_self_crowns = 0
        self.game_start = time.time()
        self.card_rec = CardRecognizer(self.ocr)
        self.last_result = 0.0
    def card_slots_rects(self):
        w = IMG_W
        h = IMG_H
        slots = []
        cw = 0.09
        ch = 0.16
        cy = 0.92
        for i in range(HAND_CARDS):
            cx = CARD_X_START + i * CARD_X_STEP
            x1 = int(max(0.0, (cx - cw * 0.5) * w))
            x2 = int(min(float(w), (cx + cw * 0.5) * w))
            y1 = int(max(0.0, (cy - ch * 0.5) * h))
            y2 = int(min(float(h), (cy + ch * 0.5) * h))
            slots.append((x1, y1, x2, y2))
        return slots
    def hp_feature(self, img):
        h, w, _ = img.shape
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        lower = np.array([45, 40, 40], dtype=np.uint8)
        upper = np.array([90, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        def ratio(x1, y1, x2, y2):
            x1i = max(0, x1)
            y1i = max(0, y1)
            x2i = min(w, x2)
            y2i = min(h, y2)
            if x2i <= x1i or y2i <= y1i:
                return 0.0
            region = mask[y1i:y2i, x1i:x2i]
            if region.size == 0:
                return 0.0
            return float(np.count_nonzero(region)) / float(region.size)
        ey1 = int(0.06 * h)
        ey2 = int(0.12 * h)
        sy1 = int(0.84 * h)
        sy2 = int(0.92 * h)
        ex_l1 = int(0.10 * w)
        ex_l2 = int(0.28 * w)
        ex_k1 = int(0.38 * w)
        ex_k2 = int(0.62 * w)
        ex_r1 = int(0.72 * w)
        ex_r2 = int(0.90 * w)
        enemy_ratios = [
            ratio(ex_l1, ey1, ex_l2, ey2),
            ratio(ex_k1, ey1, ex_k2, ey2),
            ratio(ex_r1, ey1, ex_r2, ey2)
        ]
        self_ratios = [
            ratio(ex_l1, sy1, ex_l2, sy2),
            ratio(ex_k1, sy1, ex_k2, sy2),
            ratio(ex_r1, sy1, ex_r2, sy2)
        ]
        def scale(v):
            return int(255.0 * max(0.0, min(1.0, v * 2.0)))
        enemy_lane = [scale(r) for r in enemy_ratios]
        self_lane = [scale(r) for r in self_ratios]
        enemy_hp = scale(float(sum(enemy_ratios)) / 3.0)
        self_hp = scale(float(sum(self_ratios)) / 3.0)
        return enemy_hp, self_hp, enemy_lane, self_lane
    def crown_feature(self, img):
        h, w, _ = img.shape
        y1 = int(0.18 * h)
        y2 = int(0.34 * h)
        x1 = int(0.25 * w)
        x2 = int(0.75 * w)
        if x2 <= x1 or y2 <= y1:
            return 0, 0
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return 0, 0
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        lower = np.array([15, 80, 150], dtype=np.uint8)
        upper = np.array([35, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if mask.size == 0:
            return 0, 0
        mid = mask.shape[1] // 2
        left = mask[:, :mid]
        right = mask[:, mid:]
        def count_crowns(side):
            if side.size == 0:
                return 0
            r = float(np.count_nonzero(side)) / float(side.size)
            if r < 0.01:
                return 0
            if r < 0.03:
                return 1
            if r < 0.06:
                return 2
            return 3
        enemy_c = count_crowns(left)
        self_c = count_crowns(right)
        return enemy_c, self_c
    def time_feature(self, img):
        h, w, _ = img.shape
        t = max(0.0, time.time() - self.game_start)
        logical = min(1.2, max(0.0, t / MATCH_TIME_EST))
        phase_single = 1.0 if t < 60.0 else 0.0
        phase_double = 1.0 if 60.0 <= t < 120.0 else 0.0
        phase_triple = 1.0 if t >= 120.0 else 0.0
        y1 = int(0.02 * h)
        y2 = int(0.07 * h)
        x1 = int(0.28 * w)
        x2 = int(0.72 * w)
        if x2 <= x1 or y2 <= y1:
            return logical, phase_single, phase_double, phase_triple
        roi = img[y1:y2, x1:x2]
        if roi.size == 0:
            return logical, phase_single, phase_double, phase_triple
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        lower = np.array([120, 40, 80], dtype=np.uint8)
        upper = np.array([170, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        if mask.size == 0:
            return logical, phase_single, phase_double, phase_triple
        cols = np.count_nonzero(mask, axis=0)
        vis = float(np.count_nonzero(cols > 0)) / float(cols.size if cols.size > 0 else 1)
        if vis < 0.02:
            return logical, phase_single, phase_double, phase_triple
        blended = 0.6 * logical + 0.4 * vis
        return blended, phase_single, phase_double, phase_triple
    def pressure_and_hero_feature(self, img):
        h, w, _ = img.shape
        y1 = int(FIELD_Y_MIN * h)
        y2 = int(FIELD_Y_MAX * h)
        if y2 <= y1:
            return 0.0, 0.0, 0.0, 0.0, 0.0, 0.0
        mid = (y1 + y2) // 2
        top = img[y1:mid, :]
        bottom = img[mid:y2, :]
        def seg_feats(seg, enemy_side):
            if seg is None or seg.size == 0:
                return 0.0, 0.0, 0.0, 0.0
            hsv = cv2.cvtColor(seg, cv2.COLOR_RGB2HSV)
            lower_red1 = np.array([0, 80, 80], dtype=np.uint8)
            upper_red1 = np.array([10, 255, 255], dtype=np.uint8)
            lower_red2 = np.array([160, 80, 80], dtype=np.uint8)
            upper_red2 = np.array([180, 255, 255], dtype=np.uint8)
            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
            red_mask = cv2.bitwise_or(mask1, mask2)
            lower_blue = np.array([90, 80, 80], dtype=np.uint8)
            upper_blue = np.array([140, 255, 255], dtype=np.uint8)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
            lower_gold = np.array([20, 120, 160], dtype=np.uint8)
            upper_gold = np.array([40, 255, 255], dtype=np.uint8)
            gold_mask = cv2.inRange(hsv, lower_gold, upper_gold)
            total = float(max(1, seg.shape[0] * seg.shape[1]))
            enemy_density = float(np.count_nonzero(red_mask)) / total
            self_density = float(np.count_nonzero(blue_mask)) / total
            champ_density = float(np.count_nonzero(gold_mask)) / total
            if enemy_side:
                return enemy_density, self_density, champ_density, 0.0
            else:
                return enemy_density, self_density, 0.0, champ_density
        et, st, ce_t, cs_t = seg_feats(top, True)
        eb, sb, ce_b, cs_b = seg_feats(bottom, False)
        champ_enemy = min(1.0, ce_t + ce_b)
        champ_self = min(1.0, cs_t + cs_b)
        return et, eb, st, sb, champ_enemy, champ_self
    def card_feature(self, img):
        slots = self.card_slots_rects()
        self.card_rec.update_from_image(img, slots)
        v = self.card_rec.vector()
        return v
    def read_elixir_ocr(self, img):
        h, w, _ = img.shape
        y1 = int(0.86 * h)
        y2 = int(0.98 * h)
        x1 = int(0.38 * w)
        x2 = int(0.62 * w)
        if x2 <= x1 or y2 <= y1:
            return None, 0.0
        roi = img[y1:y2, x1:x2]
        val, conf = self.ocr.read_number(roi, max_digits=2)
        return val, conf
    def update_elixir(self, img):
        now = time.time()
        dt = now - self.t_last
        self.t_last = now
        val, conf = self.read_elixir_ocr(img)
        if val is not None and 0 <= val <= 10 and conf > 0.35:
            self.elixir = float(val)
            return self.elixir
        roi = img[int(IMG_H * 0.9):, :]
        hsv = cv2.cvtColor(roi, cv2.COLOR_RGB2HSV)
        lower = np.array([120, 80, 80], dtype=np.uint8)
        upper = np.array([170, 255, 255], dtype=np.uint8)
        mask = cv2.inRange(hsv, lower, upper)
        ratio = np.count_nonzero(mask) / float(max(1, roi.shape[0] * roi.shape[1]))
        vis_e = min(10.0, ratio * 14.0)
        if abs(vis_e - self.elixir) < 1.5:
            self.elixir = min(10.0, self.elixir + dt * 0.35)
        else:
            self.elixir = vis_e * 0.8 + self.elixir * 0.2
        return self.elixir
    def reset(self):
        self.frames.clear()
        frame = self.cap.grab()
        if frame is None:
            frame = np.zeros((IMG_H, IMG_W, 3), dtype=np.uint8)
        self.card_rec = CardRecognizer(self.ocr)
        self.elixir = 5.0
        self.t_last = time.time()
        self.game_start = time.time()
        self.last_result = 0.0
        f = np.transpose(frame, (2, 0, 1))
        for _ in range(STACK):
            self.frames.append(f)
        enemy_hp, self_hp, enemy_lane, self_lane = self.hp_feature(frame)
        enemy_c, self_c = self.crown_feature(frame)
        self.prev_enemy_hp = enemy_hp
        self.prev_self_hp = self_hp
        self.prev_enemy_lane = np.array(enemy_lane, dtype=np.float32)
        self.prev_self_lane = np.array(self_lane, dtype=np.float32)
        self.prev_enemy_crowns = enemy_c
        self.prev_self_crowns = self_c
        time_ratio, phase_single, phase_double, phase_triple = self.time_feature(frame)
        enemy_top, enemy_bottom, self_top, self_bottom, champ_enemy, champ_self = self.pressure_and_hero_feature(frame)
        card_vec = self.card_feature(frame)
        s = np.concatenate(list(self.frames), axis=0)
        extra = np.zeros(EXTRA_DIM, dtype=np.float32)
        extra[0] = self.elixir / 10.0
        extra[1] = enemy_lane[0] / 255.0
        extra[2] = enemy_lane[1] / 255.0
        extra[3] = enemy_lane[2] / 255.0
        extra[4] = self_lane[0] / 255.0
        extra[5] = self_lane[1] / 255.0
        extra[6] = self_lane[2] / 255.0
        extra[7] = enemy_c / 3.0
        extra[8] = self_c / 3.0
        extra[9] = time_ratio
        extra[10] = phase_single
        extra[11] = phase_double
        extra[12] = phase_triple
        extra[13] = enemy_top
        extra[14] = enemy_bottom
        extra[15] = self_top
        extra[16] = self_bottom
        extra[17] = champ_enemy
        extra[18] = champ_self
        extra[BASE_EXTRA_DIM:] = card_vec[:CARD_TYPE_VEC_DIM]
        if patch_buffer is not None:
            patch_buffer.push((s.copy(), extra.copy()))
        return s, extra
    def observe(self):
        raw = self.cap.grab()
        if raw is None:
            return None, None, True
        e = self.update_elixir(raw)
        f = np.transpose(raw, (2, 0, 1))
        self.frames.append(f)
        if len(self.frames) < STACK:
            while len(self.frames) < STACK:
                self.frames.append(f)
        s = np.concatenate(list(self.frames), axis=0)
        enemy_hp, self_hp, enemy_lane, self_lane = self.hp_feature(raw)
        enemy_c, self_c = self.crown_feature(raw)
        time_ratio, phase_single, phase_double, phase_triple = self.time_feature(raw)
        enemy_top, enemy_bottom, self_top, self_bottom, champ_enemy, champ_self = self.pressure_and_hero_feature(raw)
        card_vec = self.card_feature(raw)
        extra = np.zeros(EXTRA_DIM, dtype=np.float32)
        extra[0] = e / 10.0
        extra[1] = enemy_lane[0] / 255.0
        extra[2] = enemy_lane[1] / 255.0
        extra[3] = enemy_lane[2] / 255.0
        extra[4] = self_lane[0] / 255.0
        extra[5] = self_lane[1] / 255.0
        extra[6] = self_lane[2] / 255.0
        extra[7] = enemy_c / 3.0
        extra[8] = self_c / 3.0
        extra[9] = time_ratio
        extra[10] = phase_single
        extra[11] = phase_double
        extra[12] = phase_triple
        extra[13] = enemy_top
        extra[14] = enemy_bottom
        extra[15] = self_top
        extra[16] = self_bottom
        extra[17] = champ_enemy
        extra[18] = champ_self
        extra[BASE_EXTRA_DIM:] = card_vec[:CARD_TYPE_VEC_DIM]
        if patch_buffer is not None:
            patch_buffer.push((s.copy(), extra.copy()))
        return s, extra, False
    def check_terminal(self, enemy_c, self_c, enemy_hp, self_hp, time_ratio):
        done = False
        result = 0.0
        if enemy_c >= 3 or self_c >= 3:
            done = True
            if self_c > enemy_c:
                result = 1.0
            elif self_c < enemy_c:
                result = -1.0
        elif time_ratio >= 1.02:
            done = True
            if self_c > enemy_c:
                result = 1.0
            elif self_c < enemy_c:
                result = -1.0
            else:
                if self_hp > enemy_hp + 10:
                    result = 1.0
                elif enemy_hp > self_hp + 10:
                    result = -1.0
                else:
                    result = 0.0
        if done:
            self.last_result = result
        return done, result
    def auto_restart(self):
        try:
            time.sleep(1.2)
            for _ in range(3):
                self.inp.click(0.5, 0.78)
                time.sleep(0.6)
            for _ in range(3):
                self.inp.click(0.5, 0.83)
                time.sleep(0.6)
        except Exception:
            pass
    def step(self, action):
        spent = 0.0
        blocked = 0.0
        if action < HAND_CARDS * N_PLACES:
            card_idx = action // N_PLACES
            place_idx = action % N_PLACES
            if 0 <= card_idx < HAND_CARDS and 0 <= place_idx < N_PLACES:
                sx = CARD_X_START + card_idx * CARD_X_STEP
                sy = 0.93
                fx, fy = DROP_POINTS[place_idx]
                est_cost = self.card_rec.hand_cost(card_idx)
                if self.elixir >= est_cost:
                    self.inp.drag(sx, sy, fx, fy)
                    spent = est_cost
                else:
                    blocked = 1.0
        time.sleep(0.08)
        raw = self.cap.grab()
        if raw is None:
            return None, None, 0.0, True
        e = self.update_elixir(raw)
        f = np.transpose(raw, (2, 0, 1))
        self.frames.append(f)
        if len(self.frames) < STACK:
            while len(self.frames) < STACK:
                self.frames.append(f)
        s_next = np.concatenate(list(self.frames), axis=0)
        enemy_hp, self_hp, enemy_lane, self_lane = self.hp_feature(raw)
        enemy_c, self_c = self.crown_feature(raw)
        if self.prev_enemy_hp is None:
            self.prev_enemy_hp = enemy_hp
            self.prev_self_hp = self_hp
            self.prev_enemy_lane = np.array(enemy_lane, dtype=np.float32)
            self.prev_self_lane = np.array(self_lane, dtype=np.float32)
        de_total = float(self.prev_enemy_hp - enemy_hp)
        ds_total = float(self.prev_self_hp - self_hp)
        de_lane = self.prev_enemy_lane - np.array(enemy_lane, dtype=np.float32)
        ds_lane = self.prev_self_lane - np.array(self_lane, dtype=np.float32)
        crowns_delta_enemy = int(enemy_c - self.prev_enemy_crowns)
        crowns_delta_self = int(self_c - self.prev_self_crowns)
        self.prev_enemy_hp = enemy_hp
        self.prev_self_hp = self_hp
        self.prev_enemy_lane = np.array(enemy_lane, dtype=np.float32)
        self.prev_self_lane = np.array(self_lane, dtype=np.float32)
        self.prev_enemy_crowns = enemy_c
        self.prev_self_crowns = self_c
        time_ratio, phase_single, phase_double, phase_triple = self.time_feature(raw)
        enemy_top, enemy_bottom, self_top, self_bottom, champ_enemy, champ_self = self.pressure_and_hero_feature(raw)
        card_vec = self.card_feature(raw)
        reward = -0.004
        if spent > 0:
            reward += 0.2 + 0.02 * min(7.0, spent)
        else:
            if blocked > 0.0:
                reward -= 0.01
        if self.elixir >= 9.5:
            reward -= 0.03
        lane_weights = np.array([1.0, 1.3, 1.0], dtype=np.float32)
        pos_damage = float(np.sum(np.maximum(de_lane, 0.0) * lane_weights))
        neg_damage = float(np.sum(np.maximum(ds_lane, 0.0) * lane_weights))
        scale_phase = 1.0 + 0.4 * phase_double + 0.8 * phase_triple
        reward += 0.0025 * pos_damage * scale_phase
        reward -= 0.0020 * neg_damage * scale_phase
        if crowns_delta_self > 0:
            reward += 12.0 * float(crowns_delta_self)
        if crowns_delta_enemy > 0:
            reward -= 12.0 * float(crowns_delta_enemy)
        pressure_self = self_top + self_bottom
        pressure_enemy = enemy_top + enemy_bottom
        reward += 0.04 * (pressure_self - pressure_enemy)
        reward += 0.06 * champ_self
        reward -= 0.04 * champ_enemy
        done, result = self.check_terminal(enemy_c, self_c, enemy_hp, self_hp, time_ratio)
        if done:
            reward += 25.0 * result
        extra = np.zeros(EXTRA_DIM, dtype=np.float32)
        extra[0] = e / 10.0
        extra[1] = enemy_lane[0] / 255.0
        extra[2] = enemy_lane[1] / 255.0
        extra[3] = enemy_lane[2] / 255.0
        extra[4] = self_lane[0] / 255.0
        extra[5] = self_lane[1] / 255.0
        extra[6] = self_lane[2] / 255.0
        extra[7] = enemy_c / 3.0
        extra[8] = self_c / 3.0
        extra[9] = time_ratio
        extra[10] = phase_single
        extra[11] = phase_double
        extra[12] = phase_triple
        extra[13] = enemy_top
        extra[14] = enemy_bottom
        extra[15] = self_top
        extra[16] = self_bottom
        extra[17] = champ_enemy
        extra[18] = champ_self
        extra[BASE_EXTRA_DIM:] = card_vec[:CARD_TYPE_VEC_DIM]
        if patch_buffer is not None:
            patch_buffer.push((s_next.copy(), extra.copy()))
        if done:
            self.auto_restart()
        return s_next, extra, reward, done

class Agent:
    def __init__(self):
        self.net = ProNet(STACK * 3, ACTION_SPACE, EXTRA_DIM).to(DEVICE)
        self.tgt = ProNet(STACK * 3, ACTION_SPACE, EXTRA_DIM).to(DEVICE)
        self.tgt.load_state_dict(self.net.state_dict())
        self.opt = optim.AdamW(self.net.parameters(), lr=LR, weight_decay=1e-5)
        self.scaler = GradScaler("cuda", enabled=(DEVICE.type == "cuda"))
        self.mem = PrioritizedReplayBuffer(MEMORY_SIZE, PER_ALPHA)
        self.demo = SimpleBuffer(DEMO_SIZE)
        self.steps = 0
        self.per_beta = PER_BETA_START
        self.seg_label_map = build_seg_label_map(self.net.grid_h, self.net.grid_w).to(DEVICE)
        self.reg_dim = self.net.reg_dim
        self.ui_steps = 0
        self.load()
        self.load_demos_from_disk()
    def load(self):
        if os.path.exists(MODEL_PATH):
            try:
                c = torch.load(MODEL_PATH, map_location=DEVICE)
                self.net.load_state_dict(c["m"])
                self.tgt.load_state_dict(self.net.state_dict())
                self.steps = int(c.get("s", 0))
                self.ui_steps = int(c.get("u", 0))
            except Exception:
                pass
    def save(self):
        tmp = MODEL_PATH + ".tmp"
        try:
            torch.save({"m": self.net.state_dict(), "s": self.steps, "u": self.ui_steps}, tmp)
            os.replace(tmp, MODEL_PATH)
        except Exception:
            pass
    def load_demos_from_disk(self):
        try:
            files = [f for f in os.listdir(POOL_DIR) if f.lower().endswith(".npz")]
        except Exception:
            return
        files.sort()
        for fn in files:
            if len(self.demo) >= DEMO_SIZE:
                break
            path = os.path.join(POOL_DIR, fn)
            try:
                data = np.load(path, allow_pickle=False)
                s_arr = data["s"]
                extra_arr = data["extra"]
                a_arr = data["a"]
            except Exception:
                continue
            if extra_arr.ndim == 1:
                extra_arr = extra_arr.reshape(-1, extra_arr.shape[0])
            if extra_arr.shape[1] != EXTRA_DIM:
                pad = np.zeros((extra_arr.shape[0], EXTRA_DIM), dtype=np.float32)
                dim = min(extra_arr.shape[1], EXTRA_DIM)
                pad[:, :dim] = extra_arr[:, :dim]
                extra_arr = pad
            n = min(len(s_arr), len(extra_arr), len(a_arr))
            for i in range(n):
                self.demo.push((s_arr[i], extra_arr[i], int(a_arr[i])))
                if len(self.demo) >= DEMO_SIZE:
                    break
    def act(self, s, extra, h, c, eps):
        if random.random() < eps:
            a = random.randint(0, ACTION_SPACE - 1)
            return a, h, c
        with torch.no_grad(), autocast(enabled=(DEVICE.type == "cuda")):
            s_t = torch.from_numpy(s).float().unsqueeze(0).to(DEVICE) / 255.0
            extra_t = torch.from_numpy(extra).float().unsqueeze(0).to(DEVICE)
            q, nh, nc = self.net(s_t, extra_t, h, c)
            a = int(q.argmax().item())
            return a, nh, nc
    def push_demo(self, s, extra, a):
        self.demo.push((s, extra, int(a)))
    def update_beta(self):
        frac = min(1.0, float(self.steps) / float(max(1, PER_BETA_FRAMES)))
        self.per_beta = PER_BETA_START + (PER_BETA_END - PER_BETA_START) * frac
    def learn_rl(self):
        if len(self.mem) < BATCH_SIZE:
            return None
        self.update_beta()
        batch, indices, weights = self.mem.sample(BATCH_SIZE, self.per_beta)
        if batch is None:
            return None
        s, extra, h, c, a, r, n_gamma, ns, nextra, nh, nc, d = zip(*batch)
        s = torch.from_numpy(np.array(s)).float().to(DEVICE) / 255.0
        extra = torch.from_numpy(np.array(extra)).float().to(DEVICE)
        ns = torch.from_numpy(np.array(ns)).float().to(DEVICE) / 255.0
        nextra = torch.from_numpy(np.array(nextra)).float().to(DEVICE)
        h = torch.stack(h).squeeze(1).to(DEVICE)
        c = torch.stack(c).squeeze(1).to(DEVICE)
        nh = torch.stack(nh).squeeze(1).to(DEVICE)
        nc = torch.stack(nc).squeeze(1).to(DEVICE)
        a = torch.LongTensor(a).unsqueeze(1).to(DEVICE)
        r = torch.FloatTensor(r).unsqueeze(1).to(DEVICE)
        d = torch.FloatTensor(d).unsqueeze(1).to(DEVICE)
        n_gamma = torch.FloatTensor(n_gamma).unsqueeze(1).to(DEVICE)
        w = torch.from_numpy(weights).unsqueeze(1).to(DEVICE)
        with autocast(enabled=(DEVICE.type == "cuda")):
            q, _, _ = self.net(s, extra, h, c)
            q_a = q.gather(1, a)
            with torch.no_grad():
                q_next_online, _, _ = self.net(ns, nextra, nh, nc)
                best_a = q_next_online.argmax(1, keepdim=True)
                q_next_tgt, _, _ = self.tgt(ns, nextra, nh, nc)
                q_next = q_next_tgt.gather(1, best_a)
                target = r + n_gamma * q_next * (1.0 - d)
            td = q_a - target
            loss_indiv = F.smooth_l1_loss(q_a, target, reduction="none")
            loss = (w * loss_indiv).mean()
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        td_err = td.detach().abs().cpu().numpy().flatten()
        self.mem.update_priorities(indices, td_err)
        val = float(loss.item())
        return val
    def learn_bc(self):
        if len(self.demo) < BATCH_SIZE:
            return None
        batch = self.demo.sample(BATCH_SIZE)
        s, extra, a = zip(*batch)
        s = torch.from_numpy(np.array(s)).float().to(DEVICE) / 255.0
        extra = torch.from_numpy(np.array(extra)).float().to(DEVICE)
        a = torch.LongTensor(a).to(DEVICE)
        h0 = torch.zeros(BATCH_SIZE, 512, device=DEVICE)
        c0 = torch.zeros(BATCH_SIZE, 512, device=DEVICE)
        with autocast(enabled=(DEVICE.type == "cuda")):
            q, _, _ = self.net(s, extra, h0, c0)
            loss = F.cross_entropy(q, a)
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        val = float(loss.item())
        return val
    def learn_ui(self):
        if patch_buffer is None or len(patch_buffer) < UI_PATCH_BATCH:
            return None
        batch = patch_buffer.sample(UI_PATCH_BATCH)
        s_arr, extra_arr = zip(*batch)
        s = torch.from_numpy(np.array(s_arr)).float().to(DEVICE) / 255.0
        extra = torch.from_numpy(np.array(extra_arr)).float().to(DEVICE)
        seg_target = self.seg_label_map.unsqueeze(0).expand(s.size(0), -1, -1)
        with autocast(enabled=(DEVICE.type == "cuda")):
            seg_logits, reg_pred = self.net.ui_heads(s)
            seg_loss = F.cross_entropy(seg_logits, seg_target)
            reg_target = extra[:, :self.reg_dim]
            reg_loss = F.mse_loss(reg_pred, reg_target)
            loss = seg_loss * 0.6 + reg_loss * 0.4
        self.opt.zero_grad(set_to_none=True)
        self.scaler.scale(loss).backward()
        self.scaler.unscale_(self.opt)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), 5.0)
        self.scaler.step(self.opt)
        self.scaler.update()
        self.ui_steps += 1
        return float(loss.item())

active = False
stop_flag = False
fps_val = 0
agent = Agent()
rl_loss_val = 0.0
bc_loss_val = 0.0
ui_loss_val = 0.0
eps_val = EPS_START
loss_hist_rl = collections.deque(maxlen=240)
loss_hist_bc = collections.deque(maxlen=240)
loss_hist_ui = collections.deque(maxlen=240)
eps_hist = collections.deque(maxlen=240)
result_hist = collections.deque(maxlen=60)
dragging = False
drag_start_rx = 0.0
drag_start_ry = 0.0
last_lbtn = 0
demo_save_buffer = []
episodes_total = 0
episodes_win = 0
episodes_loss = 0
episodes_draw = 0

def ensure_emulator():
    hwnd = windll.user32.FindWindowW(None, WIN_TITLE)
    if not hwnd and os.path.exists(EMU_PATH):
        try:
            subprocess.Popen(EMU_PATH)
            time.sleep(20)
        except Exception:
            time.sleep(5)

def adb_connect():
    if os.path.exists(ADB_PATH):
        try:
            subprocess.run([ADB_PATH, "connect", "127.0.0.1:5555"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except Exception:
            pass

def train_thread():
    global rl_loss_val, bc_loss_val, ui_loss_val, stop_flag
    target_sync = 0
    save_sync = 0
    while not stop_flag:
        rl_loss = agent.learn_rl()
        bc_loss = agent.learn_bc()
        ui_loss = agent.learn_ui()
        if rl_loss is not None:
            rl_loss_val = rl_loss * 0.05 + rl_loss_val * 0.95
            loss_hist_rl.append(rl_loss)
        if bc_loss is not None:
            bc_loss_val = bc_loss * 0.05 + bc_loss_val * 0.95
            loss_hist_bc.append(bc_loss)
        if ui_loss is not None:
            ui_loss_val = ui_loss * 0.05 + ui_loss_val * 0.95
            loss_hist_ui.append(ui_loss)
        target_sync += 1
        save_sync += 1
        if target_sync >= 2000:
            agent.tgt.load_state_dict(agent.net.state_dict())
            target_sync = 0
        if save_sync >= 5000:
            agent.save()
            save_sync = 0
        time.sleep(0.001)

def gui_thread():
    root = tk.Tk()
    root.title("Clash Royale Pro AI")
    root.geometry("480x320")
    root.minsize(460, 300)
    root.configure(bg="#050812")
    root.attributes("-topmost", True)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    style.configure("Dark.TFrame", background="#050812")
    style.configure("Card.TFrame", background="#101522", relief="flat")
    style.configure("Title.TLabel", font=("Segoe UI", 16, "bold"), foreground="#F5F5F7", background="#050812")
    style.configure("Sub.TLabel", font=("Segoe UI", 9), foreground="#7B8496", background="#050812")
    style.configure("StateOn.TLabel", font=("Segoe UI", 11, "bold"), foreground="#57e39c", background="#050812")
    style.configure("StateOff.TLabel", font=("Segoe UI", 11, "bold"), foreground="#ff6b81", background="#050812")
    style.configure("Metric.TLabel", font=("Consolas", 10), foreground="#DDE3F0", background="#101522")
    style.configure("Caption.TLabel", font=("Segoe UI", 9, "bold"), foreground="#9aa4c6", background="#101522")
    style.configure("Accent.Horizontal.TProgressbar", troughcolor="#0b0f1a", bordercolor="#0b0f1a", background="#3d7eff")
    container = ttk.Frame(root, style="Dark.TFrame", padding=14)
    container.pack(fill="both", expand=True)
    header = ttk.Frame(container, style="Dark.TFrame")
    header.pack(fill="x")
    lbl_title = ttk.Label(header, text="Clash Royale Pro AI", style="Title.TLabel")
    lbl_title.pack(anchor="w")
    lbl_sub = ttk.Label(header, text="F8 切换 AI 接管    ESC 退出    请保持模拟器在前台", style="Sub.TLabel")
    lbl_sub.pack(anchor="w", pady=(2, 6))
    lbl_device = ttk.Label(header, text="设备: {}   PyTorch: {}".format("CUDA" if DEVICE.type == "cuda" else "CPU", torch.__version__), style="Sub.TLabel")
    lbl_device.pack(anchor="e")
    state_row = ttk.Frame(container, style="Dark.TFrame")
    state_row.pack(fill="x", pady=(4, 10))
    lbl_state = ttk.Label(state_row, text="PAUSED", style="StateOff.TLabel")
    lbl_state.pack(side="left")
    bar = ttk.Progressbar(state_row, orient="horizontal", mode="determinate", maximum=1.0, style="Accent.Horizontal.TProgressbar", length=170)
    bar.pack(side="right")
    lbl_exploit = ttk.Label(container, text="", style="Sub.TLabel")
    lbl_exploit.pack(anchor="e", pady=(0, 6))
    grid = ttk.Frame(container, style="Dark.TFrame")
    grid.pack(fill="both", expand=True)
    grid.columnconfigure(0, weight=1)
    grid.columnconfigure(1, weight=1)
    card_left = ttk.Frame(grid, style="Card.TFrame", padding=10)
    card_left.grid(row=0, column=0, sticky="nsew", padx=(0, 6))
    card_right = ttk.Frame(grid, style="Card.TFrame", padding=10)
    card_right.grid(row=0, column=1, sticky="nsew", padx=(6, 0))
    lbl_rl_title = ttk.Label(card_left, text="训练与模型", style="Caption.TLabel")
    lbl_rl_title.pack(anchor="w")
    lbl_rl = ttk.Label(card_left, text="", style="Metric.TLabel", justify="left")
    lbl_rl.pack(anchor="w", pady=(4, 2))
    loss_canvas = tk.Canvas(card_left, height=64, bg="#050812", highlightthickness=0, bd=0)
    loss_canvas.pack(fill="x", pady=(4, 0))
    lbl_ui_title = ttk.Label(card_right, text="对局与运行状态", style="Caption.TLabel")
    lbl_ui_title.pack(anchor="w")
    lbl_ui = ttk.Label(card_right, text="", style="Metric.TLabel", justify="left")
    lbl_ui.pack(anchor="w", pady=(4, 2))
    eps_canvas = tk.Canvas(card_right, height=64, bg="#050812", highlightthickness=0, bd=0)
    eps_canvas.pack(fill="x", pady=(4, 0))
    def draw_loss_chart():
        loss_canvas.delete("all")
        series = {}
        if len(loss_hist_rl) > 1:
            series["rl"] = list(loss_hist_rl)
        if len(loss_hist_bc) > 1:
            series["bc"] = list(loss_hist_bc)
        if len(loss_hist_ui) > 1:
            series["ui"] = list(loss_hist_ui)
        if not series:
            return
        w = max(1, loss_canvas.winfo_width())
        h = max(1, loss_canvas.winfo_height())
        vmax = None
        vmin = None
        for vals in series.values():
            if not vals:
                continue
            mx = max(vals)
            mn = min(vals)
            vmax = mx if vmax is None else max(vmax, mx)
            vmin = mn if vmin is None else min(vmin, mn)
        if vmax is None or vmin is None:
            return
        if vmax <= vmin:
            vmax = vmin + 1e-6
        colors = {"rl": "#4dc3ff", "bc": "#ffac3d", "ui": "#ff5b7a"}
        for key, vals in series.items():
            if len(vals) < 2:
                continue
            pts = []
            n = len(vals)
            for i, v in enumerate(vals):
                x = 2 + (w - 4) * i / float(max(1, n - 1))
                y = h - 2 - (h - 4) * (v - vmin) / float(vmax - vmin)
                pts.extend([x, y])
            loss_canvas.create_line(pts, fill=colors.get(key, "#ffffff"), width=1.5, smooth=True)
    def draw_eps_chart():
        eps_canvas.delete("all")
        if len(eps_hist) < 2:
            return
        w = max(1, eps_canvas.winfo_width())
        h = max(1, eps_canvas.winfo_height())
        vals = [1.0 - v for v in list(eps_hist)]
        n = len(vals)
        pts = []
        for i, v in enumerate(vals):
            x = 2 + (w - 4) * i / float(max(1, n - 1))
            y = h - 2 - (h - 4) * v
            pts.extend([x, y])
        eps_canvas.create_line(pts, fill="#3d7eff", width=1.5, smooth=True)
    def tick():
        if stop_flag:
            root.destroy()
            return
        if active:
            lbl_state.config(text="RUNNING", style="StateOn.TLabel")
        else:
            lbl_state.config(text="PAUSED", style="StateOff.TLabel")
        bar["value"] = max(0.0, min(1.0, 1.0 - eps_val))
        lbl_exploit.config(text="Exploit {:.1f}%   Eps {:.3f}".format((1.0 - eps_val) * 100.0, eps_val))
        total_ep = max(1, episodes_total)
        wr = 100.0 * episodes_win / float(total_ep)
        lbl_rl.config(text="Step: {}\nEpisodes: {}  WR: {:.1f}%\nRL Loss: {:.4f}\nBC Loss: {:.4f}\nUI Loss: {:.4f}\nReplay: {}  Demo: {}  UI: {}".format(
            agent.steps, episodes_total, wr, rl_loss_val, bc_loss_val, ui_loss_val, len(agent.mem), len(agent.demo), len(patch_buffer)))
        recent = ""
        if result_hist:
            seq = list(result_hist)[-16:]
            for r in seq:
                if r > 0:
                    recent += "W"
                elif r < 0:
                    recent += "L"
                else:
                    recent += "D"
        lbl_ui.config(text="FPS: {}\nWin/Draw/Loss: {}/{}/{}\n最近: {}".format(
            fps_val, episodes_win, episodes_draw, episodes_loss, recent if recent else "-"))
        draw_loss_chart()
        draw_eps_chart()
        root.after(200, tick)
    tick()
    root.mainloop()

def map_drag_to_action(rx0, ry0, rx1, ry1):
    if ry0 < CARD_BAR_Y_MIN:
        return None
    if ry1 < FIELD_Y_MIN or ry1 > FIELD_Y_MAX:
        return None
    card_pos = (rx0 - CARD_X_START) / max(0.01, CARD_X_STEP)
    card_idx = int(round(card_pos))
    card_idx = max(0, min(HAND_CARDS - 1, card_idx))
    if card_idx < 0 or card_idx >= HAND_CARDS:
        return None
    if rx1 < FIELD_X_MIN or rx1 > FIELD_X_MAX:
        return None
    best = None
    best_d = 1e9
    for i, (px, py) in enumerate(DROP_POINTS):
        dx = px - rx1
        dy = py - ry1
        d = dx * dx + dy * dy
        if d < best_d:
            best_d = d
            best = i
    if best is None:
        return None
    return card_idx * N_PLACES + best

def handle_demo(env, s, extra, human_mode):
    global dragging, drag_start_rx, drag_start_ry, last_lbtn, demo_save_buffer
    if s is None or extra is None:
        return
    lbtn = windll.user32.GetAsyncKeyState(0x01) & 0x8000
    pt = POINT()
    windll.user32.GetCursorPos(byref(pt))
    rel = env.inp.to_relative(pt.x, pt.y)
    if not human_mode:
        dragging = False
        last_lbtn = lbtn
        return
    if rel is None:
        last_lbtn = lbtn
        return
    rx, ry = rel
    if lbtn and not last_lbtn:
        if ry >= CARD_BAR_Y_MIN:
            dragging = True
            drag_start_rx, drag_start_ry = rx, ry
    elif not lbtn and last_lbtn and dragging:
        dragging = False
        a = map_drag_to_action(drag_start_rx, drag_start_ry, rx, ry)
        if a is not None:
            agent.push_demo(s.copy(), extra.copy(), a)
            demo_save_buffer.append((s.copy(), extra.copy(), a))
            if len(demo_save_buffer) >= DEMO_SAVE_CHUNK:
                try:
                    ss, ee, aa = zip(*demo_save_buffer)
                    ss_arr = np.stack(ss).astype(np.uint8)
                    ee_arr = np.stack(ee).astype(np.float32)
                    aa_arr = np.array(aa, dtype=np.int64)
                    fname = os.path.join(POOL_DIR, "demo_%d.npz" % int(time.time() * 1000))
                    np.savez_compressed(fname, s=ss_arr, extra=ee_arr, a=aa_arr)
                except Exception:
                    pass
                demo_save_buffer.clear()
    last_lbtn = lbtn

def flush_demo_buffer():
    global demo_save_buffer
    if not demo_save_buffer:
        return
    try:
        ss, ee, aa = zip(*demo_save_buffer)
        ss_arr = np.stack(ss).astype(np.uint8)
        ee_arr = np.stack(ee).astype(np.float32)
        aa_arr = np.array(aa, dtype=np.int64)
        fname = os.path.join(POOL_DIR, "demo_%d.npz" % int(time.time() * 1000))
        np.savez_compressed(fname, s=ss_arr, extra=ee_arr, a=aa_arr)
    except Exception:
        pass
    demo_save_buffer.clear()

def build_nstep_transition(queue):
    R = 0.0
    g = 1.0
    last = None
    steps = 0
    for tr in queue:
        s, extra, h, c, a, r, s_next, extra_next, nh, nc, done = tr
        R += g * float(r)
        g *= GAMMA
        steps += 1
        last = (s_next, extra_next, nh, nc, done)
        if done or steps >= N_STEP:
            break
    s0, extra0, h0, c0, a0, _, _, _, _, _, _ = queue[0]
    sN, extraN, hN, cN, doneN = last
    n_gamma = g
    return s0, extra0, h0, c0, a0, R, n_gamma, sN, extraN, hN, cN, float(doneN)

def main():
    global active, stop_flag, fps_val, eps_val, episodes_total, episodes_win, episodes_loss, episodes_draw
    ensure_emulator()
    adb_connect()
    env = Env()
    t_train = threading.Thread(target=train_thread, daemon=True)
    t_train.start()
    t_gui = threading.Thread(target=gui_thread, daemon=True)
    t_gui.start()
    h = torch.zeros(1, 512, device=DEVICE)
    c = torch.zeros(1, 512, device=DEVICE)
    last_f8 = 0
    frame_timer = time.time()
    frames = 0
    s = None
    extra = None
    nstep_buffer = collections.deque(maxlen=N_STEP)
    while True:
        now = time.time()
        if now - frame_timer >= 1.0:
            fps_val = frames
            frames = 0
            frame_timer = now
        f8 = windll.user32.GetAsyncKeyState(0x77) & 0x8000
        if f8 and not last_f8:
            active = not active
            nstep_buffer.clear()
            if active:
                s, extra = env.reset()
                h.zero_()
                c.zero_()
        last_f8 = f8
        esc = windll.user32.GetAsyncKeyState(0x1B) & 0x8000
        if esc:
            stop_flag = True
            break
        if active:
            if s is None or extra is None:
                s_obs, extra_obs, done_flag = env.observe()
                if done_flag or s_obs is None:
                    time.sleep(0.1)
                    continue
                s, extra = s_obs, extra_obs
                h.zero_()
                c.zero_()
            eps = EPS_END + (EPS_START - EPS_END) * math.exp(-1.0 * agent.steps / EPS_DECAY)
            eps_val = eps
            eps_hist.append(eps)
            a, nh, nc = agent.act(s, extra, h, c, eps)
            s_next, extra_next, r, done = env.step(a)
            if s_next is not None:
                trans = (s, extra, h.detach().cpu(), c.detach().cpu(), a, float(r), s_next, extra_next, nh.detach().cpu(), nc.detach().cpu(), float(done))
                nstep_buffer.append(trans)
                if len(nstep_buffer) >= N_STEP:
                    n_tr = build_nstep_transition(nstep_buffer)
                    agent.mem.push(n_tr)
                    nstep_buffer.popleft()
                if done:
                    while nstep_buffer:
                        n_tr = build_nstep_transition(nstep_buffer)
                        agent.mem.push(n_tr)
                        nstep_buffer.popleft()
                    episodes_total += 1
                    if env.last_result > 0:
                        episodes_win += 1
                    elif env.last_result < 0:
                        episodes_loss += 1
                    else:
                        episodes_draw += 1
                    result_hist.append(env.last_result)
                    s, extra = env.reset()
                    h.zero_()
                    c.zero_()
                else:
                    s, extra, h, c = s_next, extra_next, nh, nc
                agent.steps += 1
                frames += 1
            else:
                time.sleep(0.05)
        else:
            s_obs, extra_obs, done = env.observe()
            if not done and s_obs is not None:
                s, extra = s_obs, extra_obs
                frames += 1
            time.sleep(0.03)
        handle_demo(env, s, extra, not active)
    stop_flag = True
    flush_demo_buffer()
    agent.save()
    time.sleep(0.5)

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        stop_flag = True
        flush_demo_buffer()
        agent.save()
        time.sleep(0.5)
