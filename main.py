import os
import time
import math
import random
import subprocess
import ctypes
import threading
import numpy as np
import cv2
import mss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from ctypes import windll, wintypes
import tkinter as tk
from tkinter import ttk

USER_HOME = os.path.expanduser("~")
DESKTOP_DIR = os.path.join(USER_HOME, "Desktop")
BASE_DIR = os.path.join(DESKTOP_DIR, "GameAI")
os.makedirs(BASE_DIR, exist_ok=True)

ADB_BIN = os.environ.get("CR_ADB_BIN", r"D:\LDPlayer9\adb.exe")
EMU_BIN = os.environ.get("CR_EMU_BIN", r"D:\LDPlayer9\dnplayer.exe")
WIN_NAME = os.environ.get("CR_WINDOW_NAME", "LDPlayer")
ADB_ID = os.environ.get("CR_ADB_ID", "127.0.0.1:5555")

SEED = int(os.environ.get("CR_SEED", "42"))

H = int(os.environ.get("CR_FRAME_H", "84"))
W = int(os.environ.get("CR_FRAME_W", "84"))
STACK = int(os.environ.get("CR_STACK", "4"))

GRID_SIZE = int(os.environ.get("CR_GRID", "6"))
NCARDS = int(os.environ.get("CR_NCARDS", "4"))
EXTRA_ACTIONS = int(os.environ.get("CR_EXTRA_ACT", "5"))
ACTION_DIM = GRID_SIZE * GRID_SIZE * NCARDS + EXTRA_ACTIONS

N_FEAT = int(os.environ.get("CR_NFEAT", "16"))

BUFFER_SIZE = int(os.environ.get("CR_BUFFER_SIZE", "16000"))
BATCH_SIZE_BASE = int(os.environ.get("CR_BATCH_SIZE", "64"))
GAMMA = float(os.environ.get("CR_GAMMA", "0.99"))

EPS_START = float(os.environ.get("CR_EPS_START", "0.2"))
EPS_END = float(os.environ.get("CR_EPS_END", "0.01"))
EPS_DECAY = int(os.environ.get("CR_EPS_DECAY", "300000"))

LEARN_FREQ = int(os.environ.get("CR_LEARN_FREQ", "2"))
TARGET_SYNC = int(os.environ.get("CR_TARGET_SYNC", "5000"))
SAVE_INTERVAL = int(os.environ.get("CR_SAVE_INTERVAL", "5000"))
MIN_REPLAY = int(os.environ.get("CR_MIN_REPLAY", "2000"))

NO_CHANGE_LIMIT = int(os.environ.get("CR_NOCHANGE", "80"))
IDLE_REWARD = float(os.environ.get("CR_IDLE_REWARD", "-0.02"))
MAX_EPISODE_LEN = int(os.environ.get("CR_MAX_EP_LEN", "2200"))

LR = float(os.environ.get("CR_LR", "3e-4"))
WEIGHT_DECAY = float(os.environ.get("CR_WD", "1e-5"))
GRAD_NORM_CLIP = float(os.environ.get("CR_GRAD_CLIP", "10.0"))

STEP_DELAY = float(os.environ.get("CR_STEP_DELAY", "0.08"))
FRAME_WARMUP_TRIES = int(os.environ.get("CR_WARMUP_TRIES", "60"))
FRAME_WARMUP_SLEEP = float(os.environ.get("CR_WARMUP_SLEEP", "0.5"))

PER_ALPHA = float(os.environ.get("CR_PER_ALPHA", "0.6"))
PER_BETA_START = float(os.environ.get("CR_PER_BETA_START", "0.4"))
PER_BETA_FRAMES = int(os.environ.get("CR_PER_BETA_FRAMES", "200000"))

TRAINING_ENABLED = os.environ.get("CR_TRAIN", "1").lower() not in ("0", "false", "no")

MODEL_PATH = os.path.join(BASE_DIR, "clash_ai_model_v2.pth")
BEST_MODEL_PATH = os.path.join(BASE_DIR, "clash_ai_model_v2_best.pth")
LOG_PATH = os.path.join(BASE_DIR, "training_log.txt")

VK_F8 = 0x77

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if hasattr(torch.backends, "cudnn"):
    torch.backends.cudnn.benchmark = device.type == "cuda"

use_amp = device.type == "cuda"
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler
    AMP_FROM_TORCH_AMP = True
except Exception:
    from torch.cuda.amp import autocast as _autocast, GradScaler as _GradScaler
    AMP_FROM_TORCH_AMP = False
scaler = _GradScaler(enabled=use_amp)


class _NullContext:
    def __enter__(self):
        return None
    def __exit__(self, exc_type, exc_val, exc_tb):
        return False


def amp_context():
    if not use_amp:
        return _NullContext()
    if AMP_FROM_TORCH_AMP:
        return _autocast(device_type="cuda", enabled=True)
    return _autocast(enabled=True)


def set_global_seed(seed):
    try:
        seed_int = int(seed)
    except Exception:
        seed_int = 42
    random.seed(seed_int)
    np.random.seed(seed_int)
    torch.manual_seed(seed_int)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_int)


def build_coord_map():
    ys = torch.linspace(-1.0, 1.0, H).view(1, 1, H, 1).expand(1, 1, H, W)
    xs = torch.linspace(-1.0, 1.0, W).view(1, 1, 1, W).expand(1, 1, H, W)
    return torch.cat((xs, ys), dim=1)


try:
    windll.shcore.SetProcessDpiAwareness(1)
except Exception:
    pass

try:
    KEY_TOGGLE_SUPPORTED = bool(windll.user32.GetAsyncKeyState)
except Exception:
    KEY_TOGGLE_SUPPORTED = False


class ScreenCapture:
    def __init__(self, name, height, width):
        self.name = name
        self.height = height
        self.width = width
        self.hwnd = None
        self.sct = mss.mss()
        self._enum_cb = self._create_enum_cb()
    def _create_enum_cb(self):
        def callback(hwnd, lparam):
            if not windll.user32.IsWindowVisible(hwnd):
                return True
            length = windll.user32.GetWindowTextLengthW(hwnd)
            if length <= 0:
                return True
            buf = ctypes.create_unicode_buffer(length + 1)
            windll.user32.GetWindowTextW(hwnd, buf, length + 1)
            title = buf.value
            if not title:
                return True
            if self.name.lower() in title.lower():
                self.hwnd = hwnd
                return False
            return True
        return ctypes.WINFUNCTYPE(ctypes.c_bool, wintypes.HWND, wintypes.LPARAM)(callback)
    def _update_hwnd(self):
        hwnd = windll.user32.FindWindowW(None, self.name)
        if hwnd:
            self.hwnd = hwnd
            return
        windll.user32.EnumWindows(self._enum_cb, 0)
    def grab(self):
        if not self.hwnd or not windll.user32.IsWindow(self.hwnd):
            self._update_hwnd()
            if not self.hwnd or not windll.user32.IsWindow(self.hwnd):
                return np.zeros((self.height, self.width), dtype=np.uint8)
        rect = wintypes.RECT()
        if not windll.user32.GetClientRect(self.hwnd, ctypes.byref(rect)):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        pt = wintypes.POINT()
        if not windll.user32.ClientToScreen(self.hwnd, ctypes.byref(pt)):
            return np.zeros((self.height, self.width), dtype=np.uint8)
        width = rect.right - rect.left
        height = rect.bottom - rect.top
        if width <= 0 or height <= 0:
            return np.zeros((self.height, self.width), dtype=np.uint8)
        monitor = {"top": pt.y, "left": pt.x, "width": width, "height": height}
        try:
            img = np.asarray(self.sct.grab(monitor))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            resized = cv2.resize(gray, (self.width, self.height), interpolation=cv2.INTER_AREA)
            return resized
        except Exception:
            return np.zeros((self.height, self.width), dtype=np.uint8)


class InputBot:
    def __init__(self):
        self.proc = None
        self.last_connect_time = 0.0
    def _ensure_shell(self):
        now = time.time()
        if now - self.last_connect_time > 5.0:
            self.last_connect_time = now
            if os.path.exists(ADB_BIN):
                try:
                    subprocess.run([ADB_BIN, "connect", ADB_ID], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
                except Exception:
                    pass
        if self.proc is None or self.proc.poll() is not None:
            if not os.path.exists(ADB_BIN):
                self.proc = None
                return
            creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" and hasattr(subprocess, "CREATE_NO_WINDOW") else 0
            try:
                self.proc = subprocess.Popen(
                    [ADB_BIN, "-s", ADB_ID, "shell"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    bufsize=0,
                    creationflags=creationflags
                )
            except Exception:
                self.proc = None
    def _send(self, text):
        self._ensure_shell()
        if self.proc is None or self.proc.stdin is None:
            return
        try:
            self.proc.stdin.write(text.encode("ascii"))
            self.proc.stdin.flush()
        except Exception:
            try:
                self.proc.kill()
            except Exception:
                pass
            self.proc = None
    def tap(self, x, y, screen_w, screen_h):
        tx = int(max(0, min(screen_w - 1, x)))
        ty = int(max(0, min(screen_h - 1, y)))
        self._send("input tap {} {}\n".format(tx, ty))
    def swipe(self, x1, y1, x2, y2, duration_ms, screen_w, screen_h):
        tx1 = int(max(0, min(screen_w - 1, x1)))
        ty1 = int(max(0, min(screen_h - 1, y1)))
        tx2 = int(max(0, min(screen_w - 1, x2)))
        ty2 = int(max(0, min(screen_h - 1, y2)))
        self._send("input swipe {} {} {} {} {}\n".format(tx1, ty1, tx2, ty2, int(duration_ms)))
    def back(self):
        self._send("input keyevent 4\n")
    def act(self, action_index, screen_w, screen_h):
        info = {"action_index": int(action_index), "type": "none", "from": None, "to": None, "extra": None}
        total_board_actions = GRID_SIZE * GRID_SIZE * NCARDS
        if action_index < 0:
            action_index = 0
        if action_index >= ACTION_DIM:
            action_index = ACTION_DIM - 1
        info["action_index"] = int(action_index)
        if action_index < total_board_actions:
            card = action_index // (GRID_SIZE * GRID_SIZE)
            pos = action_index % (GRID_SIZE * GRID_SIZE)
            r, c = divmod(pos, GRID_SIZE)
            card_y = int(screen_h * 0.9)
            slot_width = screen_w / (NCARDS + 2.0)
            card_x = int(slot_width * (card + 1.5))
            top_arena = int(screen_h * 0.18)
            bottom_arena = int(screen_h * 0.78)
            left_arena = int(screen_w * 0.05)
            right_arena = int(screen_w * 0.95)
            grid_w = (right_arena - left_arena) / GRID_SIZE
            grid_h = (bottom_arena - top_arena) / GRID_SIZE
            target_x = int(left_arena + (c + 0.5) * grid_w)
            target_y = int(top_arena + (r + 0.5) * grid_h)
            self.swipe(card_x, card_y, target_x, target_y, 80, screen_w, screen_h)
            info["type"] = "card_play"
            info["from"] = (card_x, card_y)
            info["to"] = (target_x, target_y)
            info["extra"] = {"card_slot": int(card), "grid_rc": (int(r), int(c))}
        else:
            rem = action_index - total_board_actions
            if rem == 0:
                info["type"] = "noop"
            elif rem == 1:
                tx = screen_w // 2
                ty = int(screen_h * 0.82)
                self.tap(tx, ty, screen_w, screen_h)
                info["type"] = "tap_bottom_center"
                info["from"] = (tx, ty)
            elif rem == 2:
                tx = screen_w // 2
                ty = int(screen_h * 0.6)
                self.tap(tx, ty, screen_w, screen_h)
                info["type"] = "tap_middle"
                info["from"] = (tx, ty)
            elif rem == 3:
                self.back()
                info["type"] = "back"
            elif rem == 4:
                rx = random.randint(int(screen_w * 0.2), int(screen_w * 0.8))
                ry = random.randint(int(screen_h * 0.25), int(screen_h * 0.75))
                self.tap(rx, ry, screen_w, screen_h)
                info["type"] = "tap_random"
                info["from"] = (rx, ry)
        return info
    def close(self):
        if self.proc is not None:
            try:
                self.proc.terminate()
            except Exception:
                try:
                    self.proc.kill()
                except Exception:
                    pass
            self.proc = None


class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        self.bias_sigma = nn.Parameter(torch.empty(out_features))
        self.register_buffer("bias_epsilon", torch.empty(out_features))
        self.sigma_init = sigma_init
        self.reset_parameters()
        self.reset_noise()
    def reset_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        nn.init.uniform_(self.weight_mu, -bound, bound)
        nn.init.uniform_(self.bias_mu, -bound, bound)
        nn.init.constant_(self.weight_sigma, self.sigma_init)
        nn.init.constant_(self.bias_sigma, self.sigma_init)
    def reset_noise(self):
        eps_in = self._scale_noise(self.in_features)
        eps_out = self._scale_noise(self.out_features)
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)
    def _scale_noise(self, size):
        x = torch.randn(size)
        return x.sign().mul_(x.abs().sqrt_())
    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias = self.bias_mu + self.bias_sigma * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)


class DuelNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer("coord", build_coord_map())
        self.cnn = nn.Sequential(
            nn.Conv2d(STACK + 2, 32, 8, 4),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 4, 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, 1),
            nn.ReLU(inplace=True)
        )
        with torch.no_grad():
            dummy = torch.zeros(1, STACK + 2, H, W)
            n_flat = self.cnn(dummy).view(1, -1).shape[1]
        self.adv_fc1 = NoisyLinear(n_flat + N_FEAT, 512)
        self.adv_fc2 = NoisyLinear(512, ACTION_DIM)
        self.val_fc1 = NoisyLinear(n_flat + N_FEAT, 512)
        self.val_fc2 = NoisyLinear(512, 1)
    def forward(self, x, feat=None):
        coord = self.coord.expand(x.size(0), -1, -1, -1)
        x = torch.cat((x, coord), dim=1)
        x = self.cnn(x).view(x.size(0), -1)
        if feat is None:
            feat = x.new_zeros((x.size(0), N_FEAT))
        else:
            if feat.dim() == 1:
                feat = feat.unsqueeze(0)
            feat = feat.to(dtype=x.dtype)
            if feat.size(0) != x.size(0):
                feat = feat[: x.size(0)]
        h = torch.cat((x, feat), dim=1)
        adv = F.relu(self.adv_fc1(h))
        adv = self.adv_fc2(adv)
        val = F.relu(self.val_fc1(h))
        val = self.val_fc2(val)
        return val + adv - adv.mean(dim=1, keepdim=True)
    def reset_noise(self):
        self.adv_fc1.reset_noise()
        self.adv_fc2.reset_noise()
        self.val_fc1.reset_noise()
        self.val_fc2.reset_noise()


class PrioritizedReplay:
    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.ptr = 0
        self.size = 0
        self.s = np.zeros((capacity, STACK, H, W), dtype=np.uint8)
        self.sf = np.zeros((capacity, N_FEAT), dtype=np.float32)
        self.a = np.zeros((capacity, 1), dtype=np.int64)
        self.r = np.zeros((capacity, 1), dtype=np.float32)
        self.ns = np.zeros((capacity, STACK, H, W), dtype=np.uint8)
        self.nsf = np.zeros((capacity, N_FEAT), dtype=np.float32)
        self.d = np.zeros((capacity, 1), dtype=np.float32)
        self.p = np.zeros((capacity, 1), dtype=np.float32)
        self.rng = np.random.default_rng(SEED)
    def push(self, s, sf, a, r, ns, nsf, d):
        i = self.ptr
        self.s[i] = s
        self.sf[i] = sf
        self.a[i, 0] = int(a)
        self.r[i, 0] = float(r)
        self.ns[i] = ns
        self.nsf[i] = nsf
        self.d[i, 0] = float(d)
        max_p = self.p.max() if self.size > 0 else 1.0
        self.p[i, 0] = max_p
        self.ptr = (self.ptr + 1) % self.capacity
        if self.size < self.capacity:
            self.size += 1
    def sample(self, batch_size, global_step):
        if self.size == 0:
            raise RuntimeError("empty buffer")
        batch_size = min(batch_size, self.size)
        probs = self.p[: self.size, 0] ** PER_ALPHA
        probs_sum = float(probs.sum())
        if probs_sum <= 0.0:
            probs[:] = 1.0
            probs_sum = float(probs.sum())
        probs /= probs_sum
        idx = self.rng.choice(self.size, batch_size, replace=False, p=probs)
        beta = PER_BETA_START + (1.0 - PER_BETA_START) * min(1.0, global_step / float(PER_BETA_FRAMES))
        weights = (self.size * probs[idx]) ** (-beta)
        weights /= weights.max()
        s = torch.from_numpy(self.s[idx]).to(device=device, dtype=torch.float32).div_(255.0)
        sf = torch.from_numpy(self.sf[idx]).to(device=device, dtype=torch.float32)
        a = torch.from_numpy(self.a[idx]).to(device=device)
        r = torch.from_numpy(self.r[idx]).to(device=device, dtype=torch.float32)
        ns = torch.from_numpy(self.ns[idx]).to(device=device, dtype=torch.float32).div_(255.0)
        nsf = torch.from_numpy(self.nsf[idx]).to(device=device, dtype=torch.float32)
        d = torch.from_numpy(self.d[idx]).to(device=device, dtype=torch.float32)
        w = torch.from_numpy(weights.reshape(-1, 1)).to(device=device, dtype=torch.float32)
        return idx, s, sf, a, r, ns, nsf, d, w
    def update_priorities(self, idx, td_errors):
        td = np.abs(td_errors) + 1e-6
        td = np.clip(td, 1e-6, 10.0)
        self.p[idx, 0] = td


def start_emulator_if_needed():
    if not os.path.exists(EMU_BIN):
        return
    tasklist_output = ""
    if os.name == "nt":
        try:
            tasklist_output = os.popen("tasklist").read().lower()
        except Exception:
            tasklist_output = ""
    if "dnplayer.exe" in tasklist_output:
        return
    try:
        subprocess.Popen(EMU_BIN, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        return
    time.sleep(20.0)


def get_screen_size(hwnd):
    if not hwnd:
        return 1280, 720
    rect = wintypes.RECT()
    if not windll.user32.GetClientRect(hwnd, ctypes.byref(rect)):
        return 1280, 720
    width = rect.right - rect.left
    height = rect.bottom - rect.top
    if width <= 0 or height <= 0:
        return 1280, 720
    return width, height


def save_checkpoint(path, model_state, optim_state, sem_state, steps):
    obj = {"model": model_state, "optim": optim_state, "sem": sem_state, "steps": int(steps)}
    try:
        torch.save(obj, path)
    except Exception:
        pass


def load_checkpoint(path, model, optimizer, sem_encoder):
    steps = 0
    if not os.path.exists(path):
        return steps
    try:
        ckpt = torch.load(path, map_location=device)
    except Exception:
        return steps
    if isinstance(ckpt, dict) and "model" in ckpt:
        try:
            model.load_state_dict(ckpt["model"])
        except Exception:
            return steps
        optim_state = ckpt.get("optim", None)
        if isinstance(optim_state, dict):
            try:
                optimizer.load_state_dict(optim_state)
            except Exception:
                optimizer.state = {}
        sem_state = ckpt.get("sem", None)
        if isinstance(sem_state, dict):
            try:
                sem_encoder.load_state_dict(sem_state)
            except Exception:
                pass
        steps = int(ckpt.get("steps", 0))
    elif isinstance(ckpt, dict):
        try:
            model.load_state_dict(ckpt)
        except Exception:
            return steps
    return steps


def init_first_frame(cap):
    frame = cap.grab()
    if frame.size != H * W:
        frame = np.zeros((H, W), dtype=np.uint8)
    if frame.mean() == 0:
        for _ in range(FRAME_WARMUP_TRIES):
            time.sleep(FRAME_WARMUP_SLEEP)
            frame = cap.grab()
            if frame.mean() > 0:
                break
    if frame.size != H * W:
        frame = np.zeros((H, W), dtype=np.uint8)
    return frame


def build_arena_weights():
    y = np.linspace(0.0, 1.0, H, dtype=np.float32)
    center = 0.5
    sigma = 0.25
    wy = np.exp(-0.5 * ((y - center) / sigma) ** 2)
    wy = 0.25 + 0.75 * (wy / np.max(wy))
    weights = np.repeat(wy.reshape(H, 1), W, axis=1)
    return weights.astype(np.float32)


class SemanticEncoder:
    def __init__(self):
        self.f_min = np.full((N_FEAT,), np.inf, dtype=np.float32)
        self.f_max = np.full((N_FEAT,), -np.inf, dtype=np.float32)
    def _region(self, img, cx, cy, wr, hr):
        w = max(1, int(W * wr))
        h = max(1, int(H * hr))
        x1 = int(cx * W - w / 2)
        y1 = int(cy * H - h / 2)
        x1 = max(0, min(W - 1, x1))
        y1 = max(0, min(H - 1, y1))
        x2 = min(W, x1 + w)
        y2 = min(H, y1 + h)
        if x2 <= x1 or y2 <= y1:
            return 0.0
        patch = img[y1:y2, x1:x2]
        return float(patch.mean())
    def extract(self, frame):
        if frame.size != H * W:
            frame = np.zeros((H, W), dtype=np.uint8)
        img = frame.astype(np.float32) / 255.0
        if img.mean() == 0.0:
            return np.zeros((N_FEAT,), dtype=np.float32)
        f = np.zeros((N_FEAT,), dtype=np.float32)
        f[0] = self._region(img, 0.5, 0.86, 0.16, 0.16)
        f[1] = self._region(img, 0.25, 0.78, 0.14, 0.14)
        f[2] = self._region(img, 0.75, 0.78, 0.14, 0.14)
        f[3] = self._region(img, 0.5, 0.14, 0.16, 0.16)
        f[4] = self._region(img, 0.25, 0.22, 0.14, 0.14)
        f[5] = self._region(img, 0.75, 0.22, 0.14, 0.14)
        f[6] = self._region(img, 0.5, 0.96, 0.7, 0.08)
        f[7] = self._region(img, 0.5, 0.04, 0.7, 0.10)
        f[8] = self._region(img, 0.18, 0.94, 0.18, 0.08)
        f[9] = self._region(img, 0.36, 0.94, 0.18, 0.08)
        f[10] = self._region(img, 0.64, 0.94, 0.18, 0.08)
        f[11] = self._region(img, 0.82, 0.94, 0.18, 0.08)
        f[12] = self._region(img, 0.5, 0.68, 0.8, 0.26)
        f[13] = self._region(img, 0.5, 0.32, 0.8, 0.26)
        mid_h = H // 2
        f[14] = float(img[mid_h:, :].mean())
        f[15] = float(img[:mid_h, :].mean())
        self.f_min = np.minimum(self.f_min, f)
        self.f_max = np.maximum(self.f_max, f)
        span = self.f_max - self.f_min
        span[span < 1e-3] = 1.0
        norm = (f - self.f_min) / span
        norm = np.clip(norm, 0.0, 1.0)
        return norm
    def state_dict(self):
        return {"f_min": self.f_min, "f_max": self.f_max}
    def load_state_dict(self, state):
        f_min = state.get("f_min", None)
        f_max = state.get("f_max", None)
        if isinstance(f_min, np.ndarray) and f_min.shape == (N_FEAT,):
            self.f_min = f_min.astype(np.float32)
        if isinstance(f_max, np.ndarray) and f_max.shape == (N_FEAT,):
            self.f_max = f_max.astype(np.float32)


SEM_ENCODER = SemanticEncoder()


def extract_semantic_features(frame):
    return SEM_ENCODER.extract(frame)


def soft_reset(bot, cap):
    hwnd = cap.hwnd
    if not hwnd:
        return
    screen_w, screen_h = get_screen_size(hwnd)
    for _ in range(3):
        bot.back()
        time.sleep(0.4)
    bot.tap(screen_w // 2, int(screen_h * 0.82), screen_w, screen_h)
    time.sleep(2.0)
    bot.tap(screen_w // 2, int(screen_h * 0.6), screen_w, screen_h)
    time.sleep(2.0)


def append_log(text):
    try:
        with open(LOG_PATH, "a", encoding="utf-8") as f:
            f.write(text + "\n")
    except Exception:
        pass


class ExperienceLogger:
    def __init__(self, base_dir):
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.session_dir = os.path.join(base_dir, "sessions", ts)
        os.makedirs(self.session_dir, exist_ok=True)
        self.frames_dir = os.path.join(self.session_dir, "frames")
        os.makedirs(self.frames_dir, exist_ok=True)
        self.meta_path = os.path.join(self.session_dir, "steps.tsv")
        self.lock = threading.Lock()
        self.step_index = 0
        self.meta_file = None
        try:
            self.meta_file = open(self.meta_path, "a", encoding="utf-8")
            header = "step_id\ttimestamp\tglobal_step\tepisode\tstep_in_episode\tai_enabled\taction_index\treward\tdone\tframe_file\taction_detail\n"
            self.meta_file.write(header)
            self.meta_file.flush()
        except Exception:
            self.meta_file = None
    def log_step(self, global_step, episode, step_in_episode, ai_enabled, frame, action_info, reward, done):
        if self.meta_file is None:
            return
        with self.lock:
            fname = "g{:09d}_e{:05d}_s{:05d}.png".format(int(global_step), int(episode), int(step_in_episode))
            fpath = os.path.join(self.frames_dir, fname)
            frame_file = ""
            try:
                cv2.imwrite(fpath, frame)
                frame_file = fname
            except Exception:
                frame_file = ""
            ts = time.time()
            if isinstance(action_info, dict):
                action_index = int(action_info.get("action_index", -1))
                detail = str(action_info).replace("\t", " ")
            else:
                action_index = -1
                detail = str(action_info)
            line = "{}\t{:.3f}\t{}\t{}\t{}\t{}\t{}\t{:.5f}\t{}\t{}\t{}\n".format(
                int(self.step_index),
                ts,
                int(global_step),
                int(episode),
                int(step_in_episode),
                int(bool(ai_enabled)),
                action_index,
                float(reward),
                int(bool(done)),
                frame_file,
                detail
            )
            try:
                self.meta_file.write(line)
                self.meta_file.flush()
            except Exception:
                pass
            self.step_index += 1
    def close(self):
        if self.meta_file is not None:
            try:
                self.meta_file.close()
            except Exception:
                pass
            self.meta_file = None


class AudioRecorder:
    def __init__(self, base_dir):
        self.base_dir = base_dir
        self.thread = None
        self.stop_event = threading.Event()
        self.enabled = False
        self.has_lib = False
        self.sd = None
        self.sf = None
        try:
            import sounddevice as sd
            import soundfile as sf
            self.sd = sd
            self.sf = sf
            self.has_lib = True
        except Exception:
            self.has_lib = False
        if self.has_lib:
            self.enabled = True
            self.audio_path = os.path.join(self.base_dir, "audio.wav")
        else:
            self.audio_path = None
    def start(self):
        if not self.enabled:
            return
        if self.thread is not None:
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
    def run(self):
        if not self.enabled or not self.has_lib or self.audio_path is None:
            return
        fs = 44100
        channels = 2
        try:
            with self.sf.SoundFile(self.audio_path, mode="w", samplerate=fs, channels=channels) as f:
                def callback(indata, frames, time_info, status):
                    if self.stop_event.is_set():
                        raise KeyboardInterrupt
                    try:
                        f.write(indata.copy())
                    except Exception:
                        pass
                with self.sd.InputStream(samplerate=fs, channels=channels, callback=callback):
                    while not self.stop_event.is_set():
                        time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        except Exception:
            pass
    def stop(self):
        if not self.enabled:
            return
        self.stop_event.set()
        if self.thread is not None:
            try:
                self.thread.join(timeout=2.0)
            except Exception:
                pass
        self.thread = None


class SharedState:
    def __init__(self):
        self.lock = threading.Lock()
        self.global_steps = 0
        self.eps = 0.0
        self.last_loss = 0.0
        self.episode_reward = 0.0
        self.episode_len = 0
        self.buffer_size = 0
        self.ai_enabled = False
        self.mode_text = "训练+对战" if TRAINING_ENABLED else "仅推理对战"
        self.status_text = "初始化中"
        self.episode_count = 0
        self.best_score = None
        self.toggle_requested = False
        self.error_text = ""
    def snapshot(self):
        with self.lock:
            return {
                "global_steps": self.global_steps,
                "eps": self.eps,
                "last_loss": self.last_loss,
                "episode_reward": self.episode_reward,
                "episode_len": self.episode_len,
                "buffer_size": self.buffer_size,
                "ai_enabled": self.ai_enabled,
                "mode_text": self.mode_text,
                "status_text": self.status_text,
                "episode_count": self.episode_count,
                "best_score": self.best_score,
                "error_text": self.error_text,
            }
    def update_stats(self, global_steps, eps, last_loss, episode_reward, episode_len, buffer_size, episode_count, best_score, ai_enabled, status_text):
        with self.lock:
            self.global_steps = global_steps
            self.eps = eps
            self.last_loss = last_loss
            self.episode_reward = episode_reward
            self.episode_len = episode_len
            self.buffer_size = buffer_size
            self.episode_count = episode_count
            self.best_score = best_score
            self.ai_enabled = ai_enabled
            self.status_text = status_text
    def request_toggle(self):
        with self.lock:
            self.toggle_requested = True
    def consume_toggle(self):
        with self.lock:
            if self.toggle_requested:
                self.toggle_requested = False
                return True
            return False
    def set_error(self, text):
        with self.lock:
            self.error_text = text
            self.status_text = text


def rl_loop(shared_state, stop_event):
    bot = None
    policy = None
    optimizer = None
    global_steps = 0
    exp_logger = None
    audio_recorder = None
    try:
        if os.name != "nt":
            shared_state.set_error("当前脚本仅支持 Windows 运行环境")
            return
        set_global_seed(SEED)
        if device.type == "cpu":
            try:
                n_threads = os.cpu_count() or 1
                torch.set_num_threads(max(1, min(8, n_threads)))
            except Exception:
                pass
        start_emulator_if_needed()
        if not os.path.exists(ADB_BIN):
            print("未找到 ADB，可在环境变量 CR_ADB_BIN 或脚本中修改路径:", ADB_BIN, flush=True)
        if not os.path.exists(EMU_BIN):
            print("未找到 LDPlayer 模拟器，可在环境变量 CR_EMU_BIN 或脚本中修改路径:", EMU_BIN, flush=True)
        print("数据和模型将保存在:", BASE_DIR, flush=True)
        print("请在 LDPlayer 中启动《部落冲突:皇室战争》，进入对战界面后按 F8 让 AI 接管，再次按 F8 切回手动，关闭窗口即可结束。", flush=True)
        cap = ScreenCapture(WIN_NAME, H, W)
        bot = InputBot()
        memory = PrioritizedReplay(BUFFER_SIZE)
        exp_logger = ExperienceLogger(BASE_DIR)
        audio_recorder = AudioRecorder(exp_logger.session_dir)
        audio_recorder.start()
        policy = DuelNet().to(device)
        target = DuelNet().to(device)
        target.load_state_dict(policy.state_dict())
        target.eval()
        optimizer = optim.AdamW(policy.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        global_steps = load_checkpoint(MODEL_PATH, policy, optimizer, SEM_ENCODER)
        target.load_state_dict(policy.state_dict())
        frames = np.zeros((STACK, H, W), dtype=np.uint8)
        first_frame = init_first_frame(cap)
        frames[:] = first_frame
        state = frames.copy()
        last_frame = first_frame
        last_feat = extract_semantic_features(first_frame)
        state_feat = last_feat.copy()
        no_change_steps = 0
        last_loss = 0.0
        arena_weights = build_arena_weights()
        episode_reward = 0.0
        episode_len = 0
        ai_enabled = not KEY_TOGGLE_SUPPORTED
        last_toggle_state = False
        best_score = None
        ema_reward = None
        episode_count = 0
        eps = EPS_START
        if KEY_TOGGLE_SUPPORTED:
            if TRAINING_ENABLED:
                print("F8 切换 AI 控制开关，当前: 关闭，模式: 训练+对战", flush=True)
            else:
                print("F8 切换 AI 控制开关，当前: 关闭，模式: 仅推理对战", flush=True)
            ai_enabled = False
        else:
            if TRAINING_ENABLED:
                print("未检测到 F8 快捷键支持，AI 将直接接管并进行训练", flush=True)
            else:
                print("未检测到 F8 快捷键支持，AI 将直接接管并仅推理", flush=True)
            ai_enabled = True
        status_text = "AI 对战中" if ai_enabled else "等待玩家操作 (按 F8 或窗口按钮开启 AI)"
        shared_state.update_stats(global_steps, eps, last_loss, episode_reward, episode_len, memory.size, episode_count, best_score, ai_enabled, status_text)
        while not stop_event.is_set():
            if KEY_TOGGLE_SUPPORTED:
                try:
                    key_state = bool(windll.user32.GetAsyncKeyState(VK_F8) & 0x8000)
                except Exception:
                    key_state = False
                if key_state and not last_toggle_state:
                    ai_enabled = not ai_enabled
                    if TRAINING_ENABLED:
                        print("AI 控制已{}，模式: 训练+对战".format("开启" if ai_enabled else "关闭"), flush=True)
                    else:
                        print("AI 控制已{}，模式: 仅推理对战".format("开启" if ai_enabled else "关闭"), flush=True)
                last_toggle_state = key_state
            if shared_state.consume_toggle():
                ai_enabled = not ai_enabled
                if TRAINING_ENABLED:
                    print("AI 控制已{} (来自 UI 按钮)，模式: 训练+对战".format("开启" if ai_enabled else "关闭"), flush=True)
                else:
                    print("AI 控制已{} (来自 UI 按钮)，模式: 仅推理对战".format("开启" if ai_enabled else "关闭"), flush=True)
            if not ai_enabled:
                status_text = "等待玩家操作 (按 F8 或窗口按钮开启 AI)"
                frame = cap.grab()
                if frame.size != H * W:
                    frame = np.zeros((H, W), dtype=np.uint8)
                frames = np.concatenate((frames[1:], frame[np.newaxis, ...]), axis=0)
                state = frames.copy()
                last_frame = frame
                last_feat = extract_semantic_features(frame)
                state_feat = last_feat.copy()
                no_change_steps = 0
                episode_reward = 0.0
                episode_len = 0
                shared_state.update_stats(global_steps, eps, last_loss, episode_reward, episode_len, memory.size, episode_count, best_score, ai_enabled, status_text)
                time.sleep(0.1)
                continue
            global_steps += 1
            episode_len += 1
            if TRAINING_ENABLED:
                eps = EPS_END + (EPS_START - EPS_END) * math.exp(-global_steps / EPS_DECAY)
            else:
                eps = EPS_END
            use_random = random.random() < eps
            screen_w, screen_h = get_screen_size(cap.hwnd)
            if use_random:
                action = random.randrange(ACTION_DIM)
            else:
                policy.train()
                policy.reset_noise()
                with torch.no_grad():
                    s_tensor = torch.tensor(state, device=device, dtype=torch.float32).unsqueeze(0).div_(255.0)
                    f_tensor = torch.tensor(state_feat, device=device, dtype=torch.float32).unsqueeze(0)
                    q_values = policy(s_tensor, f_tensor)
                    action = int(q_values.argmax(dim=1).item())
            action_info = bot.act(action, screen_w, screen_h)
            time.sleep(STEP_DELAY)
            frame = cap.grab()
            if frame.size != H * W:
                frame = np.zeros((H, W), dtype=np.uint8)
            frames = np.concatenate((frames[1:], frame[np.newaxis, ...]), axis=0)
            next_state = frames.copy()
            next_feat = extract_semantic_features(frame)
            diff_map = np.abs(frame.astype(np.float32) - last_frame.astype(np.float32)) * arena_weights
            diff_global = float(diff_map.mean() / 255.0)
            mid_h = H // 2
            diff_top = float(diff_map[:mid_h, :].mean() / 255.0)
            diff_bottom = float(diff_map[mid_h:, :].mean() / 255.0)
            last_frame = frame
            my_hp_last = float((last_feat[0] + last_feat[1] + last_feat[2]) / 3.0)
            my_hp_curr = float((next_feat[0] + next_feat[1] + next_feat[2]) / 3.0)
            opp_hp_last = float((last_feat[3] + last_feat[4] + last_feat[5]) / 3.0)
            opp_hp_curr = float((next_feat[3] + next_feat[4] + next_feat[5]) / 3.0)
            hp_delta = (opp_hp_last - opp_hp_curr) - (my_hp_last - my_hp_curr)
            reward_hp = float(np.clip(hp_delta * 4.0, -4.0, 4.0))
            if diff_global < 0.002:
                no_change_steps += 1
            else:
                no_change_steps = 0
            need_soft_reset = False
            if no_change_steps >= NO_CHANGE_LIMIT or episode_len >= MAX_EPISODE_LEN:
                need_soft_reset = True
                no_change_steps = 0
            change_core = diff_global * 1.4 + (diff_top - diff_bottom) * 2.0
            change_reward = float(np.clip(change_core, -2.5, 2.5))
            reward = 0.6 * reward_hp + 0.4 * change_reward + IDLE_REWARD
            if need_soft_reset:
                hp_gap = opp_hp_curr - my_hp_curr
                terminal_bonus = float(np.clip(hp_gap * 12.0, -8.0, 8.0))
                reward += terminal_bonus
            done_flag = 1.0 if need_soft_reset else 0.0
            episode_reward += reward
            if TRAINING_ENABLED:
                memory.push(state, state_feat, action, reward, next_state, next_feat, done_flag)
                state = next_state
                state_feat = next_feat
                last_feat = next_feat
                if memory.size >= MIN_REPLAY and global_steps % LEARN_FREQ == 0:
                    batch_size = BATCH_SIZE_BASE if device.type == "cuda" else max(32, BATCH_SIZE_BASE // 2)
                    idx, s_batch, sf_batch, a_batch, r_batch, ns_batch, nsf_batch, d_batch, w_batch = memory.sample(batch_size, global_step=global_steps)
                    policy.train()
                    target.eval()
                    policy.reset_noise()
                    target.reset_noise()
                    with amp_context():
                        q = policy(s_batch, sf_batch).gather(1, a_batch)
                        with torch.no_grad():
                            next_actions = policy(ns_batch, nsf_batch).argmax(dim=1, keepdim=True)
                            next_q = target(ns_batch, nsf_batch).gather(1, next_actions)
                            target_q = r_batch + GAMMA * next_q * (1.0 - d_batch)
                        td = target_q - q
                        abs_td = td.abs()
                        huber = torch.where(abs_td < 1.0, 0.5 * abs_td * abs_td, abs_td - 0.5)
                        loss = (huber * w_batch).mean()
                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(policy.parameters(), GRAD_NORM_CLIP)
                    scaler.step(optimizer)
                    scaler.update()
                    last_loss = float(loss.detach().item())
                    memory.update_priorities(idx, td.detach().cpu().numpy().reshape(-1))
                if global_steps % TARGET_SYNC == 0:
                    target.load_state_dict(policy.state_dict())
                if global_steps % SAVE_INTERVAL == 0:
                    line = "steps={} eps={:.3f} loss={:.4f} buffer={} ep_len={} ep_ret={:.2f}".format(global_steps, eps, last_loss, memory.size, episode_len, episode_reward)
                    print(line, flush=True)
                    append_log(time.strftime("%Y-%m-%d %H:%M:%S ") + line)
                    save_checkpoint(MODEL_PATH, policy.state_dict(), optimizer.state_dict(), SEM_ENCODER.state_dict(), global_steps)
                if need_soft_reset:
                    episode_count += 1
                    if episode_len > 0:
                        if ema_reward is None:
                            ema_reward = episode_reward
                        else:
                            ema_reward = ema_reward * 0.9 + episode_reward * 0.1
                        if best_score is None or ema_reward > best_score:
                            best_score = ema_reward
                            save_checkpoint(BEST_MODEL_PATH, policy.state_dict(), optimizer.state_dict(), SEM_ENCODER.state_dict(), global_steps)
            else:
                state = next_state
                state_feat = next_feat
                last_feat = next_feat
            if exp_logger is not None:
                exp_logger.log_step(global_steps, episode_count, episode_len, ai_enabled, frame, action_info, reward, done_flag)
            if need_soft_reset:
                soft_reset(bot, cap)
                first_frame = init_first_frame(cap)
                frames[:] = first_frame
                state = frames.copy()
                last_frame = first_frame
                last_feat = extract_semantic_features(first_frame)
                state_feat = last_feat.copy()
                episode_reward = 0.0
                episode_len = 0
            status_text = "AI 对战中"
            shared_state.update_stats(global_steps, eps, last_loss, episode_reward, episode_len, memory.size, episode_count, best_score, ai_enabled, status_text)
        if TRAINING_ENABLED and policy is not None and optimizer is not None:
            save_checkpoint(MODEL_PATH, policy.state_dict(), optimizer.state_dict(), SEM_ENCODER.state_dict(), global_steps)
    except Exception as e:
        shared_state.set_error("训练线程异常: {}".format(repr(e)))
    finally:
        if audio_recorder is not None:
            audio_recorder.stop()
        if exp_logger is not None:
            exp_logger.close()
        if bot is not None:
            bot.close()


def create_main_window(shared_state, stop_event):
    root = tk.Tk()
    root.title("Clash Royale AI 控制台")
    root.geometry("540x420")
    root.resizable(False, False)
    style = ttk.Style()
    try:
        style.theme_use("clam")
    except Exception:
        pass
    main_frame = ttk.Frame(root, padding=16)
    main_frame.pack(fill="both", expand=True)
    title_label = ttk.Label(main_frame, text="部落冲突：皇室战争 深度强化学习 AI", font=("Microsoft YaHei", 14, "bold"))
    title_label.pack(anchor="center")
    subtitle_label = ttk.Label(main_frame, text="在模拟器中先手动玩几局，然后按 F8 或点击下方按钮，让 AI 接管对战。", font=("Microsoft YaHei", 9))
    subtitle_label.pack(anchor="center", pady=(4, 12))
    info_frame = ttk.LabelFrame(main_frame, text="环境信息", padding=8)
    info_frame.pack(fill="x")
    ttk.Label(info_frame, text="模拟器窗口: {}".format(WIN_NAME), font=("Microsoft YaHei", 9)).grid(row=0, column=0, sticky="w")
    ttk.Label(info_frame, text="ADB 路径: {}".format(ADB_BIN), font=("Microsoft YaHei", 9)).grid(row=1, column=0, sticky="w", pady=(2, 0))
    ttk.Label(info_frame, text="模拟器路径: {}".format(EMU_BIN), font=("Microsoft YaHei", 9)).grid(row=2, column=0, sticky="w", pady=(2, 0))
    ttk.Label(info_frame, text="数据目录: {}".format(BASE_DIR), font=("Microsoft YaHei", 9)).grid(row=3, column=0, sticky="w", pady=(2, 0))
    status_frame = ttk.LabelFrame(main_frame, text="运行状态", padding=8)
    status_frame.pack(fill="x", pady=(10, 6))
    mode_var = tk.StringVar(value="模式: {}".format("训练+对战" if TRAINING_ENABLED else "仅推理对战"))
    ai_state_var = tk.StringVar(value="AI 状态: 初始化中")
    steps_var = tk.StringVar(value="总步数: 0")
    buffer_var = tk.StringVar(value="经验池大小: 0")
    reward_var = tk.StringVar(value="当前局奖励: 0.00")
    eps_var = tk.StringVar(value="探索率 ε: 0.00")
    loss_var = tk.StringVar(value="最近损失: 0.00")
    episode_var = tk.StringVar(value="局数: 0, 当前局时长: 0")
    best_var = tk.StringVar(value="表现评分: 暂无")
    message_var = tk.StringVar(value="后台线程启动中，请稍候…")
    ttk.Label(status_frame, textvariable=mode_var, font=("Microsoft YaHei", 9)).grid(row=0, column=0, sticky="w")
    ttk.Label(status_frame, textvariable=ai_state_var, font=("Microsoft YaHei", 9)).grid(row=0, column=1, sticky="w", padx=(16, 0))
    ttk.Label(status_frame, textvariable=steps_var, font=("Microsoft YaHei", 9)).grid(row=1, column=0, sticky="w", pady=(2, 0))
    ttk.Label(status_frame, textvariable=buffer_var, font=("Microsoft YaHei", 9)).grid(row=1, column=1, sticky="w", padx=(16, 0), pady=(2, 0))
    ttk.Label(status_frame, textvariable=reward_var, font=("Microsoft YaHei", 9)).grid(row=2, column=0, sticky="w", pady=(2, 0))
    ttk.Label(status_frame, textvariable=eps_var, font=("Microsoft YaHei", 9)).grid(row=2, column=1, sticky="w", padx=(16, 0), pady=(2, 0))
    ttk.Label(status_frame, textvariable=loss_var, font=("Microsoft YaHei", 9)).grid(row=3, column=0, sticky="w", pady=(2, 0))
    ttk.Label(status_frame, textvariable=episode_var, font=("Microsoft YaHei", 9)).grid(row=3, column=1, sticky="w", padx=(16, 0), pady=(2, 0))
    ttk.Label(status_frame, textvariable=best_var, font=("Microsoft YaHei", 9)).grid(row=4, column=0, columnspan=2, sticky="w", pady=(4, 0))
    message_label = ttk.Label(main_frame, textvariable=message_var, font=("Microsoft YaHei", 9), foreground="#444")
    message_label.pack(fill="x", pady=(6, 6))
    control_frame = ttk.Frame(main_frame)
    control_frame.pack(fill="x", pady=(4, 0))
    ai_button_text = tk.StringVar(value="开启 AI (等同 F8)")
    def on_toggle_ai():
        shared_state.request_toggle()
    ai_button = ttk.Button(control_frame, textvariable=ai_button_text, command=on_toggle_ai)
    ai_button.pack(side="left")
    def on_exit():
        stop_event.set()
        root.after(200, root.destroy)
    exit_button = ttk.Button(control_frame, text="退出程序", command=on_exit)
    exit_button.pack(side="right")
    def refresh():
        snap = shared_state.snapshot()
        if snap["ai_enabled"]:
            ai_state_var.set("AI 状态: 已开启")
            ai_button_text.set("关闭 AI (等同 F8)")
        else:
            ai_state_var.set("AI 状态: 已关闭")
            ai_button_text.set("开启 AI (等同 F8)")
        mode_var.set("模式: {}".format(snap["mode_text"]))
        steps_var.set("总步数: {}".format(snap["global_steps"]))
        buffer_var.set("经验池大小: {}".format(snap["buffer_size"]))
        reward_var.set("当前局奖励: {:.2f}".format(snap["episode_reward"]))
        eps_var.set("探索率 ε: {:.3f}".format(snap["eps"]))
        loss = snap["last_loss"]
        loss_var.set("最近损失: {:.4f}".format(loss) if loss > 0 else "最近损失: 0.0000")
        episode_var.set("局数: {}, 当前局时长: {}".format(snap["episode_count"], snap["episode_len"]))
        if snap["best_score"] is None:
            best_var.set("表现评分: 暂无")
        else:
            best_var.set("表现评分: {:.2f}".format(snap["best_score"]))
        if snap["error_text"]:
            message_var.set(snap["error_text"])
            message_label.configure(foreground="#aa0000")
        else:
            message_var.set(snap["status_text"])
            message_label.configure(foreground="#444444")
        if not stop_event.is_set():
            root.after(200, refresh)
    root.after(200, refresh)
    return root


def main():
    shared_state = SharedState()
    stop_event = threading.Event()
    worker = threading.Thread(target=rl_loop, args=(shared_state, stop_event), daemon=False)
    worker.start()
    root = create_main_window(shared_state, stop_event)
    root.mainloop()
    stop_event.set()
    try:
        worker.join(timeout=5.0)
    except Exception:
        pass


if __name__ == "__main__":
    main()
