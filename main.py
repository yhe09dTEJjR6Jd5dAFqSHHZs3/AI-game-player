import os
import sys
import threading
import time
import ctypes
import contextlib
import re
import tkinter as tk
from tkinter import ttk, messagebox
import importlib
import psutil
import torch
import numpy as np
import subprocess
from torch import nn
from torch.utils.data import Dataset, DataLoader

with contextlib.suppress(Exception):
    ctypes.windll.shcore.SetProcessDpiAwareness(1)
with contextlib.suppress(Exception):
    ctypes.windll.user32.SetProcessDPIAware()

def optional_import(module_name):
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None
    return importlib.import_module(module_name)

win32gui = optional_import("win32gui")
win32con = optional_import("win32con")
win32ui = optional_import("win32ui")
win32api = optional_import("win32api")
pynput_keyboard = optional_import("pynput.keyboard")
pynput_mouse = optional_import("pynput.mouse")
pynvml = optional_import("pynvml")
pil_module = optional_import("PIL")
Image = getattr(pil_module, "Image", None) if pil_module else None
ImageTk = getattr(pil_module, "ImageTk", None) if pil_module else None
ImageOps = getattr(pil_module, "ImageOps", None) if pil_module else None
ImageEnhance = getattr(pil_module, "ImageEnhance", None) if pil_module else None
ImageFilter = getattr(pil_module, "ImageFilter", None) if pil_module else None
ImageDraw = getattr(pil_module, "ImageDraw", None) if pil_module else None
ImageFont = getattr(pil_module, "ImageFont", None) if pil_module else None
pyautogui = optional_import("pyautogui")
easyocr = optional_import("easyocr")
wmi = optional_import("wmi")
keyboard = pynput_keyboard
mouse = pynput_mouse
MODE_INIT = "init"
MODE_LEARN = "learning"
MODE_TRAIN = "training"
MODE_RECOG = "recognizing"
MODE_OPT = "optimizing"

base_dir = os.path.join(os.path.expanduser("~"), "Desktop", "GameAI")
models_dir = os.path.join(base_dir, "models")
experience_dir = os.path.join(base_dir, "experience")
os.makedirs(models_dir, exist_ok=True)
os.makedirs(experience_dir, exist_ok=True)
model_path = os.path.join(models_dir, "policy_latest.pt")
init_model_path = os.path.join(models_dir, "policy_init.pt")

experience_buffer = []
experience_lock = threading.Lock()
experience_file_index = 0
last_user_action_vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)
last_ai_action_vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.0, 0.0, 0.0], dtype=np.float32)

policy_model = None
model_lock = threading.Lock()

recognized_values = []
recognized_lock = threading.Lock()
ocr_reader = None
ocr_lock = threading.Lock()
simple_ocr_templates = None
simple_ocr_size = (28, 36)
category_choices = ["越高越好", "越低越好", "变化越小越好", "变化越大越好", "无关", "识别错误"]
recognized_color_map = {}
recognition_running = False
recognition_attempted = False
recognition_progress = 0.0
recognition_finished_flag = False
recognition_result_msg = ""

current_mode = MODE_INIT
mode_lock = threading.Lock()

window_a_handle = None
window_a_title = ""
window_a_visible = False
window_a_rect = (0, 0, 0, 0)
window_a_occlusion = 1.0

last_frame_np = None
last_frame_lock = threading.Lock()

last_user_input_time = time.monotonic()
program_running = True

screenshot_fps = 10.0
hardware_stats = {"cpu": 0.0, "mem": 0.0, "gpu": 0.0, "vram": 0.0, "gpu_known": False, "vram_known": False, "gpu_hint": ""}

optimization_progress = 0.0
optimization_running = False
optimization_cancel_requested = False
optimization_finished_flag = False
optimization_finished_cancelled = False

gpu_available = torch.cuda.is_available()
gpu_handle = None
ai_interrupt_event = threading.Event()

if pynvml is not None:
    try:
        pynvml.nvmlInit()
        gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
    except:
        gpu_handle = None

if pyautogui is not None:
    pyautogui.FAILSAFE = False

root = tk.Tk()
root.title("GameAI 窗口智能助手")
root.geometry("1200x860")
root.minsize(960, 700)
root.configure(bg="#020617")

style = ttk.Style(root)
try:
    if "clam" in style.theme_names():
        style.theme_use("clam")
except:
    pass
style.configure("App.TFrame", background="#020617")
style.configure("Card.TLabelframe", background="#020617", foreground="#e5e7eb", padding=12, borderwidth=1, relief="solid")
style.configure("Card.TLabelframe.Label", background="#020617", foreground="#9ca3af")
style.configure("Title.TLabel", font=("Segoe UI", 18, "bold"), foreground="#f9fafb", background="#020617")
style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#9ca3af", background="#020617")
style.configure("Status.TLabel", font=("Segoe UI", 10), foreground="#e5e7eb", background="#020617")
style.configure("Metric.TLabel", font=("Consolas", 10), foreground="#a5b4fc", background="#020617")
style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"))
style.map("Accent.TButton", background=[("disabled", "#1e293b"), ("!disabled", "#38bdf8"), ("active", "#0ea5e9")], foreground=[("disabled", "#64748b"), ("!disabled", "#020617")])
style.configure("TLabel", background="#020617", foreground="#e5e7eb", padding=2)
style.configure("TFrame", background="#020617", padding=2)
style.configure("TLabelframe", background="#020617", padding=8)
style.configure("Horizontal.TProgressbar", troughcolor="#020617")

window_label_var = tk.StringVar()
visible_label_var = tk.StringVar()
size_label_var = tk.StringVar()
fps_label_var = tk.StringVar()
cpu_label_var = tk.StringVar()
mem_label_var = tk.StringVar()
gpu_label_var = tk.StringVar()
vram_label_var = tk.StringVar()
mode_label_var = tk.StringVar(value="模式: 初始化")
progress_label_var = tk.StringVar()

main_frame = ttk.Frame(root, style="App.TFrame")
main_frame.pack(fill="both", expand=True, padx=14, pady=14)

header_frame = ttk.Frame(main_frame, style="App.TFrame")
header_frame.pack(fill="x", pady=(0, 10))
title_label = ttk.Label(header_frame, text="GameAI 窗口智能助手", style="Title.TLabel")
title_label.pack(side="left", anchor="w")
mode_label = ttk.Label(header_frame, textvariable=mode_label_var, style="Status.TLabel")
mode_label.pack(side="right", anchor="e")
subtitle_label = ttk.Label(main_frame, text="学习 / 训练 / 优化 一体化控制面板", style="Subtitle.TLabel")
subtitle_label.pack(fill="x", pady=(0, 10))

status_frame = ttk.LabelFrame(main_frame, text="窗口与性能状态", style="Card.TLabelframe")
status_frame.pack(fill="x", pady=(0, 10))

status_left = ttk.Frame(status_frame, style="App.TFrame")
status_left.pack(side="left", fill="x", expand=True, padx=(0, 6))
status_right = ttk.Frame(status_frame, style="App.TFrame")
status_right.pack(side="left", fill="x", expand=True, padx=(6, 0))

window_label = ttk.Label(status_left, textvariable=window_label_var, style="Status.TLabel")
window_label.pack(anchor="w")
visible_label = ttk.Label(status_left, textvariable=visible_label_var, style="Status.TLabel")
visible_label.pack(anchor="w")
size_label = ttk.Label(status_left, textvariable=size_label_var, style="Status.TLabel")
size_label.pack(anchor="w")
fps_label = ttk.Label(status_left, textvariable=fps_label_var, style="Status.TLabel")
fps_label.pack(anchor="w")

cpu_label = ttk.Label(status_right, textvariable=cpu_label_var, style="Metric.TLabel")
cpu_label.pack(anchor="w")
mem_label = ttk.Label(status_right, textvariable=mem_label_var, style="Metric.TLabel")
mem_label.pack(anchor="w")
gpu_label = ttk.Label(status_right, textvariable=gpu_label_var, style="Metric.TLabel")
gpu_label.pack(anchor="w")
vram_label = ttk.Label(status_right, textvariable=vram_label_var, style="Metric.TLabel")
vram_label.pack(anchor="w")

controls_frame = ttk.LabelFrame(main_frame, text="控制与训练", style="Card.TLabelframe")
controls_frame.pack(fill="x", pady=(0, 10))

btn_frame = ttk.Frame(controls_frame, style="App.TFrame")
btn_frame.pack(fill="x", pady=(4, 4))

select_btn = ttk.Button(btn_frame, text="选择窗口", style="Accent.TButton")
sleep_btn = ttk.Button(btn_frame, text="Sleep", style="Accent.TButton")
getup_btn = ttk.Button(btn_frame, text="Get Up", style="Accent.TButton")
recognize_btn = ttk.Button(btn_frame, text="识别", style="Accent.TButton")
select_btn.pack(side="left", padx=(0, 8))
sleep_btn.pack(side="left", padx=8)
getup_btn.pack(side="left", padx=8)
recognize_btn.pack(side="left", padx=8)
sleep_btn.state(["disabled"])
getup_btn.state(["disabled"])
recognize_btn.state(["disabled"])

progress_frame = ttk.Frame(controls_frame, style="App.TFrame")
progress_frame.pack(fill="x", pady=(4, 4))
progress_bar = ttk.Progressbar(progress_frame, orient="horizontal", mode="determinate", maximum=100)
progress_bar.pack(fill="x", padx=2, pady=(0, 4))
progress_text_label = ttk.Label(progress_frame, textvariable=progress_label_var, style="Status.TLabel")
progress_text_label.pack(anchor="w")

preview_frame = ttk.LabelFrame(main_frame, text="窗口 A 画面", style="Card.TLabelframe")
preview_frame.pack(fill="both", expand=True)
canvas = tk.Label(preview_frame, bg="#000000", bd=0, highlightthickness=0)
canvas.pack(fill="both", expand=True, padx=6, pady=6)
canvas_image_ref = None

numbers_frame = ttk.LabelFrame(main_frame, text="数值列表", style="Card.TLabelframe")
numbers_frame.pack(fill="x", pady=(10, 0))
numbers_canvas = tk.Canvas(numbers_frame, bg="#020617", highlightthickness=0, borderwidth=0, height=140)
numbers_scrollbar = ttk.Scrollbar(numbers_frame, orient="vertical", command=numbers_canvas.yview)
numbers_inner = ttk.Frame(numbers_canvas, style="App.TFrame")
numbers_canvas.create_window((0, 0), window=numbers_inner, anchor="nw")
numbers_canvas.configure(yscrollcommand=numbers_scrollbar.set)
numbers_canvas.pack(side="left", fill="both", expand=True)
numbers_scrollbar.pack(side="right", fill="y")
numbers_inner.bind("<Configure>", lambda e: numbers_canvas.configure(scrollregion=numbers_canvas.bbox("all")))
number_row_widgets = []

user_pressing = False
user_press_start = None
user_press_path = []
user_press_time = 0.0

max_numbers = 8
category_to_id = {c: i for i, c in enumerate(category_choices)}
numeric_dim = max_numbers * (1 + len(category_choices) + 1) + 2

class PolicyNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=8, stride=4, padding=2),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU()
        )
        self.fc_img = nn.Sequential(
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU()
        )
        self.fc_num = nn.Sequential(
            nn.Linear(numeric_dim, 128),
            nn.ReLU()
        )
        self.grid_h = 21
        self.grid_w = 21
        merged = 256 + 128
        self.action_head = nn.Linear(merged, 8)
        self.control_head = nn.Linear(merged, self.grid_h * self.grid_w)
        self.rule_head = nn.Linear(merged, 4)
        self.action_type_head = nn.Linear(merged, 4)
        self.pref_head = nn.Linear(merged, numeric_dim)

    def forward(self, x, num_feat):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc_img(x)
        num_feat = num_feat.float()
        num_feat = self.fc_num(num_feat)
        merged = torch.cat([x, num_feat], dim=1)
        action = self.action_head(merged)
        control_logits = self.control_head(merged).view(-1, 1, self.grid_h, self.grid_w)
        rule_logits = self.rule_head(merged)
        type_logits = self.action_type_head(merged)
        pref_recon = self.pref_head(merged)
        pos_start = torch.tanh(action[:, :2])
        pos_end = torch.tanh(action[:, 2:4])
        pos_mid = torch.tanh(action[:, 4:6])
        press_logit = action[:, 6]
        hold_logit = action[:, 7]
        return pos_start, pos_end, pos_mid, press_logit, hold_logit, control_logits, rule_logits, type_logits, pref_recon

class ExperienceDataset(Dataset):
    def __init__(self, directory):
        self.samples = []
        if os.path.exists(directory):
            for name in os.listdir(directory):
                if name.endswith(".pt") and name.startswith("experience_"):
                    path = os.path.join(directory, name)
                    try:
                        data = torch.load(path, map_location="cpu")
                        obs = data.get("obs")
                        act = data.get("act")
                        src = data.get("src")
                        num = data.get("num")
                        if obs is not None and act is not None:
                            n = obs.shape[0]
                            for i in range(n):
                                a = act[i]
                                if a.ndim == 0:
                                    continue
                                if a.numel() < 9:
                                    pad = torch.zeros(9 - a.numel())
                                    a = torch.cat([a.view(-1), pad], dim=0)
                                elif a.numel() > 9:
                                    a = a.view(-1)[:9]
                                num_vec = None
                                if num is not None and i < num.shape[0]:
                                    num_vec = num[i].view(-1)
                                if num_vec is None or num_vec.numel() != numeric_dim:
                                    num_vec = torch.zeros(numeric_dim)
                                self.samples.append((obs[i], a, 0 if src is None else int(src[i]), num_vec))
                    except:
                        continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o, a, s, n = self.samples[idx]
        return o, a, s, n

class SyntheticDataset(Dataset):
    def __init__(self, count=256):
        obs = torch.randint(0, 256, (count, 3, 84, 84), dtype=torch.uint8)
        acts = []
        nums = []
        src = torch.zeros(count, dtype=torch.int64)
        for i in range(count):
            val = float((i * 17) % 1000)
            cat = i % len(category_choices)
            press = 1.0
            hold = (cat + 1) / len(category_choices)
            pos_seed = (cat * 13 + i * 7) % 100
            nx = (pos_seed % 10) / 10.0
            ny = ((pos_seed // 10) % 10) / 10.0
            action_vec = torch.tensor([nx, ny, 1.0 - nx, 1.0 - ny, 0.5, 0.5, press, hold, float(cat % 4)], dtype=torch.float32)
            acts.append(action_vec)
            num_vec = []
            for j in range(max_numbers):
                if j == 0:
                    onehot = [1.0 if k == cat else 0.0 for k in range(len(category_choices))]
                    num_vec.extend([np.tanh(val / 1000.0)] + onehot + [0.0])
                else:
                    num_vec.extend([0.0] * (1 + len(category_choices) + 1))
            num_vec.append(1.0 / float(max_numbers))
            num_vec.append(1.0)
            nums.append(torch.tensor(num_vec, dtype=torch.float32))
        self.obs = obs
        self.act = torch.stack(acts, dim=0)
        self.num = torch.stack(nums, dim=0)
        self.src = src

    def __len__(self):
        return self.obs.shape[0]

    def __getitem__(self, idx):
        return self.obs[idx], self.act[idx], self.src[idx], self.num[idx]

def ensure_model_exists():
    global policy_model, experience_file_index
    for name in os.listdir(experience_dir):
        if name.startswith("experience_") and name.endswith(".pt"):
            try:
                idx = int(name[len("experience_"):-3])
                if idx >= experience_file_index:
                    experience_file_index = idx + 1
            except:
                continue
    model_files = []
    try:
        for name in os.listdir(models_dir):
            if name.endswith(".pt"):
                model_files.append(os.path.join(models_dir, name))
    except:
        model_files = []
    state = None
    if model_files:
        try:
            model_files.sort(key=lambda p: os.path.getmtime(p), reverse=True)
            for path in model_files:
                try:
                    state = torch.load(path, map_location="cpu")
                    break
                except:
                    continue
        except:
            pass
    model = PolicyNet()
    if state is not None:
        try:
            model.load_state_dict(state, strict=False)
        except:
            pass
    else:
        try:
            torch.save(model.state_dict(), init_model_path)
            torch.save(model.state_dict(), model_path)
        except:
            pass
    model.eval()
    policy_model = model

def set_mode(new_mode):
    global current_mode, last_user_input_time
    with mode_lock:
        current_mode = new_mode
        if new_mode == MODE_LEARN:
            last_user_input_time = time.monotonic()

def get_mode():
    with mode_lock:
        return current_mode

def normalize_action_from_mouse(px, py, rect):
    left, top, right, bottom = rect
    w = max(1, right - left)
    h = max(1, bottom - top)
    nx = (px - left) / w
    ny = (py - top) / h
    nx = max(0.0, min(1.0, nx))
    ny = max(0.0, min(1.0, ny))
    return nx, ny

def denormalize_action_to_mouse(nx, ny, rect):
    left, top, right, bottom = rect
    w = right - left
    h = bottom - top
    x = left + nx * w
    y = top + ny * h
    return int(x), int(y)

def get_value_color(v):
    if v in recognized_color_map:
        return recognized_color_map[v]
    r = (v * 73 + 90) % 255
    g = (v * 37 + 160) % 255
    b = (v * 191 + 40) % 255
    r = max(80, r)
    g = max(80, g)
    b = max(80, b)
    recognized_color_map[v] = (r, g, b)
    return recognized_color_map[v]

def build_numeric_feature_vector():
    feat = []
    with recognized_lock:
        items = list(recognized_values)
    count = min(max_numbers, len(items))
    for i in range(max_numbers):
        if i < count:
            item = items[i]
            val = float(item.get("value", 0.0))
            category = item.get("category", "无关")
            cat_id = category_to_id.get(category, category_to_id.get("无关", 0))
            locked_flag = 1.0 if item.get("locked", False) else 0.0
            val_norm = np.tanh(val / 1000.0)
            cat_onehot = [1.0 if j == cat_id else 0.0 for j in range(len(category_choices))]
            feat.extend([val_norm] + cat_onehot + [locked_flag])
        else:
            feat.extend([0.0] * (1 + len(category_choices) + 1))
    feat.append(count / float(max_numbers))
    feat.append(1.0 - min(1.0, window_a_occlusion))
    return np.array(feat, dtype=np.float32)

def rect_intersection(a, b):
    lx = max(a[0], b[0])
    ly = max(a[1], b[1])
    rx = min(a[2], b[2])
    ry = min(a[3], b[3])
    if rx <= lx or ry <= ly:
        return None
    return (lx, ly, rx, ry)

def rect_area(r):
    return max(0, r[2] - r[0]) * max(0, r[3] - r[1])

def union_area(rects):
    if not rects:
        return 0
    xs = []
    ys = []
    for l, t, r, b in rects:
        xs.extend([l, r])
        ys.extend([t, b])
    xs = sorted(set(xs))
    ys = sorted(set(ys))
    total = 0
    for i in range(len(xs) - 1):
        for j in range(len(ys) - 1):
            cell = (xs[i], ys[j], xs[i + 1], ys[j + 1])
            for l, t, r, b in rects:
                if l < cell[2] and r > cell[0] and t < cell[3] and b > cell[1]:
                    total += (cell[2] - cell[0]) * (cell[3] - cell[1])
                    break
    return total

def get_virtual_screen_rect():
    try:
        left = ctypes.windll.user32.GetSystemMetrics(76)
        top = ctypes.windll.user32.GetSystemMetrics(77)
        width = ctypes.windll.user32.GetSystemMetrics(78)
        height = ctypes.windll.user32.GetSystemMetrics(79)
        return (left, top, left + width, top + height)
    except:
        return (0, 0, 0, 0)

def sleep_with_interrupt(duration):
    end = time.time() + max(0.0, duration)
    while time.time() < end:
        if ai_interrupt_event.is_set():
            return False
        remaining = end - time.time()
        time.sleep(max(0.0, min(0.02, remaining)))
    return not ai_interrupt_event.is_set()

def move_with_interrupt(x, y, duration):
    if pyautogui is None:
        return False
    steps = max(1, int(max(0.01, duration) / 0.02))
    try:
        cx, cy = pyautogui.position()
    except:
        cx, cy = x, y
    for i in range(steps):
        if ai_interrupt_event.is_set():
            return False
        ratio = (i + 1) / steps
        nx = int(cx + (x - cx) * ratio)
        ny = int(cy + (y - cy) * ratio)
        try:
            pyautogui.moveTo(nx, ny)
        except:
            return False
        time.sleep(max(0.0, min(0.02, duration / steps)))
    if not ai_interrupt_event.is_set():
        try:
            pyautogui.moveTo(x, y)
        except:
            return False
    return not ai_interrupt_event.is_set()

def request_ai_stop():
    ai_interrupt_event.set()
    if pyautogui is not None:
        with contextlib.suppress(Exception):
            pyautogui.mouseUp()

def is_left_button_pressed():
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
    except:
        return False

def capture_window_image(hwnd):
    global window_a_rect
                      
    image_win32 = None
    captured_with_win32 = False
    
    if win32gui is not None and win32ui is not None and Image is not None:
        try:
            left, top, right, bottom = win32gui.GetWindowRect(hwnd)
            w = right - left
            h = bottom - top
            if w > 0 and h > 0:
                hwndDC = win32gui.GetWindowDC(hwnd)
                mfcDC = win32ui.CreateDCFromHandle(hwndDC)
                saveDC = mfcDC.CreateCompatibleDC()
                saveBitMap = win32ui.CreateBitmap()
                saveBitMap.CreateCompatibleBitmap(mfcDC, w, h)
                saveDC.SelectObject(saveBitMap)
                
                                                     
                result = ctypes.windll.user32.PrintWindow(hwnd, saveDC.GetSafeHdc(), 2)
                if result != 1:
                                    
                    saveDC.BitBlt((0, 0), (w, h), mfcDC, (0, 0), win32con.SRCCOPY)
                
                bmpinfo = saveBitMap.GetInfo()
                bmpstr = saveBitMap.GetBitmapBits(True)
                image_win32 = Image.frombuffer("RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1)
                
                win32gui.DeleteObject(saveBitMap.GetHandle())
                saveDC.DeleteDC()
                mfcDC.DeleteDC()
                win32gui.ReleaseDC(hwnd, hwndDC)
                
                                              
                if image_win32:
                    stat =  np.array(image_win32).std()
                    if stat > 5.0:                      
                        captured_with_win32 = True
                    else:
                        image_win32 = None                        
        except:
            image_win32 = None

                                      
    if not captured_with_win32 and pyautogui is not None and window_a_rect is not None:
        try:
                              
            if win32gui:
                rect = win32gui.GetWindowRect(hwnd)
            else:
                rect = window_a_rect
            left, top, right, bottom = rect
            w = right - left
            h = bottom - top
            if w > 0 and h > 0:
                                                  
                img = pyautogui.screenshot(region=(left, top, w, h))
                return img.convert("RGB")
        except:
            pass
    
    return image_win32

def resize_for_model(img):
    if Image is None:
        return None
    img = img.resize((84, 84), Image.Resampling.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return arr

def record_experience(frame_arr, action_vec, source_flag, num_vec):
    global experience_buffer, experience_file_index
    if frame_arr is None or action_vec is None:
        return
    if num_vec is None:
        num_vec = np.zeros(numeric_dim, dtype=np.float32)
    if len(action_vec) < 9:
        pad_len = 9 - len(action_vec)
        action_vec = np.concatenate([action_vec, np.zeros(pad_len, dtype=np.float32)])
    elif len(action_vec) > 9:
        action_vec = action_vec[:9]
    with experience_lock:
        experience_buffer.append((frame_arr, action_vec, source_flag, num_vec))
        if len(experience_buffer) >= 128:
            try:
                obs = np.stack([x[0] for x in experience_buffer], axis=0)
                act = np.stack([x[1] for x in experience_buffer], axis=0).astype(np.float32)
                src = np.array([x[2] for x in experience_buffer], dtype=np.int64)
                num = np.stack([x[3] for x in experience_buffer], axis=0).astype(np.float32)
                tensor_obs = torch.from_numpy(obs)
                tensor_act = torch.from_numpy(act)
                tensor_src = torch.from_numpy(src)
                tensor_num = torch.from_numpy(num)
                path = os.path.join(experience_dir, f"experience_{experience_file_index}.pt")
                experience_file_index += 1
                torch.save({"obs": tensor_obs, "act": tensor_act, "src": tensor_src, "num": tensor_num}, path)
            except:
                pass
            experience_buffer = []

def flush_experience_buffer():
    global experience_buffer, experience_file_index
    with experience_lock:
        if not experience_buffer:
            return
        try:
            obs = np.stack([x[0] for x in experience_buffer], axis=0)
            act = np.stack([x[1] for x in experience_buffer], axis=0).astype(np.float32)
            src = np.array([x[2] for x in experience_buffer], dtype=np.int64)
            num = np.stack([x[3] for x in experience_buffer], axis=0).astype(np.float32)
            tensor_obs = torch.from_numpy(obs)
            tensor_act = torch.from_numpy(act)
            tensor_src = torch.from_numpy(src)
            tensor_num = torch.from_numpy(num)
            path = os.path.join(experience_dir, f"experience_{experience_file_index}.pt")
            experience_file_index += 1
            torch.save({"obs": tensor_obs, "act": tensor_act, "src": tensor_src, "num": tensor_num}, path)
        except:
            pass
        experience_buffer = []

def gather_pynvml_metrics():
    metrics = []
    if pynvml is None:
        return metrics
    try:
        count = pynvml.nvmlDeviceGetCount()
    except:
        count = 0
    for idx in range(max(0, count)):
        try:
            handle = pynvml.nvmlDeviceGetHandleByIndex(idx)
            util = pynvml.nvmlDeviceGetUtilizationRates(handle)
            meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
            metrics.append({"util": float(util.gpu), "used": float(meminfo.used), "total": float(meminfo.total)})
        except:
            continue
    return metrics

def gather_torch_metrics():
    metrics = []
    if not gpu_available:
        return metrics
    try:
        count = torch.cuda.device_count()
    except:
        count = 0
    util_fn = getattr(torch.cuda, "utilization", None)
    for idx in range(max(0, count)):
        try:
            total = torch.cuda.get_device_properties(idx).total_memory
            used = torch.cuda.memory_allocated(idx)
            vram_ratio = float(used) / float(total) * 100.0 if total > 0 else None
        except:
            vram_ratio = None
        util_val = None
        if util_fn is not None:
            try:
                util_val = float(util_fn(idx))
            except:
                util_val = None
        metrics.append({"util": util_val, "vram": vram_ratio})
    return metrics

def gather_nvidia_smi_metrics():
    metrics = []
    cmd = ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=2)
        if res.returncode == 0:
            for line in res.stdout.strip().splitlines():
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 3:
                    try:
                        util = float(parts[0])
                        used = float(parts[1]) * 1024 * 1024
                        total = float(parts[2]) * 1024 * 1024
                        metrics.append({"util": util, "used": used, "total": total})
                    except:
                        continue
    except:
        return []
    return metrics

def gather_perfmon_metrics():
    metrics = []
    if os.name != "nt":
        return metrics
    try:
        cmd = ["powershell", "-Command", "(Get-Counter '\\GPU Engine(*)\\Utilization Percentage').CounterSamples | Select-Object CookedValue"]
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if res.returncode == 0:
            vals = []
            for line in res.stdout.splitlines():
                try:
                    vals.append(float(line.strip()))
                except:
                    continue
            if vals:
                metrics.append({"util": max(vals), "vram": None})
    except:
        return []
    return metrics

def gather_dxdiag_metrics():
    metrics = []
    if os.name != "nt":
        return metrics
    try:
        tmp_path = os.path.join(os.getenv("TEMP", ""), f"dxdiag_{int(time.time())}.txt")
        res = subprocess.run(["dxdiag", "/t", tmp_path], timeout=8)
        if res.returncode == 0 and os.path.exists(tmp_path):
            used = None
            total = None
            with open(tmp_path, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    if "Dedicated Memory:" in line and total is None:
                        parts = re.findall(r"\d+", line)
                        if parts:
                            total = float(parts[0]) * 1024 * 1024
                    if "Current Usage:" in line and used is None:
                        parts = re.findall(r"\d+", line)
                        if parts:
                            used = float(parts[0]) * 1024 * 1024
            if used is not None and total not in (None, 0):
                metrics.append({"util": None, "used": used, "total": total})
        with contextlib.suppress(Exception):
            os.remove(tmp_path)
    except:
        return []
    return metrics

def gather_wmi_metrics():
    metrics = []
    if wmi is None:
        return metrics
    try:
        conn = wmi.WMI(namespace="root\\CIMV2")
        engines = conn.Win32_PerfFormattedData_GPUPerformanceCounters_GPUEngine()
        util_map = {}
        for e in engines:
            name = getattr(e, "Name", "")
            util = getattr(e, "UtilizationPercentage", None)
            if util is None:
                continue
            if "engtype_3D" in name or "engtype_Compute" in name:
                gpu_name = name.split("_")[0]
                util_map[gpu_name] = max(util_map.get(gpu_name, 0.0), float(util))
        for _, util_val in util_map.items():
            metrics.append({"util": util_val, "vram": None})
        videos = conn.Win32_VideoController()
        for idx, v in enumerate(videos):
            total = getattr(v, "AdapterRAM", None)
            used = getattr(v, "CurrentUsage", None)
            if total is not None and used is not None:
                try:
                    metrics.append({"util": None, "used": float(used), "total": float(total)})
                except:
                    continue
    except:
        return []
    return metrics

def collect_gpu_metrics():
    gpu_vals = []
    vram_vals = []
    hint_parts = []
    fail_parts = []
    sources = [
        (gather_pynvml_metrics, "pynvml"),
        (gather_torch_metrics, "torch"),
        (gather_nvidia_smi_metrics, "nvidia-smi"),
        (gather_wmi_metrics, "wmi"),
        (gather_perfmon_metrics, "perfmon"),
        (gather_dxdiag_metrics, "dxdiag"),
    ]
    for fn, label in sources:
        try:
            data = fn()
            if not data:
                fail_parts.append(f"{label}:无数据")
                continue
            hint_parts.append(label)
            for item in data:
                util_val = item.get("util")
                if util_val is not None:
                    gpu_vals.append(util_val)
                used = item.get("used")
                total = item.get("total")
                vram_ratio = item.get("vram")
                if vram_ratio is None and used is not None and total not in (None, 0):
                    vram_ratio = float(used) / float(total) * 100.0
                if vram_ratio is not None:
                    vram_vals.append(vram_ratio)
        except Exception as e:
            fail_parts.append(f"{label}:" + str(e))
    gpu_known = bool(gpu_vals)
    vram_known = bool(vram_vals)
    gpu = max(gpu_vals) if gpu_vals else 0.0
    vram = max(vram_vals) if vram_vals else 0.0
    hint = "/".join(hint_parts)
    if fail_parts:
        hint = (hint + " | " if hint else "") + ";".join(fail_parts)
    return gpu, vram, gpu_known, vram_known, hint

def hardware_monitor_loop():
    global hardware_stats, screenshot_fps, program_running
    while program_running:
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory().percent
            gpu, vram, gpu_known, vram_known, gpu_hint = collect_gpu_metrics()
            if not gpu_known and not vram_known:
                gpu_hint = gpu_hint or "GPU 指标不可用，请安装 pynvml 或更新驱动"
            hardware_stats = {"cpu": cpu, "mem": mem, "gpu": gpu, "vram": vram, "gpu_known": gpu_known, "vram_known": vram_known, "gpu_hint": gpu_hint}
            metrics = [cpu, mem]
            if gpu > 0.0:
                metrics.append(gpu)
            if vram > 0.0:
                metrics.append(vram)
            base_fps = 120.0
            if gpu_available and not gpu_known and not vram_known:
                base_fps = 60.0
            if metrics:
                stress = max(metrics)
                fps = base_fps * max(0.0, min(1.0, 1.0 - stress / 100.0))
            else:
                fps = base_fps / 2.0
            screenshot_fps = max(0.0, min(120.0, fps))
        except:
            pass

def window_visibility_check(hwnd):
    global window_a_rect
    if win32gui is None:
        return False, (0, 0, 0, 0), 1.0
    try:
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return False, (0, 0, 0, 0), 1.0
        if win32gui.IsIconic(hwnd):
            return False, (0, 0, 0, 0), 1.0
        rect = win32gui.GetWindowRect(hwnd)
        screen_rect = get_virtual_screen_rect()
        clipped = rect_intersection(rect, screen_rect)
        if clipped is None or rect_area(clipped) <= 0:
            window_a_rect = rect
            return False, rect, 1.0
        occluders = []
        try:
            hwnd_list = []
            wtop = win32gui.GetTopWindow(None)
            while wtop:
                hwnd_list.append(wtop)
                wtop = win32gui.GetWindow(wtop, win32con.GW_HWNDNEXT)
            if hwnd in hwnd_list:
                idx = hwnd_list.index(hwnd)
                blockers = hwnd_list[:idx]
                for h in blockers:
                    try:
                        if h == hwnd:
                            continue
                        if not win32gui.IsWindowVisible(h):
                            continue
                        if win32gui.IsIconic(h):
                            continue
                        r = win32gui.GetWindowRect(h)
                        inter = rect_intersection(clipped, r)
                        if inter is not None:
                            occluders.append(inter)
                    except:
                        continue
        except:
            pass
        total_area = max(1, rect_area(rect))
        clipped_area = rect_area(clipped)
        missing_area = max(0, total_area - clipped_area)
        occlusion_area = union_area(occluders)
        hidden_area = missing_area + occlusion_area
        occlusion_ratio = min(1.0, hidden_area / total_area)
        visible_full = clipped_area >= total_area * 0.999 and occlusion_ratio <= 0.001
        window_a_rect = rect
        return visible_full, rect, occlusion_ratio
    except:
        return False, (0, 0, 0, 0), 1.0

def ai_compute_action(frame_arr, num_vec):
    global policy_model
    if policy_model is None:
        return None
    try:
        with model_lock:
            model = policy_model
            model.eval()
            x = torch.from_numpy(frame_arr).unsqueeze(0)
            n = torch.from_numpy(num_vec).unsqueeze(0)
            with torch.no_grad():
                pos_start, pos_end, pos_mid, press_logit, hold_logit, _, _, type_logits, _ = model(x, n)
                pos_start = pos_start[0].cpu().numpy()
                pos_end = pos_end[0].cpu().numpy()
                pos_mid = pos_mid[0].cpu().numpy()
                press_prob = torch.sigmoid(press_logit)[0].item()
                hold_prob = torch.sigmoid(hold_logit)[0].item()
                type_idx = int(torch.argmax(type_logits, dim=1)[0].item())
                nx1 = max(0.0, min(1.0, (pos_start[0] + 1.0) / 2.0))
                ny1 = max(0.0, min(1.0, (pos_start[1] + 1.0) / 2.0))
                nx2 = max(0.0, min(1.0, (pos_end[0] + 1.0) / 2.0))
                ny2 = max(0.0, min(1.0, (pos_end[1] + 1.0) / 2.0))
                nxm = max(0.0, min(1.0, (pos_mid[0] + 1.0) / 2.0))
                nym = max(0.0, min(1.0, (pos_mid[1] + 1.0) / 2.0))
                return np.array([nx1, ny1, nx2, ny2, nxm, nym, press_prob, hold_prob, float(type_idx)], dtype=np.float32)
    except:
        return None

def frame_loop():
    global window_a_visible, window_a_title, window_a_rect, last_frame_np, program_running, last_user_action_vec, last_ai_action_vec, window_a_occlusion, user_pressing, user_press_start, user_press_path, user_press_time
    last_time = time.time()
    while program_running:
        fps = screenshot_fps
        if fps <= 0.0:
            time.sleep(0.2)
            continue
        interval = 1.0 / fps if fps > 0 else 0.1
        now = time.time()
        delay = interval - (now - last_time)
        if delay > 0:
            time.sleep(delay)
        last_time = time.time()
        hwnd = window_a_handle
        if hwnd is None:
            window_a_visible = False
            continue
        vis, rect, occlusion_ratio = window_visibility_check(hwnd)
        window_a_visible = vis
        window_a_rect = rect
        window_a_occlusion = occlusion_ratio
        if win32gui is not None:
            try:
                window_a_title_local = win32gui.GetWindowText(hwnd)
            except:
                window_a_title_local = ""
        else:
            window_a_title_local = ""
        window_a_title = window_a_title_local
        if not vis:
            continue
        
                                                 
        if get_mode() == MODE_RECOG:
            time.sleep(0.1)
            continue

        img = capture_window_image(hwnd)
        if img is None:
            continue
        frame_arr_model = resize_for_model(img)
        num_vec = build_numeric_feature_vector()
        with last_frame_lock:
            last_frame_np = np.array(img)
        mode = get_mode()
        if mode != MODE_TRAIN:
            ai_interrupt_event.clear()
        recording = mode in (MODE_LEARN, MODE_TRAIN) and vis and occlusion_ratio <= 0.001 and not optimization_running and not recognition_running
        action_vec = None
        source_flag = 0
        if mode == MODE_TRAIN:
            if not ai_interrupt_event.is_set() and frame_arr_model is not None:
                a = ai_compute_action(frame_arr_model, num_vec)
                if a is not None:
                    last_ai_action_vec = a
                    action_vec = last_ai_action_vec
                    source_flag = 1
            if action_vec is not None and pyautogui is not None and not ai_interrupt_event.is_set():
                nx1, ny1, nx2, ny2, nxm, nym, press_val, hold_val, t = [float(v) for v in action_vec]
                sx, sy = denormalize_action_to_mouse(nx1, ny1, rect)
                ex, ey = denormalize_action_to_mouse(nx2, ny2, rect)
                mx, my = denormalize_action_to_mouse(nxm, nym, rect)
                act_type = max(0, min(3, int(round(t))))
                hold_time = 0.05 + hold_val * 1.5
                press_delay = 0.02 + press_val * 0.2
                mouse_down = False
                interrupted = False
                try:
                    if act_type == 0:
                        interrupted = not move_with_interrupt(sx, sy, hold_time * 0.2)
                    elif act_type == 1:
                        interrupted = not move_with_interrupt(sx, sy, hold_time * 0.2)
                        if not interrupted:
                            pyautogui.mouseDown()
                            mouse_down = True
                            if not sleep_with_interrupt(press_delay):
                                interrupted = True
                            if not interrupted and not sleep_with_interrupt(hold_time * 0.3):
                                interrupted = True
                    elif act_type == 2:
                        interrupted = not move_with_interrupt(sx, sy, hold_time * 0.2)
                        if not interrupted:
                            pyautogui.mouseDown()
                            mouse_down = True
                            if not sleep_with_interrupt(press_delay + hold_time):
                                interrupted = True
                    else:
                        interrupted = not move_with_interrupt(sx, sy, hold_time * 0.2)
                        if not interrupted:
                            pyautogui.mouseDown()
                            mouse_down = True
                            path_pts = [(sx, sy), (mx, my), (ex, ey)]
                            filtered = []
                            for p in path_pts:
                                if not filtered or filtered[-1] != p:
                                    filtered.append(p)
                            total_steps = max(5, int(8 + press_val * 20))
                            for i in range(len(filtered) - 1):
                                if ai_interrupt_event.is_set():
                                    interrupted = True
                                    break
                                p0 = filtered[i]
                                p1 = filtered[i + 1]
                                steps = max(2, total_steps // max(1, len(filtered) - 1))
                                for s in range(1, steps + 1):
                                    if ai_interrupt_event.is_set():
                                        interrupted = True
                                        break
                                    ratio = s / steps
                                    curve = 0.5 - 0.5 * np.cos(np.pi * ratio)
                                    px = int(p0[0] + (p1[0] - p0[0]) * curve)
                                    py = int(p0[1] + (p1[1] - p0[1]) * curve)
                                    pyautogui.moveTo(px, py)
                                    if not sleep_with_interrupt(max(0.0, hold_time / total_steps)):
                                        interrupted = True
                                        break
                                if interrupted:
                                    break
                except:
                    interrupted = True
                finally:
                    if mouse_down:
                        with contextlib.suppress(Exception):
                            pyautogui.mouseUp()
                    if interrupted:
                        request_ai_stop()
        elif mode == MODE_LEARN:
            if win32gui is not None:
                try:
                    pt = win32gui.GetCursorPos()
                except:
                    pt = None
                if pt is not None and rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]:
                    nx, ny = normalize_action_from_mouse(pt[0], pt[1], rect)
                    pressed = is_left_button_pressed()
                    if pressed:
                        if not user_pressing:
                            user_pressing = True
                            user_press_start = (nx, ny)
                            user_press_path = [(nx, ny)]
                            user_press_time = time.monotonic()
                        else:
                            if len(user_press_path) < 12:
                                user_press_path.append((nx, ny))
                        action_vec = np.array([nx, ny, nx, ny, nx, ny, 1.0, 0.0, 1.0], dtype=np.float32)
                    else:
                        if user_pressing:
                            duration = time.monotonic() - user_press_time
                            path = user_press_path if user_press_path else [(nx, ny)]
                            start_pt = user_press_start if user_press_start else path[0]
                            end_pt = path[-1]
                            mid_pt = path[len(path) // 2] if path else start_pt
                            dist = np.linalg.norm(np.array(start_pt) - np.array(end_pt))
                            act_type = 1
                            if dist > 0.05:
                                act_type = 3
                            elif duration >= 0.6:
                                act_type = 2
                            hold_norm = max(0.0, min(1.0, duration / 2.0))
                            action_vec = np.array([start_pt[0], start_pt[1], end_pt[0], end_pt[1], mid_pt[0], mid_pt[1], 1.0, hold_norm, float(act_type)], dtype=np.float32)
                            user_pressing = False
                        else:
                            action_vec = np.array([nx, ny, nx, ny, nx, ny, 0.0, 0.0, 0.0], dtype=np.float32)
                    if action_vec is not None:
                        last_user_action_vec = action_vec
                        action_vec = last_user_action_vec
                        source_flag = 0
        if recording and frame_arr_model is not None and action_vec is not None:
            record_experience(frame_arr_model, action_vec, source_flag, num_vec)

class LASTINPUTINFO(ctypes.Structure):
    fields = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

def get_system_idle_seconds():
    try:
        info = LASTINPUTINFO()
        info.cbSize = ctypes.sizeof(LASTINPUTINFO)
        if ctypes.windll.user32.GetLastInputInfo(ctypes.byref(info)):
            ticks = ctypes.windll.kernel32.GetTickCount()
            delta = ticks - info.dwTime
            if delta < 0:
                delta = 0
            return delta / 1000.0
    except:
        return None
    return None

def idle_mode_manager():
    global last_user_input_time
    if not program_running:
        return
    try:
        mode = get_mode()
        if mode == MODE_LEARN and not recognition_running and not optimization_running:
            idle = get_system_idle_seconds()
            if idle is not None:
                if idle >= 10.0 and window_a_visible:
                    set_mode(MODE_TRAIN)
            else:
                now = time.monotonic()
                if now - last_user_input_time >= 10.0 and window_a_visible:
                    set_mode(MODE_TRAIN)
    except:
        pass
    root.after(500, idle_mode_manager)

def keyboard_listener_loop():
    global last_user_input_time
    if keyboard is None:
        return
    def on_press(key):
        global last_user_input_time
        last_user_input_time = time.monotonic()
        try:
            if key == keyboard.Key.esc or (hasattr(key, "vk") and key.vk == 27):
                terminate_program()
                return False
        except:
            pass
        if get_mode() == MODE_TRAIN:
            request_ai_stop()
            set_mode(MODE_LEARN)
    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()

def mouse_listener_loop():
    global last_user_input_time
    if mouse is None:
        return
    def on_move(x, y):
        global last_user_input_time
        last_user_input_time = time.monotonic()
    def on_click(x, y, button, pressed):
        global last_user_input_time
        last_user_input_time = time.monotonic()
    def on_scroll(x, y, dx, dy):
        global last_user_input_time
        last_user_input_time = time.monotonic()
    with mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll) as listener:
        listener.join()

def global_input_loop():
    global last_user_input_time
    while program_running:
        try:
            any_key = False
            for vk in range(8, 256):
                state = ctypes.windll.user32.GetAsyncKeyState(vk)
                if state & 0x8000:
                    any_key = True
                    if vk == 0x1B:
                        terminate_program()
                        return
                    else:
                        if get_mode() == MODE_TRAIN:
                            request_ai_stop()
                            set_mode(MODE_LEARN)
                    break
            if any_key:
                last_user_input_time = time.monotonic()
        except:
            pass
        time.sleep(0.05)

def optimize_model_thread():
    global optimization_running, optimization_progress, optimization_cancel_requested, optimization_finished_flag, optimization_finished_cancelled, policy_model
    optimization_running = True
    optimization_progress = 0.0
    optimization_finished_flag = False
    optimization_finished_cancelled = False
    flush_experience_buffer()
    dataset = ExperienceDataset(experience_dir)
    synthetic_only = len(dataset) == 0
    if synthetic_only:
        dataset = SyntheticDataset(320)
    device = "cuda" if gpu_available else "cpu"
    model = PolicyNet()
    try:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state, strict=False)
    except:
        pass
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    bce_control = nn.BCEWithLogitsLoss()
    ce_rule = nn.CrossEntropyLoss()
    ce_type = nn.CrossEntropyLoss()
    mse_pref = nn.MSELoss()
    scaler = None
    if device == "cuda":
        try:
            scaler = torch.amp.GradScaler("cuda")
        except:
            scaler = None
    batch_size = min(32, max(1, len(dataset)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_epochs = 4 if synthetic_only else 3
    total_steps = max(1, len(loader) * total_epochs)
    step = 0
    for epoch in range(total_epochs):
        for batch in loader:
            if optimization_cancel_requested:
                break
            obs, act, src, num = batch
            obs = obs.to(device)
            act = act.to(device)
            num = num.to(device)
            target_start = act[:, :2]
            target_end = act[:, 2:4]
            target_mid = act[:, 4:6]
            target_press = act[:, 6]
            target_hold = act[:, 7]
            target_type = act[:, 8].long().clamp(0, 3)
            optimizer.zero_grad()
            ctx = torch.amp.autocast("cuda") if device == "cuda" and scaler else contextlib.nullcontext()
            with ctx:
                pos_start, pos_end, pos_mid, press_logit, hold_logit, control_logits, _, type_logits, pref_recon = model(obs, num)
                loss_start = mse((pos_start + 1.0) / 2.0, target_start)
                loss_end = mse((pos_end + 1.0) / 2.0, target_end)
                loss_mid = mse((pos_mid + 1.0) / 2.0, target_mid)
                loss_press = bce(press_logit, target_press)
                loss_hold = bce(hold_logit, target_hold)
                gh = model.grid_h
                gw = model.grid_w
                control_target = torch.zeros((obs.size(0), 1, gh, gw), device=device)
                click_mask = target_press >= 0.5
                if click_mask.any():
                    idx = torch.nonzero(click_mask, as_tuple=False).squeeze(1)
                    cx = (target_start[idx, 0] * (gw - 1)).long().clamp(0, gw - 1)
                    cy = (target_start[idx, 1] * (gh - 1)).long().clamp(0, gh - 1)
                    control_target[idx, 0, cy, cx] = 1.0
                loss_control = bce_control(control_logits, control_target)
                loss_type = ce_type(type_logits, target_type)
                loss_pref = mse_pref(pref_recon, num)
                loss = loss_start + loss_end + loss_mid + loss_press + 0.5 * loss_hold + 0.1 * loss_control + 0.25 * loss_type + 0.2 * loss_pref
                if obs.size(0) > 0:
                    aug_list = []
                    label_list = []
                    for k in range(4):
                        aug = torch.rot90(obs, k, dims=(2, 3))
                        aug_list.append(aug)
                        label_list.append(torch.full((obs.size(0),), k, dtype=torch.long, device=device))
                    obs_aug = torch.cat(aug_list, dim=0)
                    rot_labels = torch.cat(label_list, dim=0)
                    _, _, _, rule_logits = model(obs_aug)
                    loss_rule = ce_rule(rule_logits, rot_labels)
                    loss = loss + 0.1 * loss_rule
            if scaler is not None:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
            optimizer.step()
            step += 1
            optimization_progress = min(100.0, step / total_steps * 100.0)
        if optimization_cancel_requested:
            break
    model_cpu = model.to("cpu")
    try:
        torch.save(model_cpu.state_dict(), model_path)
        ts_name = os.path.join(models_dir, f"policy_{int(time.time())}.pt")
        torch.save(model_cpu.state_dict(), ts_name)
    except:
        pass
    with model_lock:
        policy_model = model_cpu
    optimization_running = False
    optimization_finished_flag = True
    optimization_finished_cancelled = optimization_cancel_requested

def build_simple_ocr_templates():
    global simple_ocr_templates
    if simple_ocr_templates is not None:
        return simple_ocr_templates
    if Image is None or ImageDraw is None or ImageFont is None:
        simple_ocr_templates = []
        return simple_ocr_templates
    templates = []
    w, h = simple_ocr_size
    try:
        font = ImageFont.truetype("arial.ttf", h)
    except:
        try:
            font = ImageFont.load_default()
        except:
            font = None
    for d in range(10):
        canvas = Image.new("L", (w, h), 255)
        draw = ImageDraw.Draw(canvas)
        text = str(d)
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            tw = bbox[2] - bbox[0]
            th = bbox[3] - bbox[1]
        except:
            tw, th = draw.textsize(text, font=font)
        x = (w - tw) // 2
        y = (h - th) // 2
        draw.text((x, y), text, font=font, fill=0)
        arr = np.array(canvas, dtype=np.float32) / 255.0
        templates.append(arr)
    simple_ocr_templates = templates
    return simple_ocr_templates

def simple_ocr_recognize(img):
    templates = build_simple_ocr_templates()
    if Image is None or not templates:
        return []
    gray = img.convert("L")
    arr = np.array(gray, dtype=np.uint8)
    threshold = max(64, min(200, int(arr.mean())))
    mask = arr < threshold
    col_active = np.where(mask.sum(axis=0) > 0)[0]
    if col_active.size == 0:
        return []
    regions = []
    start = col_active[0]
    prev = start
    for c in col_active[1:]:
        if c != prev + 1:
            regions.append((start, prev))
            start = c
        prev = c
    regions.append((start, prev))
    results = []
    for rs, re in regions:
        if re - rs < 2:
            continue
        sub_mask = mask[:, rs:re + 1]
        rows = np.where(sub_mask.any(axis=1))[0]
        if rows.size == 0:
            continue
        top = rows.min()
        bottom = rows.max()
        if bottom - top < 4:
            continue
        crop_arr = arr[top:bottom + 1, rs:re + 1]
        crop_img = Image.fromarray(crop_arr)
        resized = crop_img.resize(simple_ocr_size, Image.Resampling.BILINEAR)
        cand = np.array(resized, dtype=np.float32) / 255.0
        best_digit = None
        best_err = None
        for d, tmpl in enumerate(templates):
            diff = np.mean(np.abs(cand - tmpl))
            if best_err is None or diff < best_err:
                best_err = diff
                best_digit = d
        if best_digit is not None and best_err is not None and best_err < 0.45:
            results.append({"value": int(best_digit), "bbox": (float(rs), float(top), float(re), float(bottom))})
    return results

def recognize_numbers_from_image(img):
    global recognition_progress
    if img is None:
        return []
    if Image is None:
        return []
    processed_list = []
    try:
        recognition_progress = 10.0
        if img.mode != "RGB":
            img = img.convert("RGB")
        w, h = img.size
        large_img = img.resize((max(1, w * 3), max(1, h * 3)), Image.Resampling.BICUBIC)
        enhancer = ImageEnhance.Contrast(large_img)
        high_contrast = enhancer.enhance(2.0)
        gray = high_contrast.convert("L")
        recognition_progress = 20.0
        processed_list.append(gray)
        processed_list.append(ImageOps.invert(gray))
        auto_gray = ImageOps.autocontrast(gray)
        processed_list.append(auto_gray)
        processed_list.append(ImageEnhance.Sharpness(auto_gray).enhance(2.0))
        processed_list.append(auto_gray.filter(ImageFilter.MedianFilter(size=3)))
        recognition_progress = 30.0
        for th in [80, 100, 128, 160, 200]:
            bin_img = gray.point(lambda x, t=th: 255 if x > t else 0, mode="L")
            processed_list.append(bin_img)
            processed_list.append(ImageOps.invert(bin_img))
        recognition_progress = 50.0
        candidates = []
        reader = None
        if easyocr is not None:
            with ocr_lock:
                global ocr_reader
                if ocr_reader is None:
                    try:
                        ocr_reader = easyocr.Reader(["en"], gpu=gpu_available)
                    except:
                        ocr_reader = None
                reader = ocr_reader
        total_ops = max(1, len(processed_list))
        op_cnt = 0
        for pm in processed_list:
            work_img = pm
            if reader is not None:
                if work_img.mode != "RGB":
                    work_img = work_img.convert("RGB")
                try:
                    res = reader.readtext(np.array(work_img), detail=1, allowlist="0123456789")
                    for bbox, text, conf in res:
                        if conf < 0.35:
                            continue
                        for n in re.findall(r"\d+", text):
                            try:
                                v = int(n)
                                xs = [p[0] for p in bbox]
                                ys = [p[1] for p in bbox]
                                bx1, by1, bx2, by2 = min(xs), min(ys), max(xs), max(ys)
                                candidates.append({"value": v, "bbox": (bx1, by1, bx2, by2)})
                            except:
                                pass
                except Exception as e:
                    print(f"OCR Error: {e}")
            else:
                basic_candidates = simple_ocr_recognize(work_img)
                candidates.extend(basic_candidates)
            op_cnt += 1
            recognition_progress = 50.0 + (float(op_cnt) / total_ops * 40.0)
        if not candidates and reader is not None:
            candidates = simple_ocr_recognize(gray)
        merged = []
        seen = set()
        for item in candidates:
            bbox = item.get("bbox")
            key = (item.get("value"), tuple(bbox) if bbox is not None else None)
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged
    except Exception as e:
        print(f"OCR Error: {e}")
        return []

def update_number_list_ui():
    global number_row_widgets, recognition_result_msg
    for w in number_row_widgets:
        try:
            w.destroy()
        except:
            pass
    number_row_widgets = []
    with recognized_lock:
        data = list(recognized_values)
        attempted = recognition_attempted
        msg = recognition_result_msg
    
    if not data:
        text = "尚未识别到数值，请在学习模式下点击“识别”按钮。"
        if attempted:
            text = "识别完成，但未检测到非负整数。\n\n" + msg
        label = ttk.Label(numbers_inner, text=text, style="Status.TLabel", wraplength=800)
        label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        number_row_widgets.append(label)
        return
    
    for idx, item in enumerate(data):
        row = ttk.Frame(numbers_inner, style="App.TFrame")
        row.grid(row=idx, column=0, sticky="we", padx=4, pady=2)
        label = ttk.Label(row, text=f"{idx + 1}. 数值: {item['value']}", style="Status.TLabel")
        label.pack(side="left")
        var = tk.StringVar(value=item.get("category", "无关"))
        locked_state = bool(item.get("locked", False))
        combo = ttk.Combobox(row, textvariable=var, values=category_choices, state="readonly" if not locked_state else "disabled", width=18)
        combo.pack(side="left", padx=(8, 0))
        def on_select(event, i=idx, v=var):
            with recognized_lock:
                if i < len(recognized_values):
                    recognized_values[i]["category"] = v.get()
        combo.bind("<<ComboboxSelected>>", on_select)
        lock_btn = ttk.Button(row, text="锁定" if not locked_state else "解锁", style="Accent.TButton")
        def on_lock_toggle(i=idx, btn=lock_btn, cb=combo):
            with recognized_lock:
                if i < len(recognized_values):
                    current = bool(recognized_values[i].get("locked", False))
                    recognized_values[i]["locked"] = not current
                    if not current:
                        cb.state(["disabled"])
                        btn.configure(text="解锁")
                    else:
                        cb.state(["!disabled"])
                        btn.configure(text="锁定")
        lock_btn.configure(command=on_lock_toggle)
        lock_btn.pack(side="left", padx=8)
        number_row_widgets.append(row)

def on_recognize_clicked():
    global recognition_running, recognition_progress
    if recognition_running:
        return
    if window_a_handle is None:
        messagebox.showerror("错误", "请先选择窗口 A。")
        return
    if get_mode() != MODE_LEARN:
        messagebox.showerror("提示", "仅在学习模式下才能进行数值识别。")
        return
    if not window_a_visible:
        messagebox.showerror("提示", "窗口 A 当前不可见或不完整，无法识别。")
        return
    if Image is None:
        messagebox.showerror("错误", "需要安装 Pillow 才能识别数值。")
        return
    
    set_mode(MODE_RECOG)
    
    def worker():
        global recognition_running, recognition_attempted, recognition_progress, recognition_finished_flag, recognition_result_msg
        recognition_running = True
        recognition_progress = 0.0
        recognition_result_msg = ""
        try:
            hwnd = window_a_handle
            img = None
            if hwnd is not None:
                recognition_progress = 5.0
                           
                time.sleep(0.5)
                           
                img = capture_window_image(hwnd)
                
                                                
                                                     
            
            values = []
            if img is not None:

                stat = np.array(img).std()
                if stat < 5.0:
                    recognition_result_msg = "警告：截取到的画面几乎为纯色（黑/白）。\n可能是硬件加速导致，请尝试将窗口移至屏幕中央并保持前台可见。"
                else:
                    values = recognize_numbers_from_image(img)
                    if not values:
                        recognition_result_msg = "图像截取成功，但OCR未能提取到数字。\n请尝试放大窗口，或检查数字是否存在遮挡。"
                    elif easyocr is None:
                        recognition_result_msg = "已使用内置数字识别引擎完成处理。"
            else:
                recognition_result_msg = "无法获取窗口图像，请确保窗口未最小化。"

            recognition_progress = 95.0
            new_list = []
            for item in values:
                v = item.get("value")
                bbox = item.get("bbox")
                color = get_value_color(v)
                new_list.append({"value": v, "category": "无关", "bbox": bbox, "color": color, "locked": False})
            with recognized_lock:
                recognized_values.clear()
                recognized_values.extend(new_list)
            recognition_attempted = True
            recognition_progress = 100.0
        except Exception as e:
            recognition_result_msg = f"发生未知错误: {str(e)}"
        finally:
            recognition_running = False
            recognition_finished_flag = True
            try:
                root.after(0, update_number_list_ui)
            except:
                pass
    t = threading.Thread(target=worker, daemon=True)
    t.start()

def update_ui_loop():
    global canvas_image_ref, optimization_finished_flag, optimization_finished_cancelled, optimization_progress, optimization_cancel_requested, recognition_finished_flag, recognition_progress
    if not program_running:
        try:
            root.quit()
        except:
            pass
        return
    try:
        if window_a_title:
            window_label_var.set("窗口 A: " + window_a_title)
        else:
            window_label_var.set("窗口 A: 未选择")
        visible_label_var.set(f"可见且完整: {'是' if window_a_visible else '否'} (遮挡 {int(window_a_occlusion * 100)}%)")
        w = window_a_rect[2] - window_a_rect[0]
        h = window_a_rect[3] - window_a_rect[1]
        size_label_var.set(f"窗口大小: {w} x {h}")
        fps_label_var.set(f"截图频率: {screenshot_fps:.1f} Hz")
        cpu_label_var.set(f"CPU 占用: {hardware_stats['cpu']:.1f}%")
        mem_label_var.set(f"内存占用: {hardware_stats['mem']:.1f}%")
        gpu_known = hardware_stats.get("gpu_known", False)
        vram_known = hardware_stats.get("vram_known", False)
        gpu_hint = hardware_stats.get("gpu_hint", "")
        if gpu_known:
            gpu_label_var.set(f"GPU 占用: {hardware_stats['gpu']:.1f}%")
        else:
            hint = f" ({gpu_hint})" if gpu_hint else ""
            gpu_label_var.set(f"GPU 占用: 未知{hint}")
        if vram_known:
            vram_label_var.set(f"显存占用: {hardware_stats['vram']:.1f}%")
        else:
            hint = f" ({gpu_hint})" if gpu_hint else ""
            vram_label_var.set(f"显存占用: 未知{hint}")
        mode_map = {MODE_INIT: "初始化", MODE_LEARN: "学习模式", MODE_TRAIN: "训练模式", MODE_OPT: "优化中", MODE_RECOG: "识别中"}
        mode_now = get_mode()
        mode_label_var.set("模式: " + mode_map.get(mode_now, mode_now))
        if optimization_running:
            progress_bar["value"] = optimization_progress
            progress_label_var.set(f"正在优化 AI 模型: {optimization_progress:.1f}%")
        elif recognition_running:
            progress_bar["value"] = recognition_progress
            progress_label_var.set(f"正在识别窗口 A 数值: {recognition_progress:.1f}%")
        else:
            if mode_now == MODE_OPT:
                progress_bar["value"] = optimization_progress
                progress_label_var.set("优化准备中...")
            elif mode_now == MODE_RECOG:
                progress_bar["value"] = recognition_progress
                progress_label_var.set("识别准备中...")
            else:
                progress_bar["value"] = 0.0
                progress_label_var.set("")
        with last_frame_lock:
            frame = last_frame_np
        if frame is not None and ImageTk is not None and Image is not None:
            img = Image.fromarray(frame)
            overlays = []
            with recognized_lock:
                for item in recognized_values:
                    bbox = item.get("bbox")
                    val = item.get("value")
                    color = item.get("color", get_value_color(val))
                    if bbox:
                        overlays.append((bbox, color))
            if overlays and ImageDraw is not None:
                img = img.convert("RGBA")
                dr = ImageDraw.Draw(img, "RGBA")
                for bbox, color in overlays:
                    x1, y1, x2, y2 = bbox
                    fill = (color[0], color[1], color[2], 80)
                    outline = (color[0], color[1], color[2], 200)
                    dr.rectangle([x1, y1, x2, y2], outline=outline, width=2)
                    dr.rectangle([x1, y1, x2, y2], fill=fill)
                img = img.convert("RGB")
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw > 10 and ch > 10:
                fw, fh = img.size
                if fw > 0 and fh > 0:
                    scale = min(cw / fw, ch / fh)
                    nw = max(1, int(fw * scale))
                    nh = max(1, int(fh * scale))
                    img = img.resize((nw, nh), Image.Resampling.BILINEAR)
                    img_tk = ImageTk.PhotoImage(img)
                    canvas.configure(image=img_tk)
                    canvas_image_ref = img_tk
        if mode_now == MODE_LEARN and window_a_handle is not None and not optimization_running and not recognition_running:
            sleep_btn.state(["!disabled"])
        else:
            sleep_btn.state(["disabled"])
        if mode_now == MODE_LEARN and window_a_handle is not None and window_a_visible and not optimization_running and not recognition_running:
            recognize_btn.state(["!disabled"])
        else:
            recognize_btn.state(["disabled"])
        if optimization_running:
            getup_btn.state(["!disabled"])
        else:
            getup_btn.state(["disabled"])
        if optimization_finished_flag:
            optimization_finished_flag = False
            if optimization_finished_cancelled:
                optimization_cancel_requested = False
                optimization_progress = 0.0
                progress_bar["value"] = 0.0
                progress_label_var.set("")
                set_mode(MODE_LEARN)
            else:
                res = messagebox.showinfo("优化完成", "模型优化完成，点击确定继续学习。")
                if res is not None:
                    optimization_progress = 0.0
                    progress_bar["value"] = 0.0
                    progress_label_var.set("")
                    set_mode(MODE_LEARN)
        if recognition_finished_flag:
            recognition_finished_flag = False
            res = messagebox.showinfo("识别完成", "数值识别过程已结束，点击确定查看结果。")
            if res is not None:
                recognition_progress = 0.0
                progress_bar["value"] = 0.0
                progress_label_var.set("")
                set_mode(MODE_LEARN)
        root.after(100, update_ui_loop)
    except:
        try:
            root.after(200, update_ui_loop)
        except:
            pass

def select_window():
    global window_a_handle, window_a_title
    if win32gui is None:
        messagebox.showerror("错误", "win32gui 不可用，无法选择窗口")
        return
    windows = []
    def enum_handler(hwnd, lParam):
        if win32gui.IsWindowVisible(hwnd):
            title = win32gui.GetWindowText(hwnd)
            if title:
                windows.append((hwnd, title))
    win32gui.EnumWindows(enum_handler, None)
    if not windows:
        messagebox.showerror("错误", "没有找到可用窗口")
        return
    top = tk.Toplevel(root)
    top.title("选择窗口 A")
    top.configure(bg="#020617")
    listbox = tk.Listbox(top, width=80, height=20, bg="#020617", fg="#e5e7eb", selectbackground="#38bdf8", highlightthickness=0, borderwidth=0)
    listbox.pack(fill="both", expand=True, padx=8, pady=8)
    for hwnd, title in windows:
        listbox.insert("end", f"{hwnd} - {title}")
    def on_ok():
        global window_a_handle, window_a_title
        sel = listbox.curselection()
        if not sel:
            return
        idx = sel[0]
        hwnd_sel, title_sel = windows[idx]
        window_a_handle = hwnd_sel
        window_a_title = title_sel
        set_mode(MODE_LEARN)
        top.destroy()
    ok_btn = ttk.Button(top, text="确定", style="Accent.TButton", command=on_ok)
    ok_btn.pack(pady=(0, 8))

def on_sleep_clicked():
    global optimization_cancel_requested
    if window_a_handle is None:
        return
    mode = get_mode()
    if mode != MODE_LEARN:
        return
    if optimization_running:
        return
    optimization_cancel_requested = False
    set_mode(MODE_OPT)
    t = threading.Thread(target=optimize_model_thread, daemon=True)
    t.start()

def on_getup_clicked():
    global optimization_cancel_requested
    if not optimization_running:
        return
    optimization_cancel_requested = True

def terminate_program():
    global program_running
    program_running = False
    flush_experience_buffer()
    try:
        root.quit()
    except:
        pass

def main():
    ensure_model_exists()
    th_hw = threading.Thread(target=hardware_monitor_loop, daemon=True)
    th_hw.start()
    th_frame = threading.Thread(target=frame_loop, daemon=True)
    th_frame.start()
    th_input = threading.Thread(target=global_input_loop, daemon=True)
    th_input.start()
    if keyboard is not None:
        th_kb = threading.Thread(target=keyboard_listener_loop, daemon=True)
        th_kb.start()
    if mouse is not None:
        th_mouse = threading.Thread(target=mouse_listener_loop, daemon=True)
        th_mouse.start()
    select_btn.configure(command=select_window)
    sleep_btn.configure(command=on_sleep_clicked)
    getup_btn.configure(command=on_getup_clicked)
    recognize_btn.configure(command=on_recognize_clicked)
    update_number_list_ui()
    root.after(100, update_ui_loop)
    root.after(500, idle_mode_manager)
    root.protocol("WM_DELETE_WINDOW", lambda: terminate_program())
    root.bind("<Escape>", lambda e: terminate_program())
    root.mainloop()
    flush_experience_buffer()

if __name__ == "__main__":
    main()
