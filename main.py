import os
import threading
import time
import ctypes
import contextlib
import re
import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import psutil
import torch
import numpy as np
from torch import nn
from torch.utils.data import Dataset, DataLoader

try:
    import win32gui
    import win32con
    import win32ui
except ImportError:
    win32gui = None
    win32con = None
    win32ui = None

try:
    from pynput import keyboard, mouse
except ImportError:
    keyboard = None
    mouse = None

try:
    import pynvml
except ImportError:
    pynvml = None

try:
    from PIL import Image, ImageTk
except ImportError:
    Image = None
    ImageTk = None

try:
    import pyautogui
except ImportError:
    pyautogui = None

try:
    import pytesseract
    import os as _os_tess
    try:
        if hasattr(pytesseract, "pytesseract") and hasattr(pytesseract.pytesseract, "tesseract_cmd") and not pytesseract.pytesseract.tesseract_cmd:
            for _p in (r"C:\\Program Files\\Tesseract-OCR\\tesseract.exe", r"C:\\Program Files (x86)\\Tesseract-OCR\\tesseract.exe"):
                if _os_tess.path.exists(_p):
                    pytesseract.pytesseract.tesseract_cmd = _p
                    break
    except:
        pass
except ImportError:
    pytesseract = None

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

experience_buffer = []
experience_lock = threading.Lock()
experience_file_index = 0
last_user_action_vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)
last_ai_action_vec = np.array([0.5, 0.5, 0.0], dtype=np.float32)

policy_model = None
model_lock = threading.Lock()

recognized_values = []
recognized_lock = threading.Lock()
category_choices = ["越高越好", "越低越好", "变化越小越好", "变化越大越好", "无关", "识别错误"]
recognition_running = False
recognition_attempted = False
recognition_progress = 0.0
recognition_finished_flag = False

current_mode = MODE_INIT
mode_lock = threading.Lock()

window_a_handle = None
window_a_title = ""
window_a_visible = False
window_a_rect = (0, 0, 0, 0)

last_frame_np = None
last_frame_lock = threading.Lock()

last_user_input_time = time.monotonic()
program_running = True

screenshot_fps = 10.0
hardware_stats = {"cpu": 0.0, "mem": 0.0, "gpu": 0.0, "vram": 0.0}

optimization_progress = 0.0
optimization_running = False
optimization_cancel_requested = False
optimization_finished_flag = False
optimization_finished_cancelled = False

gpu_available = torch.cuda.is_available()
gpu_handle = None

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
root.geometry("1200x800")
root.minsize(960, 680)
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
        self.fc = nn.Sequential(
            nn.Linear(64 * 10 * 10, 256),
            nn.ReLU()
        )
        self.grid_h = 21
        self.grid_w = 21
        self.action_head = nn.Linear(256, 3)
        self.control_head = nn.Linear(256, self.grid_h * self.grid_w)
        self.rule_head = nn.Linear(256, 4)

    def forward(self, x):
        x = x.float() / 255.0
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        action = self.action_head(x)
        control_logits = self.control_head(x).view(-1, 1, self.grid_h, self.grid_w)
        rule_logits = self.rule_head(x)
        pos = torch.tanh(action[:, :2])
        click_logit = action[:, 2]
        return pos, click_logit, control_logits, rule_logits


class ExperienceDataset(Dataset):
    def __init__(self, directory):
        self.samples = []
        for name in os.listdir(directory):
            if name.endswith(".pt") and name.startswith("experience_"):
                path = os.path.join(directory, name)
                try:
                    data = torch.load(path, map_location="cpu")
                    obs = data.get("obs")
                    act = data.get("act")
                    src = data.get("src")
                    if obs is not None and act is not None:
                        n = obs.shape[0]
                        for i in range(n):
                            self.samples.append((obs[i], act[i], 0 if src is None else int(src[i])))
                except:
                    continue

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        o, a, s = self.samples[idx]
        return o, a, s

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
        except:
            pass
        for path in model_files:
            try:
                state = torch.load(path, map_location="cpu")
                break
            except:
                continue
    model = PolicyNet()
    if state is not None:
        try:
            model.load_state_dict(state)
        except:
            pass
    try:
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

def is_left_button_pressed():
    try:
        return bool(ctypes.windll.user32.GetAsyncKeyState(0x01) & 0x8000)
    except:
        return False

def capture_window_image(hwnd):
    global window_a_rect
    if win32gui is None or win32ui is None or Image is None:
        if pyautogui is not None and window_a_rect is not None:
            left, top, right, bottom = window_a_rect
            w = right - left
            h = bottom - top
            if w > 0 and h > 0:
                try:
                    img = pyautogui.screenshot(region=(left, top, w, h))
                    return img.convert("RGB")
                except:
                    return None
        return None
    try:
        left, top, right, bottom = win32gui.GetWindowRect(hwnd)
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0:
            return None
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
        img = Image.frombuffer("RGB", (bmpinfo["bmWidth"], bmpinfo["bmHeight"]), bmpstr, "raw", "BGRX", 0, 1)
        win32gui.DeleteObject(saveBitMap.GetHandle())
        saveDC.DeleteDC()
        mfcDC.DeleteDC()
        win32gui.ReleaseDC(hwnd, hwndDC)
        try:
            import numpy as _np_internal_check
            arr = _np_internal_check.array(img)
            if arr.size == 0 or float(arr.mean()) < 1.0:
                if pyautogui is not None:
                    img2 = pyautogui.screenshot(region=(left, top, w, h))
                    return img2.convert("RGB")
        except:
            pass
        return img
    except:
        try:
            if pyautogui is not None:
                rect = win32gui.GetWindowRect(hwnd)
                left, top, right, bottom = rect
                w = right - left
                h = bottom - top
                if w > 0 and h > 0:
                    img = pyautogui.screenshot(region=(left, top, w, h))
                    return img.convert("RGB")
        except:
            pass
        return None

def resize_for_model(img):
    if Image is None:
        return None
    img = img.resize((84, 84), Image.BILINEAR)
    arr = np.array(img, dtype=np.uint8)
    if arr.ndim == 2:
        arr = np.stack([arr, arr, arr], axis=-1)
    arr = np.transpose(arr, (2, 0, 1))
    return arr

def record_experience(frame_arr, action_vec, source_flag):
    global experience_buffer, experience_file_index
    if frame_arr is None or action_vec is None:
        return
    with experience_lock:
        experience_buffer.append((frame_arr, action_vec, source_flag))
        if len(experience_buffer) >= 128:
            try:
                obs = np.stack([x[0] for x in experience_buffer], axis=0)
                act = np.stack([x[1] for x in experience_buffer], axis=0).astype(np.float32)
                src = np.array([x[2] for x in experience_buffer], dtype=np.int64)
                tensor_obs = torch.from_numpy(obs)
                tensor_act = torch.from_numpy(act)
                tensor_src = torch.from_numpy(src)
                path = os.path.join(experience_dir, f"experience_{experience_file_index}.pt")
                experience_file_index += 1
                torch.save({"obs": tensor_obs, "act": tensor_act, "src": tensor_src}, path)
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
            tensor_obs = torch.from_numpy(obs)
            tensor_act = torch.from_numpy(act)
            tensor_src = torch.from_numpy(src)
            path = os.path.join(experience_dir, f"experience_{experience_file_index}.pt")
            experience_file_index += 1
            torch.save({"obs": tensor_obs, "act": tensor_act, "src": tensor_src}, path)
        except:
            pass
        experience_buffer = []

def hardware_monitor_loop():
    global hardware_stats, screenshot_fps, program_running
    while program_running:
        try:
            cpu = psutil.cpu_percent(interval=0.5)
            mem = psutil.virtual_memory().percent
            gpu = 0.0
            vram = 0.0
            if gpu_handle is not None and pynvml is not None:
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(gpu_handle)
                    meminfo = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
                    gpu = float(util.gpu)
                    vram = float(meminfo.used) / float(meminfo.total) * 100.0 if meminfo.total > 0 else 0.0
                except:
                    gpu = 0.0
                    vram = 0.0
            elif gpu_available:
                try:
                    total = torch.cuda.get_device_properties(0).total_memory
                    used = torch.cuda.memory_allocated(0)
                    vram = float(used) / float(total) * 100.0 if total > 0 else 0.0
                except:
                    vram = 0.0
                gpu = vram
            hardware_stats = {"cpu": cpu, "mem": mem, "gpu": gpu, "vram": vram}
            metrics = [cpu, mem]
            if gpu > 0.0:
                metrics.append(gpu)
            if vram > 0.0:
                metrics.append(vram)
            if metrics:
                stress = max(metrics)
                fps = 120.0 * max(0.0, min(1.0, 1.0 - stress / 100.0))
            else:
                fps = 60.0
            screenshot_fps = max(0.0, min(120.0, fps))
        except:
            pass

def window_visibility_check(hwnd):
    global window_a_rect
    if win32gui is None:
        return False, (0, 0, 0, 0)
    try:
        if not win32gui.IsWindow(hwnd) or not win32gui.IsWindowVisible(hwnd):
            return False, (0, 0, 0, 0)
        if win32gui.IsIconic(hwnd):
            return False, (0, 0, 0, 0)
        rect = win32gui.GetWindowRect(hwnd)
        left, top, right, bottom = rect
        w = right - left
        h = bottom - top
        if w <= 0 or h <= 0:
            return False, rect
        sx = ctypes.windll.user32.GetSystemMetrics(76)
        sy = ctypes.windll.user32.GetSystemMetrics(77)
        sw = ctypes.windll.user32.GetSystemMetrics(78)
        sh = ctypes.windll.user32.GetSystemMetrics(79)
        if not (left >= sx and top >= sy and right <= sx + sw and bottom <= sy + sh):
            return False, rect
        sample_cols = 5
        sample_rows = 5
        for i in range(sample_cols):
            for j in range(sample_rows):
                px = left + (i + 0.5) * w / sample_cols
                py = top + (j + 0.5) * h / sample_rows
                if not (sx <= px < sx + sw and sy <= py < sy + sh):
                    return False, rect
                pt = (int(px), int(py))
                try:
                    hwnd_at_pt = win32gui.WindowFromPoint(pt)
                    try:
                        ga_root = getattr(win32con, "GA_ROOT", 2)
                        root_hwnd = win32gui.GetAncestor(hwnd_at_pt, ga_root)
                    except:
                        root_hwnd = hwnd_at_pt
                except:
                    root_hwnd = hwnd
                if root_hwnd != hwnd:
                    return False, rect
        window_a_rect = rect
        return True, rect
    except:
        return False, (0, 0, 0, 0)

def ai_compute_action(frame_arr):
    global policy_model
    if policy_model is None:
        return None
    try:
        with model_lock:
            model = policy_model
            model.eval()
            x = torch.from_numpy(frame_arr).unsqueeze(0)
            with torch.no_grad():
                pos, click_logit, control_logits, rule_logits = model(x)
                pos = pos[0].cpu().numpy()
                click_prob = torch.sigmoid(click_logit)[0].item()
        nx = (pos[0] + 1.0) / 2.0
        ny = (pos[1] + 1.0) / 2.0
        nx = max(0.0, min(1.0, nx))
        ny = max(0.0, min(1.0, ny))
        click_flag = 1.0 if click_prob > 0.5 else 0.0
        return np.array([nx, ny, click_flag], dtype=np.float32)
    except:
        return None

def frame_loop():
    global window_a_visible, window_a_title, window_a_rect, last_frame_np, program_running, last_user_action_vec, last_ai_action_vec
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
        vis, rect = window_visibility_check(hwnd)
        window_a_visible = vis
        window_a_rect = rect
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
        img = capture_window_image(hwnd)
        if img is None:
            continue
        frame_arr_model = resize_for_model(img)
        with last_frame_lock:
            last_frame_np = np.array(img)
        mode = get_mode()
        recording = mode in (MODE_LEARN, MODE_TRAIN) and vis and not optimization_running
        action_vec = None
        source_flag = 0
        if mode == MODE_TRAIN:
            if frame_arr_model is not None:
                a = ai_compute_action(frame_arr_model)
                if a is not None:
                    last_ai_action_vec = a
                action_vec = last_ai_action_vec
                source_flag = 1
                if action_vec is not None and pyautogui is not None:
                    nx, ny, click_flag = float(action_vec[0]), float(action_vec[1]), float(action_vec[2])
                    x, y = denormalize_action_to_mouse(nx, ny, rect)
                    try:
                        pyautogui.moveTo(x, y)
                        if click_flag >= 0.5:
                            pyautogui.click()
                    except:
                        pass
        elif mode == MODE_LEARN:
            if win32gui is not None:
                try:
                    pt = win32gui.GetCursorPos()
                except:
                    pt = None
                if pt is not None and rect[0] <= pt[0] <= rect[2] and rect[1] <= pt[1] <= rect[3]:
                    nx, ny = normalize_action_from_mouse(pt[0], pt[1], rect)
                    click_flag = 1.0 if is_left_button_pressed() else 0.0
                    action_vec = np.array([nx, ny, click_flag], dtype=np.float32)
            if action_vec is not None:
                last_user_action_vec = action_vec
            action_vec = last_user_action_vec
            source_flag = 0
        if recording and frame_arr_model is not None and action_vec is not None:
            record_experience(frame_arr_model, action_vec, source_flag)

class LASTINPUTINFO(ctypes.Structure):
    _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]

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
        if mode == MODE_LEARN:
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
    if len(dataset) == 0:
        model = PolicyNet()
        try:
            torch.save(model.state_dict(), model_path)
            ts_name = os.path.join(models_dir, f"policy_{int(time.time())}.pt")
            torch.save(model.state_dict(), ts_name)
        except:
            pass
        with model_lock:
            policy_model = model
        optimization_running = False
        optimization_finished_flag = True
        optimization_finished_cancelled = False
        return
    device = "cuda" if gpu_available else "cpu"
    model = PolicyNet()
    try:
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
    except:
        pass
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    bce = nn.BCEWithLogitsLoss()
    mse = nn.MSELoss()
    bce_control = nn.BCEWithLogitsLoss()
    ce_rule = nn.CrossEntropyLoss()
    if device == "cuda":
        scaler = torch.amp.GradScaler("cuda")
    else:
        scaler = None
    batch_size = min(32, max(1, len(dataset)))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    total_steps = max(1, len(loader) * 3)
    step = 0
    for epoch in range(3):
        for batch in loader:
            if optimization_cancel_requested:
                break
            obs, act, src = batch
            obs = obs.to(device)
            act = act.to(device)
            target_pos = act[:, :2]
            target_click = act[:, 2]
            optimizer.zero_grad()
            if device == "cuda":
                ctx = torch.amp.autocast("cuda")
            else:
                ctx = contextlib.nullcontext()
            with ctx:
                pos, click_logit, control_logits, _ = model(obs)
                loss_pos = mse((pos + 1.0) / 2.0, target_pos)
                loss_click = bce(click_logit, target_click)
                gh = model.grid_h
                gw = model.grid_w
                control_target = torch.zeros((obs.size(0), 1, gh, gw), device=device)
                click_mask = target_click >= 0.5
                if click_mask.any():
                    idx = torch.nonzero(click_mask, as_tuple=False).squeeze(1)
                    cx = (target_pos[idx, 0] * (gw - 1)).long().clamp(0, gw - 1)
                    cy = (target_pos[idx, 1] * (gh - 1)).long().clamp(0, gh - 1)
                    control_target[idx, 0, cy, cx] = 1.0
                loss_control = bce_control(control_logits, control_target)
                loss = loss_pos + loss_click + 0.1 * loss_control
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

def recognize_numbers_from_image(img):
    if img is None:
        return []
    if pytesseract is None or Image is None:
        return []
    try:
        gray = img.convert("L")
        w, h = gray.size
        scale = 1
        m = max(w, h)
        if m < 400:
            scale = 3
        elif m < 900:
            scale = 2
        nw = max(1, int(w * scale))
        nh = max(1, int(h * scale))
        if nw != w or nh != h:
            gray = gray.resize((nw, nh), Image.BILINEAR)
        candidates = [gray]
        try:
            from PIL import ImageEnhance, ImageFilter
            try:
                candidates.append(ImageEnhance.Contrast(gray).enhance(2.0))
            except:
                pass
            try:
                candidates.append(gray.filter(ImageFilter.SHARPEN))
            except:
                pass
        except:
            pass
        for th in (120, 150, 180, 210):
            try:
                candidates.append(gray.point(lambda p, t=th: 255 if p > t else 0, "L"))
            except:
                pass
        texts = []
        cfg_base = "-c tessedit_char_whitelist=0123456789"
        psms = ["--psm 6", "--psm 7", "--psm 11"]
        for im in candidates:
            for psm in psms:
                cfg = psm + " " + cfg_base
                try:
                    texts.append(pytesseract.image_to_string(im, config=cfg))
                except:
                    pass
                try:
                    if hasattr(pytesseract, "Output"):
                        data = pytesseract.image_to_data(im, config=cfg, output_type=pytesseract.Output.DICT)
                        if isinstance(data, dict) and "text" in data:
                            for s in data["text"]:
                                if s and re.fullmatch(r"\d+", s):
                                    texts.append(s)
                except:
                    pass
        joined = " ".join([t for t in texts if t])
        digits = re.findall(r"\d+", joined)
        values = []
        seen = set()
        for d in digits:
            try:
                v = int(d)
                if v >= 0 and v not in seen:
                    seen.add(v)
                    values.append(v)
            except:
                continue
        return values
    except:
        return []
def update_number_list_ui():
    global number_row_widgets
    for w in number_row_widgets:
        try:
            w.destroy()
        except:
            pass
    number_row_widgets = []
    with recognized_lock:
        data = list(recognized_values)
        attempted = recognition_attempted
    if not data:
        text = "尚未识别到数值，请在学习模式下点击“识别”按钮。"
        if attempted:
            text = "识别完成，但未检测到任何非负整数，请检查窗口内容或识别区域。"
        label = ttk.Label(numbers_inner, text=text, style="Status.TLabel")
        label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
        number_row_widgets.append(label)
        return
    for idx, item in enumerate(data):
        row = ttk.Frame(numbers_inner, style="App.TFrame")
        row.grid(row=idx, column=0, sticky="we", padx=4, pady=2)
        label = ttk.Label(row, text=f"{idx + 1}. 数值: {item['value']}", style="Status.TLabel")
        label.pack(side="left")
        var = tk.StringVar(value=item.get("category", "无关"))
        combo = ttk.Combobox(row, textvariable=var, values=category_choices, state="readonly", width=18)
        combo.pack(side="left", padx=(8, 0))
        def on_select(event, i=idx, v=var):
            with recognized_lock:
                if i < len(recognized_values):
                    recognized_values[i]["category"] = v.get()
        combo.bind("<<ComboboxSelected>>", on_select)
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
    if pytesseract is None or Image is None:
        messagebox.showerror("错误", "需要安装 Pillow 和 pytesseract 才能识别数值，请先安装后重试。")
        return
    set_mode(MODE_RECOG)
    def worker():
        global recognition_running, recognition_attempted, recognition_progress, recognition_finished_flag
        recognition_running = True
        recognition_progress = 0.0
        try:
            hwnd = window_a_handle
            img = None
            if hwnd is not None:
                recognition_progress = 10.0
                img = capture_window_image(hwnd)
            values = []
            if img is not None:
                recognition_progress = 50.0
                values = recognize_numbers_from_image(img)
            recognition_progress = 90.0
            new_list = []
            for v in values:
                new_list.append({"value": v, "category": "无关"})
            with recognized_lock:
                recognized_values.clear()
                recognized_values.extend(new_list)
                recognition_attempted = True
            recognition_progress = 100.0
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
        visible_label_var.set("可见且完整: " + ("是" if window_a_visible else "否"))
        w = window_a_rect[2] - window_a_rect[0]
        h = window_a_rect[3] - window_a_rect[1]
        size_label_var.set(f"窗口大小: {w} x {h}")
        fps_label_var.set(f"截图频率: {screenshot_fps:.1f} Hz")
        cpu_label_var.set(f"CPU 占用: {hardware_stats['cpu']:.1f}%")
        mem_label_var.set(f"内存占用: {hardware_stats['mem']:.1f}%")
        gpu_label_var.set(f"GPU 占用: {hardware_stats['gpu']:.1f}%")
        vram_label_var.set(f"显存占用: {hardware_stats['vram']:.1f}%")
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
            cw = canvas.winfo_width()
            ch = canvas.winfo_height()
            if cw > 10 and ch > 10:
                fw, fh = img.size
                if fw > 0 and fh > 0:
                    scale = min(cw / fw, ch / fh)
                    nw = max(1, int(fw * scale))
                    nh = max(1, int(fh * scale))
                    img = img.resize((nw, nh), Image.BILINEAR)
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
                res = messagebox.showinfo("优化完成", "模型优化完成，点击 Confirm 继续学习。")
                if res is not None:
                    optimization_progress = 0.0
                    progress_bar["value"] = 0.0
                    progress_label_var.set("")
                    set_mode(MODE_LEARN)
        if recognition_finished_flag:
            recognition_finished_flag = False
            res = messagebox.showinfo("识别完成", "数值识别完成，点击 Confirm 返回学习模式。")
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
