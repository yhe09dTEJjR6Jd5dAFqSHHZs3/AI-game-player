import os
import sys
import time
import math
import psutil
import threading
import datetime
import tkinter as tk
from tkinter import ttk, messagebox
import numpy as np
import cv2
from PIL import Image, ImageTk
import mss
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import win32gui
import win32con
import win32api
import win32ui
from pynput import mouse, keyboard
import shutil

class AdaptiveScaler:
    def __init__(self):
        self.cpu_usage = 0
        self.ram_usage = 0
        self.gpu_usage = 0
        self.vram_usage = 0
        self.fps = 60.0
    
    def update(self):
        self.cpu_usage = psutil.cpu_percent(interval=None)
        self.ram_usage = psutil.virtual_memory().percent
        try:
            if torch.cuda.is_available():
                d = torch.cuda.current_device()
                self.vram_usage = torch.cuda.memory_reserved(d) / torch.cuda.get_device_properties(d).total_memory * 100
                self.gpu_usage = 0 
            else:
                self.gpu_usage = 0
                self.vram_usage = 0
        except:
            self.gpu_usage = 0
            self.vram_usage = 0
        
        max_load = max(self.cpu_usage, self.ram_usage, self.gpu_usage, self.vram_usage)
        self.fps = max(1.0, 120.0 * (1.0 - (max_load / 100.0) ** 2))
        return self.fps

class ControlNet(nn.Module):
    def __init__(self):
        super(ControlNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 512)
        self.fc_mouse_x = nn.Linear(512, 1)
        self.fc_mouse_y = nn.Linear(512, 1)
        self.fc_click = nn.Linear(512, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        mx = torch.sigmoid(self.fc_mouse_x(x))
        my = torch.sigmoid(self.fc_mouse_y(x))
        mc = torch.sigmoid(self.fc_click(x))
        return mx, my, mc

class OCRNet(nn.Module):
    def __init__(self):
        super(OCRNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 11) 

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

class ExperienceBuffer:
    def __init__(self, base_path):
        self.base_path = base_path
        if not os.path.exists(base_path):
            os.makedirs(base_path)
        self.current_buffer = []

    def add(self, img, mouse_x, mouse_y, click, source):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{timestamp}_{source}.jpg"
        full_path = os.path.join(self.base_path, filename)
        cv2.imwrite(full_path, img)
        self.current_buffer.append({
            "image": filename,
            "mx": mouse_x,
            "my": mouse_y,
            "click": click,
            "source": source
        })
        if len(self.current_buffer) > 100:
            self.flush()

    def flush(self):
        pass

    def get_all_data(self):
        data = []
        files = [f for f in os.listdir(self.base_path) if f.endswith('.jpg')]
        for f in files:
            parts = f.split('_')
            data.append((os.path.join(self.base_path, f), 0.5, 0.5, 0)) 
        return data

class MainApp:
    def __init__(self, root):
        self.root = root
        self.root.title("GameAI Controller")
        self.root.geometry("1000x800")
        
        self.desktop_path = os.path.join(os.path.expanduser("~"), "Desktop", "GameAI")
        self.models_path = os.path.join(self.desktop_path, "models")
        self.exp_path = os.path.join(self.desktop_path, "experience")
        
        if not os.path.exists(self.models_path): os.makedirs(self.models_path)
        if not os.path.exists(self.exp_path): os.makedirs(self.exp_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.control_model_path = os.path.join(self.models_path, "control.pth")
        self.ocr_model_path = os.path.join(self.models_path, "ocr.pth")
        
        self.control_net = ControlNet().to(self.device)
        self.ocr_net = OCRNet().to(self.device)
        self.check_models()
        
        self.buffer = ExperienceBuffer(self.exp_path)
        self.scaler = AdaptiveScaler()
        self.sct = mss.mss()
        
        self.mode = "INIT" 
        self.target_window_name = None
        self.target_hwnd = None
        self.window_rect = (0, 0, 0, 0) 
        self.last_input_time = time.time()
        
        self.mouse_controller = mouse.Controller()
        
        self.mouse_listener = mouse.Listener(on_move=self.on_move, on_click=self.on_click)
        self.key_listener = keyboard.Listener(on_press=self.on_press)
        self.mouse_listener.start()
        self.key_listener.start()
        
        self.current_mouse_rel = (0, 0)
        self.current_click = 0
        
        self.stop_event = threading.Event()
        self.ocr_results = []
        
        self.setup_ui()
        self.update_loop()
        
    def check_models(self):
        if not os.path.exists(self.control_model_path):
            torch.save(self.control_net.state_dict(), self.control_model_path)
        else:
            try:
                self.control_net.load_state_dict(torch.load(self.control_model_path, weights_only=True))
            except:
                torch.save(self.control_net.state_dict(), self.control_model_path)

        if not os.path.exists(self.ocr_model_path):
            torch.save(self.ocr_net.state_dict(), self.ocr_model_path)
        else:
            try:
                self.ocr_net.load_state_dict(torch.load(self.ocr_model_path, weights_only=True))
            except:
                torch.save(self.ocr_net.state_dict(), self.ocr_model_path)

    def setup_ui(self):
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        info_frame = ttk.LabelFrame(main_frame, text="Status")
        info_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_freq = ttk.Label(info_frame, text="Freq: 0Hz")
        self.lbl_freq.pack(side=tk.LEFT, padx=5)
        self.lbl_win_name = ttk.Label(info_frame, text="Window: None")
        self.lbl_win_name.pack(side=tk.LEFT, padx=5)
        self.lbl_win_status = ttk.Label(info_frame, text="Status: N/A")
        self.lbl_win_status.pack(side=tk.LEFT, padx=5)
        self.lbl_win_size = ttk.Label(info_frame, text="Size: 0x0")
        self.lbl_win_size.pack(side=tk.LEFT, padx=5)
        self.lbl_mode = ttk.Label(info_frame, text="Mode: INIT")
        self.lbl_mode.pack(side=tk.LEFT, padx=5)
        
        hw_frame = ttk.Frame(info_frame)
        hw_frame.pack(side=tk.RIGHT, padx=5)
        self.lbl_cpu = ttk.Label(hw_frame, text="CPU: 0%")
        self.lbl_cpu.pack(side=tk.LEFT)
        self.lbl_mem = ttk.Label(hw_frame, text="MEM: 0%")
        self.lbl_mem.pack(side=tk.LEFT)
        self.lbl_gpu = ttk.Label(hw_frame, text="GPU: 0%")
        self.lbl_gpu.pack(side=tk.LEFT)
        self.lbl_vram = ttk.Label(hw_frame, text="VRAM: 0%")
        self.lbl_vram.pack(side=tk.LEFT)
        
        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)
        
        self.combo_windows = ttk.Combobox(control_frame)
        self.combo_windows.pack(side=tk.LEFT, padx=5)
        self.btn_select = ttk.Button(control_frame, text="Select Window A", command=self.select_window)
        self.btn_select.pack(side=tk.LEFT, padx=5)
        
        self.btn_identify = ttk.Button(control_frame, text="Identify", command=self.start_identify, state=tk.DISABLED)
        self.btn_identify.pack(side=tk.LEFT, padx=5)
        
        self.btn_sleep = ttk.Button(control_frame, text="Sleep", command=self.start_sleep, state=tk.DISABLED)
        self.btn_sleep.pack(side=tk.LEFT, padx=5)
        
        self.btn_getup = ttk.Button(control_frame, text="Get Up", command=self.stop_sleep, state=tk.DISABLED)
        self.btn_getup.pack(side=tk.LEFT, padx=5)
        
        self.progress = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progress.pack(side=tk.RIGHT, padx=5)
        self.lbl_progress = ttk.Label(control_frame, text="")
        self.lbl_progress.pack(side=tk.RIGHT)

        content_frame = ttk.Frame(main_frame)
        content_frame.pack(fill=tk.BOTH, expand=True)
        
        self.canvas = tk.Canvas(content_frame, bg="black", width=640, height=360)
        self.canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        
        right_panel = ttk.Frame(content_frame, width=300)
        right_panel.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.tree = ttk.Treeview(right_panel, columns=("Value", "Preference"), show="headings")
        self.tree.heading("Value", text="Value")
        self.tree.heading("Preference", text="Preference")
        self.tree.pack(fill=tk.BOTH, expand=True)
        self.tree.bind("<Double-1>", self.on_tree_select)
        
        self.refresh_windows()

    def refresh_windows(self):
        windows = []
        def enum_cb(hwnd, results):
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetWindowText(hwnd):
                results.append((hwnd, win32gui.GetWindowText(hwnd)))
        win32gui.EnumWindows(enum_cb, windows)
        self.window_map = {name: hwnd for hwnd, name in windows}
        self.combo_windows['values'] = list(self.window_map.keys())

    def select_window(self):
        name = self.combo_windows.get()
        if name in self.window_map:
            self.target_window_name = name
            self.target_hwnd = self.window_map[name]
            self.mode = "LEARNING"
            self.btn_select.configure(state=tk.DISABLED)
            self.combo_windows.configure(state=tk.DISABLED)
            self.btn_identify.configure(state=tk.NORMAL)
            self.btn_sleep.configure(state=tk.NORMAL)
            self.last_input_time = time.time()

    def on_move(self, x, y):
        self.last_input_time = time.time()
        if self.mode == "TRAINING":
            pass 

    def on_click(self, x, y, button, pressed):
        self.last_input_time = time.time()
        self.current_click = 1 if pressed else 0

    def on_press(self, key):
        self.last_input_time = time.time()
        if key == keyboard.Key.esc:
            self.root.quit()
            sys.exit()
        if self.mode == "TRAINING":
            self.mode = "LEARNING"

    def is_window_active_and_visible(self):
        if not self.target_hwnd: return False, (0,0,0,0)
        if not win32gui.IsWindow(self.target_hwnd): return False, (0,0,0,0)
        if not win32gui.IsWindowVisible(self.target_hwnd): return False, (0,0,0,0)
        
        rect = win32gui.GetWindowRect(self.target_hwnd)
        x, y, w, h = rect[0], rect[1], rect[2]-rect[0], rect[3]-rect[1]
        if w <= 0 or h <= 0: return False, (0,0,0,0)
        
        fg_window = win32gui.GetForegroundWindow()
        is_fg = (fg_window == self.target_hwnd)
        
        return True, (x, y, w, h) 

    def process_image(self, img_np):
        if img_np is None: return None
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_BGRA2RGB)
        img_pil = Image.fromarray(img_rgb)
        
        orig_w, orig_h = img_pil.size
        ratio = min(640/orig_w, 360/orig_h)
        new_size = (int(orig_w*ratio), int(orig_h*ratio))
        img_resized = img_pil.resize(new_size, Image.Resampling.LANCZOS)
        return ImageTk.PhotoImage(img_resized)

    def update_loop(self):
        start_time = time.time()
        fps = self.scaler.update()
        
        self.lbl_freq.config(text=f"Freq: {int(fps)}Hz")
        self.lbl_cpu.config(text=f"CPU: {self.scaler.cpu_usage}%")
        self.lbl_mem.config(text=f"MEM: {self.scaler.ram_usage}%")
        self.lbl_gpu.config(text=f"GPU: {int(self.scaler.gpu_usage)}%")
        self.lbl_vram.config(text=f"VRAM: {int(self.scaler.vram_usage)}%")
        
        if self.target_hwnd:
            visible, rect = self.is_window_active_and_visible()
            self.lbl_win_name.config(text=f"Win: {self.target_window_name[:10]}...")
            self.lbl_win_status.config(text=f"Vis: {visible}")
            self.lbl_win_size.config(text=f"{rect[2]}x{rect[3]}")
            
            img_np = None
            if visible:
                monitor = {"top": rect[1], "left": rect[0], "width": rect[2], "height": rect[3]}
                img = self.sct.grab(monitor)
                img_np = np.array(img)
                
                photo = self.process_image(img_np)
                if photo:
                    self.canvas.create_image(320, 180, image=photo, anchor=tk.CENTER)
                    self.canvas.image = photo

                mx, my = self.mouse_controller.position
                rel_x = (mx - rect[0]) / rect[2]
                rel_y = (my - rect[1]) / rect[3]
                
                if 0 <= rel_x <= 1 and 0 <= rel_y <= 1:
                    if self.mode == "LEARNING":
                        self.buffer.add(img_np[:,:,:3], rel_x, rel_y, self.current_click, "user")
                    
                    if self.mode == "TRAINING":
                        inp = cv2.resize(img_np[:,:,:3], (84, 84))
                        inp = torch.from_numpy(inp).float().permute(2,0,1).unsqueeze(0).to(self.device) / 255.0
                        with torch.no_grad():
                            px, py, pc = self.control_net(inp)
                        
                        tx = int(px.item() * rect[2] + rect[0])
                        ty = int(py.item() * rect[3] + rect[1])
                        self.mouse_controller.position = (tx, ty)
                        
                        self.buffer.add(img_np[:,:,:3], px.item(), py.item(), pc.item(), "ai")
            
            if self.mode == "LEARNING" and (time.time() - self.last_input_time > 10) and visible:
                self.mode = "TRAINING"

        self.lbl_mode.config(text=f"Mode: {self.mode}")

        delay = int(1000 / fps)
        self.root.after(delay, self.update_loop)

    def start_identify(self):
        self.btn_identify.config(state=tk.DISABLED)
        prev_mode = self.mode
        self.mode = "IDENTIFY"
        
        def worker():
            visible, rect = self.is_window_active_and_visible()
            if not visible:
                self.root.after(0, self.finish_identify_fail)
                return

            monitor = {"top": rect[1], "left": rect[0], "width": rect[2], "height": rect[3]}
            img = np.array(self.sct.grab(monitor))
            gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
            thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
            
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            rois = []
            for c in contours:
                x, y, w, h = cv2.boundingRect(c)
                if 10 < w < 100 and 10 < h < 100:
                    roi = gray[y:y+h, x:x+w]
                    rois.append(roi)
            
            total = len(rois)
            results = []
            for i, roi in enumerate(rois):
                p = int((i / total) * 100)
                self.root.after(0, lambda p=p: self.progress.configure(value=p))
                self.root.after(0, lambda p=p: self.lbl_progress.configure(text=f"OCR: {p}%"))
                
                roi_rez = cv2.resize(roi, (28, 28))
                t_roi = torch.from_numpy(roi_rez).float().unsqueeze(0).unsqueeze(0).to(self.device) / 255.0
                with torch.no_grad():
                    out = self.ocr_net(t_roi)
                    pred = out.argmax(dim=1, keepdim=True).item()
                
                if pred < 10: 
                    results.append(pred)
            
            self.ocr_results = list(set(results)) 
            self.root.after(0, lambda: self.finish_identify(prev_mode))

        threading.Thread(target=worker, daemon=True).start()

    def finish_identify_fail(self):
        messagebox.showerror("Error", "Window A not visible")
        self.btn_identify.config(state=tk.NORMAL)
        self.mode = "LEARNING"

    def finish_identify(self, prev_mode):
        messagebox.showinfo("Info", "Identification Complete")
        self.progress['value'] = 0
        self.lbl_progress.config(text="")
        self.btn_identify.config(state=tk.NORMAL)
        self.mode = "LEARNING"
        
        for item in self.tree.get_children():
            self.tree.delete(item)
        
        for val in self.ocr_results:
            self.tree.insert("", "end", values=(val, "Ignore"))

    def on_tree_select(self, event):
        item = self.tree.selection()[0]
        cur_vals = self.tree.item(item, "values")
        opts = ["Higher Better", "Lower Better", "Stable Better", "Change Better", "Ignore", "Error"]
        
        def set_val(v):
            self.tree.item(item, values=(cur_vals[0], v))
            top.destroy()

        top = tk.Toplevel(self.root)
        for o in opts:
            tk.Button(top, text=o, command=lambda o=o: set_val(o)).pack(fill=tk.X)

    def start_sleep(self):
        self.btn_sleep.config(state=tk.DISABLED)
        self.btn_getup.config(state=tk.NORMAL)
        self.mode = "SLEEP"
        self.stop_optimization = False
        
        def worker():
            scaler = torch.amp.GradScaler('cuda', enabled=True)
            optimizer = optim.Adam(self.control_net.parameters(), lr=1e-4)
            
            for i in range(100):
                if self.stop_optimization: break
                
                self.root.after(0, lambda p=i: self.progress.configure(value=p))
                self.root.after(0, lambda p=i: self.lbl_progress.configure(text=f"Opt: {p}%"))
                
                optimizer.zero_grad()
                # Dummy Loss for structure
                dummy_in = torch.randn(4, 3, 84, 84).to(self.device)
                dummy_target = torch.randn(4, 1).to(self.device)
                
                with torch.amp.autocast('cuda', enabled=True):
                    mx, my, mc = self.control_net(dummy_in)
                    loss = F.mse_loss(mx, dummy_target)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                time.sleep(0.1) 
            
            torch.save(self.control_net.state_dict(), self.control_model_path)
            self.root.after(0, self.finish_sleep)

        threading.Thread(target=worker, daemon=True).start()

    def stop_sleep(self):
        self.stop_optimization = True
        self.btn_getup.config(state=tk.DISABLED)

    def finish_sleep(self):
        messagebox.showinfo("Info", "Optimization Complete")
        self.progress['value'] = 0
        self.lbl_progress.config(text="")
        self.btn_sleep.config(state=tk.NORMAL)
        self.btn_getup.config(state=tk.DISABLED)
        self.mode = "LEARNING"

if __name__ == "__main__":
    root = tk.Tk()
    app = MainApp(root)
    root.mainloop()