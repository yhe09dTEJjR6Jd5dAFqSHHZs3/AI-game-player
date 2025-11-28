import sys
import os
import time
import json
import random
import threading
import queue
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
from torch.utils.data import Dataset, DataLoader
from pynput import mouse, keyboard
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QHBoxLayout, QLabel, QPushButton, QProgressBar,
                             QFrame, QMessageBox, QGraphicsDropShadowEffect, QDialog, QGridLayout)
from PyQt5.QtCore import QTimer, Qt, pyqtSignal, QThread, QPoint, QRect, QSize
from PyQt5.QtGui import QColor, QPainter, QPen, QFont, QBrush, QImage, QPixmap
import pyqtgraph as pg
from collections import deque
from paddleocr import PaddleOCR


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
        f.write("timestamp,img_path,mx,my,click,source,ocr\n")

if not os.path.exists(REGION_FILE):
    with open(REGION_FILE, 'w', encoding='utf-8') as f:
        json.dump([], f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_W, INPUT_H = 320, 180
SEQ_LEN = 4
MAX_OCR = 16

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

def get_screen_size():
    with mss.mss() as sct:
        monitor = sct.monitors[1]
        return monitor['width'], monitor['height'], monitor['left'], monitor['top']

SCREEN_W, SCREEN_H, SCREEN_LEFT, SCREEN_TOP = get_screen_size()

def pack_ocr(vals):
    buf = [0]*MAX_OCR
    for i, v in enumerate(vals[:MAX_OCR]):
        buf[i] = float(v)
    return buf

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
        self.ocr_enc = nn.Sequential(nn.Linear(MAX_OCR, 64), nn.SiLU())
        self.gru = nn.GRU(256 + 64, 256, batch_first=True)
        self.dir_head = nn.Linear(256, 9)
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Linear(128, 3),
            nn.Sigmoid()
        )

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
        seq = torch.cat([x, ocr_feat], dim=2)
        out, _ = self.gru(seq)
        feat = out[:, -1, :]
        dirs = self.dir_head(feat)
        return self.fc(feat), dirs

class ExperiencePool(Dataset):
    def __init__(self):
        self.data = []
        if os.path.exists(LOG_FILE):
            with open(LOG_FILE, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
                for line in lines:
                    try:
                        parts = line.strip().split(',')
                        if len(parts) >= 6:
                            ts = int(parts[0])
                            img_path = parts[1]
                            mx, my, click = float(parts[2]), float(parts[3]), float(parts[4])
                            ocr = []
                            if len(parts) >= 7:
                                ocr = [float(x) for x in parts[6].split('|') if x != '']
                            self.data.append({'ts': ts, 'img': img_path, 'mx': mx, 'my': my, 'click': click, 'ocr': pack_ocr(ocr)})
                    except:
                        pass
        self.data.sort(key=lambda x: x['ts'])

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
            img = cv2.imread(item['img'])
            if img is None:
                img = np.zeros((INPUT_H, INPUT_W, 3), dtype=np.uint8)
            img = preprocess_frame(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
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
        return torch.tensor(np.stack(seq_imgs), dtype=torch.float32), torch.tensor(np.stack(seq_ocr), dtype=torch.float32), torch.tensor([mx, my, click], dtype=torch.float32), torch.tensor(direction, dtype=torch.long)

class OptimizerThread(QThread):
    finished_sig = pyqtSignal()
    progress_sig = pyqtSignal(int)
    
    def run(self):
        ds = ExperiencePool()
        if len(ds) < 10:
            self.finished_sig.emit()
            return
            
        dl = DataLoader(ds, batch_size=16, shuffle=True, num_workers=0)
        model = Brain().to(DEVICE)

        opt = optim.AdamW(model.parameters(), lr=1e-3)
        reg_loss = nn.SmoothL1Loss()
        bce = nn.BCELoss()

        scaler = torch.amp.GradScaler('cuda') if torch.cuda.is_available() else None
        
        epochs = 5
        model.train()
        
        for ep in range(epochs):
            for i, (imgs, ocrs, targs, dirs) in enumerate(dl):
                imgs, ocrs, targs, dirs = imgs.to(DEVICE), ocrs.to(DEVICE), targs.to(DEVICE), dirs.to(DEVICE)
                opt.zero_grad(set_to_none=True)

                if scaler:
                    with torch.amp.autocast('cuda'):
                        out, logits = model(imgs, ocrs)
                        l_pos = reg_loss(out[:, :2], targs[:, :2])
                        l_click = bce(out[:, 2], targs[:, 2])
                        l_dir = F.cross_entropy(logits, dirs)
                        loss = l_pos + l_click + 0.5 * l_dir
                    scaler.scale(loss).backward()
                    scaler.step(opt)
                    scaler.update()
                else:
                    out, logits = model(imgs, ocrs)
                    l_pos = reg_loss(out[:, :2], targs[:, :2])
                    l_click = bce(out[:, 2], targs[:, 2])
                    l_dir = F.cross_entropy(logits, dirs)
                    loss = l_pos + l_click + 0.5 * l_dir
                    loss.backward()
                    opt.step()
            
            self.progress_sig.emit(int((ep + 1) / epochs * 100))
            
        torch.save(model.state_dict(), os.path.join(MODEL_DIR, "brain.pth"))
        self.finished_sig.emit()

class DataWorker(QThread):
    ocr_result = pyqtSignal(list)
    
    def __init__(self, queue):
        super().__init__()
        self.queue = queue
        self.running = True
        self.regions = []
        self.prev_rois = []
        self.prev_vals = []
        self.prev_change = []
        self.histories = []
        self.last_read_tick = []
        self.stable_counts = []
        self.tick = 0
        self.ocr = PaddleOCR(use_angle_cls=False, use_gpu=torch.cuda.is_available())
        self.reload_regions()

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

    def run(self):
        while self.running:
            try:
                data = self.queue.get(timeout=0.1)
                full_img, small_img, mx, my, click, source, save = data
                self.tick += 1

                ocr_vals = []
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
                            diff = 1.0 if self.prev_rois[idx] is None else float(np.mean(cv2.absdiff(roi, self.prev_rois[idx]))) / 255.0
                            diff = 0.5 * self.prev_change[idx] + 0.5 * diff
                            self.prev_change[idx] = diff
                            self.prev_rois[idx] = roi
                            prev_v = self.prev_vals[idx]
                            need_read = diff >= 0.003 or prev_v is None or self.tick - self.last_read_tick[idx] >= 5
                            if need_read:
                                result = self.ocr.ocr(roi, cls=False)
                                digits = ''
                                if result and len(result) > 0:
                                    texts = [item[1][0] for item in result[0] if len(item) > 1]
                                    digits = ''.join([c for c in ''.join(texts) if c.isdigit()])
                                self.last_read_tick[idx] = self.tick
                                candidate = int(digits) if digits else prev_v
                            else:
                                candidate = prev_v
                            if candidate is None:
                                candidate = 0
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
                        except:
                            ocr_vals.append(self.prev_vals[idx] if idx < len(self.prev_vals) else 0)
                    self.ocr_result.emit(ocr_vals)
                else:
                    self.ocr_result.emit([])

                if save:
                    ts = str(int(time.time() * 1000))
                    name = f"{ts}.jpg"
                    path = os.path.join(IMG_DIR, name)
                    cv2.imwrite(path, small_img)
                    with open(LOG_FILE, 'a', encoding='utf-8') as f:
                        ocr_txt = "|".join([str(int(v)) for v in ocr_vals]) if ocr_vals else ""
                        f.write(f"{ts},{path},{mx:.5f},{my:.5f},{click:.2f},{source},{ocr_txt}\n")
                
                self.queue.task_done()
            except queue.Empty:
                pass
            except Exception as e:
                pass

    def stop(self):
        self.running = False
        self.wait()

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
            'blue': QColor(50, 50, 255),
            'yellow': QColor(255, 255, 50),
            'green': QColor(50, 255, 50)
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
        if len(vals) == len(self.regions):
            self.values = vals
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
            'blue': (QColor(50, 50, 255), "蓝框(越大越好)"),
            'yellow': (QColor(255, 255, 50), "黄框(波动大)"),
            'green': (QColor(50, 255, 50), "绿框(波动小)")
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
        info = f"当前模式: {cur_txt} | [1-4]切换类型 | 鼠标拖拽创建 | 右键删除 | 回车保存 | ESC退出"
        detail = ("红框=数值越小越好 | 蓝框=数值越大越好 | 黄框=波动剧烈重点关注 | "
                  "绿框=保持稳定区域 | 请精确框选需要识别的数字或监测区域")
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
        elif e.key() == Qt.Key_3: self.current_type = 'yellow'
        elif e.key() == Qt.Key_4: self.current_type = 'green'
        elif e.key() in [Qt.Key_Return, Qt.Key_Enter]:
            with open(REGION_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.regions, f)
            self.accept()
        elif e.key() == Qt.Key_Escape:
            self.reject()
        self.update()

class MainWin(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("神经接口系统")
        self.resize(1000, 600)
        self.setStyleSheet("QMainWindow {background-color: #050a0f; color: #00f0ff; font-family: 'SimHei';}")
        
        self.brain = Brain().to(DEVICE)
        self.load_model()
        
        self.sct = mss.mss()
        self.monitor = self.sct.monitors[1]
        self.mouse = mouse.Controller()
        
        self.queue = queue.Queue()
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
        self.seq_frames = deque(maxlen=SEQ_LEN)
        self.seq_ocr = deque(maxlen=SEQ_LEN)
        
        self.init_ui()
        self.setup_listeners()
        
        self.timer = QTimer()
        self.timer.timeout.connect(self.loop)
        self.timer.start(15)
        
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
        
        header = QHBoxLayout()
        self.mode_lbl = QLabel("模式: 学习模式")
        self.mode_lbl.setStyleSheet("font-size: 22px; font-weight: bold; color: #00ff00;")
        header.addWidget(self.mode_lbl)
        header.addStretch()
        fl.addLayout(header)
        
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
        if p and b == mouse.Button.left:
            self.mouse_pressed = True
        elif not p and b == mouse.Button.left:
            self.mouse_pressed = False

    def on_key(self, key):
        if key == keyboard.Key.esc:
            self.close_app()
        elif key == keyboard.Key.space:
            if self.mode == "LEARNING":
                self.mode = "TRAINING"
            elif self.mode == "TRAINING":
                self.mode = "LEARNING"
            self.update_mode()
        else:
            if self.mode == "TRAINING":
                self.mode = "LEARNING"
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
        self.mode = "OPTIMIZING"
        self.update_mode()
        self.bar.show()
        self.bar.setValue(0)
        self.opt_thread = OptimizerThread()
        self.opt_thread.progress_sig.connect(self.bar.setValue)
        self.opt_thread.finished_sig.connect(self.opt_done)
        self.opt_thread.start()

    def opt_done(self):
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

    def update_ocr(self, vals):
        self.latest_ocr = pack_ocr(vals)
        self.seq_ocr.append(self.latest_ocr)

    def update_stats(self):
        cpu = psutil.cpu_percent()
        mem = psutil.virtual_memory().percent
        
        gpu_u, vram_u, vram_t = 0, 0, 1
        try:
            if torch.cuda.is_available():
                o = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
                u, m, t = map(float, o.decode('utf-8').strip().split(','))
                gpu_u, vram_u, vram_t = u, m, t
        except: pass
        
        disk = psutil.disk_usage(os.path.abspath(os.sep))
        scale = 100
        if platform.system() == 'Windows':
            try:
                import ctypes
                scale = ctypes.windll.shcore.GetScaleFactorForDevice(0)
            except: pass
            
        self.lbl_cpu.setText(f"处理器: {cpu}%")
        self.lbl_mem.setText(f"内存: {mem}%")
        self.lbl_gpu.setText(f"显卡: {gpu_u}%")
        self.lbl_vram.setText(f"显存: {vram_u/1024:.1f}/{vram_t/1024:.1f}GB")
        self.lbl_disk.setText(f"磁盘可用: {disk.free/(1024**4):.2f}TB")
        self.lbl_res.setText(f"分辨率: {SCREEN_W}x{SCREEN_H} ({scale}%)")
        
        for l, v in zip([self.d_cpu, self.d_mem, self.d_gpu, self.d_vram], [cpu, mem, gpu_u, (vram_u/vram_t)*100]):
            l.pop(0)
            l.append(v)
            
        self.p_cpu.plot(self.d_cpu, pen=pg.mkPen('#00ff00', width=2), clear=True)
        self.p_mem.plot(self.d_mem, pen=pg.mkPen('#00ffff', width=2), clear=True)
        self.p_gpu.plot(self.d_gpu, pen=pg.mkPen('#ff0055', width=2), clear=True)
        self.p_vram.plot(self.d_vram, pen=pg.mkPen('#ffaa00', width=2), clear=True)

    def loop(self):
        if self.mode in ["OPTIMIZING", "SELECT"]: return
        
        try:
            img = np.array(self.sct.grab(self.monitor))
            frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            proc = preprocess_frame(rgb)
            self.seq_frames.append(proc)
            self.seq_ocr.append(self.latest_ocr)

            mx, my = self.mouse.position
            nx, ny = mx/SCREEN_W, my/SCREEN_H
            nx, ny = max(0, min(1, nx)), max(0, min(1, ny))
            click = 1.0 if self.mouse_pressed else 0.0
            source = "USER"

            if self.mode == "TRAINING":
                seq_imgs = list(self.seq_frames)
                seq_ocr = list(self.seq_ocr)
                while len(seq_imgs) < SEQ_LEN:
                    seq_imgs.insert(0, np.zeros((4, INPUT_H, INPUT_W), dtype=np.float32))
                    seq_ocr.insert(0, pack_ocr([]))
                seq_imgs = torch.tensor(np.stack(seq_imgs)[None, ...], dtype=torch.float32, device=DEVICE)
                seq_ocr = torch.tensor(np.stack(seq_ocr)[None, ...], dtype=torch.float32, device=DEVICE)

                with torch.no_grad():
                    if torch.cuda.is_available():
                        with torch.amp.autocast('cuda'):
                            out, logits = self.brain(seq_imgs, seq_ocr)
                    else:
                        out, logits = self.brain(seq_imgs, seq_ocr)

                px, py, pc = out[0].cpu().numpy()
                conf = float(torch.softmax(logits, dim=1)[0].max().cpu())
                blend = 0.3 + 0.7 * conf
                self.smooth_x = (1 - blend) * self.smooth_x + blend * px
                self.smooth_y = (1 - blend) * self.smooth_y + blend * py

                self.mouse.position = (int(self.smooth_x * SCREEN_W), int(self.smooth_y * SCREEN_H))
                if pc > 0.5 and not self.dragging:
                    self.mouse.press(mouse.Button.left)
                    self.dragging = True
                elif pc <= 0.5 and self.dragging:
                    self.mouse.release(mouse.Button.left)
                    self.dragging = False
                
                nx, ny, click = self.smooth_x, self.smooth_y, pc
                source = "AI"
            else:
                self.smooth_x, self.smooth_y = nx, ny
                if self.dragging:
                    self.mouse.release(mouse.Button.left)
                    self.dragging = False

            save = False
            if (self.mode == "LEARNING" and source == "USER") or (self.mode == "TRAINING" and source == "AI"):
                if time.time() - self.last_record > 0.1:
                    save = True
                    self.last_record = time.time()

            small = letterbox(frame, INPUT_W, INPUT_H)
            self.queue.put((frame, small, nx, ny, click, source, save))

        except Exception: pass

    def close_app(self):
        self.worker.stop()
        if self.dragging:
            self.mouse.release(mouse.Button.left)
        QApplication.quit()
        sys.exit()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    w = MainWin()
    w.show()
    sys.exit(app.exec_())
