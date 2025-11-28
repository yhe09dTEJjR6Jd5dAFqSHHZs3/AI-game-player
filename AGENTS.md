这是一个通用游戏AI。

数据：时间、鼠标、电脑屏幕画面、用户划分的区域、各区域OCR读取结果。

学习模式：满足条件时，记录数据到经验池（鼠标来源：用户）
训练模式：AI根据实时数据，输出鼠标操作（点击、长按、按下-复杂拖动路径-松开），同时记录数据到经验池（鼠标来源：AI）

工作流程：
开始运行→学习模式→ESC结束运行；
学习模式→用户点击窗口上的优化按钮→退出学习模式，停止记录数据，开始离线优化→离线优化完成→生成新的AI模型→弹窗提示→用户点击确定后，切换到学习模式；
学习模式→用户点击窗口上的“识别”按钮→退出学习模式，停止记录数据，此时用户可以在电脑屏幕上划分区域→用户点击保存后，切换到学习模式；
学习模式→空格→训练模式→检测到除ESC以外的其他键盘按键时，切换到学习模式；
训练模式→ESC结束运行；

共有4种区域：
红色边框（区域内数字越小越好）
蓝色边框（区域内数字越大越好）
黄色区域（区域内数字变化幅度越大越好）
绿色区域（区域内数字变化幅度越小越好）

AI对变化幅度大的区域更加敏感。
用户划分的区域内需要识别的内容一定是一个平滑变化的非负整数。

cuDNN版本：8.9.7
Python版本：3.10.11
操作系统：Windows 11
PaddleOCR版本：v2.6.1
CUDA Toolkit版本：11.8

要求：
简体中文界面。
仅用一个无注释的Python脚本实现所有功能。
运行过程中生成的所有文件（AI模型、经验池），均位于Desktop\AAA及其子文件夹。

有一个科幻感满满的窗口，实时展示各种信息，包括但不限于：
CPU、内存、GPU、显存占用率（像心电图那样的曲线）
显示器分辨率、缩放、磁盘可用空间；
进度条（百分比，提示文本）

需要避免的报错：

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

Warning: you have set wrong precision for backend:cuda setFloat32Precision call has no effect.Please choose precision from: ieee tf32 none  (function setFloat32Precision)

UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9.

