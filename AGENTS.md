这是一个通用游戏AI。

数据：毫秒级时序标签、鼠标（特征标签/位置/运动方向/速度/预测偏差/复杂度）、全分辨率屏幕视觉帧、用户划分的区域配置、各区域OCR识别结果、各区域数值变化幅度。

学习模式：满足条件时，记录数据到经验池（鼠标来源：用户）
训练模式：AI根据实时数据，在电脑屏幕上输出鼠标操作（含点击、长按及非线性复杂拖拽路径），同时记录数据到经验池（鼠标来源：AI）

工作流程：
开始运行→学习模式→ESC结束运行；
学习模式→用户点击窗口上的优化按钮→退出学习模式，停止记录数据→锁定当前经验池快照→基于已有AI模型和经验池，开始离线优化→离线优化完成→生成新的AI模型→弹窗提示→用户点击确定后自动回滚至学习模式；
学习模式→用户点击窗口上的“识别”按钮→退出学习模式，停止记录数据→激活全屏透明标注层→此时用户可以在电脑屏幕上划分区域或对已有区域进行编辑→保存后，自动回滚至学习模式并重新加载视觉关注点；
学习模式→空格→训练模式→检测到除ESC以外的任意键盘按键信号，切断AI控制权并回退至学习模式；
训练模式→ESC结束运行；

共有2种区域：
红色边框（区域内数字越小越好）
蓝色边框（区域内数字越大越好）

经验池样本权重由以下指标决定：
越新的数据，权重越高；
模型对鼠标的预测偏差越大，权重越高；
包含“点击”、“长按”或“释放”等关键交互事件的样本，权重高于仅有鼠标移动的样本；
用户划分的区域内数值变化幅度越大，权重越高；
鼠标操作复杂度越高，权重越高；
当前帧与经验池内已有历史帧的差异越大，权重越高；

用户划分的区域内的视觉语义被约束为具有时序连续性的非负整数序列，且数值变化具备平滑的梯度特征。

用户已手动完成以下配置：
操作系统：Windows 11
Python版本：3.10.11
CUDA Toolkit：11.8
cuDNN版本：8.9.7 (for CUDA 11.x)
PaddlePaddle-gpu：v2.6.1
NumPy：1.26.4
OpenCV：4.10

经验池上限：10GB
内存上限：16GB
显存上限：4GB

要求：
简体中文界面。
仅用一个无注释的Python脚本实现所有功能。
对CPU占用率、内存占用率、GPU占用率、显存占用率、磁盘读写吞吐量、磁盘队列深度、磁盘响应时间、GPU温度、GPU功耗、风扇转速、推理延迟、数据采集循环耗时、OCR耗时、经验池填充率、截屏帧率、截屏延迟进行毫秒级监控，并建立优雅降级机制：资源负载接近饱和阈值时，采取主动降级措施。

严格约束文件I/O路径，所有运行时产物（AI模型、经验回放池、日志）必须锚定于Desktop\AAA及其子目录，确保数据管理的原子性。

有一个科幻感满满的窗口，实时展示各种信息，包括但不限于：
磁盘读写吞吐量、磁盘队列深度、磁盘响应时间、GPU温度、GPU功耗、风扇转速、推理延迟、数据采集循环耗时、OCR耗时、经验池填充率、截屏帧率、截屏延迟、显示器分辨率、缩放、进度条（百分比，提示文本）

需要避免的报错：

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

Warning: you have set wrong precision for backend:cuda setFloat32Precision call has no effect.Please choose precision from: ieee tf32 none  (function setFloat32Precision)

UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9.

