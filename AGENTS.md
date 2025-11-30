这是一个通用游戏AI。

根据毫秒级时间戳对齐数据：鼠标（特征标签/位置/运动方向/速度/预测偏差/复杂度）、全分辨率屏幕视觉帧、用户划分的区域配置、各区域OCR识别结果（需避免DPI缩放导致的坐标错位，未检测到数字时视为None）、各区域数值变化幅度。

学习模式：满足条件时，记录数据到经验池（鼠标来源：用户）
训练模式：AI根据实时数据，在电脑屏幕上输出鼠标操作（点击/长按/包含复杂路径的拖动），同时记录数据到经验池（鼠标来源：AI）

工作流程：
开始运行→初始化→一切就绪→学习模式→ESC结束运行；
学习模式→空格→训练模式→检测到除ESC以外的任意键盘按键信号→切断AI控制权→切换至学习模式；
学习模式→用户点击窗口上的优化按钮→停止记录数据→退出学习模式→锁定经验池→挑选经验池中质量最高的一批样本+最新的AI模型，开始离线优化→离线优化完成→生成新的AI模型→弹窗提示→用户点击确定→切换至学习模式；
学习模式→用户点击窗口上的“识别”按钮→停止记录数据→退出学习模式→进入编辑模式→激活全屏透明标注层→此时用户可以在电脑屏幕上划分区域或对已有区域进行编辑（也可以什么都不干）→回车→保存区域配置→退出编辑模式→切换至学习模式→用户划分的区域继续显示在电脑屏幕上；
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

用户已手动完成以下配置：
操作系统：Windows 11
Python版本：3.10.11
CUDA Toolkit：11.8
cuDNN版本：8.9.7 (for CUDA 11.x)
NumPy：1.26.4
OpenCV：4.10

经验池上限：10GB
内存上限：16GB
显存上限：4GB

直接使用调整大小后的原图进行OCR识别，不要进行可能破坏图像特征的锐化/二值化处理。
严禁对OCR数值添加防突变功能（实现“所见即所得”的瞬时响应）。

简体中文界面。

使用RapidOCR。

仅用一个无注释、无硬编码的Python脚本实现所有功能。

对CPU占用率、内存占用率、GPU占用率、显存占用率进行监控（1Hz），并建立自适应机制：
1. 资源负载接近饱和阈值时，采取降级措施（包括但不限于：降低截图频率，减小batch size）；
2. 资源负载远离饱和阈值时，采取升级措施（包括但不限于：提高截图频率，增大batch size）；

严格约束文件I/O路径，所有运行时产物（AI模型、经验回放池、日志）必须锚定于Desktop\AAA及其子目录，确保数据管理的原子性。

有一个科幻感满满的窗口，展示以下信息（1Hz）：CPU占用率、内存占用率、GPU占用率、显存占用率、截屏帧率、显示器分辨率、缩放、进度条（百分比，提示文本）

需要避免的报错：

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

DeprecationWarning: The parameter use_angle_cls has been deprecated and will be removed in the future. Please use use_textline_orientation instead.

Warning: you have set wrong precision for backend:cuda setFloat32Precision call has no effect.Please choose precision from: ieee tf32 none  (function setFloat32Precision)

FutureWarning: The pynvml package is deprecated. Please install nvidia-ml-py instead. If you did not install pynvml directly, please report this to the maintainers of the package that installed pynvml for you.

UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9.

