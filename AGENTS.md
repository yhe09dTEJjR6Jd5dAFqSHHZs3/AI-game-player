这是一个融合了卷积神经网络（CNN）、循环神经网络（RNN）及优先经验回放机制（PER）的视觉驱动型通用游戏智能体。

多模态数据流：毫秒级时序标签、HID鼠标交互遥测（轨迹/点击）、全分辨率屏幕视觉帧、用户定义的感兴趣区域（ROI）配置及基于区域的实时OCR语义解析结果。

学习模式（专家演示捕捉）： 实时采集人类专家的高保真交互轨迹与状态空间数据，将“视觉-动作”对序列化并注入优先经验回放池（PER），为行为克隆（Behavioral Cloning）提供基准样本。
训练模式（闭环推理决策）： 模型基于实时视觉观测流进行端到端推理，输出精细化鼠标动作（含点击、长按及非线性复杂拖拽路径）；系统在执行决策的同时，将自生成样本回流至经验池，实现基于自我博弈（Self-Play）的数据增强。

工作流程：
开始运行→学习模式→ESC结束运行；
学习模式→用户点击窗口上的优化按钮→退出学习模式，停止记录数据→基于已有AI模型和经验池，开始离线优化→离线优化完成→生成新的AI模型→弹窗提示→用户点击确定后，切换到学习模式；
学习模式→用户点击窗口上的“识别”按钮→退出学习模式，停止记录数据，此时用户可以在电脑屏幕上划分区域→用户点击保存后，切换到学习模式；
学习模式→空格→训练模式→检测到除ESC以外的其他键盘按键时，切换到学习模式；
训练模式→ESC结束运行；

共有2种区域：
红色边框（区域内数字越小越好）
蓝色边框（区域内数字越大越好）

AI需实现基于“优先经验回放（Priority Experience Replay）”的数据采样机制，样本权重由以下6个维度的复合特征动态决定：
1. 预测误差 (TD-Error)：模型对该样本的预测Loss越高，权重越高（难例挖掘）；
2. 动作稀疏性 (Action Sparsity)：包含“点击”、“长按”或“释放”等稀疏交互事件的样本，权重显著高于仅有鼠标移动的样本；
3. 业务指标剧变 (Semantic Criticality)：OCR识别区域（红/蓝框）数值发生阶跃性变化的时刻（代表获得奖励或遭受惩罚），视为关键帧，给予极高权重；
4. 操作复杂度 (Trajectory Entropy)：鼠标轨迹的加速度大、曲率高或发生急停急转的复杂操作，权重高于平滑匀速移动；
5. 视觉新颖性 (Visual Novelty)：当前帧与历史缓存帧的像素差异或特征距离较大（新场景/新UI）时，权重增加；
6. 时序新近度 (Recency Bias)：引入时间衰减因子，越新的数据权重越高，以对抗游戏环境的概念漂移。

用户定义的感兴趣区域（ROI）内的视觉语义被约束为具有时序连续性的非负整数序列，且数值变化具备平滑的梯度特征。

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
实时监控CPU、内存、GPU、显存并在接近阈值时采取主动降级措施。
运行过程中生成的所有文件（AI模型、经验池），均位于Desktop\AAA及其子文件夹。

有一个科幻感满满的窗口，实时展示各种信息，包括但不限于：
CPU、内存、GPU、显存占用率（像心电图那样的曲线）、显示器分辨率、缩放、磁盘剩余可用空间、进度条（百分比，提示文本）

需要避免的报错：

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

Warning: you have set wrong precision for backend:cuda setFloat32Precision call has no effect.Please choose precision from: ieee tf32 none  (function setFloat32Precision)

UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9.

