在电脑硬件配置允许的范围内，让操作窗口更有科幻感，让这个AI的智商更高，更像是一个生长在电脑里的人脑。

电脑：
显存：4GB；
内存：16GB；
Windows 11；
磁盘可用空间：2TB；
12th Gen Intel(R) Core(TM) i7-12650H；
NVIDIA GeForce RTX 3050 Ti Laptop GPU；

定义用户在操作窗口上的下拉菜单中选择的窗口为窗口A。

数据：窗口A上的鼠标操作、窗口A画面（仅在窗口A没最小化、没有任何一部分被其他窗口遮挡、没有任何一部分位于电脑屏幕外时记录数据）；

学习模式：满足条件时，记录数据到经验池（鼠标来源：用户）；
训练模式：AI根据窗口A画面，在窗口A上输出鼠标操作，同时在满足条件时记录数据到经验池（鼠标来源：AI）；

工作流程：
运行脚本→学习模式→ESC结束运行；
学习模式→回车→停止记录，开始离线优化→离线优化完成→生成新的AI模型→自动结束运行；
学习模式→空格→训练模式→ESC结束运行；

要求：
仅用一个无注释的Python脚本实现所有功能。
所有超参数根据各种合适条件自适应变化，不固定（像是一个生长在电脑里的人脑）。
运行过程中生成的所有文件（AI模型、经验池），均位于Desktop\AAA及其子文件夹。

通过电脑上的默认浏览器，实现一个科幻感满满的操作窗口，实时显示各种信息，包括但不限于：
①窗口A是否满足截图条件：若否，给出原因；
②显示器分辨率、DPI、磁盘可用空间；
③CPU、内存、GPU、显存占用率曲线；

FutureWarning: `torch.cuda.amp.GradScaler(args...)` is deprecated. Please use `torch.amp.GradScaler('cuda', args...)` instead.

UserWarning: Please use the new API settings to control TF32 behavior, such as torch.backends.cudnn.conv.fp32_precision = 'tf32' or torch.backends.cuda.matmul.fp32_precision = 'ieee'. Old settings, e.g, torch.backends.cuda.matmul.allow_tf32 = True, torch.backends.cudnn.allow_tf32 = True, allowTF32CuDNN() and allowTF32CuBLAS() will be deprecated after Pytorch 2.9.
