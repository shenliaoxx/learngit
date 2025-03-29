# 多模态数据采集系统

## 项目概述

多模态数据采集系统是一个集成了肌电信号(EMG)采集和手部动作追踪的实时数据采集平台。该系统通过Myo臂环采集肌电信号，同时利用Intel RealSense深度相机进行手部追踪，实现了多模态生物信号的同步采集、处理和可视化。

系统主要功能包括：
- 实时采集和显示8通道EMG信号
- 实时手部追踪和关节角度计算
- 多模态数据同步和可视化
- 数据记录和导出功能
- 信号滤波和处理
- 数据同步质量分析与可视化
- 动作片段标注工具
- 多模态数据互相关分析

## 系统架构

系统采用模块化设计，主要包含以下组件：

### 数据采集模块
- **Myo采集器**：负责与Myo臂环通信，采集EMG信号
- **RealSense采集器**：负责与Intel RealSense相机通信，采集RGB图像并进行手部追踪

### 数据处理模块
- **EMG滤波器**：对原始EMG信号进行带通和陷波滤波，去除噪声
- **手部角度计算器**：基于MediaPipe手部关键点，计算各关节的屈曲和外展角度
- **数据同步器**：确保不同数据流之间的时间对齐和同步
- **动作检测器**：基于EMG和关节运动特征自动检测动作片段

### 数据分析模块
- **同步质量分析**：评估多模态数据的同步质量和延迟
- **互相关分析**：分析EMG信号与关节角度之间的时间关系
- **动作标注工具**：交互式标注和分析动作片段

### 前端界面
- 基于Flask和HTML/JavaScript的实时数据可视化界面
- 支持多种数据展示模式和交互功能
- 动作标注与分析工具的可视化界面
- 使用Plotly.js实现高性能交互式数据可视化

## 技术栈

- **后端**：Python, Flask
- **前端**：HTML, CSS, JavaScript, Plotly.js, Bootstrap
- **数据采集**：Myo SDK, Intel RealSense SDK
- **计算机视觉**：OpenCV, MediaPipe
- **信号处理**：NumPy, SciPy, h5py
- **数据存储**：HDF5格式

## 安装指南

### 系统要求
- Python 3.7+
- Windows 10/11 或 Linux
- Intel RealSense相机 (推荐D435i或D455)
- Myo臂环

### 依赖安装

```bash
# 克隆仓库
git clone https://github.com/yourusername/multimodal_collection.git
cd multimodal_collection

# 安装依赖
pip install -r requirements.txt

# 安装Myo SDK (Windows)
# 请参考 https://github.com/NiklasRosenstein/myo-python 安装Myo SDK
```

### 硬件连接
1. 将Intel RealSense相机连接到USB 3.0端口
2. 将Myo臂环的蓝牙适配器连接到电脑
3. 佩戴Myo臂环并确保其已开启

## 使用指南

### 启动系统

```bash
python app.py
```

系统启动后，在浏览器中访问 `http://localhost:5000` 打开界面。

### 界面功能

界面分为多个主要区域：

1. **系统控制区**：
   - 开始/停止记录按钮
   - 重置数据按钮
   - 重置Myo连接按钮
   - 系统状态显示

2. **手部追踪区**：
   - RealSense相机实时图像
   - 相机状态和帧率信息

3. **手部关节角度数据区**：
   - 手部骨骼示意图
   - 各手指关节角度数据（屈曲和外展）

4. **EMG数据区**：
   - 8通道EMG信号实时波形
   - 原始/滤波数据切换
   - EMG信号统计信息

### 数据记录

1. 点击"开始记录"按钮开始记录数据
2. 系统会同步记录EMG信号和手部关节角度数据
3. 点击"停止记录"按钮结束记录，系统会自动将数据保存为HDF5文件

### 动作标注工具

访问 `http://localhost:5000/action_annote` 打开动作标注工具：

1. **标注功能**：
   - 交互式标注动作开始和结束点
   - 添加动作标签和备注信息
   - 标注数据可保存为JSON格式

2. **可视化功能**：
   - EMG包络线可视化
   - 关节速度和加速度可视化
   - 缩放和平移功能
   - 高亮显示已标注片段

3. **数据同步分析**：
   - 选择需分析的动作片段
   - 选择EMG和关节通道
   - 互相关分析以评估时间同步质量
   - 可视化延迟和相关性结果

## 系统特点

- **实时性**：低延迟的数据采集和处理
- **多模态**：同步采集EMG信号和手部运动数据
- **可视化**：直观的数据展示和交互界面
- **信号处理**：内置信号滤波和处理算法
- **同步质量**：提供多模态数据同步质量分析工具
- **数据标注**：交互式动作片段标注和分析
- **可扩展**：模块化设计，易于扩展新功能

## 应用场景

- 手势识别研究
- 肌电假肢控制
- 人机交互系统开发
- 康复训练和评估
- 生物信号分析
- 多模态交互系统设计
- 运动机能分析

## 文件结构

```
multimodal_collection/
├── app.py                      # 主应用程序
├── collectors/                 # 数据采集模块
│   ├── __init__.py
│   ├── myo_collector.py        # Myo臂环数据采集
│   ├── realsense_collector.py  # RealSense相机数据采集
│   └── data_collector_enhance.py # 增强数据采集器
├── data_processing/            # 数据处理模块
│   ├── emg_filter.py           # EMG信号滤波
│   ├── hand_angles.py          # 手部关节角度计算
│   ├── hand_angle_enhance.py   # 增强手部角度计算器
│   ├── datasynchronizer.py     # 数据同步器
│   └── action_detector.py      # 动作检测器
├── action_annote/              # 动作标注工具
│   ├── app.py                  # 标注工具后端
│   └── templates/              # 标注工具前端
│       └── index.html          # 标注工具界面
├── static/                     # 静态资源
│   ├── images/                 # 图片资源
│   └── videos/                 # 教程视频资源
├── templates/                  # HTML模板
│   ├── index.html              # 主界面
│   ├── angle_comparison.html   # 角度比较页面
│   └── motion_demo/            # 动作演示系统
├── utils/                      # 工具函数库
│   └── motion_lib.py           # 动作库
├── data/                       # 数据存储目录
└── README.md                   # 项目说明文档
```

## 开发者指南

### 添加新的数据采集设备

1. 在`collectors`目录下创建新的采集器类
2. 实现`start()`, `stop()`, `get_data()`等接口
3. 在`app.py`中集成新的采集器

### 添加新的数据处理算法

1. 在`data_processing`目录下创建新的处理器类
2. 实现相应的数据处理方法
3. 在采集器或应用程序中调用新的处理器

### 使用数据同步器

数据同步器提供了多种方法来同步不同数据流：

```python
from data_processing.datasynchronizer import DataSynchronizer

# 初始化同步器
synchronizer = DataSynchronizer('path/to/your/data.h5')

# 加载数据
synchronizer.load_data()

# 使用线性插值进行同步
synced_joint_data = synchronizer.linear_interpolation_sync()

# 验证同步质量
synchronizer.validate_sync_quality(synced_joint_data)
```

### 使用动作标注工具

1. 启动标注工具：`python action_annote/app.py`
2. 在浏览器中访问标注工具
3. 加载HDF5数据并进行标注
4. 分析EMG和关节角度之间的同步关系

## 同步质量分析

系统提供了多种方法来评估多模态数据的同步质量：

1. **时间对齐误差分析**：计算时间戳的对齐误差
2. **动作区间一致性分析**：评估动作和休息区间的数据特征
3. **互相关分析**：使用信号处理技术计算最优延迟
4. **可视化比较**：直观地比较原始和同步后的数据

分析结果可以帮助研究人员了解数据采集系统的性能，并指导数据处理方法的选择。

## 常见问题

1. **Myo臂环无法连接**
   - 确保Myo臂环已充电并开启
   - 检查蓝牙适配器是否正确连接
   - 尝试点击"重置Myo连接"按钮

2. **RealSense相机无法启动**
   - 确保相机已正确连接到USB 3.0端口
   - 检查是否安装了最新的RealSense SDK
   - 重启应用程序

3. **EMG信号质量差**
   - 确保Myo臂环正确佩戴，电极与皮肤充分接触
   - 尝试调整臂环位置
   - 检查电极是否清洁

4. **数据同步问题**
   - 检查数据采集设备的时间戳是否正确
   - 尝试使用不同的插值方法（线性、三次样条、最近邻）
   - 使用互相关分析确定最佳延迟补偿

5. **标注工具不显示数据**
   - 检查HDF5文件路径是否正确
   - 确认数据结构符合系统要求
   - 检查浏览器控制台是否有错误信息

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系：your.email@example.com 