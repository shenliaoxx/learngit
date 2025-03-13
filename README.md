# 多模态数据采集系统

## 项目概述

多模态数据采集系统是一个集成了肌电信号(EMG)采集和手部动作追踪的实时数据采集平台。该系统通过Myo臂环采集肌电信号，同时利用Intel RealSense深度相机进行手部追踪，实现了多模态生物信号的同步采集、处理和可视化。

系统主要功能包括：
- 实时采集和显示8通道EMG信号
- 实时手部追踪和关节角度计算
- 多模态数据同步和可视化
- 数据记录和导出功能
- 信号滤波和处理

## 系统架构

系统采用模块化设计，主要包含以下组件：

### 数据采集模块
- **Myo采集器**：负责与Myo臂环通信，采集EMG信号
- **RealSense采集器**：负责与Intel RealSense相机通信，采集RGB图像并进行手部追踪

### 数据处理模块
- **EMG滤波器**：对原始EMG信号进行带通和陷波滤波，去除噪声
- **手部角度计算器**：基于MediaPipe手部关键点，计算各关节的屈曲和外展角度

### 前端界面
- 基于Flask和HTML/JavaScript的实时数据可视化界面
- 支持多种数据展示模式和交互功能

## 技术栈

- **后端**：Python, Flask
- **前端**：HTML, CSS, JavaScript, Plotly.js
- **数据采集**：Myo SDK, Intel RealSense SDK
- **计算机视觉**：OpenCV, MediaPipe
- **信号处理**：NumPy, SciPy

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

界面分为三个主要区域：

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
3. 点击"停止记录"按钮结束记录，系统会自动将数据保存为JSON文件

## 系统特点

- **实时性**：低延迟的数据采集和处理
- **多模态**：同步采集EMG信号和手部运动数据
- **可视化**：直观的数据展示和交互界面
- **信号处理**：内置信号滤波和处理算法
- **可扩展**：模块化设计，易于扩展新功能

## 应用场景

- 手势识别研究
- 肌电假肢控制
- 人机交互系统开发
- 康复训练和评估
- 生物信号分析

## 文件结构

```
multimodal_collection/
├── app.py                  # 主应用程序
├── collectors/             # 数据采集模块
│   ├── __init__.py
│   ├── myo_collector.py    # Myo臂环数据采集
│   └── realsense_collector.py  # RealSense相机数据采集
├── data_processing/        # 数据处理模块
│   ├── emg_filter.py       # EMG信号滤波
│   └── hand_angles.py      # 手部关节角度计算
├── static/                 # 静态资源
│   └── images/             # 图片资源
│       ├── hand_skeleton.png
│       ├── myo_armband.png
│       └── realsense_camera.png
├── templates/              # HTML模板
│   └── index.html          # 主界面
└── README.md               # 项目说明文档
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

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系：your.email@example.com 