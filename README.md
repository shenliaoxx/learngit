# 多模态数据采集系统

一个用于实时采集、处理和可视化多模态生物信号数据的综合平台，特别专注于肌电图(EMG)和手部运动数据的同步获取与分析。

## 项目概述

本系统整合了肌电信号采集和手部运动跟踪技术，使用Myo手环采集肌电信号，同时通过Intel RealSense深度摄像头跟踪手部运动。系统支持实时数据可视化、数据同步采集、信号质量评估和定制化动作练习的功能。

### 核心功能

- **多模态数据同步采集**：同步采集来自Myo手环的EMG信号和来自RealSense摄像头的手部运动数据
- **实时可视化**：直观展示手部骨架、关节角度和肌电信号
- **深度信息增强**：使用深度相机提高手部关键点的三维空间精度
- **性能优化**：支持GPU加速和多线程处理，提高系统响应速度
- **数据分析**：提供角度比较、信号质量评估等分析工具
- **结构化数据存储**：使用HDF5格式高效存储多模态数据

## 技术栈

- **后端**：Python, Flask
- **前端**：HTML, CSS, JavaScript, Plotly.js
- **数据处理**：NumPy, SciPy, OpenCV, MediaPipe
- **硬件接口**：
  - Myo手环API (MyoBridge)
  - Intel RealSense SDK (pyrealsense2)
- **存储**：HDF5
- **并行处理**：ThreadPoolExecutor

## 系统架构

系统分为四个主要模块：

1. **数据采集模块**：
   - `collectors/myo_collector.py` - 肌电信号采集
   - `collectors/realsense_collector_enhance.py` - 手部运动数据采集

2. **数据处理模块**：
   - `data_processing/hand_angle_enhance.py` - 手部关节角度计算
   - `data_processing/emg_filter.py` - 肌电信号滤波
   - `data_processing/datasynchronizer.py` - 多模态数据同步

3. **数据存储模块**：
   - `collectors/data_collector_enhance.py` - 优化的HDF5数据存储

4. **Web界面**：
   - `templates/` - 前端界面模板
   - `app.py` - Flask应用程序和API路由

## 安装说明

### 前提条件

# 基础库
numpy==1.19.5
scipy==1.5.4
h5py==3.1.0
opencv-python==4.5.1.48
mediapipe==0.8.5

# TensorFlow GPU 版本
tensorflow==2.6.0

# Flask 及其相关库
Flask==2.0.1
Flask-Cors==3.0.10

# 其他可能需要的库
plotly==5.1.0

### 步骤

1. 克隆仓库
```bash
git clone https://github.com/yourusername/multimodel-acquisition.git
cd multimodel-acquisition
```

2. 创建并激活虚拟环境
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. 安装依赖
```bash
pip install -r requirements.txt
```

4. 安装特定版本的依赖(如果需要GPU加速)
```bash
pip install tensorflow==2.6.0
```

5. 安装Intel RealSense SDK和Myo SDK
查看官方文档获取详细安装步骤：
- [Intel RealSense SDK](https://github.com/IntelRealSense/librealsense)
- [Myo SDK](https://developer.thalmic.com/downloads)

## 使用指南

1. 连接设备
   - 确保Myo手环已连接并同步
   - 确保RealSense摄像头已正确连接

2. 启动应用
```bash
python app.py
```

3. 访问Web界面
   打开浏览器并访问 http://localhost:5000

4. 主要功能
   - **主页**：显示实时手部跟踪和EMG数据
   - **角度比较**：分析原始和滤波后的角度数据
   - **动作演示**：学习和练习预定义动作

## 数据格式

系统使用HDF5格式存储数据，包含以下主要组：

- **emg**：肌电信号数据
  - timestamps：时间戳
  - raw_data：原始肌电数据
  - filtered_data：滤波后的肌电数据

- **hand**：手部运动数据
  - timestamps：时间戳
  - joint_angles：关节角度数据
  - angle_labels：关节角度标签

- **metadata**：会话元数据
  - subject_id：受试者ID
  - session_id：会话ID
  - recording_date：记录日期
  - 等其他元数据

## 性能优化

- 使用多线程并行计算手部关节角度
- 支持MediaPipe GPU加速
- 实现自适应平滑算法处理手部关键点
- 优化HDF5文件存储，支持压缩和分块
- 深度数据批处理和缓存

## 未来计划

- 添加机器学习模型进行手势识别
- 改进多模态数据同步算法
- 扩展支持更多生物信号设备
- 开发离线分析工具

## 贡献指南

欢迎提交问题和功能请求。如需贡献代码，请遵循以下步骤：

1. Fork仓库
2. 创建功能分支(`git checkout -b feature/amazing-feature`)
3. 提交更改(`git commit -m 'Add some amazing feature'`)
4. 推送到分支(`git push origin feature/amazing-feature`)
5. 开启Pull Request

## 许可证

[MIT License](LICENSE)

## 联系方式

项目维护者 - 您的名字 - 您的邮箱

项目链接：[https://github.com/yourusername/multimodel-acquisition](https://github.com/yourusername/multimodel-acquisition) 