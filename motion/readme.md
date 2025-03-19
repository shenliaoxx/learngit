以下是基于所提供文档为多模态数据采集系统项目撰写的中文版 `README.md` 文件，内容包括项目概述、功能、安装指南、使用说明、文件结构以及其他相关信息，适合开发者和用户参考。

---

# 多模态动作演示系统

## 项目概述

**多模态动作演示系统** 是一个基于Web的平台，旨在演示和采集与手部及腕部动作相关的多模态数据。该系统集成了预定义的动作序列，并提供用户友好的界面，用于动作视频播放、演示以及数据采集准备。系统支持动作视频、关键点和采集进度的实时可视化，非常适合用于手势识别研究、肌电假肢控制、人机交互系统开发以及康复训练等场景。

该项目结合了 Flask 后端与动态前端，使用模块化的动作库来管理动作序列和采集工作流程，注重扩展性和易用性。

## 主要功能

- **动作库管理**：存储和管理预定义的动作序列（如“捏取”、“抓握”、“屈腕”、“伸腕”），包括动作描述、关键点和视频资源。
- **动作序列演示**：通过视频播放和关键点展示，指导用户完成动作序列。
- **数据采集准备**：支持动作的重复采集，包含准备时间、采集时间和休息时间的配置。
- **实时界面**：基于HTML和JavaScript的交互式界面，显示动作信息、序列进度和采集状态。
- **状态管理**：实时跟踪采集会话状态（如准备中、采集中、休息中、已完成）。
- **响应式设计**：适配不同设备，提供良好的用户体验。

## 系统要求

- **操作系统**：Windows 10/11 或 Linux
- **Python版本**：3.7 或更高
- **依赖工具**：
  - Flask >= 2.0.0
  - 其他依赖详见 `requirements.txt`

## 安装指南

### 1. 克隆仓库
```bash
git clone https://github.com/yourusername/multimodal-motion-demo.git
cd multimodal-motion-demo
```

### 2. 创建虚拟环境（可选但推荐）
```bash
python -m venv venv
source venv/bin/activate  # Linux
venv\Scripts\activate     # Windows
```

### 3. 安装依赖
```bash
pip install -r requirements.txt
```

### 4. 配置环境
- 确保视频资源路径（如 `videos/web_videos/`）正确，且文件存在。
- 可选：修改 `app.py` 中的 `app.secret_key` 为自定义密钥。

## 使用指南

### 启动系统
```bash
python app.py
```
启动后，在浏览器中访问 `http://localhost:5000` 打开界面。

### 界面功能
1. **动作序列选择**：
   - 在顶部下拉菜单中选择动作序列（如“基础动作”）。
   - 系统将加载序列中的第一个动作。

2. **视频演示**：
   - 点击“播放”按钮查看动作视频。
   - 使用“上一个”和“下一个”按钮切换动作。
   - 查看动作名称、描述和关键点。

3. **采集流程**：
   - 点击“准备采集”按钮进入准备状态，显示倒计时。
   - 点击“开始采集”启动数据采集，界面显示采集进度。
   - 点击“停止采集”暂停当前轮次，进入休息状态或完成采集。
   - 点击“完成采集”结束整个会话。

4. **状态监控**：
   - 底部状态栏显示当前操作状态。
   - 采集进度区域显示当前轮次和总轮次。

### 数据采集说明
- 每次采集会生成一个会话ID，记录动作名称、重复次数和数据文件。
- 数据文件以 `.h5` 格式保存（需自行实现保存逻辑，当前仅生成文件名）。

## 文件结构

```
multimodal-motion-demo/
├── motion_lib.py           # 动作库定义和管理
├── app.py                  # Flask主应用程序
├── static/                 # 静态资源
│   ├── css/                # 样式表
│   │   └── style.css       # 主样式文件
│   ├── js/                 # JavaScript文件
│   │   └── script.js       # 前端逻辑
│   └── videos/             # 视频资源目录（需手动添加）
│       └── web_videos/     # 示例视频存放路径
├── templates/              # HTML模板
│   └── index.html          # 主界面模板
└── requirements.txt        # 项目依赖
```

## 技术栈

- **后端**：Python, Flask
- **前端**：HTML, CSS, JavaScript
- **样式**：自定义CSS，Font Awesome图标，Noto Sans SC字体
- **依赖管理**：pip, requirements.txt

## 开发者指南

### 添加新动作
1. 在 `motion_lib.py` 的 `MotionLibrary` 类中添加新动作到 `self.motions` 字典。
   - 示例：
     ```python
     "new_motion": {
         "id": "new_motion",
         "name": "新动作",
         "description": "动作描述",
         "duration": 5,
         "video": "videos/web_videos/new_motion.mp4",
         "key_points": ["关键点1", "关键点2"],
         "collection": {
             "repeat_times": 5,
             "duration": 3,
             "rest_time": 2,
             "preparation_time": 3
         }
     }
     ```
2. 将对应的视频文件放入 `static/videos/web_videos/` 目录。

### 扩展动作序列
1. 在 `motion_lib.py` 的 `self.sequences` 中添加新序列。
   - 示例：
     ```python
     "新序列": ["pinch", "new_motion"]
     ```

### 集成数据采集
1. 在 `app.py` 的 `/api/collection/start` 和 `/api/collection/stop` 路由中添加实际的传感器数据采集逻辑。
2. 使用 `h5py` 将采集数据保存为 `.h5` 文件。

### 修改前端
1. 编辑 `templates/index.html` 添加新UI元素。
2. 在 `static/js/script.js` 中实现对应的JavaScript逻辑。
3. 更新 `static/css/style.css` 调整样式。

## 常见问题

1. **视频无法播放**
   - 确保视频路径正确且文件存在。
   - 检查浏览器是否支持MP4格式。

2. **采集按钮不可用**
   - 确保已选择动作序列并加载了动作。
   - 检查会话状态是否正确。

3. **界面样式异常**
   - 清除浏览器缓存或检查CSS文件是否加载。

## 性能优化

- 使用异步请求（`fetch`）加载动作序列，减少页面阻塞。
- 通过CSS动画（如`pulse`）提升用户体验。
- 动态调整倒计时逻辑，避免频繁DOM操作。

## 许可证

[MIT License](LICENSE)

## 联系方式

如有问题或建议，请联系：your.email@example.com

---

此 `README.md` 文件提供了项目的全面介绍，涵盖安装、使用和开发扩展的详细信息，适合中文用户和开发者使用。你可以根据实际需求调整内容，例如添加具体的视频文件路径或补充数据采集的具体实现方法。