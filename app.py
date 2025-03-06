from flask import Flask, render_template, jsonify
<<<<<<< HEAD
from collectors.myo_collector import MyoManager
from collectors.realsense_collector import RealSenseCollector
from threading import Thread
import time
import base64
import cv2

app = Flask(__name__)

# 全局变量
myo_manager = None
=======
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from threading import Lock, Thread
import time
import base64
from hand_angles import HandAngleCalculator

app = Flask(__name__)

class RealSenseCollector:
    def __init__(self):
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        self.calculator = HandAngleCalculator()
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing_styles = mp.solutions.drawing_styles 
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        self.frame = None
        self.hand_data = None
        self.lock = Lock()
        self.is_running = False
        
    def start(self):
        print("启动RealSense相机...")
        self.pipeline.start(self.config)
        self.is_running = True
        print("RealSense相机已启动")
        
    def stop(self):
        self.is_running = False
        self.pipeline.stop()
        self.hands.close()
        print("RealSense相机已停止")
        
    def get_frame(self):
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
            
    def get_hand_data(self):
        with self.lock:
            return self.hand_data.copy() if self.hand_data is not None else None
        
    def process_frame(self):
        try:
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return
                
            color_image = np.asanyarray(color_frame.get_data())
            results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    # 计算关节角度
                    angles = self.calculator.calculate_joint_angles(hand_landmarks)
                    
                    # 调试输出中指MCP外展角
                    if 'middle_mcp_abduction' in angles:
                        print(f"中指MCP外展角: {angles['middle_mcp_abduction']:.2f}°")
                    
                    # 绘制手部关键点和连接线
                    self.mp_drawing.draw_landmarks(
                        image=color_image,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    # # 可视化角度
                    # self.calculator.visualize_angles(color_image, hand_landmarks, angles)

                    with self.lock:
                        self.hand_data = angles
                    
            with self.lock:
                self.frame = color_image
                
        except Exception as e:
            print(f"处理帧错误: {e}")
            import traceback
            print(traceback.format_exc())
            
# 全局变量
>>>>>>> 3711920965b577d3e77a69fa1704c988053377e4
realsense_collector = None

def process_realsense():
    global realsense_collector
<<<<<<< HEAD
    while realsense_collector.is_running:
        try:
            realsense_collector.process_frame()
            time.sleep(0.01)
=======
    print("开始处理RealSense数据流")
    while realsense_collector.is_running:
        try:
            realsense_collector.process_frame()
            time.sleep(0.01)  # 控制帧率
>>>>>>> 3711920965b577d3e77a69fa1704c988053377e4
        except Exception as e:
            print(f"RealSense处理错误: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
<<<<<<< HEAD
    global myo_manager, realsense_collector
    try:
        # 获取EMG数据
        emg_data = myo_manager.get_latest_data() if myo_manager else [0] * 8
        
        # 获取手部数据和相机帧
        hand_data = realsense_collector.get_hand_data() if realsense_collector else {}
        frame = realsense_collector.get_frame()
        
        # 转换相机帧为base64
=======
    global realsense_collector
    try:
        # 获取手部数据
        hand_data = realsense_collector.get_hand_data() if realsense_collector else None
        
        # 获取当前帧并转换为base64
        frame = realsense_collector.get_frame()
>>>>>>> 3711920965b577d3e77a69fa1704c988053377e4
        frame_base64 = ''
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
<<<<<<< HEAD
        return jsonify({
            'status': 'success',
            'emg_data': emg_data,
            'hand_data': hand_data,
            'frame': frame_base64
        })
    except Exception as e:
        print(f"获取数据错误: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
=======
        # 添加调试信息
        print("发送的手部数据:", hand_data)  # 调试输出
        
        response_data = {
            'status': 'success',
            'hand_data': hand_data if hand_data is not None else {},
            'frame': frame_base64,
            'timestamp': time.time()
        }
        
        return jsonify(response_data)
        
    except Exception as e:
        print(f"获取数据错误: {e}")
        import traceback
        print(traceback.format_exc())  # 打印完整错误堆栈
        return jsonify({
            'status': 'error',
            'message': str(e),
            'hand_data': {},
            'frame': ''
>>>>>>> 3711920965b577d3e77a69fa1704c988053377e4
        })

if __name__ == '__main__':
    try:
        # 初始化RealSense
        realsense_collector = RealSenseCollector()
        realsense_collector.start()
        
        # 启动RealSense处理线程
        realsense_thread = Thread(target=process_realsense)
        realsense_thread.daemon = True
        realsense_thread.start()
        
<<<<<<< HEAD
        # 初始化并启动Myo线程
        myo_manager = MyoManager()
        myo_thread = Thread(target=myo_manager.run_collection)
        myo_thread.daemon = True
        myo_thread.start()
        
=======
>>>>>>> 3711920965b577d3e77a69fa1704c988053377e4
        # 启动Flask服务器
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"启动错误: {e}")
    finally:
        if realsense_collector:
            realsense_collector.stop()