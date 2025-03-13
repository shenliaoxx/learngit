import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from threading import Lock
from data_processing.hand_angles import HandAngleCalculator
import time

class RealSenseCollector:
    def __init__(self):
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
        # 初始化MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,  # 降低模型复杂度以提高速度
            min_detection_confidence=0.7,  # 降低检测置信度以提高检测率
            min_tracking_confidence=0.7    # 降低跟踪置信度以提高跟踪率
        )
        
        # 初始化角度计算器
        self.calculator = HandAngleCalculator()
        
        self.frame = None
        self.hand_data = None
        self.lock = Lock()
        self.is_running = False
        
        # 添加处理性能统计
        self.process_times = []
        self.max_process_times = 100
        self.last_process_time = 0
        self.avg_process_time = 0

        # 帧率计算相关变量
        self.frame_count = 0
        self.total_frames = 0
        self.current_fps = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # 每秒更新一次帧率
        
        # 手部检测统计
        self.hand_detected_count = 0
        self.hand_detection_rate = 0
        
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
        
    def get_camera_stats(self):
        """获取相机统计信息"""
        with self.lock:
            return {
                'camera_fps': round(self.current_fps, 1),
                'camera_total_frames': self.total_frames,
                'avg_process_time': round(self.avg_process_time * 1000, 2),  # 毫秒
                'hand_detection_rate': round(self.hand_detection_rate * 100, 1)  # 百分比
            } 
               
    def process_frame(self):
        try:
            process_start = time.perf_counter()
            
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                return

            # 更新帧率统计
            current_time = time.time()
            self.frame_count += 1
            self.total_frames += 1
            
            # 每秒更新一次帧率
            if current_time - self.last_fps_update >= self.fps_update_interval:
                self.current_fps = self.frame_count / (current_time - self.last_fps_update)
                self.frame_count = 0
                self.last_fps_update = current_time
                
                # 更新手部检测率
                if self.total_frames > 0:
                    self.hand_detection_rate = self.hand_detected_count / self.total_frames

            color_image = np.asanyarray(color_frame.get_data())
            
            # 降低分辨率以提高处理速度
            # scale_percent = 75  # 缩放到原来的75%
            # width = int(color_image.shape[1] * scale_percent / 100)
            # height = int(color_image.shape[0] * scale_percent / 100)
            # resized_image = cv2.resize(color_image, (width, height))
            
            # 使用原始分辨率进行处理
            results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
            hand_detected = False
            if results.multi_hand_landmarks:
                hand_detected = True
                self.hand_detected_count += 1
                
                for hand_landmarks in results.multi_hand_landmarks:
                    # 绘制手部关键点
                    self.mp_drawing.draw_landmarks(
                        color_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS)
                    
                    # 计算并绘制手部坐标系
                    origin, rotation_matrix = self.calculator.create_hand_coordinate_system(hand_landmarks)
                    self.calculator.draw_hand_coordinate_system(color_image, origin, rotation_matrix)
                    
                    # 计算关节角度
                    angles = self.calculator.calculate_joint_angles(hand_landmarks)
                    
                    # 绘制手部关键点和连接线
                    self.mp_drawing.draw_landmarks(
                        image=color_image,
                        landmark_list=hand_landmarks,
                        connections=self.mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
                    )
                    
                    with self.lock:
                        self.hand_data = angles
            else:
                with self.lock:
                    # 保持上一帧的手部数据，避免数据丢失
                    # self.hand_data = None
                    pass

            with self.lock:
                self.frame = color_image
                
            # 计算处理时间
            process_end = time.perf_counter()
            process_time = process_end - process_start
            
            # 更新平均处理时间
            self.process_times.append(process_time)
            if len(self.process_times) > self.max_process_times:
                self.process_times.pop(0)
            
            self.avg_process_time = sum(self.process_times) / len(self.process_times)
            
            # 定期打印处理性能
            if self.total_frames % 100 == 0:
                print(f"相机处理性能 - 帧率: {self.current_fps:.1f} FPS | "
                      f"处理时间: {self.avg_process_time*1000:.2f} ms | "
                      f"手部检测率: {self.hand_detection_rate*100:.1f}%")
                
        except Exception as e:
            print(f"处理帧错误: {e}")
            import traceback
            print(traceback.format_exc())