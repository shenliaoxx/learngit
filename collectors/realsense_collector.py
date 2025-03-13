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
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 初始化角度计算器
        self.calculator = HandAngleCalculator()
        
        self.frame = None
        self.hand_data = None
        self.lock = Lock()
        self.is_running = False

        # 帧率计算相关变量
        self.frame_count = 0
        self.total_frames = 0
        self.current_fps = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 1.0  # 每秒更新一次帧率
        
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
                'camera_total_frames': self.total_frames
            } 
               
    def process_frame(self):
        try:
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

            color_image = np.asanyarray(color_frame.get_data())
            results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            

            
            if results.multi_hand_landmarks:
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
                    self.hand_data = None

                               
            with self.lock:
                self.frame = color_image
                
        except Exception as e:
            print(f"处理帧错误: {e}")
            import traceback
            print(traceback.format_exc())