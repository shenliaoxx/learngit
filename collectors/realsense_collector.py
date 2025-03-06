# import cv2
# import mediapipe as mp
# import pyrealsense2 as rs
# import numpy as np
# from threading import Lock

# class RealSenseCollector:
#     def __init__(self):
#         # 初始化RealSense
#         self.pipeline = rs.pipeline()
#         self.config = rs.config()
#         self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        
#         # 初始化MediaPipe
#         self.mp_hands = mp.solutions.hands
#         self.mp_drawing = mp.solutions.drawing_utils
#         self.hands = self.mp_hands.Hands(
#             static_image_mode=False,
#             max_num_hands=2,
#             min_detection_confidence=0.7,
#             min_tracking_confidence=0.5
#         )
        
#         self.frame = None
#         self.hand_data = None
#         self.lock = Lock()
#         self.is_running = False
        
#     def start(self):
#         print("启动RealSense相机...")
#         self.pipeline.start(self.config)
#         self.is_running = True
#         print("RealSense相机已启动")
        
#     def stop(self):
#         self.is_running = False
#         self.pipeline.stop()
#         self.hands.close()
#         print("RealSense相机已停止")
        
#     def get_frame(self):
#         with self.lock:
#             return self.frame.copy() if self.frame is not None else None
            
#     def get_hand_data(self):
#         with self.lock:
#             return self.hand_data.copy() if self.hand_data is not None else None
        
#     def process_frame(self):
#         try:
#             frames = self.pipeline.wait_for_frames()
#             color_frame = frames.get_color_frame()
#             if not color_frame:
#                 return
                
#             # 转换为numpy数组
#             color_image = np.asanyarray(color_frame.get_data())
            
#             # MediaPipe处理
#             results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
#             # 绘制手部关键点
#             if results.multi_hand_landmarks:
#                 for hand_landmarks in results.multi_hand_landmarks:
#                     self.mp_drawing.draw_landmarks(
#                         color_image,
#                         hand_landmarks,
#                         self.mp_hands.HAND_CONNECTIONS
#                     )
                    
#                 # 提取手部关节角度数据
#                 hand_angles = self.calculate_hand_angles(results.multi_hand_landmarks[0])
#                 with self.lock:
#                     self.hand_data = hand_angles
                    
#             with self.lock:
#                 self.frame = color_image
                
#         except Exception as e:
#             print(f"处理帧错误: {e}")
            
#     def calculate_hand_angles(self, hand_landmarks):
#         """计算手部20个关节的角度"""
#         angles = {}
        
#         # 定义手指关节链
#         finger_chains = {
#             'thumb': [0, 1, 2, 3, 4],           # 拇指
#             'index': [0, 5, 6, 7, 8],           # 食指
#             'middle': [0, 9, 10, 11, 12],       # 中指
#             'ring': [0, 13, 14, 15, 16],        # 无名指
#             'pinky': [0, 17, 18, 19, 20]        # 小指
#         }
        
#         try:
#             # 将landmarks转换为numpy数组，便于计算
#             points = []
#             for landmark in hand_landmarks.landmark:
#                 points.append(np.array([landmark.x, landmark.y, landmark.z]))
#             points = np.array(points)
            
#             # 计算每个手指的关节角度
#             for finger, chain in finger_chains.items():
#                 # MCP关节 (掌指关节)
#                 mcp_angle = self.calculate_angle(
#                     points[chain[0]],  # 手掌中心
#                     points[chain[1]],  # MCP关节
#                     points[chain[2]]   # PIP关节
#                 )
#                 angles[f'{finger}_mcp'] = round(mcp_angle, 2)
                
#                 # PIP关节 (近节指关节)
#                 pip_angle = self.calculate_angle(
#                     points[chain[1]],  # MCP关节
#                     points[chain[2]],  # PIP关节
#                     points[chain[3]]   # DIP关节
#                 )
#                 angles[f'{finger}_pip'] = round(pip_angle, 2)
                
#                 # DIP关节 (远节指关节)
#                 dip_angle = self.calculate_angle(
#                     points[chain[2]],  # PIP关节
#                     points[chain[3]],  # DIP关节
#                     points[chain[4]]   # 指尖
#                 )
#                 angles[f'{finger}_dip'] = round(dip_angle, 2)
                
#                 # 对于拇指，额外计算CMC关节角度
#                 if finger == 'thumb':
#                     cmc_angle = self.calculate_angle(
#                         points[0],     # 手掌中心
#                         points[1],     # CMC关节
#                         points[2]      # MCP关节
#                     )
#                     angles['thumb_cmc'] = round(cmc_angle, 2)
            
#             # 计算手腕角度
#             wrist_angle = self.calculate_wrist_angle(points)
#             angles['wrist'] = round(wrist_angle, 2)
            
#         except Exception as e:
#             print(f"计算手部角度错误: {e}")
#             return {}
            
#         return angles
        
#     def calculate_angle(self, p1, p2, p3):
#         """计算三个点形成的角度"""
#         try:
#             v1 = p1 - p2
#             v2 = p3 - p2
#             cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
#             # 处理数值误差
#             cos_angle = np.clip(cos_angle, -1.0, 1.0)
#             angle = np.degrees(np.arccos(cos_angle))
#             return angle
#         except Exception as e:
#             print(f"计算角度错误: {e}")
#             return 0
            
#     def calculate_wrist_angle(self, points):
#         """计算手腕角度"""
#         try:
#             # 使用手掌中心和食指、小指的MCP关节来计算手腕角度
#             palm_center = points[0]
#             index_mcp = points[5]
#             pinky_mcp = points[17]
            
#             # 计算手掌平面的法向量
#             v1 = index_mcp - palm_center
#             v2 = pinky_mcp - palm_center
#             normal = np.cross(v1, v2)
            
#             # 计算与垂直方向的夹角
#             vertical = np.array([0, 0, 1])
#             angle = np.degrees(np.arccos(np.dot(normal, vertical) / np.linalg.norm(normal)))
#             return angle
#         except Exception as e:
#             print(f"计算手腕角度错误: {e}")
#             return 0


import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from threading import Lock
from hand_angles import HandAngleCalculator

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
            else:
                with self.lock:
                    self.hand_data = None
                    
            with self.lock:
                self.frame = color_image
                
        except Exception as e:
            print(f"处理帧错误: {e}")
            import traceback
            print(traceback.format_exc())