import cv2
import mediapipe as mp
import pyrealsense2 as rs
import numpy as np
from threading import Lock
from data_processing.hand_angle_enhance import HandAngleCalculator
import time
from collections import deque
import copy

class RealSenseCollector:
    def __init__(self, high_performance=True):
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()

        # 高性能模式标志
        self.high_performance = high_performance
        
        # 根据模式设置不同的帧率
        fps = 60 if high_performance else 30

        #启用深度流
        self.config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, fps)
        self.config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, fps)

        # 创建对齐对象
        self.align = rs.align(rs.stream.color)


        self._init_hand_tracking()
        
        # 初始化角度计算器
        self.calculator = HandAngleCalculator()
        self._init_depth_processing()

        # 平滑参数配置
        self.smooth_config = {
            'base_window_size': 5,
            'min_window_size': 3,
            'max_window_size': 7,
            'velocity_threshold': {
                'static': 0.01,
                'slow': 0.05,
                'fast': 0.1
            }
        }
        self.landmark_history = {}
        
        # 存储相机内参
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
        
    def _init_depth_processing(self) -> None:
        """初始化深度处理相关设置，添加高性能配置"""
        # 创建空间滤波器对象 - 使用更轻量级的滤波设置
        self.spatial_filter = rs.spatial_filter()
        self.spatial_filter.set_option(rs.option.filter_magnitude, 1)
        self.spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.25)
        self.spatial_filter.set_option(rs.option.filter_smooth_delta, 10)
        
        # 创建时间滤波器对象
        self.temporal_filter = rs.temporal_filter()
        
        # 创建孔洞填充滤波器
        self.hole_filling_filter = rs.hole_filling_filter()
        
        # 高性能模式标志
        self.high_performance = True

    def _update_fps(self, current_time: float) -> None:
        """更新帧率统计"""
        self.frame_count += 1
        self.total_frames += 1
        
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time

    def _init_hand_tracking(self) -> None:
        """初始化MediaPipe手部追踪"""
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
              
    def start(self) -> None:
        """启动相机采集，优化相机参数"""
        print("启动RealSense相机...")
        
        # 启动相机
        profile = self.pipeline.start(self.config)
        
        # 设置相机参数以提高帧率
        try:
            # 获取设备和深度传感器
            device = profile.get_device()
            depth_sensor = device.first_depth_sensor()
            
            # 设置低延迟模式和自动曝光模式
            if depth_sensor.supports(rs.option.enable_auto_exposure):
                depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
                print("已启用自动曝光模式")
            
   
            if depth_sensor.supports(rs.option.enable_motion_correction):
                depth_sensor.set_option(rs.option.enable_motion_correction, 1)
                print("已启用运动校正")

            
            # 如果支持激光器功能，优化激光功率
            if depth_sensor.supports(rs.option.laser_power):
                # 设置激光功率为中等水平，平衡性能与准确性
                max_power = depth_sensor.get_option_range(rs.option.laser_power).max
                depth_sensor.set_option(rs.option.laser_power, max_power * 0.7)
                print(f"已优化激光功率: {max_power * 0.7:.1f}")

        except Exception as e:
            print(f"相机参数优化失败: {e}")
        
        self.is_running = True
        print("RealSense相机已启动")

    def stop(self) -> None:
        """停止相机采集"""
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
            } 
        
    def _process_hand_landmarks(self, image, landmarks): 
        """处理手部关键点，计算角度并绘制可视化"""

        # 计算并绘制手部坐标系
        origin, rotation_matrix = self.calculator.create_hand_coordinate_system(landmarks)

        self.calculator.draw_hand_coordinate_system(image, origin, rotation_matrix)
        
        # 计算关节角度
        angles = self.calculator.parallel_calculate_joint_angles(landmarks)
        
        # 绘制手部关键点和连接线
        self.mp_drawing.draw_landmarks(
            image=image,
            landmark_list=landmarks,
            connections=self.mp_hands.HAND_CONNECTIONS,
            landmark_drawing_spec=self.mp_drawing_styles.get_default_hand_landmarks_style(),
            connection_drawing_spec=self.mp_drawing_styles.get_default_hand_connections_style()
        )
        
        cv2.putText(image, "Depth Enhanced", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return angles
    

    
    
    def _enhance_landmarks_with_depth(self, landmarks, depth_frame, color_image):
        """使用深度信息增强手部关键点的空间位置 - 优化版"""
        smoothed_landmarks = self._smooth_landmarks(landmarks)

        # 如果提供了深度数据，则使用深度增强
        if depth_frame is not None and color_image is not None:
            depth_image = np.asanyarray(depth_frame.get_data())
            enhanced_landmarks = self._enhance_landmarks_with_depth_array(smoothed_landmarks, depth_image, color_image)
        else:
            enhanced_landmarks = smoothed_landmarks

        
        return enhanced_landmarks
    
    def _smooth_landmarks(self, landmarks):
        """改进的自适应平滑处理"""
        smoothed_landmarks = copy.deepcopy(landmarks)
        
        for i, landmark in enumerate(landmarks.landmark):
            # 初始化历史记录
            if i not in self.landmark_history:
                self.landmark_history[i] = {
                    'positions': deque(maxlen=self.smooth_config['max_window_size']),
                    'velocities': deque(maxlen=3)
                }
            
            # 当前位置
            current_pos = np.array([landmark.x, landmark.y, landmark.z])
            
            # 计算速度
            if len(self.landmark_history[i]['positions']) > 0:
                prev_pos = self.landmark_history[i]['positions'][-1]
                velocity = np.linalg.norm(current_pos - prev_pos)
                self.landmark_history[i]['velocities'].append(velocity)
            
            # 添加当前位置到历史记录
            self.landmark_history[i]['positions'].append(current_pos)
            
            # 根据速度调整平滑参数
            if len(self.landmark_history[i]['positions']) >= 3:
                # 计算平均速度
                avg_velocity = np.mean(list(self.landmark_history[i]['velocities'])) if \
                    self.landmark_history[i]['velocities'] else 0
                
                # 确定运动状态
                if avg_velocity < self.smooth_config['velocity_threshold']['static']:
                    window_size = self.smooth_config['max_window_size']
                    smooth_factor = 0.8
                elif avg_velocity < self.smooth_config['velocity_threshold']['slow']:
                    window_size = self.smooth_config['base_window_size']
                    smooth_factor = 0.6
                else:
                    window_size = self.smooth_config['min_window_size']
                    smooth_factor = 0.4
                
                # 生成自适应权重
                positions_list = list(self.landmark_history[i]['positions'])
                if len(positions_list) > window_size:
                    positions_list = positions_list[-window_size:]
                
                weights = np.exp(np.linspace(-2 * smooth_factor, 0, len(positions_list)))
                weights /= weights.sum()
                
                # 计算加权平均
                smoothed_pos = np.average(positions_list, weights=weights, axis=0)
                
                # 更新平滑后的坐标
                smoothed_landmarks.landmark[i].x = smoothed_pos[0]
                smoothed_landmarks.landmark[i].y = smoothed_pos[1]
                smoothed_landmarks.landmark[i].z = smoothed_pos[2]
        
        return smoothed_landmarks

    def _enhance_landmarks_with_depth_array(self, landmarks, depth_image, color_image):
        """使用深度图像数组增强手部关键点的空间位置"""
        enhanced_landmarks = copy.deepcopy(landmarks)
        h, w = color_image.shape[:2]
        depth_h, depth_w = depth_image.shape[:2]
        
        # 调整尺度，确保像素坐标能对应
        scale_x = depth_w / w if w != depth_w else 1.0
        scale_y = depth_h / h if h != depth_h else 1.0
        
        # 一次性获取所有关键点的像素坐标
        landmark_pixels = np.array([[int(lm.x * w), int(lm.y * h)] 
                                  for lm in landmarks.landmark])
        
        # 批量检查像素坐标的有效性
        valid_mask = (landmark_pixels[:, 0] >= 0) & (landmark_pixels[:, 0] < w) & \
                    (landmark_pixels[:, 1] >= 0) & (landmark_pixels[:, 1] < h)
        
        # 获取手腕深度（关键点0）作为参考
        wrist_px, wrist_py = landmark_pixels[0]
        depth_px = min(int(wrist_px * scale_x), depth_w - 1)
        depth_py = min(int(wrist_py * scale_y), depth_h - 1)
        
        # 获取深度值 (毫米)，转换为米
        wrist_depth = depth_image[depth_py, depth_px] * 0.001  # 通常深度值是毫米
        if wrist_depth <= 0 or wrist_depth > 3.0:  # 3米是一个合理的最大距离
            wrist_depth = self._get_surrounding_depth_array(depth_image, depth_px, depth_py)
        
        # 处理每个有效的关键点
        for i, (px, py) in enumerate(landmark_pixels):
            if not valid_mask[i]:
                continue
            
            depth_px = min(int(px * scale_x), depth_w - 1)
            depth_py = min(int(py * scale_y), depth_h - 1)
            
            # 获取深度值
            depth = depth_image[depth_py, depth_px] * 0.001  # 转换为米
            if depth <= 0 or depth > 3.0:
                depth = self._get_surrounding_depth_array(depth_image, depth_px, depth_py)
            
            if depth > 0 and wrist_depth > 0:
                # 计算相对深度
                relative_depth = depth - wrist_depth
                
                # 使用固定缩放因子
                z_scale_factor = 0.5
                
                # 更新z坐标
                enhanced_landmarks.landmark[i].z += relative_depth * z_scale_factor
                
                # 更新可见度
                enhanced_landmarks.landmark[i].visibility = max(
                    landmarks.landmark[i].visibility,
                    0.8
                )
        
        return enhanced_landmarks

    def _get_surrounding_depth_array(self, depth_image, x, y, window_size=5):
        """从深度图像数组中获取周围区域的平均深度值"""
        h, w = depth_image.shape[:2]
        half_window = window_size // 2
        
        # 提取窗口区域
        x_min = max(0, x - half_window)
        x_max = min(w, x + half_window + 1)
        y_min = max(0, y - half_window)
        y_max = min(h, y + half_window + 1)
        
        window = depth_image[y_min:y_max, x_min:x_max]
        
        # 过滤有效深度值 (非零且在合理范围内)
        valid_depths = window[(window > 0) & (window < 3000)]  # 3000mm = 3m
        
        if len(valid_depths) > 0:
            # 返回中位数值（毫米转米）
            return np.median(valid_depths) * 0.001
        else:
            return 0.0
        
    def process_frame(self):
        try:
            # 获取帧并对齐
            frames = self.pipeline.wait_for_frames()
            aligned_frames = self.align.process(frames)

            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()


            # 应用优化的深度滤波链
            if depth_frame:
                # 首先应用空间滤波器 - 增强深度图像的空间一致性
                filtered_depth = self.spatial_filter.process(depth_frame)
                
                # 只在非高性能模式下使用更多滤波
                if not self.high_performance:
                    # 然后应用时间滤波器 - 减少时间上的波动
                    filtered_depth = self.temporal_filter.process(filtered_depth)
                    # 最后应用孔洞填充滤波器 - 填补缺失深度
                    filtered_depth = self.hole_filling_filter.process(filtered_depth)
                
                depth_frame = filtered_depth

            if not color_frame or not depth_frame:
                return
        
            color_image = np.asanyarray(color_frame.get_data())

            # 处理图像
            results = self.hands.process(cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB))
            
            self._update_fps(time.time())

            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:

                    # 使用深度信息增强关节点坐标
                    enhanced_landmarks = self._enhance_landmarks_with_depth(
                    hand_landmarks, depth_frame, color_image)
                    
                    angles = self._process_hand_landmarks(color_image, enhanced_landmarks)                    
                    with self.lock:
                        self.hand_data = angles

            with self.lock:
                self.frame = color_image
                
        except Exception as e:
            print(f"处理帧错误: {e}")
            import traceback
            print(traceback.format_exc())

