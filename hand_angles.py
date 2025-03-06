import cv2
import numpy as np


class HandAngleCalculator:
    def __init__(self):
        # 定义手指关节链
        self.finger_chains = {
            'thumb': [0, 1, 2, 3, 4],      # 拇指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'index': [0, 5, 6, 7, 8],      # 食指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'middle': [0, 9, 10, 11, 12],  # 中指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'ring': [0, 13, 14, 15, 16],   # 无名指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'pinky': [0, 17, 18, 19, 20]   # 小指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
        }

        self.angle_history = {}  # 存储历史角度数据
        self.history_length = 5  # 历史数据长度
        self.kalman_filters = {}  # 卡尔曼滤波器字典

    def apply_kalman_filter(self, angle_name, value):
        """应用卡尔曼滤波"""
        if angle_name not in self.kalman_filters:
            self.kalman_filters[angle_name] = {
                'x': value,  # 状态估计
                'P': 1.0,    # 估计误差协方差
                'Q': 0.1,    # 过程噪声协方差
                'R': 1.0     # 测量噪声协方差
            }
        
        kf = self.kalman_filters[angle_name]
        
        # 预测
        x_pred = kf['x']
        P_pred = kf['P'] + kf['Q']
        
        # 更新
        K = P_pred / (P_pred + kf['R'])  # 卡尔曼增益
        kf['x'] = x_pred + K * (value - x_pred)
        kf['P'] = (1 - K) * P_pred
        
        return kf['x']

    def smooth_angle(self, angle_name, value):
        """平滑角度数据"""
        if angle_name not in self.angle_history:
            self.angle_history[angle_name] = []
            
        history = self.angle_history[angle_name]
        history.append(value)
        
        # 保持固定长度的历史数据
        if len(history) > self.history_length:
            history.pop(0)
            
        # 使用加权移动平均
        weights = np.exp(np.linspace(-1, 0, len(history)))
        weights /= weights.sum()
        
        smoothed = np.average(history, weights=weights)
        
        # 应用卡尔曼滤波
        filtered = self.apply_kalman_filter(angle_name, smoothed)
        
        return round(filtered, 2)
    
    def process_angle(self, angle_name, value, min_val, max_val):
        """处理角度数据：限制范围、平滑和滤波"""
        # 首先限制角度范围
        clipped = np.clip(value, min_val, max_val)
        # 然后应用平滑和滤波
        return self.smooth_angle(angle_name, clipped) 
               
    def create_hand_coordinate_system(self, landmarks):
        """创建手部解剖坐标系"""
        # 转换landmarks为numpy数组
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # 手掌中心（原点）
        origin = points[0]
        
        # 建立手部坐标系
        # Z轴：垂直于手掌平面（手掌法向量）
        palm_normal = np.cross(
            points[5] - points[0],    # 食指MCP到手掌中心的向量
            points[17] - points[0]    # 小指MCP到手掌中心的向量
        )
        z_axis = palm_normal / np.linalg.norm(palm_normal)
        
        # Y轴：从手掌中心指向中指MCP
        y_temp = points[9] - points[0]  # 中指MCP方向
        y_axis = y_temp - np.dot(y_temp, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # X轴：右手定则，垂直于Y和Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 创建旋转矩阵
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        
        return origin, rotation_matrix
    
    def transform_to_local(self, point, origin, rotation_matrix):
        """将点转换到局部坐标系"""
        return np.dot(rotation_matrix.T, (point - origin))
        
    def calculate_flexion_angle(self, p1, p2, p3):
        """计算屈曲角度"""
        try:
            # 将关键点转换为numpy数组
            point1 = np.array([p1.x, p1.y, p1.z])
            point2 = np.array([p2.x, p2.y, p2.z])
            point3 = np.array([p3.x, p3.y, p3.z])
            
            # 计算向量
            vector1 = point2 - point1
            vector2 = point3 - point2
            
            # 计算角度
            cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            
            # 确保角度在合理范围内
            if angle > 180:
                angle = 360 - angle
            
            return angle
        
        except Exception as e:
            print(f"计算屈曲角度错误: {e}")
            return 0
        
    def calculate_joint_angles(self, hand_landmarks):
        """计算所有关节角度"""
        angles = {}
        try:
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            origin, rotation_matrix = self.create_hand_coordinate_system(hand_landmarks)
            
            # 计算每个手指的角度
            for finger, chain in self.finger_chains.items():
                # 计算基础角度
                flexion = self.calculate_flexion_angle(
                    hand_landmarks.landmark[chain[0]],
                    hand_landmarks.landmark[chain[1]],
                    hand_landmarks.landmark[chain[2]]
                )
                
                # 计算外展角度
                if finger == 'middle':
                    # 特殊处理中指外展角度
                    index_mcp = points[5]
                    middle_mcp = points[9]
                    ring_mcp = points[13]
                    palm_center = points[0]
                    
                    adjacent_midpoint = (index_mcp + ring_mcp) / 2
                    mid_vector = adjacent_midpoint - palm_center
                    mid_vector = mid_vector / np.linalg.norm(mid_vector)
                    
                    middle_vector = middle_mcp - palm_center
                    middle_vector = middle_vector / np.linalg.norm(middle_vector)
                    
                    dot_product = np.dot(middle_vector, mid_vector)
                    angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
                    
                    cross_product = np.cross(mid_vector, middle_vector)
                    direction = np.sign(np.dot(cross_product, rotation_matrix[:, 2]))
                    
                    abduction = direction * angle
                else:
                    mcp_vector = points[chain[1]] - points[chain[0]]
                    abduction = self.calculate_abduction_angle(mcp_vector, rotation_matrix)
                
                # 处理角度数据
                if finger == 'thumb':
                    # 拇指角度
                    angles[f'{finger}_mcp_flexion'] = self.process_angle(
                        f'{finger}_mcp_flexion', flexion, 0, 50)
                    angles[f'{finger}_mcp_abduction'] = self.process_angle(
                        f'{finger}_mcp_abduction', abduction, 0, 70)
                    
                    # PIP屈曲
                    pip_flexion = self.calculate_flexion_angle(
                        hand_landmarks.landmark[chain[1]],
                        hand_landmarks.landmark[chain[2]],
                        hand_landmarks.landmark[chain[3]]
                    )
                    angles[f'{finger}_pip_flexion'] = self.process_angle(
                        f'{finger}_pip_flexion', pip_flexion, 0, 80)
                    
                    # DIP屈曲
                    dip_flexion = self.calculate_flexion_angle(
                        hand_landmarks.landmark[chain[2]],
                        hand_landmarks.landmark[chain[3]],
                        hand_landmarks.landmark[chain[4]]
                    )
                    angles[f'{finger}_dip_flexion'] = self.process_angle(
                        f'{finger}_dip_flexion', dip_flexion, 0, 90)
                else:
                    # 其他手指角度
                    angles[f'{finger}_mcp_flexion'] = self.process_angle(
                        f'{finger}_mcp_flexion', flexion, 0, 90)
                    angles[f'{finger}_mcp_abduction'] = self.process_angle(
                        f'{finger}_mcp_abduction', abduction, -20, 20)
                    
                    # PIP屈曲
                    pip_flexion = self.calculate_flexion_angle(
                        hand_landmarks.landmark[chain[1]],
                        hand_landmarks.landmark[chain[2]],
                        hand_landmarks.landmark[chain[3]]
                    )
                    angles[f'{finger}_pip_flexion'] = self.process_angle(
                        f'{finger}_pip_flexion', pip_flexion, 0, 100)
                    
                    # DIP屈曲
                    dip_flexion = self.calculate_flexion_angle(
                        hand_landmarks.landmark[chain[2]],
                        hand_landmarks.landmark[chain[3]],
                        hand_landmarks.landmark[chain[4]]
                    )
                    angles[f'{finger}_dip_flexion'] = self.process_angle(
                        f'{finger}_dip_flexion', dip_flexion, 0, 90)
                
        except Exception as e:
            print(f"计算关节角度错误: {e}")
            import traceback
            print(traceback.format_exc())
            return {}
            
        return angles        

            


    def calculate_abduction_angle(self, vector, rotation_matrix):
        """计算外展角度"""
        try:
            # 将向量投影到手掌坐标系的XY平面
            vector_normalized = vector / np.linalg.norm(vector)
            vector_local = np.dot(rotation_matrix.T, vector_normalized)
            
            # 只取XY平面的投影
            vector_proj = np.array([vector_local[0], vector_local[1], 0])
            
            # 如果投影向量太小，说明手指竖直，此时外展角接近0
            if np.linalg.norm(vector_proj) < 0.1:
                return 0
            
            vector_proj = vector_proj / np.linalg.norm(vector_proj)
            
            # 计算与Y轴的夹角
            y_axis = np.array([0, 1, 0])
            angle = np.degrees(np.arccos(np.clip(np.dot(vector_proj, y_axis), -1.0, 1.0)))
            
            # 根据x分量判断外展方向（左负右正）
            if vector_proj[0] < 0:
                angle = -angle
            
            return angle
            
        except Exception as e:
            print(f"计算外展角度错误: {e}")
            return 0

    def get_joint_index(self, finger, joint):
        """获取关节索引"""
        if finger not in self.finger_chains:
            return None
            
        chain = self.finger_chains[finger]
        
        if joint == 'mcp':
            return chain[1]
        elif joint == 'pip':
            return chain[2]
        elif joint == 'dip':
            return chain[3]
        
        return None
    
    # def visualize_angles(self, frame, landmarks, angles):
    #     """可视化关节角度"""
    #     height, width = frame.shape[:2]
        
    #     # 绘制坐标系
    #     origin, rotation_matrix = self.create_hand_coordinate_system(landmarks)
    #     origin_px = tuple(map(int, [
    #         origin[0] * width,
    #         origin[1] * height
    #     ]))
        
    #     # 绘制关节角度
    #     for name, angle in angles.items():
    #         # 解析角度名称
    #         parts = name.split('_')
    #         if len(parts) < 3:
    #             continue
                
    #         finger, joint, movement = parts
            
    #         # 获取关节位置
    #         joint_idx = self.get_joint_index(finger, joint)
    #         if joint_idx is None:
    #             continue
                
    #         px = int(landmarks.landmark[joint_idx].x * width)
    #         py = int(landmarks.landmark[joint_idx].y * height)
            
    #         # 根据运动类型选择颜色
    #         color = (0, 255, 0) if movement == 'flexion' else (255, 0, 0)
            
    #         # 显示角度值
    #         cv2.putText(frame, f"{angle:.1f}°", 
    #                    (px, py), cv2.FONT_HERSHEY_SIMPLEX, 
    #                    0.3, color, 1)

