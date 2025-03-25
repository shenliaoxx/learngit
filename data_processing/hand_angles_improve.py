import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R
import logging

class HandAngleCalculator:
    def __init__(self):
        # 初始化数据存储属性
        self.raw_angles = {}        # 存储原始角度数据
        self.filtered_angles = {}   # 存储滤波后的角度数据
        self.angle_history = {}     # 存储历史角度数据
        self.kalman_filters = {}    # 存储卡尔曼滤波器状态
        
        # 动态历史长度配置
        self.history_config = {
            'static': 8,    # 静止状态使用较长的历史数据
            'slow': 5,      # 低速运动使用中等长度
            'fast': 3       # 高速运动使用较短的历史数据
        }
        
        self.motion_state = 'slow'  # 默认为低速状态
        self.history_length = self.history_config['slow']

        # 定义手指关节链
        self.finger_chains = {
            'thumb': [0, 1, 2, 3, 4],      # 拇指: CMC(2DOF), MCP(2DOF), IP(1DOF)
            'index': [0, 5, 6, 7, 8],      # 食指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'middle': [0, 9, 10, 11, 12],  # 中指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'ring': [0, 13, 14, 15, 16],   # 无名指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
            'pinky': [0, 17, 18, 19, 20]   # 小指: MCP(2DOF), PIP(1DOF), DIP(1DOF)
        }

        # 设置基础角度补偿
        self.base_angles = {
            'thumb_cmc_flexion': 15,
            'thumb_mcp_flexion': 10,
            'thumb_ip_flexion': 0,
            'thumb_mcp_abduction': 20,
            'index_mcp_flexion': 15,
            'middle_mcp_flexion': 8,
            'ring_mcp_flexion': 1,
            'pinky_mcp_flexion': 8
        }

        # 卡尔曼滤波参数配置
        self.kalman_params = {
            'thumb': {
                'cmc_flexion': {'P': 1.0, 'Q': 0.2, 'R': 1.5},
                'mcp_flexion': {'P': 1.0, 'Q': 0.25, 'R': 1.2},
                'ip_flexion': {'P': 1.0, 'Q': 0.2, 'R': 1.0},
                'mcp_abduction': {'P': 1.0, 'Q': 0.15, 'R': 1.8}
            },
            'index': {
                'mcp_flexion': {'P': 1.0, 'Q': 0.15, 'R': 1.2},
                'pip_flexion': {'P': 1.0, 'Q': 0.12, 'R': 1.0},
                'dip_flexion': {'P': 1.0, 'Q': 0.1, 'R': 0.8},
                'mcp_abduction': {'P': 1.0, 'Q': 0.08, 'R': 1.5}
            },
            'middle': {
                'mcp_flexion': {'P': 1.0, 'Q': 0.15, 'R': 1.2},
                'pip_flexion': {'P': 1.0, 'Q': 0.12, 'R': 1.0},
                'dip_flexion': {'P': 1.0, 'Q': 0.1, 'R': 0.8},
                'mcp_abduction': {'P': 1.0, 'Q': 0.08, 'R': 1.5}
            },
            'ring': {
                'mcp_flexion': {'P': 1.0, 'Q': 0.12, 'R': 1.3},
                'pip_flexion': {'P': 1.0, 'Q': 0.1, 'R': 1.1},
                'dip_flexion': {'P': 1.0, 'Q': 0.08, 'R': 0.9},
                'mcp_abduction': {'P': 1.0, 'Q': 0.06, 'R': 1.6}
            },
            'pinky': {
                'mcp_flexion': {'P': 1.0, 'Q': 0.1, 'R': 1.5},
                'pip_flexion': {'P': 1.0, 'Q': 0.08, 'R': 1.2},
                'dip_flexion': {'P': 1.0, 'Q': 0.06, 'R': 1.0},
                'mcp_abduction': {'P': 1.0, 'Q': 0.05, 'R': 1.8}
            }
        }

    def apply_kalman_filter(self, angle_name, value):
        """应用卡尔曼滤波"""
        # 获取对应手指和关节的参数
        finger = next((f for f in ['thumb', 'index', 'middle', 'ring', 'pinky'] 
                      if f in angle_name), None)
        joint_type = next((j for j in ['cmc_flexion', 'mcp_flexion', 'pip_flexion', 
                                      'dip_flexion', 'mcp_abduction', 'ip_flexion'] 
                          if j in angle_name), None)
        
        if finger and joint_type and finger in self.kalman_params:
            params = self.kalman_params[finger].get(joint_type, 
                    {'P': 1.0, 'Q': 0.1, 'R': 1.0})
        else:
            params = {'P': 1.0, 'Q': 0.1, 'R': 1.0}
        
        if angle_name not in self.kalman_filters:
            self.kalman_filters[angle_name] = {
                'x': value,  # 状态估计
                'P': params['P'],  # 估计误差协方差
                'Q': params['Q'],  # 过程噪声
                'R': params['R']   # 测量噪声
            }
        
        kf = self.kalman_filters[angle_name]
        
        # 预测步骤
        x_pred = kf['x']
        P_pred = kf['P'] + kf['Q']
        
        # 更新步骤
        K = P_pred / (P_pred + kf['R'])  # 卡尔曼增益
        kf['x'] = x_pred + K * (value - x_pred)
        kf['P'] = (1 - K) * P_pred
        
        # 确保返回float类型
        return float(kf['x'])

    def detect_motion_state(self, angle_name, current_value):
        """检测运动状态"""
        if angle_name not in self.raw_angles:
            self.raw_angles[angle_name] = []
        
        history = self.raw_angles[angle_name]
        history.append(float(current_value))  # 确保使用float类型
        
        # 保持最近20帧的历史数据
        if len(history) > 20:
            history.pop(0)
        
        if len(history) < 3:
            return 'slow'
        
        # 计算角速度和加速度
        velocities = np.diff(history)
        accelerations = np.diff(velocities)
        
        mean_velocity = np.mean(np.abs(velocities))
        mean_acceleration = np.mean(np.abs(accelerations)) if len(accelerations) > 0 else 0
        
        # 根据速度和加速度判断运动状态
        if mean_velocity < 1.0 and mean_acceleration < 0.5:
            return 'static'
        elif mean_velocity < 5.0 and mean_acceleration < 2.0:
            return 'slow'
        else:
            return 'fast'

    def smooth_angle(self, angle_name, value):
        """平滑角度数据，根据运动状态自适应调整"""
        try:
            # 使用列表的固定长度来优化内存使用
            max_history = 100
            
            # 初始化或更新数据存储
            if angle_name not in self.raw_angles:
                self.raw_angles[angle_name] = []
            if angle_name not in self.filtered_angles:
                self.filtered_angles[angle_name] = []
            if angle_name not in self.angle_history:
                self.angle_history[angle_name] = []
            
            # 高效地更新历史数据
            raw_angles = self.raw_angles[angle_name]
            raw_angles.append(float(value))
            if len(raw_angles) > max_history:
                raw_angles.pop(0)
            
            # 检测运动状态并获取适当的历史长度
            motion_state = self.detect_motion_state(angle_name, value)
            history_length = self.history_config[motion_state]
            
            # 更新角度历史
            history = self.angle_history[angle_name]
            history.append(value)
            while len(history) > history_length:
                history.pop(0)
            
            # 计算权重
            weights = np.exp(np.linspace(
                -2 if motion_state == 'fast' else -1 if motion_state == 'slow' else -0.5,
                0,
                len(history)
            ))
            weights /= weights.sum()
            
            # 计算平滑值
            smoothed = np.average(history, weights=weights)
            
            # 应用卡尔曼滤波
            filtered = self.apply_kalman_filter(angle_name, smoothed)
            
            # 更新滤波后的数据
            filtered_angles = self.filtered_angles[angle_name]
            filtered_angles.append(float(filtered))
            if len(filtered_angles) > max_history:
                filtered_angles.pop(0)
            
            return round(filtered, 2)
            
        except Exception as e:
            print(f"平滑角度数据错误: {e}")
            return value  # 发生错误时返回原始值
    
    def process_angle(self, angle_name, value, min_val, max_val):
        """处理角度数据：限制范围、平滑和滤波"""
        # 首先限制角度范围
        clipped = np.clip(value, min_val, max_val)
        # 然后应用平滑和滤波
        return self.smooth_angle(angle_name, clipped) 
               
    def create_hand_coordinate_system(self, landmarks):
        """创建手部解剖坐标系"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # 手掌中心（原点）
        origin = points[0]
        
        # Z轴：垂直于手掌平面（手掌法向量）
        palm_normal = np.cross(
            points[5] - points[0],    # 食指MCP到手掌中心的向量
            points[17] - points[0]    # 小指MCP到手掌中心的向量
        )
        z_axis = palm_normal / np.linalg.norm(palm_normal)
        
        # Y轴：使用手掌中轴（考虑所有手指MCP关节）
        mcp_points = np.array([points[5], points[9], points[13]])  # 食指、中指、无名指的MCP
        palm_midline = np.mean(mcp_points, axis=0) - origin
        y_temp = palm_midline - np.dot(palm_midline, z_axis) * z_axis
        y_axis = y_temp / np.linalg.norm(y_temp)
        
        # X轴：右手定则，垂直于Y和Z
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 创建旋转矩阵
        rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
        
        return origin, rotation_matrix
    

    
    def transform_to_local(self, point, origin, rotation_matrix):
        """将点转换到局部坐标系"""
        return np.dot(rotation_matrix.T, (point - origin))
        
    def calculate_flexion_angle(self, p1, p2, p3, origin, rotation_matrix, base_angle=0, joint_type=None):
        """计算屈曲角度，适用于垂直手掌姿势
        Args:
            p1, p2, p3: 世界坐标系下的三个关键点（近端、中端、远端）
            origin: 手部坐标系原点
            rotation_matrix: 手部坐标系旋转矩阵
            base_angle: 初始姿态下的基准角度（默认为0）
        Returns:
            屈曲角度（度）
        """
        try:
            # 1. 将关键点转换为numpy数组（世界坐标系）
            point1 = np.array([p1.x, p1.y, p1.z])
            point2 = np.array([p2.x, p2.y, p2.z])
            point3 = np.array([p3.x, p3.y, p3.z])
            
            # 2. 将点转换到手部坐标系
            point1_local = self.transform_to_local(point1, origin, rotation_matrix)
            point2_local = self.transform_to_local(point2, origin, rotation_matrix)
            point3_local = self.transform_to_local(point3, origin, rotation_matrix)
            
            # 3. 计算局部坐标系下的向量
            vector1 = point2_local - point1_local  # 近端到中端的向量
            vector2 = point3_local - point2_local  # 中端到远端的向量
            
            # 4. 检查向量长度
            if np.linalg.norm(vector1) < 0.001 or np.linalg.norm(vector2) < 0.001:
                return 0
            
            # 5. 计算夹角 - 直接使用向量点积公式
            cos_angle = np.dot(vector1, vector2) / (np.linalg.norm(vector1) * np.linalg.norm(vector2))
            angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
            # 6. 特殊处理某些关节类型
            if joint_type == 'thumb_cmc':
                # 拇指CMC关节三维运动更复杂，需考虑其他因素
                # 计算CMC到MCP向量相对于手掌平面的倾斜程度
                palm_normal = rotation_matrix[:, 2]  # Z轴为手掌法向量
                thumb_dir = vector2 / np.linalg.norm(vector2)
                
                # 计算向量与手掌法向量的夹角
                thumb_elevation = 90 - np.degrees(np.arccos(np.clip(np.abs(np.dot(thumb_dir, palm_normal)), -1.0, 1.0)))
                
                # 根据拇指抬高程度调整基准角度
                adjusted_base = base_angle * (1 - thumb_elevation / 90)
                angle = angle - adjusted_base
            else:
                # 其他关节使用简单的基准角度补偿
                angle = angle - base_angle
            # # 6. 减去基础角度补偿
            # compensated_angle = predicted_angle - base_angle

            return max(0, angle)
            
        except Exception as e:
            print(f"计算屈曲角度错误: {e}")
            return 0
        
        
    def calculate_abduction_angle(self, vector, rotation_matrix, finger_name, hand_landmarks, base_angle=0):
        """计算外展角度 - 简化版（无方向）
        - 五指并拢时角度接近0度
        - 五指张开时角度增大
        """
        try:
            # 1. 获取手部关键点
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            
            # 2. 将输入向量归一化并转换到手部坐标系
            vector_normalized = vector / np.linalg.norm(vector)
            vector_local = np.dot(rotation_matrix.T, vector_normalized)
            
            # 3. 投影到手掌平面（XY平面）
            z_axis = rotation_matrix[:, 2]  # 手掌法向量
            vector_proj = vector_local - np.dot(vector_local, z_axis) * z_axis
            
            proj_length = np.linalg.norm(vector_proj)
            # 如果投影向量太小，说明手指接近垂直于手掌
            if proj_length < 0.05:
                return 0
            
            vector_proj = vector_proj / proj_length
            
            # 4. 选择合适的参考向量
            if finger_name == 'thumb':
                # 拇指：使用食指MCP方向作为参考
                index_vector = points[6] - points[5]  # 食指MCP方向
                index_vector = index_vector / np.linalg.norm(index_vector)
                reference_local = np.dot(rotation_matrix.T, index_vector)
            elif finger_name == 'pinky':
                # 小指：使用无名指MCP方向作为参考
                ring_vector = points[14] - points[13]  # 无名指MCP方向
                ring_vector = ring_vector / np.linalg.norm(ring_vector)
                reference_local = np.dot(rotation_matrix.T, ring_vector)
            elif finger_name == 'index':
                # 食指：混合使用Y轴和中指方向
                middle_vector = points[10] - points[9]  # 中指PIP到MCP的向量
                middle_vector = middle_vector / np.linalg.norm(middle_vector)
                middle_local = np.dot(rotation_matrix.T, middle_vector)
                
                # 80% Y轴 + 20% 中指方向
                reference_local = 0.8 * rotation_matrix[:, 1] + 0.2 * middle_local
                reference_local = reference_local / np.linalg.norm(reference_local)
            else:
                # 其他手指：使用手掌Y轴作为参考
                reference_local = rotation_matrix[:, 1]            

            
            # 5. 将参考向量投影到手掌平面
            reference_proj = reference_local - np.dot(reference_local, z_axis) * z_axis
            if np.linalg.norm(reference_proj) < 0.05:
                # 备用方案：如果参考向量投影太小，使用X轴
                reference_proj = rotation_matrix[:, 0] - np.dot(rotation_matrix[:, 0], z_axis) * z_axis
            
            reference_proj = reference_proj / np.linalg.norm(reference_proj)
            
            # 6. 计算外展角度（无方向，只有大小）
            dot_product = np.dot(vector_proj, reference_proj)
            angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
            angle = min(angle, 180-angle)
            
            # 7. 减去基础角度补偿并确保角度非负
            compensated_angle = max(0, angle - base_angle)

            return compensated_angle
            
    
        except Exception as e:
            print(f"计算外展角度错误: {e}")
            return 0

    def calculate_thumb_angles(self, hand_landmarks, chain, origin, rotation_matrix):
        """计算拇指的所有角度，考虑其独特解剖结构"""
        try:
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
            

            
            # 1. CMC (Carpometacarpal) 屈曲角度
            # 使用特殊基准角度考虑拇指CMC关节的自然角度
            cmc_flexion = self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[0]],  # 手掌中心
                hand_landmarks.landmark[chain[1]],  # CMC
                hand_landmarks.landmark[chain[2]],  # MCP
                origin, rotation_matrix,
                base_angle=self.base_angles['thumb_cmc_flexion'],  # 考虑CMC关节自然角度
                joint_type='thumb_cmc'
            )
            
            # 2. MCP (Metacarpophalangeal) 屈曲角度
            mcp_flexion = self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[1]],  # CMC
                hand_landmarks.landmark[chain[2]],  # MCP
                hand_landmarks.landmark[chain[3]],  # IP
                origin, rotation_matrix,
                base_angle=self.base_angles['thumb_mcp_flexion']  # 考虑MCP关节自然角度
            )
            
            # 3. IP (Interphalangeal) 屈曲角度
            ip_flexion = self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[2]],  # MCP
                hand_landmarks.landmark[chain[3]],  # IP
                hand_landmarks.landmark[chain[4]],  # 指尖
                origin, rotation_matrix,
                base_angle=self.base_angles['thumb_ip_flexion']  # 考虑IP关节自然角度
            )
            
            
            # 4.MCP外展角度（使用标准方法）
            mcp_vector = points[chain[2]] - points[chain[1]]  # 使用MCP到CMC的向量
            mcp_abduction = self.calculate_abduction_angle(
                mcp_vector, 
                rotation_matrix, 
                'thumb',
                hand_landmarks,
                base_angle=self.base_angles['thumb_mcp_abduction']
            )
            
            # 处理角度数据
            angles = {
                'thumb_cmc_flexion': self.process_angle('thumb_cmc_flexion', cmc_flexion, 0, 50),
                'thumb_mcp_flexion': self.process_angle('thumb_mcp_flexion', mcp_flexion, 0, 80),
                'thumb_ip_flexion': self.process_angle('thumb_ip_flexion', ip_flexion, 0, 90),
                'thumb_mcp_abduction': self.process_angle('thumb_mcp_abduction', mcp_abduction, 0, 70)
            }
            
            return angles
            
        except Exception as e:
            print(f"计算拇指角度错误: {e}")
            return {}
        
                

    def calculate_finger_angles(self, hand_landmarks, chain, finger_name, origin, rotation_matrix):
        """计算其他手指的所有角度"""
        try:


            # 1. 获取手部关键点
            points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])

            # 2. MCP屈曲角度（添加基础角度补偿）
            mcp_flexion = self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[0]],  # 手掌中心
                hand_landmarks.landmark[chain[1]],  # MCP
                hand_landmarks.landmark[chain[2]],  # PIP
                origin, rotation_matrix,
                base_angle=self.base_angles[f'{finger_name}_mcp_flexion']  # 获取对应手指的基础角度
            )
            
            # 3. PIP屈曲角度
            pip_flexion = self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[1]],  # MCP
                hand_landmarks.landmark[chain[2]],  # PIP
                hand_landmarks.landmark[chain[3]],  # DIP
                origin, rotation_matrix
            )
            
            # 4. DIP屈曲角度
            dip_flexion = self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[2]],  # PIP
                hand_landmarks.landmark[chain[3]],  # DIP
                hand_landmarks.landmark[chain[4]],  # 指尖
                origin, rotation_matrix
            )
            
            mcp_vector = points[chain[2]] - points[chain[1]]
            mcp_abduction = self.calculate_abduction_angle(
                mcp_vector, 
                rotation_matrix, 
                finger_name,
                hand_landmarks
            )
                
            # 在calculate_finger_angles函数中:
            angles = {
                f'{finger_name}_mcp_flexion': self.process_angle(
                    f'{finger_name}_mcp_flexion', mcp_flexion, 0, 90),
                f'{finger_name}_pip_flexion': self.process_angle(
                    f'{finger_name}_pip_flexion', pip_flexion, 0, 100),
                f'{finger_name}_dip_flexion': self.process_angle(
                    f'{finger_name}_dip_flexion', dip_flexion, 0, 90),
                f'{finger_name}_mcp_abduction': self.process_angle(
                    f'{finger_name}_mcp_abduction', mcp_abduction, 0, 40)  # 只有正值，统一限制在0-30度
            }
            
            return angles
        except Exception as e:
            print(f"计算{finger_name}角度错误: {e}")
            return {}
         
       
    def calculate_joint_angles(self, hand_landmarks):
        """计算所有关节角度"""
        try:
            # 1. 准备数据
            origin, rotation_matrix = self.create_hand_coordinate_system(hand_landmarks)
            all_angles = {}
            
            # 2. 计算每个手指的角度
            for finger, chain in self.finger_chains.items():
                if finger == 'thumb':
                    # 计算拇指角度
                    thumb_angles = self.calculate_thumb_angles(
                        hand_landmarks, chain, origin, rotation_matrix)
                    all_angles.update(thumb_angles)
                else:
                    # 计算其他手指角度
                    finger_angles = self.calculate_finger_angles(
                        hand_landmarks, chain, finger, origin, rotation_matrix)
                    all_angles.update(finger_angles)
                    
            return all_angles
            
        except Exception as e:
            print(f"计算关节角度错误: {e}")
            import traceback
            print(traceback.format_exc())
            return {}

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
    
    def draw_global_coordinate_system(self, image, scale=30):
        """绘制全局坐标系"""
        h, w = image.shape[:2]
        origin = (50, h - 50)  # 将原点放在左下角附近
        
        # 绘制坐标轴
        cv2.line(image, origin, (origin[0] + scale, origin[1]), (0, 0, 255), 2)  # X轴-红色
        cv2.line(image, origin, (origin[0], origin[1] - scale), (0, 255, 0), 2)  # Y轴-绿色
        cv2.line(image, origin, (origin[0] + int(scale/2), origin[1] - int(scale/2)), (255, 0, 0), 2)  # Z轴-蓝色
        
        # 添加标签
        cv2.putText(image, 'X', (origin[0] + scale + 10, origin[1]), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        cv2.putText(image, 'Y', (origin[0], origin[1] - scale - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(image, 'Z', (origin[0] + int(scale/2) + 10, origin[1] - int(scale/2)), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)
        cv2.putText(image, 'Global', (origin[0] - 30, origin[1] + 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

  

            
 
            

    def draw_hand_coordinate_system(self, image, origin, rotation_matrix, scale=100):
        """绘制手部坐标系，简化显示效果
        Args:
            image: OpenCV图像
            origin: 坐标系原点 [x, y, z]
            rotation_matrix: 旋转矩阵
            scale: 坐标轴长度缩放因子
        """
        # 将3D原点转换为2D图像坐标
        origin_2d = (int(origin[0] * image.shape[1]), int(origin[1] * image.shape[0]))
        
        # 优化颜色方案
        colors = [
            (0, 0, 255),    # X轴：纯红色
            (0, 255, 0),    # Y轴：纯绿色
            (255, 0, 0)     # Z轴：纯蓝色
        ]
        labels = ['X', 'Y', 'Z']
        
        # 绘制原点标记（仅保留白色圆点）
        cv2.circle(image, origin_2d, 3, (255, 255, 255), -1)
        
        for i in range(3):
            axis = rotation_matrix[:, i]
            # 调整缩放因子以获得合适的轴长
            endpoint = origin + axis * (scale/800)
            endpoint_2d = (int(endpoint[0] * image.shape[1]), 
                        int(endpoint[1] * image.shape[0]))
            
            # 绘制坐标轴
            cv2.line(image, origin_2d, endpoint_2d, colors[i], 2, cv2.LINE_AA)
            
            # 计算标签位置
            label_pos = (
                endpoint_2d[0] + (5 if endpoint_2d[0] > origin_2d[0] else -15),
                endpoint_2d[1] + (15 if endpoint_2d[1] > origin_2d[1] else -5)
            )
            
            # 直接绘制文本，无边框
            cv2.putText(image, labels[i], label_pos, 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2, cv2.LINE_AA) 
            



# def create_hand_coordinate_system(self, landmarks):
#     """创建更稳定的手部解剖坐标系"""
#     points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    
#     # 手掌中心（原点）
#     origin = points[0]
    
#     # 收集所有MCP关节点和手腕点
#     mcp_points = np.array([points[1], points[5], points[9], points[13], points[17]])
    
#     # 使用主成分分析创建更稳定的坐标系
#     # 收集更多手掌关键点
#     palm_points = np.vstack([origin.reshape(1, 3), mcp_points])
    
#     # 中心化数据
#     centered_points = palm_points - np.mean(palm_points, axis=0)
    
#     # 计算协方差矩阵
#     cov = np.dot(centered_points.T, centered_points) / (len(centered_points) - 1)
    
#     # 计算特征值和特征向量
#     eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
#     # 按特征值大小排序特征向量
#     idx = eigenvalues.argsort()[::-1]  
#     eigenvectors = eigenvectors[:, idx]
    
#     # 使用特征向量作为坐标系
#     # 第一特征向量（最大方差方向）应该大致对应于手掌的宽度方向
#     x_axis_temp = eigenvectors[:, 0]
#     # 第二特征向量通常对应手掌长度方向
#     y_axis_temp = eigenvectors[:, 1]
#     # 第三特征向量对应法向量
#     z_axis_temp = eigenvectors[:, 2]
    
#     # 确保坐标系是右手系
#     if np.dot(np.cross(x_axis_temp, y_axis_temp), z_axis_temp) < 0:
#         z_axis_temp = -z_axis_temp
    
#     # 进一步校正：使Y轴朝向手指方向
#     palm_midline = np.mean([points[9]], axis=0) - origin  # 使用中指MCP到手腕的向量
#     if np.dot(y_axis_temp, palm_midline) < 0:
#         y_axis_temp = -y_axis_temp
#         x_axis_temp = -x_axis_temp  # 保持右手系
    
#     # 正交化
#     z_axis = z_axis_temp / np.linalg.norm(z_axis_temp)
#     y_axis = y_axis_temp / np.linalg.norm(y_axis_temp)
#     x_axis = np.cross(y_axis, z_axis)
#     x_axis = x_axis / np.linalg.norm(x_axis)
    
#     # 创建旋转矩阵
#     rotation_matrix = np.vstack([x_axis, y_axis, z_axis]).T
    
#     return origin, rotation_matrix



# def apply_anatomical_constraints(self, angles, finger_name):
#     """应用解剖学约束修正角度"""


#     # 1. DIP和PIP关节的关系约束
#     pip_key = f'{finger_name}_pip_flexion'
#     dip_key = f'{finger_name}_dip_flexion'
    
#     if angles[pip_key] > 15:
#         # 当PIP明显弯曲时，DIP通常约为PIP的2/3
#         expected_dip = angles[pip_key] * 0.67
#         actual_dip = angles[dip_key]
        
#         # 如果差异过大，使用加权平均调整
#         if abs(actual_dip - expected_dip) > 15:
#             # 实际值权重0.6，预期值权重0.4
#             angles[dip_key] = 0.6 * actual_dip + 0.4 * expected_dip
    
#     # 2. 手指完全伸直时的约束
#     mcp_key = f'{finger_name}_mcp_flexion'
#     if angles[mcp_key] < 5 and angles[pip_key] < 5:
#         # 手指伸直时，所有关节角度接近零
#         angles[dip_key] = min(angles[dip_key], 10)
    
#     # 3. 外展角度的合理性检查
#     abd_key = f'{finger_name}_mcp_abduction'
    
#     # 屈曲程度大时，外展角度的置信度应降低
#     if angles[mcp_key] > 60:
#         # 当MCP屈曲超过60度时，外展测量可能不准确
#         # 随着屈曲增加，降低外展角度（线性衰减）
#         reduction_factor = 1.0 - (angles[mcp_key] - 60) / 30
#         reduction_factor = max(0.3, reduction_factor)  # 至少保留30%
#         angles[abd_key] *= reduction_factor
    
#     return angles 