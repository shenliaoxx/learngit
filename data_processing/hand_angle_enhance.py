import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

class HandAngleCalculator:
    def __init__(self):
        self._init_data_structures()
        self._init_finger_chains()
        self._init_base_angles()
        self._init_kalman_params()

    def _init_data_structures(self):
        self.kalman_filters = {}

    def _init_finger_chains(self):
        self.finger_chains = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [0, 5, 6, 7, 8],
            'middle': [0, 9, 10, 11, 12],
            'ring': [0, 13, 14, 15, 16],
            'pinky': [0, 17, 18, 19, 20]
        }

    def _init_base_angles(self):
        self.base_angles = {
            'thumb_cmc_flexion': 15, 'thumb_mcp_flexion': 10, 'thumb_ip_flexion': 0,
            'thumb_mcp_abduction': 15, 'index_mcp_flexion': 15, 'middle_mcp_flexion': 8,
            'ring_mcp_flexion': 1, 'pinky_mcp_flexion': 8, 'pinky_mcp_abduction': 7
        }

    def _init_kalman_params(self):
        self.kalman_params = {
            'thumb': {
                'cmc_flexion': {'P': 1.5, 'Q': 0.3, 'R': 1.2},     # 增大P和Q，降低R
                'mcp_flexion': {'P': 1.2, 'Q': 0.25, 'R': 1.2},    # 保持不变
                'ip_flexion': {'P': 0.9, 'Q': 0.2, 'R': 1.2},      # 降低P，增加R
                'mcp_abduction': {'P': 1.3, 'Q': 0.15, 'R': 1.5}   # 降低R
            },
            'index': {
                'mcp_flexion': {'P': 1.3, 'Q': 0.15, 'R': 1.2},    # 增大P
                'pip_flexion': {'P': 1.0, 'Q': 0.12, 'R': 1.2},    # 增加R
                'dip_flexion': {'P': 0.8, 'Q': 0.1, 'R': 1.0},     # 降低P
                'mcp_abduction': {'P': 1.2, 'Q': 0.08, 'R': 1.3}   # 降低R
            },
            'middle': {
                'mcp_flexion': {'P': 1.2, 'Q': 0.13, 'R': 1.2},    # 调整Q
                'pip_flexion': {'P': 1.0, 'Q': 0.11, 'R': 1.2},    # 调整Q和R
                'dip_flexion': {'P': 0.8, 'Q': 0.09, 'R': 1.0},    # 调整P和Q
                'mcp_abduction': {'P': 1.1, 'Q': 0.07, 'R': 1.3}   # 调整所有参数
            },
            'ring': {
                'mcp_flexion': {'P': 1.1, 'Q': 0.12, 'R': 1.3},    # 保持不变
                'pip_flexion': {'P': 0.9, 'Q': 0.1, 'R': 1.2},     # 增加R
                'dip_flexion': {'P': 0.8, 'Q': 0.08, 'R': 1.1},    # 增加R
                'mcp_abduction': {'P': 1.0, 'Q': 0.06, 'R': 1.4}   # 降低R
            },
            'pinky': {
                'mcp_flexion': {'P': 1.0, 'Q': 0.15, 'R': 1.3},    # 增大Q，降低R
                'pip_flexion': {'P': 0.9, 'Q': 0.12, 'R': 1.1},    # 增大Q，降低R
                'dip_flexion': {'P': 0.8, 'Q': 0.1, 'R': 1.0},     # 增大Q
                'mcp_abduction': {'P': 1.0, 'Q': 0.08, 'R': 1.5}   # 增大Q，降低R
            }
        }

    def apply_kalman_filter(self, angle_name, value):
        finger = next((f for f in self.kalman_params if f in angle_name), None)
        joint_type = next((j for j in ['cmc_flexion', 'mcp_flexion', 'pip_flexion', 
                                     'dip_flexion', 'mcp_abduction', 'ip_flexion'] 
                         if j in angle_name), None)
        
        params = self.kalman_params.get(finger, {}).get(joint_type, {'P': 1.0, 'Q': 0.1, 'R': 1.0})
        
        if angle_name not in self.kalman_filters:
            self.kalman_filters[angle_name] = {'x': value, 'P': params['P'], 
                                             'Q': params['Q'], 'R': params['R']}
        
        kf = self.kalman_filters[angle_name]
        x_pred = kf['x']
        P_pred = kf['P'] + kf['Q']
        K = P_pred / (P_pred + kf['R'])
        kf['x'] = x_pred + K * (value - x_pred)
        kf['P'] = (1 - K) * P_pred
        
        return float(kf['x'])


    def process_angle(self, angle_name, value, min_val, max_val):
        """移除平滑处理，只保留卡尔曼滤波"""
        # 限制角度范围
        clipped_value = np.clip(value, min_val, max_val)
        # 应用卡尔曼滤波
        filtered_value = self.apply_kalman_filter(angle_name, clipped_value)
        return round(filtered_value, 2)

    def create_hand_coordinate_system(self, landmarks):
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        origin = points[0]
        edge_vectors = np.array([points[5]-points[0], points[17]-points[0], 
                               points[9]-points[0], points[13]-points[0]])
        
        cross_prods = np.cross(edge_vectors[[0,0,0,1,1,2]], edge_vectors[[1,2,3,2,3,3]])
        z_axis = np.mean(cross_prods, axis=0)
        z_axis /= np.linalg.norm(z_axis)
        
        mcp_points = points[[2,5,9,13,17]]
        palm_center = np.mean(mcp_points, axis=0)
        y_temp = palm_center - origin
        y_temp = y_temp - np.dot(y_temp, z_axis) * z_axis
        y_axis = y_temp / np.linalg.norm(y_temp)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= np.linalg.norm(x_axis)
        
        return origin, np.column_stack((x_axis, y_axis, z_axis))

    def calculate_flexion_angle(self, p1, p2, p3, origin, rotation_matrix, base_angle=0, joint_type=None):
        points = np.array([[p1.x, p1.y, p1.z], [p2.x, p2.y, p2.z], [p3.x, p3.y, p3.z]])
        points_local = (rotation_matrix.T @ (points - origin).T).T
        
        vector1 = points_local[1] - points_local[0]
        vector2 = points_local[2] - points_local[1]
        norm1, norm2 = np.linalg.norm(vector1), np.linalg.norm(vector2)
        
        if norm1 < 1e-6 or norm2 < 1e-6:
            return 0.0
        
        cos_angle = np.dot(vector1, vector2) / (norm1 * norm2)
        angle = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0)))
        
        if joint_type == 'thumb_cmc':
            palm_normal = rotation_matrix[:, 2]
            thumb_dir = vector2 / norm2
            thumb_elevation = 90 - np.degrees(np.arccos(np.clip(np.abs(np.dot(thumb_dir, palm_normal)), -1.0, 1.0)))
            adjusted_base = base_angle * (1 - thumb_elevation / 90)
            angle = angle - adjusted_base
        else:
            angle = angle - base_angle
            
        return max(0.0, angle)

    # def calculate_abduction_angle(self, vector, rotation_matrix, finger_name, hand_landmarks, base_angle=0):
    #     points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
    #     vector_norm = vector / np.linalg.norm(vector)
        
    #     # 将向量转换到局部坐标系
    #     vector_local = rotation_matrix.T @ vector_norm
    #     z_axis = rotation_matrix[:, 2]
        
    #     # 将向量投影到手掌平面
    #     vector_proj = vector_local - np.dot(vector_local, z_axis) * z_axis
        
    #     if np.linalg.norm(vector_proj) < 0.05:
    #         return 0  # 如果投影过小，返回0
        
    #     vector_proj = vector_proj / np.linalg.norm(vector_proj)
        
    #     # 选择参考向量
    #     if finger_name == 'thumb':
    #         reference_vector = points[6] - points[5]  # 拇指参考
    #     elif finger_name == 'pinky':
    #         reference_vector = points[14] - points[13]  # 小指参考
    #     elif finger_name == 'index':
    #         middle_vector = points[10] - points[9]
    #         middle_local = rotation_matrix.T @ (middle_vector / np.linalg.norm(middle_vector))
    #         reference_local = 0.8 * rotation_matrix[:, 1] + 0.2 * middle_local
    #         reference_vector = reference_local / np.linalg.norm(reference_local)  # 食指参考
    #     else:
    #         reference_vector = rotation_matrix[:, 1]  # 其他手指参考
        
    #     # 计算参考向量的投影
    #     ref_proj = reference_vector - np.dot(reference_vector, z_axis) * z_axis
    #     ref_norm = np.linalg.norm(ref_proj)
        
    #     if ref_norm < 0.05:
    #         ref_proj = rotation_matrix[:, 0] - np.dot(rotation_matrix[:, 0], z_axis) * z_axis
    #         ref_proj = ref_proj / np.linalg.norm(ref_proj)
    #     else:
    #         ref_proj = ref_proj / ref_norm
        
    #     # 计算夹角
    #     dot_product = np.dot(vector_proj, ref_proj)
    #     angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
    #     angle = min(angle, 180 - angle)  # 确保角度在0到180度之间
    #     return max(0.0, angle - base_angle)


    def calculate_abduction_angle(self, vector, rotation_matrix, finger_name, hand_landmarks, base_angle=0):
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        vector_norm = vector / np.linalg.norm(vector)
        
        # 将向量转换到局部坐标系
        vector_local = rotation_matrix.T @ vector_norm
        z_axis = rotation_matrix[:, 2]
        
        # 将向量投影到手掌平面
        vector_proj = vector_local - np.dot(vector_local, z_axis) * z_axis
        
        if np.linalg.norm(vector_proj) < 0.05:
            return 0  # 如果投影过小，返回0
        
        vector_proj = vector_proj / np.linalg.norm(vector_proj)
        
        # 选择参考向量
        if finger_name == 'thumb':
            reference_vector = points[6] - points[5]  # 拇指参考
        elif finger_name == 'pinky':
            reference_vector = points[14] - points[13]  # 小指参考
        elif finger_name == 'index':
            middle_vector = points[10] - points[9]
            middle_local = rotation_matrix.T @ (middle_vector / np.linalg.norm(middle_vector))
            reference_local = 0.8 * rotation_matrix[:, 1] + 0.2 * middle_local
            reference_vector = reference_local / np.linalg.norm(reference_local)  # 食指参考
        else:
            reference_vector = rotation_matrix[:, 1]  # 其他手指参考
        
        # 计算参考向量的投影
        ref_proj = reference_vector - np.dot(reference_vector, z_axis) * z_axis
        ref_norm = np.linalg.norm(ref_proj)
        
        if ref_norm < 0.05:
            ref_proj = rotation_matrix[:, 0] - np.dot(rotation_matrix[:, 0], z_axis) * z_axis
            ref_proj = ref_proj / np.linalg.norm(ref_proj)
        else:
            ref_proj = ref_proj / ref_norm
        
        # 计算夹角
        dot_product = np.dot(vector_proj, ref_proj)
        angle = np.degrees(np.arccos(np.clip(dot_product, -1.0, 1.0)))
        angle = min(angle, 180 - angle)  # 确保角度在0到180度之间

        # 计算角度符号
        cross_product = np.cross(ref_proj, vector_proj)
        sign = np.sign(np.dot(cross_product, z_axis))  # 根据z轴确定角度的正负

        # 应用符号并减去基准角度
        signed_angle = angle * sign - base_angle
        
        # 根据手指类型限制角度范围
        if finger_name == 'thumb':
            return np.clip(signed_angle, -45, 45)
        elif finger_name == 'index':
            return np.clip(signed_angle, -20, 20)
        elif finger_name == 'middle':
            return np.clip(signed_angle, -15, 15)
        elif finger_name == 'ring':
            return np.clip(signed_angle, -10, 10)
        else:  # pinky
            return np.clip(signed_angle, -15, 15)

    def calculate_thumb_angles(self, hand_landmarks, chain, origin, rotation_matrix):
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        angles = {
            'thumb_cmc_flexion': self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[0]], hand_landmarks.landmark[chain[1]], 
                hand_landmarks.landmark[chain[2]], origin, rotation_matrix,
                self.base_angles['thumb_cmc_flexion'], 'thumb_cmc'),
            'thumb_mcp_flexion': self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[1]], hand_landmarks.landmark[chain[2]], 
                hand_landmarks.landmark[chain[3]], origin, rotation_matrix,
                self.base_angles['thumb_mcp_flexion']),
            'thumb_ip_flexion': self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[2]], hand_landmarks.landmark[chain[3]], 
                hand_landmarks.landmark[chain[4]], origin, rotation_matrix,
                self.base_angles['thumb_ip_flexion']),
            'thumb_mcp_abduction': self.calculate_abduction_angle(
                points[chain[2]] - points[chain[1]], rotation_matrix, 'thumb',
                hand_landmarks, self.base_angles['thumb_mcp_abduction'])
        }
        
        return {k: self.process_angle(k, v, *self._get_angle_limits(k)) for k, v in angles.items()}

    def calculate_finger_angles(self, hand_landmarks, chain, finger_name, origin, rotation_matrix):
        points = np.array([[lm.x, lm.y, lm.z] for lm in hand_landmarks.landmark])
        
        angles = {
            f'{finger_name}_mcp_flexion': self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[0]], hand_landmarks.landmark[chain[1]], 
                hand_landmarks.landmark[chain[2]], origin, rotation_matrix,
                self.base_angles.get(f'{finger_name}_mcp_flexion', 0)),
            f'{finger_name}_pip_flexion': self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[1]], hand_landmarks.landmark[chain[2]], 
                hand_landmarks.landmark[chain[3]], origin, rotation_matrix),
            f'{finger_name}_dip_flexion': self.calculate_flexion_angle(
                hand_landmarks.landmark[chain[2]], hand_landmarks.landmark[chain[3]], 
                hand_landmarks.landmark[chain[4]], origin, rotation_matrix),
            f'{finger_name}_mcp_abduction': self.calculate_abduction_angle(
                points[chain[2]] - points[chain[1]], rotation_matrix, finger_name,
                hand_landmarks, self.base_angles.get(f'{finger_name}_mcp_abduction', 0))
        }
        
        return {k: self.process_angle(k, v, *self._get_angle_limits(k)) for k, v in angles.items()}
    
    def _get_angle_limits(self, angle_name):
        """更新角度限制范围，为外展角度添加负值范围"""
        if 'thumb' in angle_name:
            if 'cmc_flexion' in angle_name: return (0, 50)
            if 'mcp_flexion' in angle_name: return (0, 80)
            if 'ip_flexion' in angle_name: return (0, 90)
            if 'abduction' in angle_name: return (0, 70)
        else:
            if 'mcp_flexion' in angle_name: return (0, 90)
            if 'pip_flexion' in angle_name: return (0, 100)
            if 'dip_flexion' in angle_name: return (0, 90)
            if 'abduction' in angle_name: return(0, 40)
        return (0, 180)

    # def _get_angle_limits(self, angle_name):
    #     """更新角度限制范围，为外展角度添加负值范围"""
    #     if 'thumb' in angle_name:
    #         if 'cmc_flexion' in angle_name: return (0, 50)
    #         if 'mcp_flexion' in angle_name: return (0, 80)
    #         if 'ip_flexion' in angle_name: return (0, 90)
    #         if 'abduction' in angle_name: return (0, 70)
    #     else:
    #         if 'mcp_flexion' in angle_name: return (0, 90)
    #         if 'pip_flexion' in angle_name: return (0, 100)
    #         if 'dip_flexion' in angle_name: return (0, 90)
    #         if 'abduction' in angle_name: return(0, 40)
    #     return (0, 180)

    def calculate_joint_angles(self, hand_landmarks):
        try:
            origin, rotation_matrix = self.create_hand_coordinate_system(hand_landmarks)
            all_angles = {}
            
            with ThreadPoolExecutor() as executor:
                futures = []
                for finger, chain in self.finger_chains.items():
                    if finger == 'thumb':
                        futures.append(executor.submit(
                            self.calculate_thumb_angles, 
                            hand_landmarks, chain, origin, rotation_matrix))
                    else:
                        futures.append(executor.submit(
                            self.calculate_finger_angles,
                            hand_landmarks, chain, finger, origin, rotation_matrix))
                
                for future in futures:
                    all_angles.update(future.result())
                    
            return all_angles
        except Exception as e:
            print(f"计算关节角度错误: {e}")
            return {}

    def draw_global_coordinate_system(self, image, scale=30):
        h, w = image.shape[:2]
        origin = (50, h - 50)
        cv2.line(image, origin, (origin[0]+scale, origin[1]), (0,0,255), 2)
        cv2.line(image, origin, (origin[0], origin[1]-scale), (0,255,0), 2)
        cv2.line(image, origin, (origin[0]+int(scale/2), origin[1]-int(scale/2)), (255,0,0), 2)
        cv2.putText(image, 'X', (origin[0]+scale+10, origin[1]), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255), 2)
        cv2.putText(image, 'Y', (origin[0], origin[1]-scale-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        cv2.putText(image, 'Z', (origin[0]+int(scale/2)+10, origin[1]-int(scale/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,0,0), 2)
        cv2.putText(image, 'Global', (origin[0]-30, origin[1]+30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    def draw_hand_coordinate_system(self, image, origin, rotation_matrix, scale=100):
        origin_2d = (int(origin[0]*image.shape[1]), int(origin[1]*image.shape[0]))
        colors = [(0,0,255), (0,255,0), (255,0,0)]
        labels = ['X', 'Y', 'Z']
        
        cv2.circle(image, origin_2d, 3, (255,255,255), -1)
        
        for i in range(3):
            endpoint = origin + rotation_matrix[:,i] * (scale/800)
            endpoint_2d = (int(endpoint[0]*image.shape[1]), int(endpoint[1]*image.shape[0]))
            cv2.line(image, origin_2d, endpoint_2d, colors[i], 2, cv2.LINE_AA)
            label_pos = (endpoint_2d[0]+(5 if endpoint_2d[0]>origin_2d[0] else -15),
                        endpoint_2d[1]+(15 if endpoint_2d[1]>origin_2d[1] else -5))
            cv2.putText(image, labels[i], label_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[i], 2, cv2.LINE_AA)