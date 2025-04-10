import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial
import copy
class HandAngleCalculator:
    def __init__(self):
        self._init_data_structures()
        self._init_finger_chains()
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

    def _init_kalman_params(self):
        """初始化卡尔曼滤波参数"""
        self.kalman_params = {
            'thumb': {
                'cmc_flexion': {'P': 1.2, 'Q': 0.3, 'R': 1.0},
                'mcp_flexion': {'P': 1.2, 'Q': 0.25, 'R': 1.0},
                'ip_flexion': {'P': 0.9, 'Q': 0.2, 'R': 1.0},
                'mcp_abduction': {'P': 1.3, 'Q': 0.15, 'R': 1.0}
            },
            'index': {
                'mcp_flexion': {'P': 1.3, 'Q': 0.15, 'R': 1.0},
                'pip_flexion': {'P': 1.0, 'Q': 0.12, 'R': 1.0},
                'dip_flexion': {'P': 0.8, 'Q': 0.1, 'R': 1.0},
                'mcp_abduction': {'P': 1.2, 'Q': 0.08, 'R': 1.0}
            },
            'middle': {
                'mcp_flexion': {'P': 1.2, 'Q': 0.13, 'R': 1.0},
                'pip_flexion': {'P': 1.0, 'Q': 0.11, 'R': 1.0},
                'dip_flexion': {'P': 0.8, 'Q': 0.09, 'R': 1.0},
                'mcp_abduction': {'P': 1.1, 'Q': 0.07, 'R': 1.0}
            },
            'ring': {
                'mcp_flexion': {'P': 1.1, 'Q': 0.12, 'R': 1.0},
                'pip_flexion': {'P': 0.9, 'Q': 0.1, 'R': 1.0},
                'dip_flexion': {'P': 0.8, 'Q': 0.08, 'R': 1.0},
                'mcp_abduction': {'P': 1.0, 'Q': 0.06, 'R': 1.0}
            },
            'pinky': {
                'mcp_flexion': {'P': 1.0, 'Q': 0.15, 'R': 1.0},
                'pip_flexion': {'P': 0.9, 'Q': 0.12, 'R': 1.0},
                'dip_flexion': {'P': 0.8, 'Q': 0.1, 'R': 1.0},
                'mcp_abduction': {'P': 1.0, 'Q': 0.08, 'R': 1.0}
            }
        }




    def process_angle(self, angle_name, value):
        """处理角度：应用限制和卡尔曼滤波"""
        # 获取角度限制
        limits = self._get_angle_limits(angle_name)
        
        # 限制角度范围
        clipped_value = np.clip(value, limits[0], limits[1])
        
        # 应用卡尔曼滤波
        filtered_value = self.apply_kalman_filter(angle_name, clipped_value)
        
        return round(filtered_value, 2)


    def _get_angle_limits(self, angle_name):
        """获取角度限制"""
        if 'thumb' in angle_name:
            if 'cmc_flexion' in angle_name: return (-25, 60)
            if 'mcp_flexion' in angle_name: return (0, 90)
            if 'ip_flexion' in angle_name: return (0, 100)
            if 'abduction' in angle_name: return (-50, 50)
        elif 'index' in angle_name:
            if 'mcp_flexion' in angle_name: return (0, 90)
            if 'pip_flexion' in angle_name: return (0, 100)
            if 'dip_flexion' in angle_name: return (0, 80)
            if 'abduction' in angle_name: return (-20, 20)
        elif 'middle' in angle_name:
            if 'mcp_flexion' in angle_name: return (0, 90)
            if 'pip_flexion' in angle_name: return (0, 100)
            if 'dip_flexion' in angle_name: return (0, 80)
            if 'abduction' in angle_name: return (-15, 15)
        elif 'ring' in angle_name:
            if 'mcp_flexion' in angle_name: return (0, 90)
            if 'pip_flexion' in angle_name: return (0, 100)
            if 'dip_flexion' in angle_name: return (0, 80)
            if 'abduction' in angle_name: return (-10, 10)
        elif 'pinky' in angle_name:
            if 'mcp_flexion' in angle_name: return (0, 90)
            if 'pip_flexion' in angle_name: return (0, 100)
            if 'dip_flexion' in angle_name: return (0, 80)
            if 'abduction' in angle_name: return (-15, 15)
        return (0, 180)  # 默认范围
    
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

    def create_hand_coordinate_system(self, landmarks):
        """创建手部局部坐标系统，更准确处理外展角度"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        origin = points[0]  # 手腕作为原点
        
        # 手掌基本向量：掌心中心到手腕
        mcp_points = points[[5, 9, 13, 17]]  # 使用食指到小指的MCP关节
        palm_center = np.mean(mcp_points, axis=0)
        wrist_to_palm = palm_center - origin
        
        # Y轴：手腕到中指MCP的方向
        y_axis_temp = points[9] - origin
        
        # Z轴：手掌法向量 (使用叉积)
        # 使用两个向量：手腕到食指MCP，手腕到小指MCP
        v1 = points[5] - origin  # 手腕到食指MCP
        v2 = points[17] - origin  # 手腕到小指MCP
        z_axis = np.cross(v1, v2)
        
        # 确保Z轴与掌心垂直，方向朝外
        if np.dot(z_axis, y_axis_temp) > 0:
            z_axis = -z_axis
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # 修正Y轴，确保与Z轴垂直
        y_axis = wrist_to_palm - np.dot(wrist_to_palm, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # X轴：使用右手坐标系规则
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 返回原点和旋转矩阵
        return origin, np.column_stack((x_axis, y_axis, z_axis))


    def parallel_calculate_joint_angles(self, landmarks):
        """使用多线程并行计算手部关节角度，优化性能"""
        try:
            # 1. 创建手部坐标系
            origin, rotation_matrix = self.create_hand_coordinate_system(landmarks)
            
            # 2. 将关键点转换到局部坐标系
            local_points = self._transform_to_local_coordinates(landmarks, origin, rotation_matrix)
            
            # 3. 使用线程池并行计算各手指角度
            angles = {}
            
            # 并行处理各手指
            with ThreadPoolExecutor(max_workers=5) as executor:
                # 拇指角度计算任务
                thumb_future = executor.submit(
                    self._calculate_thumb_angles, 
                    local_points, rotation_matrix
                )
                
                # 其他四指角度计算任务
                finger_futures = {
                    finger: executor.submit(
                        self._calculate_finger_angles, 
                        finger, local_points, rotation_matrix
                    ) for finger in ['index', 'middle', 'ring', 'pinky']
                }
                
                # 收集结果
                angles.update(thumb_future.result())
                for finger, future in finger_futures.items():
                    angles.update(future.result())
            
            # 4. 应用关节限制和约束
            constrained_angles = self._apply_joint_constraints(angles)
            
            # 5. 应用卡尔曼滤波平滑角度
            filtered_angles = {k: self.process_angle(k, v) for k, v in constrained_angles.items()}
            
            return filtered_angles
        
        except Exception as e:
            print(f"并行计算关节角度错误: {e}")
            import traceback
            traceback.print_exc()
            return {}
        
    def _calculate_thumb_angles(self, local_points, rotation_matrix):
        """计算拇指角度的独立任务"""
        angles = {}
        
        thumb_chain = self.finger_chains['thumb']
        wrist_pos = local_points[thumb_chain[0]]
        cmc_pos = local_points[thumb_chain[1]]
        mcp_pos = local_points[thumb_chain[2]]
        ip_pos = local_points[thumb_chain[3]]
        tip_pos = local_points[thumb_chain[4]]
        
        # 计算向量
        wrist_to_cmc = cmc_pos - wrist_pos
        cmc_to_mcp = mcp_pos - cmc_pos
        mcp_to_ip = ip_pos - mcp_pos
        ip_to_tip = tip_pos - ip_pos
        
        # 计算拇指角度，传递关节类型
        angles['thumb_cmc_flexion'] = self._compute_flexion_angle(wrist_to_cmc, cmc_to_mcp, rotation_matrix, 'thumb_cmc')
        angles['thumb_cmc_abduction'] = self._compute_abduction_angle(wrist_to_cmc, cmc_to_mcp, rotation_matrix[:, 2])
        angles['thumb_mcp_flexion'] = self._compute_flexion_angle(cmc_to_mcp, mcp_to_ip, rotation_matrix, 'thumb_mcp')
        angles['thumb_mcp_abduction'] = self._compute_abduction_angle(cmc_to_mcp, mcp_to_ip, rotation_matrix[:, 2])
        angles['thumb_ip_flexion'] = self._compute_flexion_angle(mcp_to_ip, ip_to_tip, rotation_matrix, 'thumb_ip')
        
        return angles

    def _calculate_finger_angles(self, finger_name, local_points, rotation_matrix):
        """计算单个手指角度的独立任务"""
        angles = {}
        
        chain = self.finger_chains[finger_name]
        wrist_pos = local_points[chain[0]]
        mcp_pos = local_points[chain[1]]
        pip_pos = local_points[chain[2]]
        dip_pos = local_points[chain[3]]
        tip_pos = local_points[chain[4]]
        
        # 计算向量
        wrist_to_mcp = mcp_pos - wrist_pos
        mcp_to_pip = pip_pos - mcp_pos
        pip_to_dip = dip_pos - pip_pos
        dip_to_tip = tip_pos - dip_pos
        
        # 计算手指角度
        angles[f'{finger_name}_mcp_flexion'] = self._compute_flexion_angle(wrist_to_mcp, mcp_to_pip, rotation_matrix)
        angles[f'{finger_name}_mcp_abduction'] = self._compute_abduction_angle(wrist_to_mcp, mcp_to_pip, rotation_matrix[:, 2])
        angles[f'{finger_name}_pip_flexion'] = self._compute_flexion_angle(mcp_to_pip, pip_to_dip, rotation_matrix)
        angles[f'{finger_name}_dip_flexion'] = self._compute_flexion_angle(pip_to_dip, dip_to_tip, rotation_matrix)
        
        return angles


    def create_hand_coordinate_system(self, landmarks):
        """创建手部局部坐标系统，更准确处理外展角度"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        origin = points[0]  # 手腕作为原点
        
        # 手掌基本向量：掌心中心到手腕
        mcp_points = points[[5, 9, 13, 17]]  # 使用食指到小指的MCP关节
        palm_center = np.mean(mcp_points, axis=0)
        wrist_to_palm = palm_center - origin
        
        # Y轴：手腕到中指MCP的方向
        y_axis_temp = points[9] - origin
        
        # Z轴：手掌法向量 (使用叉积)
        # 使用两个向量：手腕到食指MCP，手腕到小指MCP
        v1 = points[5] - origin  # 手腕到食指MCP
        v2 = points[17] - origin  # 手腕到小指MCP
        z_axis = np.cross(v1, v2)
        
        # 确保Z轴与掌心垂直，方向朝外
        if np.dot(z_axis, y_axis_temp) > 0:
            z_axis = -z_axis
        z_axis = z_axis / np.linalg.norm(z_axis)
        
        # 修正Y轴，确保与Z轴垂直
        y_axis = wrist_to_palm - np.dot(wrist_to_palm, z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # X轴：使用右手坐标系规则
        x_axis = np.cross(y_axis, z_axis)
        x_axis = x_axis / np.linalg.norm(x_axis)
        
        # 返回原点和旋转矩阵
        return origin, np.column_stack((x_axis, y_axis, z_axis))
           
    def _transform_to_local_coordinates(self, landmarks, origin, rotation_matrix):
        """将世界坐标系中的关键点转换到手部局部坐标系"""
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        local_points = np.zeros_like(points)
        
        for i, point in enumerate(points):
            # 平移到原点
            translated = point - origin
            # 旋转到局部坐标系
            local_points[i] = rotation_matrix.T @ translated
            
        return local_points
    
    def _compute_flexion_angle(self, parent_vec, child_vec, rotation_matrix, joint_type=None):
        """优化的屈曲角度计算方法，针对拇指关节改进"""
        # 确保向量单位化
        if np.linalg.norm(parent_vec) < 1e-6 or np.linalg.norm(child_vec) < 1e-6:
            return 0.0
        
        parent_norm = parent_vec / np.linalg.norm(parent_vec)
        child_norm = child_vec / np.linalg.norm(child_vec)
        
        # 计算夹角
        dot_product = np.clip(np.dot(parent_norm, child_norm), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        
        # 其他关节保持原计算方式
        return max(0.0, angle)

    def _compute_abduction_angle(self, parent_vec, child_vec, normal_vec):
        """直接计算外展角度，无需IK"""
        # 投影到与法向量垂直的平面
        parent_proj = parent_vec - np.dot(parent_vec, normal_vec) * normal_vec
        child_proj = child_vec - np.dot(child_vec, normal_vec) * normal_vec
        
        # 确保向量非零
        if np.linalg.norm(parent_proj) < 1e-6 or np.linalg.norm(child_proj) < 1e-6:
            return 0.0
        
        # 单位化
        parent_proj = parent_proj / np.linalg.norm(parent_proj)
        child_proj = child_proj / np.linalg.norm(child_proj)
        
        # 计算平面内夹角
        dot_product = np.clip(np.dot(parent_proj, child_proj), -1.0, 1.0)
        angle = np.degrees(np.arccos(dot_product))
        
        # 确定角度符号
        cross_product = np.cross(parent_proj, child_proj)
        sign = np.sign(np.dot(cross_product, normal_vec))
        
        # 应用符号并返回
        return angle * sign
    

    def _apply_joint_constraints(self, angles):
        """改进的约束系统，针对对握动作优化"""
        constrained_angles = copy.deepcopy(angles)
        
        # 检测是否可能处于对握状态
        opposition_mode = False
        opposition_finger = None
        
        # 1. 检测对握状态 - 通过拇指和其他手指的屈曲角度判断
        if ('thumb_cmc_flexion' in angles and 
            'thumb_mcp_flexion' in angles and 
            'thumb_ip_flexion' in angles):
            
            thumb_flex_sum = abs(angles['thumb_cmc_flexion']) + angles['thumb_mcp_flexion'] + angles['thumb_ip_flexion']
            
            # 检查其他手指是否有明显屈曲
            for finger in ['index', 'middle']:
                if (f'{finger}_mcp_flexion' in angles and 
                    f'{finger}_pip_flexion' in angles):
                    
                    finger_flex = angles[f'{finger}_mcp_flexion'] + angles[f'{finger}_pip_flexion']
                    
                    # 如果拇指和手指都有适度屈曲，可能是对握
                    if thumb_flex_sum > 50 and finger_flex > 60:
                        opposition_mode = True
                        opposition_finger = finger
                        break
        
        # 2. 在对握模式下使用特殊约束
        if opposition_mode and opposition_finger:
            print(f"检测到与{opposition_finger}的对握模式")
            
            # A. 对握时拇指MCP和IP关节协调 - 关键优化点
            if 'thumb_mcp_flexion' in angles and 'thumb_ip_flexion' in angles:
                mcp_flex = angles['thumb_mcp_flexion']
                # 让IP关节有更灵活的屈曲以适应对握
                constrained_angles['thumb_ip_flexion'] = max(
                    angles['thumb_ip_flexion'],
                    min(80, mcp_flex * 0.8)  # IP屈曲至少为MCP的80%
                )
            
            # B. 对握时禁用DIP-PIP耦合约束
            # 仅保留最小限度的约束以防止不自然姿势
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if (f'{finger}_pip_flexion' in angles and 
                    f'{finger}_dip_flexion' in angles):
                    
                    pip_angle = angles[f'{finger}_pip_flexion']
                    dip_angle = angles[f'{finger}_dip_flexion']
                    
                    # 只确保DIP不小于PIP的一定比例，不硬性设定值
                    min_dip = pip_angle * 0.4
                    if dip_angle < min_dip:
                        constrained_angles[f'{finger}_dip_flexion'] = min_dip
            
            # C. 对握指间关节协调 - 确保对握手指的指尖有适当弯曲
            if f'{opposition_finger}_pip_flexion' in angles and f'{opposition_finger}_dip_flexion' in angles:
                pip_angle = angles[f'{opposition_finger}_pip_flexion']
                # 对握时允许DIP更灵活，而非严格遵循比例
                if pip_angle > 45:
                    # 对握时保持指尖适当弯曲
                    min_dip = 20
                    if angles[f'{opposition_finger}_dip_flexion'] < min_dip:
                        constrained_angles[f'{opposition_finger}_dip_flexion'] = min_dip
        
        else:
            # 3. 非对握模式下保留少量基础约束
            # 仅应用最基本的生理约束，避免不自然的姿势
            for finger in ['index', 'middle', 'ring', 'pinky']:
                if f'{finger}_pip_flexion' in angles and f'{finger}_dip_flexion' in angles:
                    pip_angle = angles[f'{finger}_pip_flexion']
                    dip_angle = angles[f'{finger}_dip_flexion']
                    
                    # 只应用最低限度的约束：DIP不应超过PIP的2倍
                    if dip_angle > pip_angle * 2:
                        constrained_angles[f'{finger}_dip_flexion'] = pip_angle * 2
        
        # 4. 外展角度联动(保留原来的)
        if 'index_mcp_abduction' in angles:
            for finger, factor in zip(['middle', 'ring', 'pinky'], [0.7, 0.5, 0.6]):
                key = f'{finger}_mcp_abduction'
                if key in angles:
                    if abs(angles[key]) < abs(angles['index_mcp_abduction'] * factor * 0.5):
                        constrained_angles[key] = angles['index_mcp_abduction'] * factor * 0.5
        
        # 5. 应用角度限制
        for key, value in constrained_angles.items():
            limits = self._get_angle_limits(key)
            constrained_angles[key] = np.clip(value, limits[0], limits[1])
        
        return constrained_angles


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









    # def fast_calculate_joint_angles(self, landmarks):
    #     """快速计算手部20个关节角度，优化性能"""
    #     try:
    #         # 1. 创建手部坐标系
    #         origin, rotation_matrix = self.create_hand_coordinate_system(landmarks)
            
    #         # 2. 将关键点转换到局部坐标系
    #         local_points = self._transform_to_local_coordinates(landmarks, origin, rotation_matrix)
            
    #         # 3. 直接使用几何方法计算角度而不是IK求解
    #         angles = {}
            
    #         # 处理拇指的5个角度 (cmc_flexion, cmc_abduction, mcp_flexion, mcp_abduction, ip_flexion)
    #         thumb_chain = self.finger_chains['thumb']
    #         wrist_pos = local_points[thumb_chain[0]]
    #         cmc_pos = local_points[thumb_chain[1]]
    #         mcp_pos = local_points[thumb_chain[2]]
    #         ip_pos = local_points[thumb_chain[3]]
    #         tip_pos = local_points[thumb_chain[4]]
            
    #         # 计算向量
    #         wrist_to_cmc = cmc_pos - wrist_pos
    #         cmc_to_mcp = mcp_pos - cmc_pos
    #         mcp_to_ip = ip_pos - mcp_pos
    #         ip_to_tip = tip_pos - ip_pos
            
    #         # 计算拇指角度，传递关节类型
    #         angles['thumb_cmc_flexion'] = self._compute_flexion_angle(wrist_to_cmc, cmc_to_mcp, rotation_matrix, 'thumb_cmc')
    #         angles['thumb_cmc_abduction'] = self._compute_abduction_angle(wrist_to_cmc, cmc_to_mcp, rotation_matrix[:, 2])
    #         angles['thumb_mcp_flexion'] = self._compute_flexion_angle(cmc_to_mcp, mcp_to_ip, rotation_matrix, 'thumb_mcp')
    #         angles['thumb_mcp_abduction'] = self._compute_abduction_angle(cmc_to_mcp, mcp_to_ip, rotation_matrix[:, 2])
    #         angles['thumb_ip_flexion'] = self._compute_flexion_angle(mcp_to_ip, ip_to_tip, rotation_matrix, 'thumb_ip')
            
    #         # 处理其他四个手指，每个3个屈曲角度和1个外展角度
    #         finger_names = ['index', 'middle', 'ring', 'pinky']
            
    #         for finger_name in finger_names:
    #             chain = self.finger_chains[finger_name]
    #             wrist_pos = local_points[chain[0]]
    #             mcp_pos = local_points[chain[1]]
    #             pip_pos = local_points[chain[2]]
    #             dip_pos = local_points[chain[3]]
    #             tip_pos = local_points[chain[4]]
                
    #             # 计算向量
    #             wrist_to_mcp = mcp_pos - wrist_pos
    #             mcp_to_pip = pip_pos - mcp_pos
    #             pip_to_dip = dip_pos - pip_pos
    #             dip_to_tip = tip_pos - dip_pos
                
    #             # 计算手指角度
    #             angles[f'{finger_name}_mcp_flexion'] = self._compute_flexion_angle(wrist_to_mcp, mcp_to_pip, rotation_matrix)
    #             angles[f'{finger_name}_mcp_abduction'] = self._compute_abduction_angle(wrist_to_mcp, mcp_to_pip, rotation_matrix[:, 2])
    #             angles[f'{finger_name}_pip_flexion'] = self._compute_flexion_angle(mcp_to_pip, pip_to_dip, rotation_matrix)
    #             angles[f'{finger_name}_dip_flexion'] = self._compute_flexion_angle(pip_to_dip, dip_to_tip, rotation_matrix)
            
    #         # 4. 应用关节限制和约束
    #         constrained_angles = self._apply_joint_constraints(angles)
            
    #         # 5. 应用卡尔曼滤波平滑角度
    #         filtered_angles = {k: self.process_angle(k, v) for k, v in constrained_angles.items()}
            
    #         return filtered_angles
        
    #     except Exception as e:
    #         print(f"快速计算关节角度错误: {e}")
    #         import traceback
    #         traceback.print_exc()
    #         return {}
