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
                # IK求解参数
        self.ik_params = {
            'max_iterations': 10,
            'convergence_threshold': 0.001,
            'dip_pip_ratio': 0.67,  # DIP约为PIP的2/3
            'optimization_method': 'SLSQP'
        }




    def process_angle(self, angle_name, value):
        """处理角度：应用限制和卡尔曼滤波"""
        # 获取角度限制
        limits = self._get_angle_limits(angle_name)
        
        # 限制角度范围
        clipped_value = np.clip(value, limits[0], limits[1])
        
        # 应用卡尔曼滤波
        filtered_value = self.apply_kalman_filter(angle_name, clipped_value)
        
        return filtered_value


    def _get_angle_limits(self, angle_name):
        """获取角度限制"""
        if 'thumb' in angle_name:
            if 'cmc_flexion' in angle_name: return (-60, 90)
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
        

    def _calculate_thumb_cmc_angles(self, local_points, rotation_matrix):
        """计算拇指CMC关节的屈曲和外展角度
        
        使用简化的几何方法:
        1. 屈曲角度: 反向手腕到CMC向量与投影到手掌平面的CMC到MCP向量之间的夹角
        2. 外展角度: CMC到MCP向量与手掌平面法向量的夹角
        
        Args:
            local_points: 手部关键点的局部坐标
            rotation_matrix: 手部坐标系的旋转矩阵
        
        Returns:
            tuple: (cmc_flexion, cmc_abduction)
        """
        # 1. 提取关键点
        wrist = local_points[0]  # 手腕
        thumb_cmc = local_points[1]  # 拇指CMC
        thumb_mcp = local_points[2]  # 拇指MCP
        index_mcp = local_points[5]  # 食指MCP
        middle_mcp = local_points[9]    # 中指MCP
        pinky_mcp = local_points[17]  # 小指MCP
        
        # 2. 计算手掌平面法向量 (使用手腕到食指和手腕到小指的叉积)
        v1 = index_mcp - wrist
        v2 = pinky_mcp - wrist
        n = np.cross(v1, v2)
        n_normalized = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-6 else np.zeros(3)
        
        # 3. 计算向量a(手腕到CMC)和向量b(CMC到MCP)
        a = thumb_cmc - wrist
        b = thumb_mcp - thumb_cmc
        
        # 4. 将向量b投影到手掌平面
        b_proj_plane = b - np.dot(b, n_normalized) * n_normalized
        
        reference = middle_mcp - wrist
        reference_proj = reference - np.dot(reference, n_normalized) * n_normalized


        # 5. 计算屈曲角度 (反向向量a与投影后的b的夹角)
        flexion_angle = 0.0
        if np.linalg.norm(reference_proj) > 1e-6 and np.linalg.norm(b_proj_plane) > 1e-6:
            # 归一化向量
            reference_proj_norm = reference_proj / np.linalg.norm(reference_proj)
            b_proj_plane_norm = b_proj_plane / np.linalg.norm(b_proj_plane)
            
            # 计算夹角
            cos_theta = np.dot(reference_proj_norm, b_proj_plane_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在[-1, 1]范围内
            flexion_angle = np.degrees(np.arccos(cos_theta))

            flex_direction = -np.sign(np.dot(np.cross(reference_proj_norm, b_proj_plane_norm), n_normalized))
            flexion_angle *= flex_direction
        
        # 6. 计算外展角度 (向量b与法向量n的夹角的余弦值)
        abduction_angle = 0.0
        if np.linalg.norm(b) > 1e-6:
            # 计算b与法向量的夹角余弦值 (90°减去夹角)
            b_normalized = b / np.linalg.norm(b)
            sin_phi = np.dot(b_normalized, n_normalized)
            sin_phi = np.clip(sin_phi, -1.0, 1.0)  # 确保在[0, 1]范围内
            abduction_angle = np.degrees(np.arcsin(sin_phi))
        
        # 7. 调试信息
        print(f"屈曲角度: {flexion_angle:.2f}, 外展角度: {abduction_angle:.2f}")
        
        # 8. 应用生理学角度限制
        # flexion_angle = np.clip(flexion_angle, 0.0, 90)  # 屈曲角度通常为0-90度
        # abduction_angle = np.clip(abduction_angle, 0.0, 70.0)  # 外展角度通常为0-70度
        
        return flexion_angle, abduction_angle
    




          
    def _calculate_thumb_angles(self, local_points, rotation_matrix):
        """计算拇指所有关节角度"""
        angles = {}
        
        # 1. 使用新方法计算CMC关节角度
        flex_angle, abd_angle = self._calculate_thumb_cmc_angles(
            local_points,
            rotation_matrix
        )
        angles['thumb_cmc_flexion'] = flex_angle
        angles['thumb_cmc_abduction'] = abd_angle
        
        # 2. 计算其他关节角度（保持原有方法）
        thumb_chain = self.finger_chains['thumb']
        cmc_pos = local_points[thumb_chain[1]]
        mcp_pos = local_points[thumb_chain[2]]
        ip_pos = local_points[thumb_chain[3]]
        tip_pos = local_points[thumb_chain[4]]
        
        cmc_to_mcp = mcp_pos - cmc_pos
        mcp_to_ip = ip_pos - mcp_pos
        ip_to_tip = tip_pos - ip_pos
        
        angles['thumb_mcp_flexion'] = self._compute_flexion_angle(
            cmc_to_mcp, mcp_to_ip, rotation_matrix, 'thumb_mcp'
        )
        angles['thumb_mcp_abduction'] = self._compute_abduction_angle(
            cmc_to_mcp, mcp_to_ip, rotation_matrix[:, 2]
        )
        angles['thumb_ip_flexion'] = self._compute_flexion_angle(
            mcp_to_ip, ip_to_tip, rotation_matrix, 'thumb_ip'
        )
        
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
        """创建手部局部坐标系统，更准确稳定地处理外展角度
        
        参数:
            landmarks: MediaPipe手部关键点结果
            
        返回:
            origin: 坐标系原点
            rotation_matrix: 旋转矩阵，列向量分别为x轴、y轴、z轴
        """
        # 转换关键点为numpy数组
        points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
        
        # 步骤1: 使用手腕作为原点
        origin = points[0]
        
        # 步骤2: 计算手掌中心 (使用四指MCP关节，排除拇指)
        mcp_indices = [5, 9, 13, 17]  # 食指、中指、无名指、小指的MCP关节
        palm_center = np.mean(points[mcp_indices], axis=0)
        
        # 步骤3: 计算Y轴 - 从手腕指向手掌中心
        y_axis_raw = palm_center - origin
        
        # 步骤4: 计算手掌平面和法向量 (Z轴)
        # 选择手腕点和MCP点构成稳定的手掌平面
        palm_plane_points = np.vstack([points[0:1], points[mcp_indices]])
        palm_points_centered = palm_plane_points - np.mean(palm_plane_points, axis=0)
        
        # 使用SVD计算平面法向量
        _, _, vh = np.linalg.svd(palm_points_centered)
        palm_normal = vh[2]  # 最小奇异值对应的向量(第三个)
        
        # 确保法向量朝向掌心向外
        # 计算两个向量分别从手腕到食指MCP和小指MCP
        v1 = points[5] - origin  # 手腕到食指MCP
        v2 = points[17] - origin  # 手腕到小指MCP
        
        # 使用叉积计算掌心法向量
        palm_cross = np.cross(v1, v2)
        
        # 确保SVD计算的法向量与叉积方向一致（掌心向外）
        if np.dot(palm_normal, palm_cross) < 0:
            palm_normal = -palm_normal
        
        # 步骤5: 应用格拉姆-施密特正交化，确保坐标轴相互正交
        # 首先将palm_normal作为z轴
        z_axis = palm_normal / np.linalg.norm(palm_normal)
        
        # 将y_axis_raw与z_axis正交化
        y_axis = y_axis_raw - (y_axis_raw @ z_axis) * z_axis
        y_axis = y_axis / np.linalg.norm(y_axis)
        
        # 使用叉积计算x轴，保证右手坐标系
        x_axis = np.cross(y_axis, z_axis)
        
        # 构建旋转矩阵 (列向量分别为x、y、z轴)
        rotation_matrix = np.column_stack((x_axis, y_axis, z_axis))
        
        # 步骤6: 验证坐标系合法性并修正
        det = np.linalg.det(rotation_matrix)
        if not 0.99 < abs(det) < 1.01:
            # 如果不是标准正交基，使用SVD重建
            u, _, vh = np.linalg.svd(rotation_matrix)
            rotation_matrix = u @ vh
        
        return origin, rotation_matrix
    


           
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
        """直接计算外展角度"""
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
        """应用生物力学约束和联动规则，优化拇指约束"""
        constrained_angles = copy.deepcopy(angles)
        
        # # 1. 标准DIP-PIP耦合约束(不变)
        # for finger in ['index', 'middle', 'ring', 'pinky']:
        #     pip_angle = angles.get(f'{finger}_pip_flexion', 0)
        #     dip_angle = self.ik_params['dip_pip_ratio'] * pip_angle
        #     constrained_angles[f'{finger}_dip_flexion'] = dip_angle
        
        # # 2. 手指协同运动约束(不变)
        # if 'middle_mcp_flexion' in angles:
        #     if 'ring_mcp_flexion' in angles:
        #         constrained_angles['ring_mcp_flexion'] = angles['middle_mcp_flexion'] * 0.9
        #     if 'pinky_mcp_flexion' in angles:
        #         constrained_angles['pinky_mcp_flexion'] = angles['middle_mcp_flexion'] * 0.8
        
        # 3. 外展角度联动(不变)
        if 'index_mcp_abduction' in angles:
            for finger, factor in zip(['middle', 'ring', 'pinky'], [0.7, 0.5, 0.6]):
                key = f'{finger}_mcp_abduction'
                if key in angles:
                    if abs(angles[key]) < abs(angles['index_mcp_abduction'] * factor * 0.5):
                        constrained_angles[key] = angles['index_mcp_abduction'] * factor * 0.5
    
        
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





    # def _calculate_thumb_cmc_angles(self, local_points, rotation_matrix):
    #     """计算拇指CMC关节的屈曲和外展角度
        
    #     使用简化的几何方法:
    #     1. 屈曲角度: 反向手腕到CMC向量与投影到手掌平面的CMC到MCP向量之间的夹角
    #     2. 外展角度: CMC到MCP向量与手掌平面法向量的夹角
        
    #     Args:
    #         local_points: 手部关键点的局部坐标
    #         rotation_matrix: 手部坐标系的旋转矩阵
        
    #     Returns:
    #         tuple: (cmc_flexion, cmc_abduction)
    #     """
    #     # 1. 提取关键点
    #     wrist = local_points[0]  # 手腕
    #     thumb_cmc = local_points[1]  # 拇指CMC
    #     thumb_mcp = local_points[2]  # 拇指MCP
    #     index_mcp = local_points[5]  # 食指MCP
    #     pinky_mcp = local_points[17]  # 小指MCP
        
    #     # 2. 计算手掌平面法向量 (使用手腕到食指和手腕到小指的叉积)
    #     v1 = index_mcp - wrist
    #     v2 = pinky_mcp - wrist
    #     n = np.cross(v1, v2)
    #     n_normalized = n / np.linalg.norm(n) if np.linalg.norm(n) > 1e-6 else np.zeros(3)
        
    #     # 3. 计算向量a(手腕到CMC)和向量b(CMC到MCP)
    #     a = thumb_cmc - wrist
    #     b = thumb_mcp - thumb_cmc
        
    #     # 4. 将向量b投影到手掌平面
    #     b_proj_plane = b - np.dot(b, n_normalized) * n_normalized
        
    #     # 5. 计算屈曲角度 (反向向量a与投影后的b的夹角)
    #     a_reversed = -a  # 反向手腕到CMC的向量
    #     flexion_angle = 0.0
    #     if np.linalg.norm(a_reversed) > 1e-6 and np.linalg.norm(b_proj_plane) > 1e-6:
    #         # 归一化向量
    #         a_reversed_norm = a_reversed / np.linalg.norm(a_reversed)
    #         b_proj_plane_norm = b_proj_plane / np.linalg.norm(b_proj_plane)
            
    #         # 计算夹角
    #         cos_theta = np.dot(a_reversed_norm, b_proj_plane_norm)
    #         cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 确保在[-1, 1]范围内
    #         flexion_angle = np.degrees(np.arccos(cos_theta))
        
    #     # 6. 计算外展角度 (向量b与法向量n的夹角的余弦值)
    #     abduction_angle = 0.0
    #     if np.linalg.norm(b) > 1e-6:
    #         # 计算b与法向量的夹角余弦值 (90°减去夹角)
    #         b_normalized = b / np.linalg.norm(b)
    #         sin_phi = np.dot(b_normalized, n_normalized)
    #         sin_phi = np.clip(sin_phi, -1.0, 1.0)  # 确保在[0, 1]范围内
    #         abduction_angle = np.degrees(np.arcsin(sin_phi))
        
    #     # 7. 调试信息
    #     print(f"屈曲角度: {flexion_angle:.2f}, 外展角度: {abduction_angle:.2f}")
        
    #     # 8. 应用生理学角度限制
    #     flexion_angle = np.clip(flexion_angle, 0.0, 90)  # 屈曲角度通常为0-90度
    #     abduction_angle = np.clip(abduction_angle, 0.0, 70.0)  # 外展角度通常为0-70度
        
    #     return flexion_angle, abduction_angle
    




          
    # def _calculate_thumb_angles(self, local_points, rotation_matrix):
    #     """计算拇指所有关节角度"""
    #     angles = {}
        
    #     # 1. 使用新方法计算CMC关节角度
    #     flex_angle, abd_angle = self._calculate_thumb_cmc_angles(
    #         local_points,
    #         rotation_matrix
    #     )
    #     angles['thumb_cmc_flexion'] = flex_angle
    #     angles['thumb_cmc_abduction'] = abd_angle
        
    #     # 2. 计算其他关节角度（保持原有方法）
    #     thumb_chain = self.finger_chains['thumb']
    #     cmc_pos = local_points[thumb_chain[1]]
    #     mcp_pos = local_points[thumb_chain[2]]
    #     ip_pos = local_points[thumb_chain[3]]
    #     tip_pos = local_points[thumb_chain[4]]
        
    #     cmc_to_mcp = mcp_pos - cmc_pos
    #     mcp_to_ip = ip_pos - mcp_pos
    #     ip_to_tip = tip_pos - ip_pos
        
    #     angles['thumb_mcp_flexion'] = self._compute_flexion_angle(
    #         cmc_to_mcp, mcp_to_ip, rotation_matrix, 'thumb_mcp'
    #     )
    #     angles['thumb_mcp_abduction'] = self._compute_abduction_angle(
    #         cmc_to_mcp, mcp_to_ip, rotation_matrix[:, 2]
    #     )
    #     angles['thumb_ip_flexion'] = self._compute_flexion_angle(
    #         mcp_to_ip, ip_to_tip, rotation_matrix, 'thumb_ip'
    #     )
        
    #     return angles
