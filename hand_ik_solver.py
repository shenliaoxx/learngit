# hand_ik_solver.py
import numpy as np
import cv2
import mediapipe as mp
from scipy.optimize import minimize
from collections import deque
import copy
import time
import h5py
import json
from datetime import datetime
import argparse
import os
import scipy

class HandIKSolver:
    """使用逆运动学优化MediaPipe手部关键点生成的关节角度"""
    
    def __init__(self):
        # 初始化参数
        self._init_kalman_params()
        self._init_finger_chains()
        
        # 缓存和历史数据
        self.landmark_history = {}
        self.angle_history = {}
        self.kalman_filters = {}
        
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
        
        # IK求解参数
        self.ik_params = {
            'max_iterations': 10,
            'convergence_threshold': 0.001,
            'dip_pip_ratio': 0.67,  # DIP约为PIP的2/3
            'optimization_method': 'SLSQP'
        }
        
        
        # 指间协同运动约束
        self.finger_coupling = {
            'flexion': {
                'ring_mcp': {'middle_mcp': 0.9},  # 环指MCP屈曲约为中指的90%
                'pinky_mcp': {'ring_mcp': 0.9},   # 小指MCP屈曲约为环指的90%
                'ring_pip': {'middle_pip': 0.95}, # 环指PIP屈曲约为中指的95%
                'pinky_pip': {'ring_pip': 0.95}   # 小指PIP屈曲约为环指的95%
            },
            'abduction': {
                'middle_mcp': {'index_mcp': 0.8}, # 中指外展约为食指的80%
                'ring_mcp': {'middle_mcp': 0.7},  # 环指外展约为中指的70%
                'pinky_mcp': {'ring_mcp': 0.8}    # 小指外展约为环指的80%
            }
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


        
    def _init_finger_chains(self):
        """初始化手指关键点链"""
        self.finger_chains = {
            'thumb': [0, 1, 2, 3, 4],
            'index': [0, 5, 6, 7, 8],
            'middle': [0, 9, 10, 11, 12],
            'ring': [0, 13, 14, 15, 16],
            'pinky': [0, 17, 18, 19, 20]
        }
        
    def apply_kalman_filter(self, angle_name, value):
        """应用卡尔曼滤波"""
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
    
    # def create_hand_coordinate_system(self, landmarks):
    #     """创建手部局部坐标系统，更准确处理外展角度"""
    #     points = np.array([[lm.x, lm.y, lm.z] for lm in landmarks.landmark])
    #     origin = points[0]  # 手腕作为原点
        
    #     # 手掌基本向量：掌心中心到手腕
    #     mcp_points = points[[5, 9, 13, 17]]  # 使用食指到小指的MCP关节
    #     palm_center = np.mean(mcp_points, axis=0)
    #     wrist_to_palm = palm_center - origin
        
    #     # Y轴：手腕到中指MCP的方向
    #     y_axis_temp = points[9] - origin
        
    #     # Z轴：手掌法向量 (使用叉积)
    #     # 使用两个向量：手腕到食指MCP，手腕到小指MCP
    #     v1 = points[5] - origin  # 手腕到食指MCP
    #     v2 = points[17] - origin  # 手腕到小指MCP
    #     z_axis = np.cross(v1, v2)
        
    #     # 确保Z轴与掌心垂直，方向朝外
    #     if np.dot(z_axis, y_axis_temp) > 0:
    #         z_axis = -z_axis
    #     z_axis = z_axis / np.linalg.norm(z_axis)
        
    #     # 修正Y轴，确保与Z轴垂直
    #     y_axis = wrist_to_palm - np.dot(wrist_to_palm, z_axis) * z_axis
    #     y_axis = y_axis / np.linalg.norm(y_axis)
        
    #     # X轴：使用右手坐标系规则
    #     x_axis = np.cross(y_axis, z_axis)
    #     x_axis = x_axis / np.linalg.norm(x_axis)
        
    #     # 返回原点和旋转矩阵
    #     return origin, np.column_stack((x_axis, y_axis, z_axis))
    
    
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
    
    
    def _smooth_landmarks(self, landmarks):
        """对手部关键点进行自适应平滑处理"""
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
    

    def _get_surrounding_depth(self, depth_frame, x, y, window_size=5):
        """获取像素周围区域的平均深度值"""
        valid_depths = []
        half_window = window_size // 2
        
        for j in range(max(0, y - half_window), min(depth_frame.get_height(), y + half_window + 1)):
            for i in range(max(0, x - half_window), min(depth_frame.get_width(), x + half_window + 1)):
                depth = depth_frame.get_distance(i, j)
                if depth > 0:  # 只考虑有效深度值
                    valid_depths.append(depth)
        
        if valid_depths:
            # 使用中位数可以更好地处理异常值
            return np.median(valid_depths)
        else:
            return 0.0  # 如果周围没有有效深度，返回0
    
    def process_hand_landmarks(self, landmarks, depth_frame=None, color_image=None):
        """处理手部关键点，计算关节角度，优化版"""
        # 1. 平滑关键点
        smoothed_landmarks = self._smooth_landmarks(landmarks)
        
        # 2. 如果提供了深度数据，则使用深度增强
        if depth_frame is not None and color_image is not None:
            try:
                # 将depth_frame转换为numpy数组处理
                depth_image = np.asanyarray(depth_frame.get_data())
                enhanced_landmarks = self._enhance_landmarks_with_depth_array(smoothed_landmarks, depth_image, color_image)
            except:
                # 失败时使用原始关键点
                enhanced_landmarks = smoothed_landmarks
        else:
            enhanced_landmarks = smoothed_landmarks
        
        # 3. 使用快速方法计算关节角度
        angles = self.fast_calculate_joint_angles(enhanced_landmarks)
        
        return enhanced_landmarks, angles

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

    def fast_calculate_joint_angles(self, landmarks):
        """快速计算手部20个关节角度，优化性能"""
        try:
            # 1. 创建手部坐标系
            origin, rotation_matrix = self.create_hand_coordinate_system(landmarks)
            
            # 2. 将关键点转换到局部坐标系 - 这一步必须的，但可以优化
            local_points = self._transform_to_local_coordinates(landmarks, origin, rotation_matrix)
            
            # 3. 直接使用几何方法计算角度而不是IK求解
            angles = {}
            
            # 处理拇指的5个角度 (cmc_flexion, cmc_abduction, mcp_flexion, mcp_abduction, ip_flexion)
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
            
            # 处理其他四个手指，每个3个屈曲角度和1个外展角度
            finger_names = ['index', 'middle', 'ring', 'pinky']
            
            for finger_name in finger_names:
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
            
            # 4. 应用关节限制和约束
            constrained_angles = self._apply_joint_constraints(angles)
            
            # 5. 应用卡尔曼滤波平滑角度
            filtered_angles = {k: self.process_angle(k, v) for k, v in constrained_angles.items()}
            
            return filtered_angles
        
        except Exception as e:
            print(f"快速计算关节角度错误: {e}")
            import traceback
            traceback.print_exc()
            return {}

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

    # def _apply_joint_constraints(self, angles):
    #     """应用生物力学约束和联动规则，优化拇指约束"""
    #     constrained_angles = copy.deepcopy(angles)
        
    #     # 1. 标准DIP-PIP耦合约束(不变)
    #     for finger in ['index', 'middle', 'ring', 'pinky']:
    #         pip_angle = angles.get(f'{finger}_pip_flexion', 0)
    #         dip_angle = self.ik_params['dip_pip_ratio'] * pip_angle
    #         constrained_angles[f'{finger}_dip_flexion'] = dip_angle
        
    #     # 2. 手指协同运动约束(不变)
    #     if 'middle_mcp_flexion' in angles:
    #         if 'ring_mcp_flexion' in angles:
    #             constrained_angles['ring_mcp_flexion'] = angles['middle_mcp_flexion'] * 0.9
    #         if 'pinky_mcp_flexion' in angles:
    #             constrained_angles['pinky_mcp_flexion'] = angles['middle_mcp_flexion'] * 0.8
        
    #     # 3. 外展角度联动(不变)
    #     if 'index_mcp_abduction' in angles:
    #         for finger, factor in zip(['middle', 'ring', 'pinky'], [0.7, 0.5, 0.6]):
    #             key = f'{finger}_mcp_abduction'
    #             if key in angles:
    #                 if abs(angles[key]) < abs(angles['index_mcp_abduction'] * factor * 0.5):
    #                     constrained_angles[key] = angles['index_mcp_abduction'] * factor * 0.5
    
        
    #     for key, value in constrained_angles.items():
    #         limits = self._get_angle_limits(key)
    #         constrained_angles[key] = np.clip(value, limits[0], limits[1])
        
    #     return constrained_angles

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
        """绘制手部坐标系"""
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
    
    def visualize_ik_results(self, image, landmarks, angles):
        """可视化IK求解结果并显示外展角度信息"""
        h, w = image.shape[:2]
        
        # 强调显示外展角度
        abduction_angles = {k: v for k, v in angles.items() if 'abduction' in k}
        
        # 在图像右上角显示外展角度
        y_offset = 30
        cv2.putText(image, "外展角度:", (w - 300, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y_offset += 25
        
        for angle_name, value in abduction_angles.items():
            color = (0, 255, 255) if abs(value) > 5 else (150, 150, 150)  # 突出显示有明显外展的角度
            cv2.putText(image, f"{angle_name}: {value:.1f}", (w - 300, y_offset), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            y_offset += 20
        
        # 添加其他角度信息
        y_offset += 10
        cv2.putText(image, "其他角度:", (w - 300, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        y_offset += 20
        
        for angle_name, value in angles.items():
            if 'abduction' not in angle_name:
                cv2.putText(image, f"{angle_name}: {value:.1f}", (w - 300, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                y_offset += 20
        
        # 在图像左下角显示IK求解信息
        y_offset = h - 120
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            if finger in self.ik_debug_info['iterations']:
                cv2.putText(image, f"{finger}: it={self.ik_debug_info['iterations'][finger]}, "
                                 f"err={self.ik_debug_info['error'][finger]:.4f}", 
                           (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                y_offset += 20
        
        # 创建并绘制手部坐标系
        origin, rotation_matrix = self.create_hand_coordinate_system(landmarks)
        self.draw_hand_coordinate_system(image, origin, rotation_matrix)
        
        return image

    def save_joint_angles_to_hdf5(self, angles, file_path, fps=None):
        """将关节角度数据保存为HDF5格式，记录平均帧率信息
        
        Args:
            angles (dict): 关节角度数据字典
            file_path (str): HDF5文件保存路径
            fps (float, optional): 当前的帧率
        """
        try:
            with h5py.File(file_path, 'a') as f:
                # 获取当前时间戳
                current_time = datetime.now()
                timestamp = current_time.timestamp()
                
                # 如果是新文件，创建必要的组和数据集
                if 'hand' not in f:
                    hand_group = f.create_group('hand')
                    hand_group.create_dataset('timestamps', data=[timestamp], maxshape=(None,), dtype='float64', 
                                            chunks=True, compression="gzip", compression_opts=1)
                    
                    # 按照指定顺序定义关节角度标签
                    angle_labels = [
                        'thumb_cmc_flexion',      # THUMB_CMC_FE
                        'thumb_cmc_abduction',    # THUMB_CMC_AA
                        'thumb_mcp_flexion',      # THUMB_MCP_FE
                        'thumb_ip_flexion',       # THUMB_IP_FE
                        'index_mcp_abduction',    # INDEX_MCP_AA
                        'index_mcp_flexion',      # INDEX_MCP_FE
                        'index_pip_flexion',      # INDEX_PIP_FE
                        'index_dip_flexion',      # INDEX_DIP_FE
                        'middle_mcp_abduction',   # MIDDLE_MCP_AA
                        'middle_mcp_flexion',     # MIDDLE_MCP_FE
                        'middle_pip_flexion',     # MIDDLE_PIP_FE
                        'middle_dip_flexion',     # MIDDLE_DIP_FE
                        'ring_mcp_abduction',     # RING_MCP_AA
                        'ring_mcp_flexion',       # RING_MCP_FE
                        'ring_pip_flexion',       # RING_PIP_FE
                        'ring_dip_flexion',       # RING_DIP_FE
                        'pinky_mcp_abduction',    # PINKY_MCP_AA
                        'pinky_mcp_flexion',      # PINKY_MCP_FE
                        'pinky_pip_flexion',      # PINKY_PIP_FE
                        'pinky_dip_flexion'       # PINKY_DIP_FE
                    ]
                    
                    # 创建关节角度数据集
                    angles_data = [[angles.get(label, 0.0) for label in angle_labels]]
                    joint_angles = hand_group.create_dataset('joint_angles', 
                                                           data=angles_data,
                                                           maxshape=(None, len(angle_labels)),
                                                           dtype='float32',
                                                           chunks=True, 
                                                           compression="gzip", 
                                                           compression_opts=1)
                    
                    # 保存角度标签作为属性
                    joint_angles.attrs['angle_labels'] = angle_labels
                    
                    # 创建帧率数据集
                    fps_dataset = hand_group.create_dataset('framerate', 
                                                          data=[[timestamp, fps if fps else 0.0, fps if fps else 0.0]], 
                                                          maxshape=(None, 3),
                                                          dtype='float32')
                    fps_dataset.attrs['columns'] = ['timestamp', 'current_fps', 'average_fps']
                    
                    # 创建统计信息组
                    stats = f.create_group('stats')
                    stats.attrs.update({
                        'hand_samples': 1,
                        'start_time': str(current_time),
                        'last_update_time': str(current_time),
                        'average_fps': fps if fps else 0.0
                    })
                
                else:
                    # 追加新数据到现有数据集
                    hand_group = f['hand']
                    
                    # 获取当前数据集大小
                    current_size = hand_group['timestamps'].shape[0]
                    
                    # 计算平均帧率
                    if fps is not None:
                        if 'framerate' in hand_group:
                            # 读取前一个平均帧率
                            prev_avg_fps = hand_group['framerate'][-1, 2]
                            # 计算新的平均帧率 (累积移动平均)
                            avg_fps = (prev_avg_fps * current_size + fps) / (current_size + 1)
                        else:
                            avg_fps = fps
                    else:
                        avg_fps = 0.0
                    
                    # 扩展时间戳数据集
                    hand_group['timestamps'].resize((current_size + 1,))
                    hand_group['timestamps'][current_size] = timestamp
                    
                    # 获取角度标签
                    angle_labels = hand_group['joint_angles'].attrs['angle_labels']
                    
                    # 准备新的角度数据
                    angles_data = [angles.get(label, 0.0) for label in angle_labels]
                    
                    # 扩展并添加关节角度数据
                    hand_group['joint_angles'].resize((current_size + 1, len(angle_labels)))
                    hand_group['joint_angles'][current_size] = angles_data
                    
                    # 更新帧率信息
                    if 'framerate' in hand_group:
                        hand_group['framerate'].resize((current_size + 1, 3))
                        hand_group['framerate'][current_size] = [timestamp, fps if fps else 0.0, avg_fps]
                    else:
                        # 如果不存在帧率数据集，创建它
                        fps_dataset = hand_group.create_dataset('framerate', 
                                                              data=[[timestamp, fps if fps else 0.0, avg_fps]], 
                                                              maxshape=(None, 3),
                                                              dtype='float32')
                        fps_dataset.attrs['columns'] = ['timestamp', 'current_fps', 'average_fps']
                    
                    # 更新统计信息
                    stats = f['stats']
                    stats.attrs['hand_samples'] = current_size + 1
                    stats.attrs['last_update_time'] = str(current_time)
                    stats.attrs['average_fps'] = avg_fps
                    
                    # 每100帧刷新文件以确保数据及时写入
                    if current_size % 100 == 0:
                        f.flush()
                
            return True
                
        except Exception as e:
            print(f"保存HDF5文件错误: {e}")
            import traceback
            traceback.print_exc()
            return False

    def interpolate_joint_angles_to_200hz(self, file_path):
        """将HDF5文件中的关节角度数据插值到200Hz
        
        Args:
            file_path (str): HDF5文件路径
            
        Returns:
            dict: 包含插值后的时间戳和关节角度数据的字典
        """
        try:
            # 读取原始数据
            data = self.read_joint_angles_from_hdf5(file_path)
            if data is None:
                print("无法读取数据文件")
                return None
                
            # 获取原始时间戳和关节角度数据
            original_timestamps = data['timestamps']
            original_angles = data['joint_angles']
            angle_labels = data['angle_labels']
            
            # 计算目标时间戳（200Hz采样）
            start_time = original_timestamps[0]
            end_time = original_timestamps[-1]
            target_dt = 1/200.0  # 200Hz的时间间隔
            target_timestamps = np.arange(start_time, end_time, target_dt)
            
            # 为每个关节角度进行插值
            interpolated_angles = np.zeros((len(target_timestamps), len(angle_labels)))
            
            for i in range(original_angles.shape[1]):
                # 使用三次样条插值，保持平滑性
                interpolator = scipy.interpolate.CubicSpline(original_timestamps, original_angles[:, i])
                interpolated_angles[:, i] = interpolator(target_timestamps)
                
                # 确保插值结果在有效范围内
                for j, label in enumerate(angle_labels):
                    limits = self._get_angle_limits(label)
                    interpolated_angles[:, j] = np.clip(interpolated_angles[:, j], limits[0], limits[1])
            
            # 保存插值后的数据到新文件
            output_file = file_path.replace('.h5', '_200hz.h5')
            with h5py.File(output_file, 'w') as f:
                # 创建手部数据组
                hand_group = f.create_group('hand')
                
                # 保存插值后的时间戳
                hand_group.create_dataset('timestamps', data=target_timestamps,
                                        compression="gzip", compression_opts=1)
                
                # 保存插值后的关节角度
                joint_angles = hand_group.create_dataset('joint_angles', 
                                                    data=interpolated_angles,
                                                    compression="gzip", 
                                                    compression_opts=1)
                
                # 保存角度标签
                joint_angles.attrs['angle_labels'] = angle_labels
                
                # 更新统计信息
                stats = f.create_group('stats')
                stats.attrs.update({
                    'hand_samples': len(target_timestamps),
                    'original_samples': len(original_timestamps),
                    'sampling_rate': 200,
                    'interpolation_method': 'cubic_spline',
                    'start_time': data['stats']['start_time'],
                    'last_update_time': str(datetime.now())
                })
            
            print(f"数据已成功插值并保存到: {output_file}")
            print(f"原始样本数: {len(original_timestamps)}")
            print(f"插值后样本数: {len(target_timestamps)}")
            print(f"采样率: 200 Hz")
            
            return {
                'timestamps': target_timestamps,
                'joint_angles': interpolated_angles,
                'angle_labels': angle_labels,
                'file_path': output_file
            }
            
        except Exception as e:
            print(f"插值处理错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def read_joint_angles_from_hdf5(self, file_path, start_idx=None, end_idx=None):
        """从HDF5文件读取关节角度数据
        
        Args:
            file_path (str): HDF5文件路径
            start_idx (int, optional): 起始索引
            end_idx (int, optional): 结束索引
            
        Returns:
            dict: 包含时间戳和关节角度数据的字典，如果读取失败则返回None
        """
        try:
            # 1. 文件验证
            if not os.path.exists(file_path):
                print(f"错误：文件 {file_path} 不存在")
                return None
            
            with h5py.File(file_path, 'r') as f:
                # 2. 数据结构验证
                if 'hand' not in f:
                    print("错误：文件中没有手部数据组")
                    return None
                
                hand_group = f['hand']
                required_datasets = ['timestamps', 'joint_angles']
                for dataset in required_datasets:
                    if dataset not in hand_group:
                        print(f"错误：缺少必要的数据集 '{dataset}'")
                        return None
                
                # 3. 获取数据范围
                total_samples = hand_group['timestamps'].shape[0]
                if total_samples == 0:
                    print("错误：数据集为空")
                    return None
                
                start_idx = 0 if start_idx is None else max(0, min(start_idx, total_samples-1))
                end_idx = total_samples if end_idx is None else min(end_idx, total_samples)
                
                if start_idx >= end_idx:
                    print("错误：无效的索引范围")
                    return None
                
                # 4. 读取数据
                try:
                    timestamps = hand_group['timestamps'][start_idx:end_idx]
                    joint_angles = hand_group['joint_angles'][start_idx:end_idx]
                    angle_labels = hand_group['joint_angles'].attrs['angle_labels']
                    
                    # 5. 读取统计信息
                    stats = {}
                    if 'stats' in f:
                        stats_group = f['stats']
                        for key in stats_group.attrs.keys():
                            stats[key] = stats_group.attrs[key]
                        
                    # 6. 读取帧率信息
                    framerate_data = None
                    if 'framerate' in hand_group:
                        framerate_data = {
                            'data': hand_group['framerate'][start_idx:end_idx],
                            'columns': hand_group['framerate'].attrs['columns']
                        }
                    
                    # 7. 数据验证
                    if len(timestamps) != len(joint_angles):
                        print("错误：时间戳和关节角度数据长度不匹配")
                        return None
                    
                    # 8. 返回完整的数据字典
                    return {
                        'timestamps': timestamps,
                        'joint_angles': joint_angles,
                        'angle_labels': angle_labels,
                        'stats': stats,
                        'framerate': framerate_data,
                        'data_range': {
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'total_samples': total_samples
                        }
                    }
                    
                except Exception as e:
                    print(f"读取数据集时发生错误: {e}")
                    return None
                
        except Exception as e:
            print(f"读取HDF5文件错误: {e}")
            import traceback
            traceback.print_exc()
            return None

    def read_joint_angles_from_hdf5_chunked(self, file_path, chunk_size=1000):
        """以分块方式从HDF5文件读取关节角度数据，用于处理大文件
        
        Args:
            file_path (str): HDF5文件路径
            chunk_size (int): 每次读取的数据块大小
            
        Returns:
            generator: 生成包含时间戳和关节角度数据的字典
        """
        try:
            with h5py.File(file_path, 'r') as f:
                if 'hand' not in f:
                    print("错误：文件中没有手部数据组")
                    return
                
                hand_group = f['hand']
                total_samples = hand_group['timestamps'].shape[0]
                
                # 获取基本信息
                angle_labels = hand_group['joint_angles'].attrs['angle_labels']
                
                # 分块读取数据
                for start_idx in range(0, total_samples, chunk_size):
                    end_idx = min(start_idx + chunk_size, total_samples)
                    
                    chunk_data = {
                        'timestamps': hand_group['timestamps'][start_idx:end_idx],
                        'joint_angles': hand_group['joint_angles'][start_idx:end_idx],
                        'angle_labels': angle_labels,
                        'chunk_info': {
                            'start_idx': start_idx,
                            'end_idx': end_idx,
                            'total_samples': total_samples,
                            'progress': f"{end_idx}/{total_samples} ({end_idx/total_samples*100:.1f}%)"
                        }
                    }
                    
                    yield chunk_data
                
        except Exception as e:
            print(f"分块读取HDF5文件错误: {e}")
            import traceback
            traceback.print_exc()
            return None


class MediaPipeHandIK:
    """MediaPipe手部跟踪与IK逆运动学集成"""
    
    def __init__(self, high_fps=False):
        # 初始化MediaPipe手部跟踪
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        
        # 针对高帧率场景的配置
        if high_fps:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,  # 使用轻量级模型提高处理速度
                min_detection_confidence=0.7,  # 降低检测置信度提高帧率
                min_tracking_confidence=0.7   # 降低跟踪置信度提高帧率
            )
        else:
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=1,
                model_complexity=1,
                min_detection_confidence=0.7,
                min_tracking_confidence=0.7
            )
        
        # 初始化IK求解器
        self.ik_solver = HandIKSolver()
        
        # 统计信息
        self.frame_count = 0
        self.total_frames = 0
        self.current_fps = 0
        self.last_fps_update = time.time()
        self.fps_update_interval = 0.5  # 每0.5秒更新一次帧率，更快响应
        self.fps_history = deque(maxlen=30)  # 记录最近30个FPS值计算平均帧率
    
    def _update_fps(self, current_time):
        """更新帧率统计"""
        self.frame_count += 1
        self.total_frames += 1
        
        if current_time - self.last_fps_update >= self.fps_update_interval:
            self.current_fps = self.frame_count / (current_time - self.last_fps_update)
            self.frame_count = 0
            self.last_fps_update = current_time
    
    def process_image(self, image, depth_frame=None, save_data=False, save_path=None):
        """处理图像，检测手部关键点并计算关节角度，优化版
        
        Args:
            image: 输入图像
            depth_frame: 深度图像帧（可选）
            save_data (bool): 是否保存关节角度数据
            save_path (str): HDF5文件保存路径
        """
        # 更新帧率
        self._update_fps(time.time())
        
        # 图像预处理
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # MediaPipe处理
        results = self.hands.process(rgb_image)
        
        # 创建结果副本
        output_image = image.copy()
        angles = None
        
        # 处理检测结果
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # 使用优化后的处理方法
                enhanced_landmarks, angles = self.ik_solver.process_hand_landmarks(
                    hand_landmarks, depth_frame, image
                )
                
                # 如果需要保存数据
                if save_data and angles and save_path:
                    self.ik_solver.save_joint_angles_to_hdf5(angles, save_path)
                
                # 绘制手部关键点
                self.mp_drawing.draw_landmarks(
                    output_image,
                    enhanced_landmarks,
                    self.mp_hands.HAND_CONNECTIONS,
                    self.mp_drawing_styles.get_default_hand_landmarks_style(),
                    self.mp_drawing_styles.get_default_hand_connections_style()
                )
                
        # 显示帧率
        cv2.putText(output_image, f"FPS: {self.current_fps:.1f}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        return output_image, angles







def demo_with_realsense(high_fps=True):
    """使用RealSense相机演示MediaPipe+IK"""
    import pyrealsense2 as rs
    from datetime import datetime
    import os
    
    # 创建保存目录
    save_dir = "hand_joint_data"
    os.makedirs(save_dir, exist_ok=True)
    
    # 创建HDF5文件名（使用日期和时间）
    save_path = os.path.join(save_dir, f"hand_joint_angles_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
    
    # 初始化RealSense
    pipeline = rs.pipeline()
    config = rs.config()
    
    # 启用深度流和彩色流，提高帧率
    if high_fps:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)  # 提高到60FPS
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)   # 提高到60FPS
    else:
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    
    # 创建对齐对象
    align = rs.align(rs.stream.color)
    
    # 创建更轻量级的深度处理滤波器，提高性能
    spatial_filter = rs.spatial_filter()
    if high_fps:
        # 高帧率模式下使用更轻量级的滤波设置
        spatial_filter.set_option(rs.option.filter_magnitude, 1)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.25)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 10)
        use_temporal_filter = False  # 高帧率模式下禁用时间滤波
    else:
        spatial_filter.set_option(rs.option.filter_magnitude, 2)
        spatial_filter.set_option(rs.option.filter_smooth_alpha, 0.5)
        spatial_filter.set_option(rs.option.filter_smooth_delta, 20)
        use_temporal_filter = True
    
    # 只在标准模式下使用时间滤波
    if use_temporal_filter:
        temporal_filter = rs.temporal_filter()
        hole_filling_filter = rs.hole_filling_filter()
    
    # 启动相机
    profile = pipeline.start(config)
    
    # 设置相机参数以提高帧率
    device = profile.get_device()
    depth_sensor = device.first_depth_sensor()
    
    # 设置低延迟模式和自动曝光模式
    if depth_sensor.supports(rs.option.enable_auto_exposure):
        depth_sensor.set_option(rs.option.enable_auto_exposure, 1)
    
    # 初始化MediaPipe+IK
    hand_ik = MediaPipeHandIK(high_fps=high_fps)
    
    # 性能统计
    start_time = time.time()
    frame_counter = 0
    
    try:
        while True:
            # 计时
            frame_start = time.time()
            
            # 获取帧
            frames = pipeline.wait_for_frames()
            
            # 对齐深度帧和彩色帧
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            # 应用深度滤波（根据高帧率模式决定使用哪些滤波）
            if depth_frame:
                filtered_depth = spatial_filter.process(depth_frame)
                if use_temporal_filter:
                    filtered_depth = temporal_filter.process(filtered_depth)
                    filtered_depth = hole_filling_filter.process(filtered_depth)
                depth_frame = filtered_depth
            
            if not depth_frame or not color_frame:
                continue
            
            # 转换为NumPy数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 处理图像并保存数据
            output_image, angles = hand_ik.process_image(
                color_image, 
                depth_frame,
                save_data=True,
                save_path=save_path
            )
            
            # 更新性能统计
            frame_counter += 1
            elapsed = time.time() - start_time
            avg_fps = frame_counter / elapsed if elapsed > 0 else 0
            
            # 显示性能信息
            if output_image is not None:
                cv2.putText(output_image, f"总平均FPS: {avg_fps:.1f}", (10, 90), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(output_image, f"帧处理时间: {(time.time()-frame_start)*1000:.1f}ms", (10, 120), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # 显示结果
                cv2.imshow('MediaPipe Hand IK with RealSense', output_image)
            
            # 按ESC退出
            if cv2.waitKey(1) & 0xFF == 27:
                break
                
    finally:
        # 释放资源
        hand_ik.close()
        pipeline.stop()
        cv2.destroyAllWindows()
        
        # 显示最终性能统计
        elapsed = time.time() - start_time
        print(f"\n性能统计:")
        print(f"总运行时间: {elapsed:.2f}秒")
        print(f"处理帧数: {frame_counter}")
        print(f"平均帧率: {frame_counter/elapsed:.2f} FPS")


if __name__ == "__main__":

    # # 直接设置高帧率模式
    high_fps = True  # 或者设置为 False 以禁用高帧率模式

 
    print("检测到RealSense相机，使用RealSense模式")
    demo_with_realsense(high_fps=high_fps)

    # # 创建HandIKSolver实例
    # solver = HandIKSolver()

    # # 读取并插值数据
    # file_path = "E:\multimodel-acquisition\hand_joint_data\hand_joint_angles_20250409_213732.h5"
    # interpolated_data = solver.interpolate_joint_angles_to_200hz(file_path)

