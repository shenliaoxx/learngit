import numpy as np
import matplotlib.pyplot as plt
from data_processing.hand_angles_improve import HandAngleCalculator
import time
import matplotlib.pyplot as plt
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

class AdvancedAngleFilterTester:
    def __init__(self):
        # 为不同手指定义不同的参数配置
        self.finger_configs = {
            'thumb': {
                'static': {'P': 1.0, 'Q': 0.2, 'R': 1.5, 'history_length': 5},
                'slow': {'P': 1.0, 'Q': 0.3, 'R': 1.2, 'history_length': 4},
                'fast': {'P': 1.0, 'Q': 0.4, 'R': 0.8, 'history_length': 3}
            },
            'index': {
                'static': {'P': 1.0, 'Q': 0.1, 'R': 1.2, 'history_length': 5},
                'slow': {'P': 1.0, 'Q': 0.15, 'R': 1.0, 'history_length': 4},
                'fast': {'P': 1.0, 'Q': 0.25, 'R': 0.8, 'history_length': 3}
            },
            'middle': {
                'static': {'P': 1.0, 'Q': 0.1, 'R': 1.2, 'history_length': 5},
                'slow': {'P': 1.0, 'Q': 0.15, 'R': 1.0, 'history_length': 4},
                'fast': {'P': 1.0, 'Q': 0.25, 'R': 0.8, 'history_length': 3}
            },
            'ring': {
                'static': {'P': 1.0, 'Q': 0.08, 'R': 1.2, 'history_length': 6},
                'slow': {'P': 1.0, 'Q': 0.12, 'R': 1.0, 'history_length': 5},
                'fast': {'P': 1.0, 'Q': 0.2, 'R': 0.8, 'history_length': 4}
            },
            'pinky': {
                'static': {'P': 1.0, 'Q': 0.05, 'R': 1.5, 'history_length': 6},
                'slow': {'P': 1.0, 'Q': 0.08, 'R': 1.2, 'history_length': 5},
                'fast': {'P': 1.0, 'Q': 0.15, 'R': 1.0, 'history_length': 4}
            }
        }

    def generate_motion_data(self, scenario='static', num_points=200):
        """生成不同场景的测试数据"""
        t = np.linspace(0, 10, num_points)
        
        if scenario == 'static':
            # 静止场景：小幅度随机抖动
            true_angles = 45 + np.random.normal(0, 1, num_points)
            noise_std = 2.0
            
        elif scenario == 'slow':
            # 低速运动：缓慢的正弦运动
            true_angles = 45 + 30 * np.sin(0.5 * t) + 10 * np.sin(0.2 * t)
            noise_std = 3.0
            
        else:  # fast
            # 高速运动：快速的正弦运动加上突变
            true_angles = 45 + 30 * np.sin(2 * t) + 15 * np.sin(4 * t)
            # 添加突变
            sudden_changes = np.random.randint(0, num_points, 5)
            for idx in sudden_changes:
                true_angles[idx:idx+10] += 20
            noise_std = 5.0

        # 添加噪声
        noisy_angles = true_angles + np.random.normal(0, noise_std, num_points)
        return t, true_angles, noisy_angles

    def test_finger_config(self, finger_name, scenario, noisy_angles):
        """测试特定手指在特定场景下的滤波效果"""
        config = self.finger_configs[finger_name][scenario]
        calculator = HandAngleCalculator()
        
        # 设置参数
        calculator.kalman_filters = {}
        calculator.history_length = config['history_length']
        
        filtered_angles = []
        processing_times = []
        
        for angle in noisy_angles:
            start_time = time.time()
            filtered = calculator.smooth_angle(f'{finger_name}_test', angle)
            processing_times.append(time.time() - start_time)
            filtered_angles.append(filtered)
            
            # 更新卡尔曼滤波器参数
            if f'{finger_name}_test' in calculator.kalman_filters:
                calculator.kalman_filters[f'{finger_name}_test'].update({
                    'P': config['P'],
                    'Q': config['Q'],
                    'R': config['R']
                })
        
        return np.array(filtered_angles), np.mean(processing_times)

    def calculate_metrics(self, true_angles, filtered_angles):
        """计算性能指标"""
        mse = np.mean((true_angles - filtered_angles)**2)
        max_error = np.max(np.abs(true_angles - filtered_angles))
        smoothness = np.mean(np.abs(np.diff(filtered_angles)))
        delay = np.correlate(filtered_angles - np.mean(filtered_angles),
                           true_angles - np.mean(true_angles))[0]
        return {
            'MSE': mse,
            '最大误差': max_error,
            '平滑度': smoothness,
            '延迟': delay
        }

    def visualize_scenario(self, scenario, t, true_angles, noisy_angles, results):
        """可视化特定场景下所有手指的滤波效果"""
        plt.figure(figsize=(15, 10))
        plt.subplot(211)
        
        plt.plot(t, true_angles, 'k-', label='真实角度', alpha=0.5)
        plt.plot(t, noisy_angles, 'gray', label='噪声数据', alpha=0.3)
        
        colors = {'thumb': 'r', 'index': 'g', 'middle': 'b', 
                 'ring': 'm', 'pinky': 'c'}
        
        for finger_name, color in colors.items():
            filtered_angles = results[finger_name]['filtered_angles']
            plt.plot(t, filtered_angles, color, 
                    label=f"{finger_name}", alpha=0.7)
        
        plt.title(f'{scenario}场景下的滤波效果对比')
        plt.xlabel('时间 (s)')
        plt.ylabel('角度 (度)')
        plt.legend()
        plt.grid(True)
        
        # 绘制误差分析
        plt.subplot(212)
        for finger_name, color in colors.items():
            filtered_angles = results[finger_name]['filtered_angles']
            error = filtered_angles - true_angles
            plt.plot(t, error, color, label=f"{finger_name}误差", alpha=0.7)
        
        plt.title('滤波误差分析')
        plt.xlabel('时间 (s)')
        plt.ylabel('误差 (度)')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        
        # 打印详细指标
        print(f"\n{scenario}场景评估结果:")
        print("-" * 50)
        for finger_name in colors.keys():
            metrics = results[finger_name]['metrics']
            config = self.finger_configs[finger_name][scenario]
            print(f"\n{finger_name}:")
            print(f"参数配置: P={config['P']}, Q={config['Q']}, R={config['R']}, "
                  f"history_length={config['history_length']}")
            print(f"均方误差: {metrics['MSE']:.2f}")
            print(f"最大误差: {metrics['最大误差']:.2f}")
            print(f"平滑度: {metrics['平滑度']:.2f}")
            print(f"处理时间: {results[finger_name]['proc_time']*1000:.2f}ms")

    def run_comprehensive_test(self):
        """运行完整的测试流程"""
        scenarios = ['static', 'slow', 'fast']
        
        for scenario in scenarios:
            print(f"\n测试{scenario}场景...")
            t, true_angles, noisy_angles = self.generate_motion_data(scenario)
            
            results = {}
            for finger_name in self.finger_configs.keys():
                filtered_angles, proc_time = self.test_finger_config(
                    finger_name, scenario, noisy_angles)
                metrics = self.calculate_metrics(true_angles, filtered_angles)
                results[finger_name] = {
                    'filtered_angles': filtered_angles,
                    'metrics': metrics,
                    'proc_time': proc_time
                }
            
            self.visualize_scenario(scenario, t, true_angles, noisy_angles, results)
            plt.show()

# 运行测试
if __name__ == "__main__":
    tester = AdvancedAngleFilterTester()
    tester.run_comprehensive_test()

# import numpy as np
# import matplotlib.pyplot as plt

# class HandAngleFilter:
#     def __init__(self, history_length=5):
#         """手部关节角度滤波器初始化
#         Args:
#             history_length: 移动平均的历史帧数（建议3-10帧，30FPS视频常用5帧）
#         """
#         self.history_length = history_length
#         self.angle_history = {}  # 存储各关节历史角度，格式：{"thumb_flexion": [15,16,17...]}
#         self.kalman_filters = {} # 存储各关节的卡尔曼滤波器状态

#     def apply_kalman_filter(self, joint_type, measured_angle):
#         """应用卡尔曼滤波到指定关节
#         Args:
#             joint_type: 关节类型，如"thumb_mcp_flexion" 
#             measured_angle: 从传感器获得的原始角度测量值
#         """
#         if joint_type not in self.kalman_filters:
#             # 初始化滤波器参数（不同关节可设置不同参数）
#             self.kalman_filters[joint_type] = {
#                 'x': measured_angle,  # 状态估计（当前最优角度估计）
#                 'P': 1.0,    # 估计误差协方差（初始不确定性较大）
#                 'Q': 0.1,    # 过程噪声（手指运动突变程度，拇指建议0.2，小指建议0.05）
#                 'R': 1.0     # 测量噪声（检测算法误差，MediaPipe通常0.5-2.0）
#             }
        
#         kf = self.kalman_filters[joint_type]
        
#         # 预测阶段（假设角度变化不大）
#         predicted_angle = kf['x']
#         predicted_uncertainty = kf['P'] + kf['Q']  # 预测后不确定性增加
        
#         # 更新阶段（融合测量值）
#         kalman_gain = predicted_uncertainty / (predicted_uncertainty + kf['R'])
#         kf['x'] = predicted_angle + kalman_gain * (measured_angle - predicted_angle)
#         kf['P'] = (1 - kalman_gain) * predicted_uncertainty
        
#         return kf['x']

#     def smooth_joint_angle(self, joint_type, raw_angle):
#         """处理关节角度数据：加权平均 + 卡尔曼滤波
#         Args:
#             joint_type: 关节类型，如"index_pip_flexion"
#             raw_angle: 原始检测角度（可能含噪声）
#         """
#         if joint_type not in self.angle_history:
#             self.angle_history[joint_type] = []
            
#         history = self.angle_history[joint_type]
#         history.append(raw_angle)
        
#         # 保持固定长度的历史数据（滑动窗口）
#         if len(history) > self.history_length:
#             history.pop(0)
            
#         # 加权移动平均（近期数据权重更高）
#         weights = np.exp(np.linspace(-1, 0, len(history)))  # 指数衰减权重
#         weights /= weights.sum()  # 归一化
#         smoothed = np.average(history, weights=weights)
        
#         # 应用卡尔曼滤波进一步去噪
#         filtered_angle = self.apply_kalman_filter(joint_type, smoothed)
        
#         return round(filtered_angle, 2)  # 保留2位小数

# # 模拟手部跟踪数据（示例：食指PIP关节屈曲角度）
# np.random.seed(42)
# true_flexion = np.linspace(0, 90, 100)  # 真实屈曲角度从0°到90°
# noisy_flexion = true_flexion + np.random.normal(0, 5, 100)  # 添加测量噪声（模拟MediaPipe检测抖动）

# # 初始化滤波器（针对食指PIP关节）
# finger_filter = HandAngleFilter(history_length=5)

# # 处理每一帧数据
# filtered_angles = []
# for angle in noisy_flexion:
#     filtered_angles.append(finger_filter.smooth_joint_angle("index_pip_flexion", angle))

# # 可视化结果
# plt.figure(figsize=(12, 6))
# plt.plot(true_flexion, label="真实角度", linestyle="--", color="green", linewidth=2)
# plt.plot(noisy_flexion, label="原始检测值（含噪声）", alpha=0.5, color="blue")
# plt.plot(filtered_angles, label="滤波后角度", linewidth=2, color="red")
# plt.title("食指PIP关节屈曲角度滤波效果（模拟数据）", fontsize=14)
# plt.xlabel("视频帧序号（30FPS）", fontsize=12)
# plt.ylabel("关节角度（度）", fontsize=12)
# plt.legend(fontsize=12)
# plt.grid(True, linestyle='--', alpha=0.7)
# plt.show()