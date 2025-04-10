# import numpy as np
# from scipy import signal


# ########## SNR
# # import numpy as np
# # import matplotlib.pyplot as plt

# # def calculate_snr(emg_data, rest_indices, active_indices, fs=200):
# #     """
# #     计算多通道EMG的SNR
# #     参数：
# #         emg_data: [n_samples, n_channels] 的EMG数据
# #         rest_indices: 静息状态的样本索引（列表或切片）
# #         active_indices: 活动状态的样本索引
# #         fs: 采样率（Hz）
# #     返回：
# #         snr_db: 各通道的SNR（dB）
# #         noise_rms: 静息噪声RMS值（μV）
# #         signal_rms: 活动信号RMS值（μV）
# #     """
# #     # 计算静息噪声功率（RMS）
# #     noise_rms = np.sqrt(np.mean(emg_data[rest_indices, :]**2, axis=0))
    
# #     # 计算活动信号功率（RMS）
# #     signal_rms = np.sqrt(np.mean(emg_data[active_indices, :]**2, axis=0))
    
# #     # 计算SNR（dB），避免除以0
# #     snr_db = 20 * np.log10((signal_rms + 1e-10) / (noise_rms + 1e-10))
    
# #     return snr_db, noise_rms, signal_rms

# # # 示例数据生成（实际使用时替换为你的数据）
# # fs = 200  # 采样率200Hz
# # t = np.arange(0, 5, 1/fs)  # 5秒数据
# # emg_data = np.zeros((len(t), 8))  # 8通道

# # # 模拟噪声（静息段：0-2秒）
# # noise_amplitude = 20  # μV
# # emg_data[:2*fs, :] = noise_amplitude * np.random.randn(2*fs, 8)

# # # 模拟信号+噪声（活动段：2-4秒）
# # signal_amplitude = 100  # μV
# # emg_data[2*fs:4*fs, :] += signal_amplitude * np.random.randn(2*fs, 8)

# # # 定义静息和活动段
# # rest_indices = slice(0, 2*fs)  # 前2秒为静息
# # active_indices = slice(2*fs, 4*fs)  # 2-4秒为活动

# # # 计算SNR
# # snr_db, noise_rms, signal_rms = calculate_snr(emg_data, rest_indices, active_indices, fs)

# # # 可视化结果
# # plt.figure(figsize=(12, 6))
# # plt.subplot(2, 1, 1)
# # plt.plot(t, emg_data[:, 0], label='Channel 1 Raw EMG')
# # plt.axvspan(0, 2, color='gray', alpha=0.2, label='Rest')
# # plt.axvspan(2, 4, color='green', alpha=0.2, label='Active')
# # plt.xlabel('Time (s)')
# # plt.ylabel('Amplitude (μV)')
# # plt.legend()

# # plt.subplot(2, 1, 2)
# # plt.bar(range(8), snr_db, color='skyblue')
# # plt.axhline(y=10, color='r', linestyle='--', label='Minimum SNR (10 dB)')
# # plt.xlabel('Channel')
# # plt.ylabel('SNR (dB)')
# # plt.xticks(range(8), [f'Ch{i+1}' for i in range(8)])
# # plt.legend()
# # plt.tight_layout()
# # plt.show()

# # # 打印结果
# # print(f"噪声RMS（μV）: {noise_rms.round(2)}")
# # print(f"信号RMS（μV）: {signal_rms.round(2)}")
# # print(f"SNR（dB）: {snr_db.round(2)}")
# # A = np.array([1, 2, 3])
# # B = np.array([0, 1, 0.5])

# # lags_full = signal.correlation_lags(len(A), len(B), mode='full')
# # corr = signal.correlate(A, B, mode='full')
# # print(corr)  # 输出: [0.5, 2. , 3.5, 2. , 0.5]
# # print(lags_full)


# # # B超前于A
# # A = np.array([0, 0, 1, 2, 3])  # A在t=2时激活
# # B = np.array([1, 2, 3, 0, 0])   # B在t=0时激活（比A早2个时间点）

# # lags = signal.correlation_lags(len(A), len(B), mode='full')
# # corr = signal.correlate(A, B, mode='full')
# # print("lags:", lags)    # 输出: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# # print("corr:", corr)    # 峰值在 lag=-2

# # # B超前于A
# # A = np.array([0, 0, 1, 2, 3])  # A在t=2时激活
# # B = np.array([0,0, 1, 2, 3])   # B在t=0时激活（比A早2个时间点）

# # lags = signal.correlation_lags(len(A), len(B), mode='full')
# # corr = signal.correlate(A, B, mode='full')
# # print("lags:", lags)    # 输出: [-4, -3, -2, -1, 0, 1, 2, 3, 4]
# # print("corr:", corr)    # 峰值在 lag=-2

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.interpolate import interp1d
# import matplotlib.pyplot as plt
# from data_processing.datasynchronizer import DataSynchronizer

# # 设置全局字体为支持中文的字体
# plt.rcParams['font.sans-serif'] = ['SimHei']  # Windows系统常用
# # plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 也可选
# plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方块的问题

# import numpy as np
# import matplotlib.pyplot as plt
# from scipy import signal
# from scipy.interpolate import interp1d

# # 设置中文字体
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False

# # 初始化同步器 (替换为您的实际路径)
# synchronizer = DataSynchronizer(r'E:\multimodel-acquisition\data\20250326_210703_1_right_6.h5')
# synchronizer.load_data()
# # 执行同步
# synced_joint_data = synchronizer.linear_interpolation_sync()
# # =============================================
# # 替换时间戳数据 (关键修改点)
# t_emg = synchronizer.emg_timestamps  # EMG时间戳数组
# t_angle = synchronizer.joint_timestamps  # 关节角度时间戳数组
# # =============================================

# # 动作段定义 (来自您的JSON数据)
# action_start = 42823.5337359
# action_end = 42827.7606673






# # 动作段数据提取
# def extract_action_segment(timestamps, data, start, end):
#     mask = (timestamps >= start) & (timestamps <= end)
#     return data[mask], timestamps[mask]

# # 提取动作段
# emg_segment, t_emg_seg = extract_action_segment(t_emg, synchronizer.emg_data[:, 3], action_start, action_end)
# joint_segment, _ = extract_action_segment(t_emg, synced_joint_data[:, 2], action_start, action_end)

# # 动作段互相关分析
# def action_cross_corr(signal1, signal2, fs=200):
#     corr = signal.correlate(signal1 - np.mean(signal1), 
#                            signal2 - np.mean(signal2), 
#                            mode='full')
#     lags = signal.correlation_lags(len(signal1), len(signal2), mode='full')
#     lag_time = lags / fs  # 转换为秒
#     peak_lag = lags[np.argmax(corr)]
#     return corr, lag_time, peak_lag / fs

# corr, lag_time, peak_delay = action_cross_corr(emg_segment, joint_segment)

# # 可视化
# plt.figure(figsize=(12, 8))

# # 原始信号对比
# plt.subplot(2, 1, 1)
# plt.plot(t_emg_seg - t_emg_seg[0], emg_segment, 'b', label='EMG通道1')
# plt.plot(t_emg_seg - t_emg_seg[0], joint_segment, 'r', label='关节角度1(同步后)')
# plt.axvline(x=peak_delay, color='k', linestyle='--', 
#             label=f'最佳延迟: {peak_delay*1000:.1f}ms')
# plt.xlabel('时间 (秒)')
# plt.ylabel('幅值')
# plt.title('捏取动作段信号对比')
# plt.legend()
# plt.grid(True)

# # 互相关分析
# plt.subplot(2, 1, 2)
# plt.plot(lag_time*1000, corr, 'g', label='互相关')
# plt.axvline(x=peak_delay*1000, color='r', linestyle='--', 
#             label=f'峰值延迟: {peak_delay*1000:.1f}ms')
# plt.xlabel('时间偏移 (毫秒)')
# plt.ylabel('相关性')
# plt.title('动作段互相关分析 (EMG vs 关节角度)')
# plt.legend()
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# print(f"\n动作段分析结果:")
# print(f"- 开始时间: {action_start}")
# print(f"- 结束时间: {action_end}")
# print(f"- 持续时间: {action_end - action_start:.3f}秒")
# print(f"- EMG与关节角度最佳延迟: {peak_delay*1000:.1f}ms")




# # # 生成EMG信号 (8通道)
# # emg = np.random.randn(2000, 8) * 0.5  # 基线噪声
# # pulse_start = int(2000 * 0.3)  # 3秒时激活
# # emg[pulse_start:pulse_start+50, 1] += np.sin(np.linspace(0, 2*np.pi, 50)) * 10  # 通道2（索引1）

# # # 生成关节角度信号 (20通道)
# # angle = np.random.randn(300, 20) * 2  # 基线噪声
# # pulse_start_angle = int(300 * 0.3) + 5  # 延迟5个时间点（约0.167秒）
# # angle[pulse_start_angle:pulse_start_angle+20, 4] += np.sin(np.linspace(0, 2*np.pi, 20)) * 5  # 通道5（索引4）

# # # 2. 重采样关节角度到200Hz
# # f = interp1d(t_angle, angle[:, 4], kind='cubic', fill_value="extrapolate")
# # angle_resampled = f(t_emg)

# # import h5py
# # import numpy as np
# # from scipy import signal
# # from scipy.stats import pearsonr
# # import matplotlib.pyplot as plt
# # import matplotlib
# # from dtw import dtw
# # from sklearn.preprocessing import StandardScaler
# # # matplotlib.rcParams['font.family'] = 'SimSun'
# # matplotlib.rcParams['font.family'] = 'SimHei'
# # plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题
# # def enhanced_cross_correlation(emg_data, joint_data, emg_ts, joint_ts, fs=1000):
# #     """
# #     增强版多模态数据互相关分析测试脚本
# #     改进点：
# #     1. 多通道并行分析
# #     2. 动态时间规整(DTW)辅助验证
# #     3. 统计显著性检验
# #     4. 更全面的可视化
# #     """
# #     # 数据预处理
# #     min_len = min(len(emg_data), len(joint_data))
# #     emg_data = emg_data[:min_len]
# #     joint_data = joint_data[:min_len]
# #     emg_ts = emg_ts[:min_len]
# #     joint_ts = joint_ts[:min_len]

# #     # 多通道分析 (示例分析前3个EMG通道和前2个关节)
# #     emg_channels = min(3, emg_data.shape[1])
# #     joint_channels = min(2, joint_data.shape[1])
    
# #     # 创建可视化画布
# #     fig = plt.figure(figsize=(18, 6 * (emg_channels * joint_channels)))
# #     gs = fig.add_gridspec(emg_channels * joint_channels, 3)
    
# #     results = {}
    
# #     for emg_ch in range(emg_channels):
# #         for joint_ch in range(joint_channels):
# #             # 当前通道数据标准化
# #             emg_norm = StandardScaler().fit_transform(emg_data[:, emg_ch].reshape(-1, 1)).flatten()
# #             joint_norm = StandardScaler().fit_transform(joint_data[:, joint_ch].reshape(-1, 1)).flatten()
            
# #             # 互相关分析
# #             cross_corr = signal.correlate(emg_norm, joint_norm, mode='full', method='auto')
# #             lags = signal.correlation_lags(len(emg_norm), len(joint_norm), mode='full')
# #             time_lags = lags / fs
            
# #             # 寻找主峰和次峰
# #             peaks, _ = signal.find_peaks(np.abs(cross_corr), height=0.5*np.max(np.abs(cross_corr)), distance=fs//2)
# #             sorted_peaks = sorted(peaks, key=lambda x: np.abs(cross_corr[x]), reverse=True)
            
# #             # 主峰信息
# #             main_lag = time_lags[sorted_peaks[0]] if len(sorted_peaks) > 0 else 0
# #             main_corr = cross_corr[sorted_peaks[0]] if len(sorted_peaks) > 0 else 0
            
# #             # 动态时间规整(DTW)验证
# #             def dtw_distance(x, y):
# #                 from dtw import dtw
# #                 alignment = dtw(x, y, keep_internals=True)
# #                 return alignment.normalizedDistance
            
# #             try:
# #                 dtw_dist = dtw_distance(emg_norm, joint_norm)
# #                 dtw_dist_aligned = dtw_distance(emg_norm, np.roll(joint_norm, lags[sorted_peaks[0]]))
# #             except:
# #                 dtw_dist = dtw_dist_aligned = np.nan
            
# #             # 统计检验
# #             pearson_r, pearson_p = pearsonr(emg_norm, np.roll(joint_norm, lags[sorted_peaks[0]]))
            
# #             # 存储结果
# #             key = f"EMG_CH{emg_ch+1}_vs_Joint{joint_ch+1}"
# #             results[key] = {
# #                 'optimal_lag': main_lag,
# #                 'max_correlation': main_corr,
# #                 'pearson_r': pearson_r,
# #                 'pearson_p': pearson_p,
# #                 'dtw_distance': dtw_dist,
# #                 'dtw_distance_aligned': dtw_dist_aligned
# #             }
            
# #             # 绘制结果
# #             ax_idx = emg_ch * joint_channels + joint_ch
# #             ax1 = fig.add_subplot(gs[ax_idx, 0])
# #             ax2 = fig.add_subplot(gs[ax_idx, 1])
# #             ax3 = fig.add_subplot(gs[ax_idx, 2])
            
# #             # 子图1: 原始信号对比
# #             ax1.plot(emg_ts, emg_norm, label=f'EMG CH{emg_ch+1}')
# #             ax1.plot(joint_ts, joint_norm, label=f'Joint {joint_ch+1}')
# #             ax1.set_title(f"Signal Comparison\nPearson r={pearson_r:.3f} (p={pearson_p:.3f})")
# #             ax1.legend()
            
# #             # 子图2: 互相关函数
# #             ax2.plot(time_lags, cross_corr)
# #             for peak in sorted_peaks[:2]:  # 标记前两个峰值
# #                 ax2.plot(time_lags[peak], cross_corr[peak], 'ro')
# #                 ax2.text(time_lags[peak], cross_corr[peak], 
# #                         f'{time_lags[peak]:.3f}s', ha='center', va='bottom')
# #             ax2.set_title(f"Cross-Correlation\nOptimal Lag: {main_lag:.3f}s")
# #             ax2.set_xlabel("Time Lag (s)")
            
# #             # 子图3: 对齐后信号
# #             aligned_joint = np.roll(joint_norm, lags[sorted_peaks[0]])
# #             ax3.plot(emg_ts, emg_norm, label='EMG')
# #             ax3.plot(joint_ts, aligned_joint, label=f'Joint (aligned)')
# #             ax3.set_title(f"Aligned Signals\nDTW dist: {dtw_dist:.2f}→{dtw_dist_aligned:.2f}")
# #             ax3.legend()
    
# #     plt.tight_layout()
# #     plt.show()
# #     return results

# # # 示例使用
# # if __name__ == "__main__":
# #     # 模拟数据生成（实际使用时替换为真实HDF5数据）
# #     fs = 1000
# #     t = np.arange(0, 10, 1/fs)
    
# #     # 生成多通道EMG信号
# #     emg_data = np.column_stack([
# #         np.sin(2*np.pi*5*t) * (t % 2 < 1) + 0.2*np.random.randn(len(t)),  # CH1
# #         0.8*np.sin(2*np.pi*3*t) * (t % 1.5 < 1) + 0.3*np.random.randn(len(t)),  # CH2
# #         np.random.randn(len(t))  # CH3 (噪声)
# #     ])
    
# #     # 生成多关节角度信号（不同延迟）
# #     joint_data = np.column_stack([
# #         np.roll(emg_data[:,0], int(0.1*fs)) + 0.1*np.random.randn(len(t)),  # Joint1 (100ms延迟)
# #         np.roll(emg_data[:,1], int(-0.05*fs)) + 0.15*np.random.randn(len(t))  # Joint2 (50ms提前)
# #     ])
    
# #     # 运行分析
# #     print("开始多模态时间同步验证...")
# #     metrics = enhanced_cross_correlation(
# #         emg_data=emg_data,
# #         joint_data=joint_data,
# #         emg_ts=t,
# #         joint_ts=t,
# #         fs=fs
# #     )
    
# #     # 打印结果摘要
# #     print("\n=== 同步验证结果摘要 ===")
# #     for key, val in metrics.items():
# #         print(f"\n{key}:")
# #         print(f"最优延迟: {val['optimal_lag']:.4f}s")
# #         print(f"最大互相关系数: {val['max_correlation']:.2f}")
# #         print(f"Pearson相关性: r={val['pearson_r']:.3f} (p={val['pearson_p']:.3f})")
# #         print(f"DTW距离: {val['dtw_distance']:.2f} → {val['dtw_distance_aligned']:.2f}")


# # import numpy as np
# # import matplotlib.pyplot as plt
# # from data_processing.hand_angles_improve import HandAngleCalculator
# # import time
# # import matplotlib.pyplot as plt
# # from matplotlib import rcParams

# # # 设置中文字体
# # rcParams['font.sans-serif'] = ['SimHei']  # 使用SimHei字体
# # rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# # class AdvancedAngleFilterTester:
# #     def __init__(self):
# #         # 为不同手指定义不同的参数配置
# #         self.finger_configs = {
# #             'thumb': {
# #                 'static': {'P': 1.0, 'Q': 0.2, 'R': 1.5, 'history_length': 5},
# #                 'slow': {'P': 1.0, 'Q': 0.3, 'R': 1.2, 'history_length': 4},
# #                 'fast': {'P': 1.0, 'Q': 0.4, 'R': 0.8, 'history_length': 3}
# #             },
# #             'index': {
# #                 'static': {'P': 1.0, 'Q': 0.1, 'R': 1.2, 'history_length': 5},
# #                 'slow': {'P': 1.0, 'Q': 0.15, 'R': 1.0, 'history_length': 4},
# #                 'fast': {'P': 1.0, 'Q': 0.25, 'R': 0.8, 'history_length': 3}
# #             },
# #             'middle': {
# #                 'static': {'P': 1.0, 'Q': 0.1, 'R': 1.2, 'history_length': 5},
# #                 'slow': {'P': 1.0, 'Q': 0.15, 'R': 1.0, 'history_length': 4},
# #                 'fast': {'P': 1.0, 'Q': 0.25, 'R': 0.8, 'history_length': 3}
# #             },
# #             'ring': {
# #                 'static': {'P': 1.0, 'Q': 0.08, 'R': 1.2, 'history_length': 6},
# #                 'slow': {'P': 1.0, 'Q': 0.12, 'R': 1.0, 'history_length': 5},
# #                 'fast': {'P': 1.0, 'Q': 0.2, 'R': 0.8, 'history_length': 4}
# #             },
# #             'pinky': {
# #                 'static': {'P': 1.0, 'Q': 0.05, 'R': 1.5, 'history_length': 6},
# #                 'slow': {'P': 1.0, 'Q': 0.08, 'R': 1.2, 'history_length': 5},
# #                 'fast': {'P': 1.0, 'Q': 0.15, 'R': 1.0, 'history_length': 4}
# #             }
# #         }

# #     def generate_motion_data(self, scenario='static', num_points=200):
# #         """生成不同场景的测试数据"""
# #         t = np.linspace(0, 10, num_points)
        
# #         if scenario == 'static':
# #             # 静止场景：小幅度随机抖动
# #             true_angles = 45 + np.random.normal(0, 1, num_points)
# #             noise_std = 2.0
            
# #         elif scenario == 'slow':
# #             # 低速运动：缓慢的正弦运动
# #             true_angles = 45 + 30 * np.sin(0.5 * t) + 10 * np.sin(0.2 * t)
# #             noise_std = 3.0
            
# #         else:  # fast
# #             # 高速运动：快速的正弦运动加上突变
# #             true_angles = 45 + 30 * np.sin(2 * t) + 15 * np.sin(4 * t)
# #             # 添加突变
# #             sudden_changes = np.random.randint(0, num_points, 5)
# #             for idx in sudden_changes:
# #                 true_angles[idx:idx+10] += 20
# #             noise_std = 5.0

# #         # 添加噪声
# #         noisy_angles = true_angles + np.random.normal(0, noise_std, num_points)
# #         return t, true_angles, noisy_angles

# #     def test_finger_config(self, finger_name, scenario, noisy_angles):
# #         """测试特定手指在特定场景下的滤波效果"""
# #         config = self.finger_configs[finger_name][scenario]
# #         calculator = HandAngleCalculator()
        
# #         # 设置参数
# #         calculator.kalman_filters = {}
# #         calculator.history_length = config['history_length']
        
# #         filtered_angles = []
# #         processing_times = []
        
# #         for angle in noisy_angles:
# #             start_time = time.time()
# #             filtered = calculator.smooth_angle(f'{finger_name}_test', angle)
# #             processing_times.append(time.time() - start_time)
# #             filtered_angles.append(filtered)
            
# #             # 更新卡尔曼滤波器参数
# #             if f'{finger_name}_test' in calculator.kalman_filters:
# #                 calculator.kalman_filters[f'{finger_name}_test'].update({
# #                     'P': config['P'],
# #                     'Q': config['Q'],
# #                     'R': config['R']
# #                 })
        
# #         return np.array(filtered_angles), np.mean(processing_times)

# #     def calculate_metrics(self, true_angles, filtered_angles):
# #         """计算性能指标"""
# #         mse = np.mean((true_angles - filtered_angles)**2)
# #         max_error = np.max(np.abs(true_angles - filtered_angles))
# #         smoothness = np.mean(np.abs(np.diff(filtered_angles)))
# #         delay = np.correlate(filtered_angles - np.mean(filtered_angles),
# #                            true_angles - np.mean(true_angles))[0]
# #         return {
# #             'MSE': mse,
# #             '最大误差': max_error,
# #             '平滑度': smoothness,
# #             '延迟': delay
# #         }

# #     def visualize_scenario(self, scenario, t, true_angles, noisy_angles, results):
# #         """可视化特定场景下所有手指的滤波效果"""
# #         plt.figure(figsize=(15, 10))
# #         plt.subplot(211)
        
# #         plt.plot(t, true_angles, 'k-', label='真实角度', alpha=0.5)
# #         plt.plot(t, noisy_angles, 'gray', label='噪声数据', alpha=0.3)
        
# #         colors = {'thumb': 'r', 'index': 'g', 'middle': 'b', 
# #                  'ring': 'm', 'pinky': 'c'}
        
# #         for finger_name, color in colors.items():
# #             filtered_angles = results[finger_name]['filtered_angles']
# #             plt.plot(t, filtered_angles, color, 
# #                     label=f"{finger_name}", alpha=0.7)
        
# #         plt.title(f'{scenario}场景下的滤波效果对比')
# #         plt.xlabel('时间 (s)')
# #         plt.ylabel('角度 (度)')
# #         plt.legend()
# #         plt.grid(True)
        
# #         # 绘制误差分析
# #         plt.subplot(212)
# #         for finger_name, color in colors.items():
# #             filtered_angles = results[finger_name]['filtered_angles']
# #             error = filtered_angles - true_angles
# #             plt.plot(t, error, color, label=f"{finger_name}误差", alpha=0.7)
        
# #         plt.title('滤波误差分析')
# #         plt.xlabel('时间 (s)')
# #         plt.ylabel('误差 (度)')
# #         plt.legend()
# #         plt.grid(True)
# #         plt.tight_layout()
        
# #         # 打印详细指标
# #         print(f"\n{scenario}场景评估结果:")
# #         print("-" * 50)
# #         for finger_name in colors.keys():
# #             metrics = results[finger_name]['metrics']
# #             config = self.finger_configs[finger_name][scenario]
# #             print(f"\n{finger_name}:")
# #             print(f"参数配置: P={config['P']}, Q={config['Q']}, R={config['R']}, "
# #                   f"history_length={config['history_length']}")
# #             print(f"均方误差: {metrics['MSE']:.2f}")
# #             print(f"最大误差: {metrics['最大误差']:.2f}")
# #             print(f"平滑度: {metrics['平滑度']:.2f}")
# #             print(f"处理时间: {results[finger_name]['proc_time']*1000:.2f}ms")

# #     def run_comprehensive_test(self):
# #         """运行完整的测试流程"""
# #         scenarios = ['static', 'slow', 'fast']
        
# #         for scenario in scenarios:
# #             print(f"\n测试{scenario}场景...")
# #             t, true_angles, noisy_angles = self.generate_motion_data(scenario)
            
# #             results = {}
# #             for finger_name in self.finger_configs.keys():
# #                 filtered_angles, proc_time = self.test_finger_config(
# #                     finger_name, scenario, noisy_angles)
# #                 metrics = self.calculate_metrics(true_angles, filtered_angles)
# #                 results[finger_name] = {
# #                     'filtered_angles': filtered_angles,
# #                     'metrics': metrics,
# #                     'proc_time': proc_time
# #                 }
            
# #             self.visualize_scenario(scenario, t, true_angles, noisy_angles, results)
# #             plt.show()

# # # 运行测试
# # if __name__ == "__main__":
# #     tester = AdvancedAngleFilterTester()
# #     tester.run_comprehensive_test()

# # import numpy as np
# # import matplotlib.pyplot as plt

# # class HandAngleFilter:
# #     def __init__(self, history_length=5):
# #         """手部关节角度滤波器初始化
# #         Args:
# #             history_length: 移动平均的历史帧数（建议3-10帧，30FPS视频常用5帧）
# #         """
# #         self.history_length = history_length
# #         self.angle_history = {}  # 存储各关节历史角度，格式：{"thumb_flexion": [15,16,17...]}
# #         self.kalman_filters = {} # 存储各关节的卡尔曼滤波器状态

# #     def apply_kalman_filter(self, joint_type, measured_angle):
# #         """应用卡尔曼滤波到指定关节
# #         Args:
# #             joint_type: 关节类型，如"thumb_mcp_flexion" 
# #             measured_angle: 从传感器获得的原始角度测量值
# #         """
# #         if joint_type not in self.kalman_filters:
# #             # 初始化滤波器参数（不同关节可设置不同参数）
# #             self.kalman_filters[joint_type] = {
# #                 'x': measured_angle,  # 状态估计（当前最优角度估计）
# #                 'P': 1.0,    # 估计误差协方差（初始不确定性较大）
# #                 'Q': 0.1,    # 过程噪声（手指运动突变程度，拇指建议0.2，小指建议0.05）
# #                 'R': 1.0     # 测量噪声（检测算法误差，MediaPipe通常0.5-2.0）
# #             }
        
# #         kf = self.kalman_filters[joint_type]
        
# #         # 预测阶段（假设角度变化不大）
# #         predicted_angle = kf['x']
# #         predicted_uncertainty = kf['P'] + kf['Q']  # 预测后不确定性增加
        
# #         # 更新阶段（融合测量值）
# #         kalman_gain = predicted_uncertainty / (predicted_uncertainty + kf['R'])
# #         kf['x'] = predicted_angle + kalman_gain * (measured_angle - predicted_angle)
# #         kf['P'] = (1 - kalman_gain) * predicted_uncertainty
        
# #         return kf['x']

# #     def smooth_joint_angle(self, joint_type, raw_angle):
# #         """处理关节角度数据：加权平均 + 卡尔曼滤波
# #         Args:
# #             joint_type: 关节类型，如"index_pip_flexion"
# #             raw_angle: 原始检测角度（可能含噪声）
# #         """
# #         if joint_type not in self.angle_history:
# #             self.angle_history[joint_type] = []
            
# #         history = self.angle_history[joint_type]
# #         history.append(raw_angle)
        
# #         # 保持固定长度的历史数据（滑动窗口）
# #         if len(history) > self.history_length:
# #             history.pop(0)
            
# #         # 加权移动平均（近期数据权重更高）
# #         weights = np.exp(np.linspace(-1, 0, len(history)))  # 指数衰减权重
# #         weights /= weights.sum()  # 归一化
# #         smoothed = np.average(history, weights=weights)
        
# #         # 应用卡尔曼滤波进一步去噪
# #         filtered_angle = self.apply_kalman_filter(joint_type, smoothed)
        
# #         return round(filtered_angle, 2)  # 保留2位小数

# # # 模拟手部跟踪数据（示例：食指PIP关节屈曲角度）
# # np.random.seed(42)
# # true_flexion = np.linspace(0, 90, 100)  # 真实屈曲角度从0°到90°
# # noisy_flexion = true_flexion + np.random.normal(0, 5, 100)  # 添加测量噪声（模拟MediaPipe检测抖动）

# # # 初始化滤波器（针对食指PIP关节）
# # finger_filter = HandAngleFilter(history_length=5)

# # # 处理每一帧数据
# # filtered_angles = []
# # for angle in noisy_flexion:
# #     filtered_angles.append(finger_filter.smooth_joint_angle("index_pip_flexion", angle))

# # # 可视化结果
# # plt.figure(figsize=(12, 6))
# # plt.plot(true_flexion, label="真实角度", linestyle="--", color="green", linewidth=2)
# # plt.plot(noisy_flexion, label="原始检测值（含噪声）", alpha=0.5, color="blue")
# # plt.plot(filtered_angles, label="滤波后角度", linewidth=2, color="red")
# # plt.title("食指PIP关节屈曲角度滤波效果（模拟数据）", fontsize=14)
# # plt.xlabel("视频帧序号（30FPS）", fontsize=12)
# # plt.ylabel("关节角度（度）", fontsize=12)
# # plt.legend(fontsize=12)
# # plt.grid(True, linestyle='--', alpha=0.7)
# # plt.show()



