import h5py
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
import numpy as np
from scipy import signal
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler

# matplotlib.rcParams['font.family'] = 'SimSun'
matplotlib.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示为方框的问题

class DataSynchronizer:
    def __init__(self, hdf5_file_path):
        self.file_path = hdf5_file_path
        self.emg_data = None
        self.emg_timestamps = None
        self.joint_data = None
        self.joint_timestamps = None
        
    def load_data(self):
        """加载HDF5文件中的数据"""
        with h5py.File(self.file_path, 'r') as f:
            # 加载数据
            self.emg_data = f['emg/filtered_data'][:]
            self.emg_timestamps = f['emg/timestamps'][:]
            self.joint_data = f['hand/joint_angles'][:]
            self.joint_timestamps = f['hand/timestamps'][:]
            
    def find_common_time_range(self):
        """找到两个数据流的共同时间范围"""
        # 找到共同的开始和结束时间
        start_time = max(self.emg_timestamps[0], self.joint_timestamps[0])
        end_time = min(self.emg_timestamps[-1], self.joint_timestamps[-1])
        
        # 打印时间范围信息
        print(f"EMG数据时间范围: {self.emg_timestamps[0]} 到 {self.emg_timestamps[-1]}")
        print(f"关节数据时间范围: {self.joint_timestamps[0]} 到 {self.joint_timestamps[-1]}")
        print(f"共同时间范围: {start_time} 到 {end_time}")
        
        return start_time, end_time
        
    def trim_data_to_common_range(self, start_time, end_time):
        """将数据修剪到共同时间范围"""
        # 修剪EMG数据
        emg_mask = (self.emg_timestamps >= start_time) & (self.emg_timestamps <= end_time)
        self.emg_data = self.emg_data[emg_mask]
        self.emg_timestamps = self.emg_timestamps[emg_mask]
        
        # 修剪关节数据
        joint_mask = (self.joint_timestamps >= start_time) & (self.joint_timestamps <= end_time)
        self.joint_data = self.joint_data[joint_mask]
        self.joint_timestamps = self.joint_timestamps[joint_mask]




        
    def linear_interpolation_sync(self,method='linear'):
        """使用线性插值进行数据同步"""
        # 首先找到共同时间范围
        start_time, end_time = self.find_common_time_range()
        
        # 修剪数据到共同范围
        self.trim_data_to_common_range(start_time, end_time)
        
        # 创建插值函数
        joint_interpolator = interpolate.interp1d(
            self.joint_timestamps,
            self.joint_data,
            axis=0,
            kind=method,
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # 在EMG时间戳上进行插值
        synced_joint_data = joint_interpolator(self.emg_timestamps)
        
        return synced_joint_data
    
    def interpolate_data(self, method):


        if method == 'linear':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data, kind='linear', fill_value='extrapolate')
        elif method == 'cubic':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data, kind='cubic', fill_value='extrapolate')
        elif method == 'nearest':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data, kind='nearest', fill_value='extrapolate')
        else:
            raise ValueError("不支持的插值方法: {}".format(method))
        
        return f(self.emg_timestamps)


    def validate_sync_quality(self, synced_joint_data):
        """验证同步质量，包含扩展评估标准"""
        emg_intervals = np.diff(self.emg_timestamps)
        joint_intervals = np.diff(self.joint_timestamps)

        print("\n=== 同步质量报告 ===")
        print("\nEMG数据统计：")
        print(f"数据点数量: {len(self.emg_timestamps)}")
        print(f"采样间隔均值: {np.mean(emg_intervals) * 1000:.6f}毫秒")
        print(f"采样间隔标准差: {np.std(emg_intervals) * 1000:.6f}毫秒")
        
        print("\n关节数据统计：")
        print(f"数据点数量: {len(self.joint_timestamps)}")
        print(f"采样间隔均值: {np.mean(joint_intervals) * 1000:.6f}毫秒")
        print(f"采样间隔标准差: {np.std(joint_intervals) * 1000:.6f}毫秒")
        
        print("\n同步后数据检查：")
        print(f"NaN值数量: {np.isnan(synced_joint_data).sum()}")
        print(f"同步后数据点数量: {len(synced_joint_data)}")

        # 时间对齐误差
        print("\n时间对齐误差分析：")
        synced_times = self.emg_timestamps
        original_times = self.joint_timestamps
        time_errors = []
        for synced_t in synced_times:
            closest_idx = np.argmin(np.abs(original_times - synced_t))
            time_error = synced_t - original_times[closest_idx]
            time_errors.append(time_error)
        time_errors = np.array(time_errors)
        print(f"平均时间对齐误差: {np.mean(np.abs(time_errors)) * 1000:.6f} 毫秒")
        print(f"最大时间对齐误差: {np.max(np.abs(time_errors)) * 1000:.6f} 毫秒")
        print(f"时间对齐误差标准差: {np.std(time_errors) * 1000:.6f} 毫秒")


        # 动作区间一致性
        print("\n动作区间一致性分析：")
        cycle_duration = 9
        t0 = self.emg_timestamps[0]
        total_duration = self.emg_timestamps[-1] - t0
        num_cycles = int(total_duration // cycle_duration) + 1
        for joint_idx in range(min(20, synced_joint_data.shape[1])):
            action_amplitudes = []
            rest_amplitudes = []
            for cycle in range(num_cycles):
                start = cycle * cycle_duration + t0
                action_end = start + 4
                rest_end = start + 9
                if action_end <= self.emg_timestamps[-1]:
                    action_mask = (self.emg_timestamps >= start) & (self.emg_timestamps < action_end)
                    rest_mask = (self.emg_timestamps >= action_end) & (self.emg_timestamps < rest_end)
                    action_data = synced_joint_data[action_mask, joint_idx]
                    rest_data = synced_joint_data[rest_mask, joint_idx]
                    action_amplitudes.append(np.max(action_data) - np.min(action_data))
                    rest_amplitudes.append(np.max(rest_data) - np.min(rest_data))
            avg_action_amplitude = np.mean(action_amplitudes)
            avg_rest_amplitude = np.mean(rest_amplitudes)
            print(f"关节 {joint_idx+1}:")
            print(f"  动作区间平均幅度: {avg_action_amplitude:.6f}")
            print(f"  休息区间平均幅度: {avg_rest_amplitude:.6f}")
            print(f"  动作/休息幅度比: {avg_action_amplitude / avg_rest_amplitude:.2f}")


class DataVisualizer:
    def __init__(self, emg_data, emg_timestamps, joint_data, joint_timestamps):
        self.emg_data = emg_data
        self.emg_timestamps = emg_timestamps
        self.joint_data = joint_data
        self.joint_timestamps = joint_timestamps

    def _plot_scatter(self, ax, x_data, y_data, label, color, marker, title, ylabel, 
                      show_legend=True, show_points=True):
        """辅助函数：绘制散点图并设置通用属性"""
        ax.scatter(x_data, y_data, label=label, color=color, marker=marker, 
                   s=2, alpha=0.6)
        ax.set_title(title, fontsize=11, pad=5)
        ax.set_ylabel(ylabel, fontsize=10)
        if show_legend:
            ax.legend(loc='upper right', fontsize=9, frameon=False)
        if show_points:
            num_points = len(y_data)
            ax.text(0.02, 0.95, f'数据点: {num_points}', 
                    transform=ax.transAxes, fontsize=9, verticalalignment='top')
            

    def visualize_sync_results(self, synced_joint_data):
        """可视化关节数据同步结果，适合小论文投稿"""
        # 设置全局字体和样式
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体，符合中文规范
        plt.rcParams['axes.unicode_minus'] = False   # 确保负号正常显示
        plt.rcParams['font.size'] = 10               # 默认字体大小适合论文

        # 创建画布，调整为论文常用尺寸
        fig, axs = plt.subplots(3, 1, figsize=(8, 6), sharex=True)  # 共享x轴

        # 子图1：原始关节数据
        self._plot_scatter(axs[0], self.joint_timestamps, self.joint_data[:, 0], 
                    label='原始数据', color='blue', marker='o', 
                    title='原始关节数据', ylabel='角度 (度)')

        # 子图2：同步关节数据
        self._plot_scatter(axs[1], self.emg_timestamps, synced_joint_data[:, 0], 
                    label='同步数据', color='red', marker='x', 
                    title='同步关节数据', ylabel='角度 (度)')

        # 子图3：原始与同步数据比较
        self._plot_scatter(axs[2], self.joint_timestamps, self.joint_data[:, 0], 
                    label='原始数据', color='blue', marker='o', 
                    title='原始与同步数据比较', ylabel='角度 (度)', show_points=False)
        axs[2].scatter(self.emg_timestamps, synced_joint_data[:, 0], 
                    label='同步数据', color='red', marker='x', s=4, alpha=0.6)
        axs[2].legend(loc='upper right', fontsize=9, frameon=False)
        axs[2].set_xlabel('时间 (s)', fontsize=10)

        for ax in axs:
            self.annotate_action_intervals(ax, self.joint_timestamps)

        # 调整布局
        plt.tight_layout(pad=1.2)

        # 保存为高分辨率图片（可选）
        # plt.savefig('sync_results.png', dpi=300, bbox_inches='tight')

        plt.show()

    def plot_interpolated_results(self):
        """绘制插值结果"""
        # 设置全局字体和样式
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体，符合中文规范
        plt.rcParams['axes.unicode_minus'] = False   # 确保负号正常显示
        plt.rcParams['font.size'] = 10               # 默认字体大小适合论文


        # 进行插值
        linear_interpolated = self.interpolate_data('linear')
        cubic_interpolated = self.interpolate_data('cubic')
        nearest_interpolated = self.interpolate_data('nearest')


        # 创建画布，调整为论文常用尺寸
        fig, axs = plt.subplots(4, 1, figsize=(8, 6), sharex=True)  # 共享x轴

 
        self._plot_scatter(axs[1], self.emg_timestamps, linear_interpolated[:, 0], 
                    label='线性插值', color='blue', marker='o', 
                    title='线性插值', ylabel='角度 (度)')


        self._plot_scatter(axs[2], self.emg_timestamps, cubic_interpolated  [:, 0], 
                    label='三次样条插值', color='red', marker='x', 
                    title='三次样条插值', ylabel='角度 (度)')

        self._plot_scatter(axs[3], self.emg_timestamps, nearest_interpolated[:, 0], 
                    label='最近邻插值', color='green', marker='x', 
                    title='最近邻插值', ylabel='角度 (度)')
        
        self._plot_scatter(axs[0], self.joint_timestamps, self.joint_data[:, 0], 
                    label='原始数据', color='black', marker='o', 
                    title='原始数据', ylabel='角度 (度)')
                

    # 添加动作区间标注并优化
        for ax in axs:
            self.annotate_action_intervals(ax, self.joint_timestamps)
            ax.tick_params(axis='both', labelsize=8)  # 调整刻度标签大小
            # 优化图例样式
            ax.legend(loc='upper right', fontsize=9, frameon=True, 
                    edgecolor='gray', framealpha=0.9, fancybox=True)

        # 调整布局
        plt.tight_layout(pad=1.2)

        # 保存为高分辨率图片（可选）
        # plt.savefig('sync_results.png', dpi=300, bbox_inches='tight')

        plt.show()
        

    def interpolate_data(self, method):


        if method == 'linear':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data,axis=0, kind='linear', fill_value='extrapolate')
        elif method == 'cubic':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data,axis=0, kind='cubic', fill_value='extrapolate')
        elif method == 'nearest':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data,axis=0, kind='nearest', fill_value='extrapolate')
        else:
            raise ValueError("不支持的插值方法: {}".format(method))
        
        return f(self.emg_timestamps)
    

    def annotate_action_intervals(self, ax, timestamps, action_duration=5, rest_duration=5):
            """在指定轴上标注动作和休息区间"""
            cycle_duration = action_duration + rest_duration  # 总周期（9秒）

            # 将时间戳转换为相对时间
            t0 = timestamps[0]
            relative_timestamps = timestamps - t0
            total_duration = relative_timestamps[-1]

            # 计算动作次数
            num_cycles = int(total_duration // cycle_duration) + 1

            # 标注动作和休息区间
            for cycle in range(num_cycles):
                action_start = cycle * cycle_duration
                action_end = action_start + action_duration
                rest_end = action_start + cycle_duration

                if action_start < total_duration:  # 确保不超过数据范围
                    # 动作区间
                    ax.axvspan(action_start + t0, min(action_end + t0, timestamps[-1]), 
                            alpha=0.2, color='gray', label='动作' if cycle == 0 else None)
                    # 休息区间（可选）
                    if rest_end <= total_duration:
                        ax.axvspan(action_end + t0, min(rest_end + t0, timestamps[-1]), 
                                alpha=0.1, color='lightgreen', label='休息' if cycle == 0 else None)
        
    



    def visualize_emg_data(self):
        """可视化8通道肌电信号，适合小论文投稿"""
        # 设置全局字体和负号显示
        plt.rcParams['font.sans-serif'] = ['SimSun']  # 使用宋体，符合中文论文规范
        plt.rcParams['axes.unicode_minus'] = False  # 确保负号正常显示
        plt.rcParams['font.size'] = 10  # 默认字体大小适合论文

        # 创建画布，调整尺寸为论文常用大小
        plt.figure(figsize=(8, 6))  # 宽度8英寸，高度6英寸，适合单栏或双栏排版

        # 计算偏移量，避免通道信号重叠
        emg_amplitude_range = np.max(self.emg_data) - np.min(self.emg_data)
        offset = emg_amplitude_range * 0.8  # 调整偏移量，优化间距

        # 绘制8通道肌电信号，从CH8到CH1（逆序绘制）
        colors = plt.cm.tab10(np.linspace(0, 1, 8))  # 使用tab10颜色方案，专业且区分度高
        for i in range(7, -1, -1):  # 从7倒序到0，绘制CH8到CH1
            offset_data = self.emg_data[:, i] + (i * offset)
            plt.plot(self.emg_timestamps, offset_data, label=f'CH{i+1}', 
                    color=colors[i], linewidth=1.2)  # 细线条更清晰

        self.annotate_action_intervals(plt.gca(), self.emg_timestamps)

        # 设置标题和轴标签（简洁规范）
        plt.title('8通道肌电信号', fontsize=12, pad=10)
        plt.xlabel('时间 (s)', fontsize=10)
        plt.ylabel('肌电信号 (mV)', fontsize=10)

        # 添加图例（从上到下为CH8到CH1）
        plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), 
                fontsize=9, frameon=False, ncol=1)

        # 添加数据点数量（可选，若论文不需要可删除）
        num_emg_points = len(self.emg_data)
        plt.text(0.02, 0.98, f'数据点: {num_emg_points}', 
                transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')

        # 调整布局并优化边距
        plt.tight_layout(pad=1.5)

        # 保存为高分辨率图片（可选，用于投稿）
        # plt.savefig('emg_data.png', dpi=300, bbox_inches='tight')

        plt.show()

    def visualize_single_finger_joints(self, synced_joint_data, finger_name="食指", joint_indices=[4,5,6,7]):
        """
        将单个手指的四个自由度关节角度数据按2x2布局可视化
        finger_name: 手指名称
        joint_indices: 对应的四个关节角度索引 [MCP屈曲, MCP外展/内收, PIP屈曲, DIP屈曲]
        """
        # 设置全局字体和样式
        plt.rcParams['font.sans-serif'] = ['SimHei']  
        plt.rcParams['axes.unicode_minus'] = False    
        plt.rcParams['font.size'] = 10               

        # 创建2x2的子图布局
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{finger_name}关节角度分析', fontsize=14, y=0.95)
        
        # 定义四个自由度的名称和位置
        joint_plots = [
            ('MCP屈曲', joint_indices[0], axes[0,0]),
            ('MCP外展/内收', joint_indices[1], axes[0,1]),
            ('PIP屈曲', joint_indices[2], axes[1,0]),
            ('DIP屈曲', joint_indices[3], axes[1,1])
        ]
        
        # 在2x2布局中绘制每个自由度的数据
        for joint_name, joint_idx, ax in joint_plots:
            # 绘制关节角度数据
            ax.plot(self.emg_timestamps, 
                    synced_joint_data[:, joint_idx],
                    color='blue',
                    linewidth=1.2,
                    label=joint_name)
            
            # 设置子图标题和标签
            ax.set_title(joint_name, pad=10)
            ax.set_xlabel('时间 (秒)', fontsize=9)
            ax.set_ylabel('角度 (度)', fontsize=9)
            
            # 添加网格线
            ax.grid(True, linestyle='--', alpha=0.3)
            
            self.annotate_action_intervals(ax, self.emg_timestamps)
            
            # 添加数据点数量
            num_points = len(synced_joint_data)
            ax.text(0.02, 0.98, f'数据点: {num_points}',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top')

            # 添加最大值和最小值标注
            max_val = np.max(synced_joint_data[:, joint_idx])
            min_val = np.min(synced_joint_data[:, joint_idx])
            ax.text(0.02, 0.90, 
                    f'最大值: {max_val:.1f}°\n最小值: {min_val:.1f}°',
                    transform=ax.transAxes,
                    fontsize=8,
                    verticalalignment='top')
        
        # 调整子图之间的间距
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # 添加整体说明
        fig.text(0.5, 0.02,
                '注：正值表示屈曲/外展，负值表示伸直/内收',
                ha='center',
                fontsize=9)
        
        plt.show()

    def visualize_all_fingers(self, synced_joint_data):
        """
        可视化所有手指的关节角度数据
        """
        # 定义每个手指的关节索引
        fingers = {
            '拇指': [0, 1, 2, 3],      # 示例索引，请根据实际数据调整
            '食指': [4, 5, 6, 7],
            '中指': [8, 9, 10, 11],
            '无名指': [12, 13, 14, 15],
            '小指': [16, 17, 18, 19]
        }
        
        # 为每个手指创建单独的图
        for finger_name, indices in fingers.items():
            self.visualize_single_finger_joints(synced_joint_data, 
                                            finger_name=finger_name,
                                            joint_indices=indices)
        


    def visualize_all_joint_angles(self):
            """在一张图上可视化20个关节角度的原始数据，适合小论文投稿"""
            # 设置全局字体和样式
            plt.rcParams['font.sans-serif'] = ['SimSun']  # 宋体，符合中文规范
            plt.rcParams['axes.unicode_minus'] = False   # 确保负号正常显示
            plt.rcParams['font.size'] = 10               # 默认字体大小适合论文

            # 创建画布，调整为论文常用尺寸
            plt.figure(figsize=(8, 6))  # 宽度8英寸，高度6英寸

            # 计算偏移量，避免关节角度重叠
            joint_amplitude_range = np.max(self.joint_data) - np.min(self.joint_data)
            offset = joint_amplitude_range * 0.8  # 调整偏移量，可根据数据范围微调

            # 使用颜色方案区分20个关节
            colors = plt.cm.tab20(np.linspace(0, 1, 20))  # tab20提供20种颜色

            # 绘制20个关节角度
            for i in range(4):
                offset_data = self.joint_data[:, i] + (i * offset)
                plt.plot(self.joint_timestamps, offset_data, label=f'关节 {i+1}', 
                        color=colors[i], linewidth=1.2)

            # 设置标题和轴标签
            plt.title('20个关节角度的原始数据', fontsize=12, pad=10)
            plt.xlabel('时间 (s)', fontsize=10)
            plt.ylabel('角度 (度, 带偏移)', fontsize=10)

            # 添加图例
            plt.legend(loc='upper right', bbox_to_anchor=(1.15, 1), 
                    fontsize=9, frameon=False, ncol=1)

            # 添加数据点数量（可选）
            num_points = len(self.joint_data)
            plt.text(0.02, 0.98, f'数据点: {num_points}', 
                    transform=plt.gca().transAxes, fontsize=9, verticalalignment='top')


            # 调整布局
            plt.tight_layout(pad=1.5)

            # 保存为高分辨率图片（可选）
            # plt.savefig('all_joint_angles.png', dpi=300, bbox_inches='tight')

            plt.show()

    def evaluate_joint_coordination(self, synced_joint_data):
        """评估关节协调性并可视化相关性矩阵"""

        # 创建 DataFrame
        joint_df = pd.DataFrame(synced_joint_data, columns=[f'关节_{i+1}' for i in range(synced_joint_data.shape[1])])

        # 计算相关性矩阵
        correlation_matrix = joint_df.corr(method='pearson')

        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("关节角度相关性矩阵")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def evaluate_emg_coordination(self, emg_data):
        """评估肌肉协同性并可视化相关矩阵"""
        emg_df = pd.DataFrame(emg_data , columns=[f'肌电通道_{i+1}' for i in range(emg_data.shape[1])])

        # 计算相关矩阵
        correlation_matrix = emg_df.corr(method='pearson')

        # 可视化相关性矩阵
        plt.figure(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True, cbar_kws={"shrink": .8})
        plt.title("肌电通道相关性矩阵")
        plt.xticks(rotation=45)
        plt.yticks(rotation=45)
        plt.tight_layout()
        plt.show()

    def analyze_multimodal_data(self, synced_joint_data, emg_data):
        """
        多模态数据分析：分析同步后的关节角度数据和EMG数据之间的关系
        
        参数:
        synced_joint_data: 同步后的关节角度数据 shape=(n_samples, n_joints)
        emg_data: EMG数据 shape=(n_samples, n_channels)
        """
        
        def plot_correlation_analysis(emg_channel, joint_angle, ch_name, joint_name):
            """绘制EMG和关节角度的相关性分析图"""
            plt.figure(figsize=(15, 5))
            
            # 创建三个子图
            plt.subplot(131)
            plt.plot(emg_channel, label='EMG')
            plt.plot(joint_angle, label='Joint Angle')
            plt.title(f'{ch_name} vs {joint_name}\n原始信号对比')
            plt.legend()
            
            # 散点图
            plt.subplot(132)
            plt.scatter(emg_channel, joint_angle, alpha=0.5)
            plt.xlabel('EMG')
            plt.ylabel('Joint Angle')
            corr, p_value = pearsonr(emg_channel, joint_angle)
            plt.title(f'相关系数: {corr:.3f}\np值: {p_value:.3e}')
            
            # 互相关
            plt.subplot(133)
            correlation = signal.correlate(emg_channel, joint_angle, mode='full')
            lags = signal.correlation_lags(len(emg_channel), len(joint_angle))
            plt.plot(lags, correlation)
            plt.title('互相关分析')
            plt.xlabel('时间延迟 (样本)')
            
            plt.tight_layout()
            plt.show()
            
            return corr, p_value

        def analyze_frequency_coherence(emg_channel, joint_angle, fs=1000):
            """分析EMG和关节角度的频率相干性"""
            f, Cxy = signal.coherence(emg_channel, joint_angle, fs=fs)
            
            plt.figure(figsize=(10, 4))
            plt.plot(f, Cxy)
            plt.xlabel('频率 (Hz)')
            plt.ylabel('相干性')
            plt.title('EMG-关节角度频率相干性分析')
            plt.grid(True)
            plt.show()
            
            return f, Cxy

        def analyze_time_frequency(data, fs=1000):
            """时频分析"""
            f, t, Sxx = signal.spectrogram(data, fs=fs)
            
            plt.figure(figsize=(10, 4))
            plt.pcolormesh(t, f, 10 * np.log10(Sxx))
            plt.ylabel('频率 (Hz)')
            plt.xlabel('时间 (s)')
            plt.colorbar(label='功率/频率 (dB/Hz)')
            plt.show()
            
            return f, t, Sxx

        # 1. 数据预处理
        scaler = StandardScaler()
        emg_normalized = scaler.fit_transform(emg_data)
        joint_normalized = scaler.fit_transform(synced_joint_data)
        
        # 2. 全局相关性分析
        correlation_matrix = np.zeros((emg_data.shape[1], synced_joint_data.shape[1]))
        p_values_matrix = np.zeros_like(correlation_matrix)
        
        for i in range(emg_data.shape[1]):
            for j in range(synced_joint_data.shape[1]):
                correlation_matrix[i, j], p_values_matrix[i, j] = pearsonr(
                    emg_normalized[:, i], joint_normalized[:, j]
                )
        
        # 绘制相关性热力图
        plt.figure(figsize=(12, 8))
        sns.heatmap(correlation_matrix, 
                    annot=True, 
                    fmt='.2f', 
                    cmap='coolwarm',
                    xticklabels=[f'Joint_{i+1}' for i in range(synced_joint_data.shape[1])],
                    yticklabels=[f'EMG_{i+1}' for i in range(emg_data.shape[1])])
        plt.title('EMG-关节角度相关性矩阵')
        plt.tight_layout()
        plt.show()
        
        # 3. 选择最强相关的EMG-关节对进行详细分析
        max_corr_idx = np.unravel_index(np.argmax(np.abs(correlation_matrix)), correlation_matrix.shape)
        emg_idx, joint_idx = max_corr_idx
        
        print(f"\n最强相关对: EMG通道 {emg_idx+1} - 关节 {joint_idx+1}")
        print(f"相关系数: {correlation_matrix[emg_idx, joint_idx]:.3f}")
        print(f"p值: {p_values_matrix[emg_idx, joint_idx]:.3e}")
        
        # 4. 对最强相关对进行详细分析
        # 相关性分析
        plot_correlation_analysis(
            emg_normalized[:, emg_idx],
            joint_normalized[:, joint_idx],
            f'EMG_{emg_idx+1}',
            f'Joint_{joint_idx+1}'
        )
        
        # 频率相干性分析
        analyze_frequency_coherence(
            emg_normalized[:, emg_idx],
            joint_normalized[:, joint_idx]
        )
        
        # 时频分析
        print("\nEMG信号的时频分析:")
        analyze_time_frequency(emg_normalized[:, emg_idx])
        print("\n关节角度的时频分析:")
        analyze_time_frequency(joint_normalized[:, joint_idx])
        
        # 5. 动作段分析
        def analyze_action_segments(emg_data, joint_data, segment_duration=5):
            """分析动作段内的相关性"""
            n_samples = len(emg_data)
            segment_samples = int(segment_duration * 1000)  # 假设采样率为1000Hz
            n_segments = n_samples // segment_samples
            
            segment_correlations = []
            for i in range(n_segments):
                start_idx = i * segment_samples
                end_idx = start_idx + segment_samples
                
                corr, _ = pearsonr(
                    emg_data[start_idx:end_idx],
                    joint_data[start_idx:end_idx]
                )
                segment_correlations.append(corr)
            
            plt.figure(figsize=(10, 4))
            plt.plot(segment_correlations, '-o')
            plt.xlabel('段落编号')
            plt.ylabel('相关系数')
            plt.title('动作段相关性变化')
            plt.grid(True)
            plt.show()
            
            return segment_correlations
        
        print("\n分析动作段相关性:")
        segment_correlations = analyze_action_segments(
            emg_normalized[:, emg_idx],
            joint_normalized[:, joint_idx]
        )
        
        return {
            'correlation_matrix': correlation_matrix,
            'p_values_matrix': p_values_matrix,
            'max_correlation': correlation_matrix[max_corr_idx],
            'max_correlation_indices': max_corr_idx,
            'segment_correlations': segment_correlations
        }


# 使用示例
def main():
    # 初始化同步器
    synchronizer = DataSynchronizer(r'E:\multimodel-acquisition\data\20250319_212126_8_right_6 (1).h5')
    
    # 加载数据
    synchronizer.load_data()



    
    # 执行同步
    synced_joint_data = synchronizer.linear_interpolation_sync()



    # 创建可视化器实例
    visualizer = DataVisualizer(synchronizer.emg_data, synchronizer.emg_timestamps, 
                                synchronizer.joint_data, synchronizer.joint_timestamps)

    visualizer.analyze_multimodal_data(synced_joint_data, visualizer.emg_data)        
    # visualizer.plot_interpolated_results()
    

    
    # # # 可视化结果
    # visualizer.visualize_sync_results(synced_joint_data)  # 可视化同步结果
    # visualizer.visualize_emg_data()                       # 可视化肌电数据
    # visualizer.visualize_all_fingers(synced_joint_data)   # 可视化所有手指的关节数据

    # visualizer.evaluate_joint_coordination(synced_joint_data)
    # visualizer.evaluate_emg_coordination(visualizer.emg_data)

    # # 验证同步质量
    # synchronizer.validate_sync_quality(synced_joint_data)


if __name__ == '__main__':
    main()






# def detect_action_boundaries(self):
#     """检测动作的精确起始和结束时间"""
#     def calculate_signal_energy(data, window_size=100):
#         """计算信号能量"""
#         # 使用滑动窗口计算能量
#         energy = np.zeros(len(data))
#         for i in range(len(data)):
#             start = max(0, i - window_size//2)
#             end = min(len(data), i + window_size//2)
#             energy[i] = np.mean(np.square(data[start:end]))
#         return energy

#     def find_action_boundaries(energy, threshold_factor=2):
#         """找出动作的起始和结束点"""
#         # 计算基线能量（休息状态）
#         baseline = np.percentile(energy, 25)  # 使用25%分位数作为基线
#         threshold = baseline * threshold_factor  # 设置阈值

#         # 初始化结果列表
#         action_segments = []
#         in_action = False
#         action_start = 0

#         # 检测动作边界
#         for i in range(len(energy)):
#             if not in_action and energy[i] > threshold:
#                 # 动作开始
#                 action_start = i
#                 in_action = True
#             elif in_action and energy[i] < threshold:
#                 # 动作结束
#                 if i - action_start > 100:  # 最小动作持续时间（可调整）
#                     action_segments.append((action_start, i))
#                 in_action = False

#         return action_segments

#     # 对每个EMG通道进行分析
#     all_segments = []
#     for channel in range(self.emg_data.shape[1]):
#         # 计算信号能量
#         energy = calculate_signal_energy(self.emg_data[:, channel])
#         # 检测动作边界
#         segments = find_action_boundaries(energy)
#         all_segments.append(segments)

#     # 合并所有通道的结果
#     merged_segments = self.merge_segments(all_segments)
    
#     # 转换为时间戳
#     time_segments = [(self.emg_timestamps[start], self.emg_timestamps[end]) 
#                     for start, end in merged_segments]
    
#     return time_segments

# def merge_segments(self, all_segments):
#     """合并多个通道检测到的动作段"""
#     # 将所有段落展平并排序
#     all_points = []
#     for channel_segments in all_segments:
#         for start, end in channel_segments:
#             all_points.append((start, 'start'))
#             all_points.append((end, 'end'))
    
#     all_points.sort(key=lambda x: x[0])
    
#     # 合并重叠的段落
#     merged = []
#     active_count = 0
#     current_start = None
    
#     for point, point_type in all_points:
#         if point_type == 'start':
#             active_count += 1
#             if active_count == 1:
#                 current_start = point
#         else:  # point_type == 'end'
#             active_count -= 1
#             if active_count == 0:
#                 merged.append((current_start, point))
    
#     return merged

# def visualize_action_detection(self):
#     """可视化动作检测结果"""
#     # 获取动作段
#     action_segments = self.detect_action_boundaries()
    
#     # 创建多子图
#     fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
#     # 绘制EMG信号
#     for i in range(self.emg_data.shape[1]):
#         offset_data = self.emg_data[:, i] + (i * np.max(np.abs(self.emg_data)))
#         ax1.plot(self.emg_timestamps, offset_data, label=f'CH{i+1}')
#     ax1.set_title('EMG信号与动作检测')
#     ax1.set_ylabel('EMG (mV)')
#     ax1.legend()
    
#     # 绘制关节角度数据
#     for i in range(min(4, self.joint_data.shape[1])):  # 显示前4个关节
#         ax2.plot(self.joint_timestamps, self.joint_data[:, i], 
#                 label=f'关节{i+1}')
#     ax2.set_title('关节角度数据')
#     ax2.set_xlabel('时间 (s)')
#     ax2.set_ylabel('角度 (度)')
#     ax2.legend()
    
#     # 在两个子图中标注动作段
#     for start, end in action_segments:
#         ax1.axvspan(start, end, color='gray', alpha=0.2)
#         ax2.axvspan(start, end, color='gray', alpha=0.2)
    
#     plt.tight_layout()
#     plt.show()
    
#     return action_segments

# def extract_action_data(self, action_segments):
#     """提取每个动作段的数据"""
#     action_data = []
    
#     for start_time, end_time in action_segments:
#         # 提取EMG数据
#         emg_mask = (self.emg_timestamps >= start_time) & (self.emg_timestamps <= end_time)
#         emg_segment = self.emg_data[emg_mask]
        
#         # 提取关节角度数据
#         joint_mask = (self.joint_timestamps >= start_time) & (self.joint_timestamps <= end_time)
#         joint_segment = self.joint_data[joint_mask]
        
#         action_data.append({
#             'start_time': start_time,
#             'end_time': end_time,
#             'emg_data': emg_segment,
#             'joint_data': joint_segment
#         })
    
#     return action_data