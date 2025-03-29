import streamlit as st
import h5py
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler

# 设置中文显示
matplotlib.rcParams['font.family'] = 'SimHei'
plt.rcParams['axes.unicode_minus'] = False

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
            self.emg_data = f['emg/filtered_data'][:]
            self.emg_timestamps = f['emg/timestamps'][:]
            self.joint_data = f['hand/joint_angles'][:]
            self.joint_timestamps = f['hand/timestamps'][:]
            
    def find_common_time_range(self):
        """找到两个数据流的共同时间范围"""
        start_time = max(self.emg_timestamps[0], self.joint_timestamps[0])
        end_time = min(self.emg_timestamps[-1], self.joint_timestamps[-1])
        return start_time, end_time
        
    def trim_data_to_common_range(self, start_time, end_time):
        """将数据修剪到共同时间范围"""
        emg_mask = (self.emg_timestamps >= start_time) & (self.emg_timestamps <= end_time)
        self.emg_data = self.emg_data[emg_mask]
        self.emg_timestamps = self.emg_timestamps[emg_mask]
        
        joint_mask = (self.joint_timestamps >= start_time) & (self.joint_timestamps <= end_time)
        self.joint_data = self.joint_data[joint_mask]
        self.joint_timestamps = self.joint_timestamps[joint_mask]

    def interpolate_data(self, method='linear'):
        """使用插值进行数据同步"""
        start_time, end_time = self.find_common_time_range()
        self.trim_data_to_common_range(start_time, end_time)
        
        if method == 'linear':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data, axis=0, kind='linear', fill_value='extrapolate')
        elif method == 'cubic':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data, axis=0, kind='cubic', fill_value='extrapolate')
        elif method == 'nearest':
            f = interpolate.interp1d(self.joint_timestamps, self.joint_data, axis=0, kind='nearest', fill_value='extrapolate')
        else:
            raise ValueError("不支持的插值方法")
        
        return f(self.emg_timestamps)

class DataVisualizer:
    def __init__(self, emg_data, emg_timestamps, joint_data, joint_timestamps):
        self.emg_data = emg_data
        self.emg_timestamps = emg_timestamps
        self.joint_data = joint_data
        self.joint_timestamps = joint_timestamps

    def annotate_action_intervals(self, ax, timestamps, action_duration=4, rest_duration=4):
        """在指定轴上标注动作和休息区间"""
        cycle_duration = action_duration + rest_duration
        t0 = timestamps[0]
        relative_timestamps = timestamps - t0
        total_duration = relative_timestamps[-1]
        num_cycles = int(total_duration // cycle_duration) + 1

        for cycle in range(num_cycles):
            action_start = cycle * cycle_duration
            action_end = action_start + action_duration
            rest_end = action_start + cycle_duration

            if action_start < total_duration:
                ax.axvspan(action_start + t0, min(action_end + t0, timestamps[-1]), 
                          alpha=0.2, color='gray', label='动作' if cycle == 0 else None)
                if rest_end <= total_duration:
                    ax.axvspan(action_end + t0, min(rest_end + t0, timestamps[-1]), 
                              alpha=0.1, color='lightgreen', label='休息' if cycle == 0 else None)

    def visualize_emg_data(self):
        """可视化8通道肌电信号"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        emg_amplitude_range = np.max(self.emg_data) - np.min(self.emg_data)
        offset = emg_amplitude_range * 0.8
        colors = plt.cm.tab10(np.linspace(0, 1, 8))
        
        for i in range(7, -1, -1):
            offset_data = self.emg_data[:, i] + (i * offset)
            ax.plot(self.emg_timestamps, offset_data, label=f'CH{i+1}', color=colors[i], linewidth=1.2)
        
        self.annotate_action_intervals(ax, self.emg_timestamps)
        ax.set_title('8通道肌电信号')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('肌电信号 (mV)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9, frameon=False, ncol=1)
        
        st.pyplot(fig)

    def visualize_joint_data(self, joint_indices=None):
        """可视化关节角度数据"""
        if joint_indices is None:
            joint_indices = range(min(4, self.joint_data.shape[1]))
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        joint_amplitude_range = np.max(self.joint_data) - np.min(self.joint_data)
        offset = joint_amplitude_range * 0.8
        colors = plt.cm.tab20(np.linspace(0, 1, len(joint_indices)))
        
        for i, idx in enumerate(joint_indices):
            offset_data = self.joint_data[:, idx] + (i * offset)
            ax.plot(self.joint_timestamps, offset_data, label=f'关节 {idx+1}', color=colors[i], linewidth=1.2)
        
        self.annotate_action_intervals(ax, self.joint_timestamps)
        ax.set_title('关节角度数据')
        ax.set_xlabel('时间 (s)')
        ax.set_ylabel('角度 (度, 带偏移)')
        ax.legend(loc='upper right', bbox_to_anchor=(1.15, 1), fontsize=9, frameon=False, ncol=1)
        
        st.pyplot(fig)

    def visualize_sync_comparison(self, synced_joint_data, joint_idx=0):
        """可视化同步前后的数据比较"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        
        # 原始关节数据
        ax1.plot(self.joint_timestamps, self.joint_data[:, joint_idx], label='原始数据', color='blue')
        ax1.set_title('原始关节数据')
        ax1.set_ylabel('角度 (度)')
        ax1.legend()
        
        # 同步关节数据
        ax2.plot(self.emg_timestamps, synced_joint_data[:, joint_idx], label='同步数据', color='red')
        ax2.set_title('同步关节数据')
        ax2.set_xlabel('时间 (s)')
        ax2.set_ylabel('角度 (度)')
        ax2.legend()
        
        for ax in [ax1, ax2]:
            self.annotate_action_intervals(ax, self.joint_timestamps)
        
        st.pyplot(fig)

    def visualize_single_finger_joints(self, synced_joint_data, finger_name="手指", joint_indices=None):
        """可视化单个手指的关节角度"""
        if joint_indices is None:
            joint_indices = range(4)  # 默认显示前4个关节
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        fig.suptitle(f'{finger_name}关节角度分析', fontsize=14, y=0.95)
        
        # 定义关节名称
        joint_names = ['MCP屈曲', 'MCP外展/内收', 'PIP屈曲', 'DIP屈曲']
        
        for i, (joint_name, joint_idx, ax) in enumerate(zip(
            joint_names,
            joint_indices,
            [axes[0,0], axes[0,1], axes[1,0], axes[1,1]]
        )):
            if joint_idx >= synced_joint_data.shape[1]:
                continue  # 跳过不存在的关节索引

            self.annotate_action_intervals(ax, self.emg_timestamps)   
            ax.plot(self.emg_timestamps, synced_joint_data[:, joint_idx],
                    color='blue', linewidth=1.2)
            ax.set_title(joint_name, pad=10)
            ax.set_xlabel('时间 (秒)')
            ax.set_ylabel('角度 (度)')
            ax.grid(True, linestyle='--', alpha=0.3)
            
            # 添加数据统计信息
            joint_data = synced_joint_data[:, joint_idx]
            stats_text = f"最大值: {np.max(joint_data):.1f}°\n最小值: {np.min(joint_data):.1f}°\n均值: {np.mean(joint_data):.1f}°"
            ax.text(0.02, 0.95, stats_text, transform=ax.transAxes,
                    fontsize=8, verticalalignment='top', bbox=dict(facecolor='white', alpha=0.7))
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        st.pyplot(fig)

    def visualize_all_fingers(self, synced_joint_data):
        """
        可视化所有手指的关节角度数据
        """
        # 定义每个手指的关节索引
        fingers = {
            '拇指': [0, 1, 2, 3],
            '食指': [4, 5, 6, 7],
            '中指': [8, 9, 10, 11],
            '无名指': [12, 13, 14, 15],
            '小指': [16, 17, 18, 19]
        }
        
        # 为每个手指创建单独的标签页
        tabs = st.tabs(list(fingers.keys()))
        
        for tab, (finger_name, indices) in zip(tabs, fingers.items()):
            with tab:
                self.visualize_single_finger_joints(
                    synced_joint_data, 
                    finger_name=finger_name,
                    joint_indices=indices
                )

def main():
    st.title("多模态数据同步与可视化系统")
    st.markdown("""
    ### 功能说明
    1. 上传HDF5格式的实验数据文件
    2. 选择插值方法进行时间同步
    3. 可视化EMG信号和关节角度数据
    4. 比较同步前后的数据
    """)
    
    # 文件上传
    uploaded_file = st.file_uploader("上传HDF5数据文件", type=['h5'])
    
    if uploaded_file is not None:
        # 保存临时文件
        with open("temp.h5", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # 初始化同步器
        try:
            synchronizer = DataSynchronizer("temp.h5")
            synchronizer.load_data()
            st.success("数据加载成功！")
            
            # 显示基本信息
            st.subheader("数据基本信息")
            col1, col2 = st.columns(2)
            with col1:
                st.write(f"EMG数据点数: {len(synchronizer.emg_timestamps)}")
                st.write(f"EMG采样率: {1/np.mean(np.diff(synchronizer.emg_timestamps)):.2f} Hz")
            with col2:
                st.write(f"关节数据点数: {len(synchronizer.joint_timestamps)}")
                st.write(f"关节采样率: {1/np.mean(np.diff(synchronizer.joint_timestamps)):.2f} Hz")
            
            # 插值方法选择
            st.subheader("时间同步设置")
            method = st.selectbox("选择插值方法", ['linear', 'cubic', 'nearest'], index=0)
            
            if st.button("执行时间同步"):
                with st.spinner("正在执行时间同步..."):
                    synced_joint_data = synchronizer.interpolate_data(method)
                    st.session_state['synced_data'] = synced_joint_data
                    st.success("时间同步完成！")
                    
                    # 显示同步质量信息
                    st.subheader("同步质量评估")
                    emg_intervals = np.diff(synchronizer.emg_timestamps)
                    joint_intervals = np.diff(synchronizer.joint_timestamps)
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("EMG采样间隔均值", f"{np.mean(emg_intervals)*1000:.2f} ms")
                        st.metric("EMG采样间隔标准差", f"{np.std(emg_intervals)*1000:.2f} ms")
                    with col2:
                        st.metric("关节采样间隔均值", f"{np.mean(joint_intervals)*1000:.2f} ms")
                        st.metric("关节采样间隔标准差", f"{np.std(joint_intervals)*1000:.2f} ms")
                    
                    st.write(f"同步后数据点数: {len(synced_joint_data)}")
                    st.write(f"NaN值数量: {np.isnan(synced_joint_data).sum()}")
            
            if 'synced_data' in st.session_state:
                visualizer = DataVisualizer(
                    synchronizer.emg_data, synchronizer.emg_timestamps,
                    synchronizer.joint_data, synchronizer.joint_timestamps
                )
                
                st.subheader("数据可视化")
                tab1, tab2, tab3, tab4 = st.tabs(["EMG信号", "关节角度", "同步比较", "手指关节"])
                
                with tab1:
                    st.write("### 8通道肌电信号")
                    visualizer.visualize_emg_data()
                
                with tab2:
                    st.write("### 关节角度数据")
                    num_joints = synchronizer.joint_data.shape[1]
                    joints_to_plot = st.multiselect(
                        "选择要显示的关节", 
                        range(num_joints), 
                        default=range(min(4, num_joints)),
                        format_func=lambda x: f"关节 {x+1}"
                    )
                    visualizer.visualize_joint_data(joints_to_plot)
                
                with tab3:
                    st.write("### 同步前后数据比较")
                    joint_idx = st.selectbox(
                        "选择要比较的关节", 
                        range(synchronizer.joint_data.shape[1]), 
                        format_func=lambda x: f"关节 {x+1}"
                    )
                    visualizer.visualize_sync_comparison(st.session_state['synced_data'], joint_idx)
                
                with tab4:
                    st.write("### 各手指关节角度分析")
                    visualizer.visualize_all_fingers(st.session_state['synced_data'])
        
        except Exception as e:
            st.error(f"数据处理出错: {str(e)}")

if __name__ == "__main__":
    main()


######### streamlit run data_visual.py