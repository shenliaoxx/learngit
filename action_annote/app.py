import numpy as np
import json
from datetime import datetime
import sys
from pathlib import Path
from scipy.signal import find_peaks, correlate, correlation_lags
from scipy import interpolate
import pandas as pd
from sklearn.cluster import KMeans
import h5py

project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))
from data_processing.datasynchronizer import DataSynchronizer
from flask import Flask, jsonify, render_template, request, send_from_directory

app = Flask(__name__, static_folder='static')

class DataProcessor:
    def __init__(self, file_path):
        self.synchronizer = DataSynchronizer(file_path)
        self.synchronizer.load_data()
        self.emg_data = self.synchronizer.emg_data
        self.joint_data = self.synchronizer.linear_interpolation_sync()
        self.timestamps = self.synchronizer.emg_timestamps
        self.fs = 200
        self.file_path = file_path
        
    def extract_features(self):
        """提取EMG包络和关节速度特征"""
        # EMG包络 - 使用RMS方法
        window = int(0.05 * self.fs)  # 50ms窗口
        emg_squared = self.emg_data**2
        emg_sum = np.sum(emg_squared, axis=1)
        kernel = np.ones(window) / window
        emg_envelope = np.sqrt(np.convolve(emg_sum, kernel, mode='same'))
        
        # 关节速度 - 计算欧几里得距离并平滑
        joint_velocity = np.zeros(len(self.joint_data))
        for i in range(1, len(self.joint_data)):
            joint_velocity[i] = np.linalg.norm(
                self.joint_data[i] - self.joint_data[i-1]
            )
        joint_velocity = np.convolve(joint_velocity, kernel, mode='same')
        
        # 计算加速度 - 速度的一阶导数
        acceleration = np.zeros(len(joint_velocity))
        for i in range(1, len(joint_velocity)):
            acceleration[i] = joint_velocity[i] - joint_velocity[i-1]
        acceleration = np.convolve(acceleration, kernel, mode='same')
        
        # 确保数据为一维数组
        if len(emg_envelope.shape) > 1:
            emg_envelope = emg_envelope.flatten()
        
        # 确保长度一致
        min_length = min(len(emg_envelope), len(joint_velocity), len(acceleration), len(self.timestamps))
        
        return {
            'emg_envelope': emg_envelope[:min_length].tolist(),
            'joint_velocity': joint_velocity[:min_length].tolist(),
            'acceleration': acceleration[:min_length].tolist(),
            'timestamps': self.timestamps[:min_length].tolist()
        }
    
    def get_dataset_info(self):
        """获取数据集信息"""
        file_name = Path(self.file_path).name
        channels = self.emg_data.shape[1] if len(self.emg_data.shape) > 1 else 1
        duration = self.timestamps[-1] - self.timestamps[0]
        
        return {
            'file_name': file_name,
            'channels': channels,
            'duration': duration,
            'sample_rate': self.fs,
            'data_points': len(self.timestamps)
        }
    
    def get_previous_annotations(self):
        """读取之前的标注结果"""
        data_dir = Path(self.file_path).parent
        file_base = Path(self.file_path).stem
        
        json_files = list(data_dir.glob(f"*{file_base}*.json"))
        
        if not json_files:
            return []
        
        # 获取最新的标注文件
        latest_file = max(json_files, key=lambda x: x.stat().st_mtime)
        
        try:
            with open(latest_file, 'r') as f:
                data = json.load(f)
                return data.get('segments', [])
        except:
            return []
    
    def get_channel_info(self):
        """获取通道信息"""
        emg_channels = self.emg_data.shape[1] if len(self.emg_data.shape) > 1 else 1
        joint_channels = self.joint_data.shape[1] if len(self.joint_data.shape) > 1 else 1
        
        return {
            'emg_channels': [{'id': i, 'name': f'EMG通道 {i+1}'} for i in range(emg_channels)],
            'joint_channels': [{'id': i, 'name': f'关节通道 {i+1}'} for i in range(joint_channels)]
        }
    
    # def analyze_segment_correlation(self, segment, emg_channel=0, joint_channel=0):
    #     """使用DataSynchronizer分析选定片段的互相关"""
    #     # 获取片段起止时间
    #     start_time = segment['start_time']
    #     end_time = segment['end_time']
        
    #     # 克隆一个新的同步器实例，避免修改原始数据
    #     synchronizer = DataSynchronizer(self.file_path)
    #     synchronizer.load_data()
        
    #     # 修剪数据到指定时间范围
    #     emg_mask = (synchronizer.emg_timestamps >= start_time) & (synchronizer.emg_timestamps <= end_time)
    #     joint_mask = (synchronizer.joint_timestamps >= start_time) & (synchronizer.joint_timestamps <= end_time)
        
    #     segment_emg_data = synchronizer.emg_data[emg_mask]
    #     segment_emg_timestamps = synchronizer.emg_timestamps[emg_mask]
    #     segment_joint_data = synchronizer.joint_data[joint_mask]
    #     segment_joint_timestamps = synchronizer.joint_timestamps[joint_mask]
        
    #     # 使用线性插值进行同步
    #     joint_interpolator = interpolate.interp1d(
    #         segment_joint_timestamps,
    #         segment_joint_data[:, joint_channel],
    #         kind='linear', 
    #         bounds_error=False,
    #         fill_value='extrapolate'
    #     )
        
    #     # 在EMG时间戳上进行插值
    #     synced_joint_data = joint_interpolator(segment_emg_timestamps)
        
    #     # 获取特定通道的EMG数据
    #     segment_emg = segment_emg_data[:, emg_channel]
        
    #     # 计算互相关
    #     corr = correlate(segment_emg, synced_joint_data, mode='full')
    #     lags = correlation_lags(len(segment_emg), len(synced_joint_data), mode='full')
    #     fs = 1.0 / np.mean(np.diff(segment_emg_timestamps))  # 估计采样率
    #     lag_time = lags / fs
        
    #     # 找到最大相关性位置
    #     max_corr_idx = np.argmax(np.abs(corr))
    #     peak_lag = lags[max_corr_idx]
    #     peak_time = peak_lag / fs
        
    #     # 归一化互相关结果
    #     norm_corr = corr / np.max(np.abs(corr))
        
    #     return {
    #         'peak_time': float(peak_time),
    #         'peak_lag': int(peak_lag),
    #         'corr_data': {
    #             'x': lag_time.tolist(),
    #             'y': norm_corr.tolist()
    #         },
    #         'segment_data': {
    #             'timestamps': segment_emg_timestamps.tolist(),
    #             'emg': segment_emg.tolist(),
    #             'joint': synced_joint_data.tolist()
    #         },
    #         'emg_channel': emg_channel,
    #         'joint_channel': joint_channel,
    #         'segment_duration': float(end_time - start_time)
    #     }
    
    def analyze_segment_correlation(self, segment, emg_channel=0, joint_channel=0):
        """使用DataSynchronizer分析选定片段的互相关（仅添加归一化）"""
        # 获取片段起止时间
        start_time = segment['start_time']
        end_time = segment['end_time']
        
        # 克隆一个新的同步器实例，避免修改原始数据
        synchronizer = DataSynchronizer(self.file_path)
        synchronizer.load_data()
        
        # 修剪数据到指定时间范围
        emg_mask = (synchronizer.emg_timestamps >= start_time) & (synchronizer.emg_timestamps <= end_time)
        joint_mask = (synchronizer.joint_timestamps >= start_time) & (synchronizer.joint_timestamps <= end_time)
        
        segment_emg_data = synchronizer.emg_data[emg_mask]
        segment_emg_timestamps = synchronizer.emg_timestamps[emg_mask]
        segment_joint_data = synchronizer.joint_data[joint_mask]
        segment_joint_timestamps = synchronizer.joint_timestamps[joint_mask]
        
        # 使用线性插值进行同步
        joint_interpolator = interpolate.interp1d(
            segment_joint_timestamps,
            segment_joint_data[:, joint_channel],
            kind='linear', 
            bounds_error=False,
            fill_value='extrapolate'
        )
        
        # 在EMG时间戳上进行插值
        synced_joint_data = joint_interpolator(segment_emg_timestamps)
        
        # 获取特定通道的EMG数据
        segment_emg = segment_emg_data[:, emg_channel]
        
        # === 新增部分：信号归一化（Z-score）===
        segment_emg_norm = (segment_emg - np.mean(segment_emg)) / np.std(segment_emg)
        synced_joint_norm = (synced_joint_data - np.mean(synced_joint_data)) / np.std(synced_joint_data)
        
        # 计算互相关（使用归一化后的信号）
        corr = correlate(segment_emg_norm, synced_joint_norm, mode='full')
        lags = correlation_lags(len(segment_emg_norm), len(synced_joint_norm), mode='full')
        fs = 1.0 / np.mean(np.diff(segment_emg_timestamps))  # 估计采样率
        lag_time = lags / fs
        
        # 找到最大相关性位置
        max_corr_idx = np.argmax(np.abs(corr))
        peak_lag = lags[max_corr_idx]
        peak_time = peak_lag / fs
        
        # 归一化互相关结果
        norm_corr = corr / np.max(np.abs(corr))
        
        return {
            'peak_time': float(peak_time),
            'peak_lag': int(peak_lag),
            'corr_data': {
                'x': lag_time.tolist(),
                'y': norm_corr.tolist()
            },
            'segment_data': {
                'timestamps': segment_emg_timestamps.tolist(),
                # 'emg': segment_emg.tolist(),          # 返回原始EMG（兼容旧代码）
                'emg_norm': segment_emg_norm.tolist(), # 新增归一化后的EMG
                # 'joint': synced_joint_data.tolist(),   # 返回原始关节角度（兼容旧代码）
                'joint_norm': synced_joint_norm.tolist() # 新增归一化后的关节角度
            },
            'emg_channel': emg_channel,
            'joint_channel': joint_channel,
            'segment_duration': float(end_time - start_time)
        }

# 初始化数据处理器
data_file = r'E:\multimodel-acquisition\data\20250326_210703_1_right_6.h5'
data_processor = DataProcessor(data_file)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    features = data_processor.extract_features()
    return jsonify(features)

@app.route('/get_dataset_info')
def get_dataset_info():
    info = data_processor.get_dataset_info()
    return jsonify(info)

@app.route('/get_channel_info')
def get_channel_info():
    info = data_processor.get_channel_info()
    return jsonify(info)

@app.route('/get_previous_annotations')
def get_previous_annotations():
    annotations = data_processor.get_previous_annotations()
    return jsonify({'annotations': annotations})

@app.route('/analyze_segment', methods=['POST'])
def analyze_segment():
    data = request.json
    segment = data.get('segment')
    emg_channel = data.get('emg_channel', 0)
    joint_channel = data.get('joint_channel', 0)
    
    result = data_processor.analyze_segment_correlation(
        segment, 
        emg_channel=emg_channel, 
        joint_channel=joint_channel
    )
    
    return jsonify(result)

@app.route('/save_segments', methods=['POST'])
def save_segments():
    segments = request.json
    file_base = Path(data_file).stem
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f'annotations_{file_base}_{timestamp}.json'
    save_path = Path(project_root) / 'data' / filename
    
    with open(save_path, 'w') as f:
        json.dump(segments, f, indent=4)
    
    return jsonify({'status': 'success', 'filename': filename})

if __name__ == '__main__':
    app.run(debug=True)