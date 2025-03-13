import threading
import time
import os
from datetime import datetime
import h5py
import numpy as np
class DataCollector:
    """多模态数据采集器"""
    def __init__(self, myo_manager, realsense_collector):
        self.myo_manager = myo_manager
        self.realsense_collector = realsense_collector

        # 数据缓冲区
        self.emg_buffer = {
            'timestamps': [],
            'raw_data': [],
            'filtered_data': []
        }
        self.hand_buffer = {
            'timestamps': [],
            'joint_angles': []
        }
        
        # 记录状态
        self.is_recording = False
        self.recording_start_time = None
        self.recording_metadata = {}


        self.stats = {
            'emg_rate': 0,
            'hand_rate': 0,
            'emg_samples': 0,
            'hand_samples': 0,
            'last_emg_time': 0,
            'last_hand_time': 0
        }

        self.lock = threading.Lock()

    def add_emg_data(self, timestamp, data):
        """添加EMG数据"""
        with self.lock:
            if self.is_recording:
                self.emg_buffer['timestamps'].append(timestamp)
                self.emg_buffer['raw_data'].append(data['raw_emg'])
                self.emg_buffer['filtered_data'].append(data['filtered_emg'])


                self.stats['emg_samples'] += 1
                if self.stats['last_emg_time'] > 0:
                    interval = timestamp - self.stats['last_emg_time']
                    self.stats['emg_rate'] = 1.0 / interval
                self.stats['last_emg_time'] = timestamp
        

    def add_hand_data(self, timestamp, data):
        """添加手部数据"""
        with self.lock:
            if self.is_recording:
                try:

                    if data and isinstance(data, dict):
                        self.hand_buffer['timestamps'].append(timestamp)

                        angles_data = []
                        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                            
                            # MCP关节2个自由度，屈曲和外展
                            angles_data.append(data.get(f"{finger}_mcp_flexion", 0.0))
                            angles_data.append(data.get(f"{finger}_mcp_abduction", 0.0))

                            # PIP,DIP关节1个自由度，屈曲
                            angles_data.append(data.get(f"{finger}_pip_flexion", 0.0))
                            angles_data.append(data.get(f"{finger}_dip_flexion", 0.0))

                                                        
                        self.hand_buffer['joint_angles'].append(angles_data)



                        self.stats['hand_samples'] += 1
                        if self.stats['last_hand_time'] > 0:
                            interval = timestamp - self.stats['last_hand_time']
                            self.stats['hand_rate'] = 1.0 / interval
                        self.stats['last_hand_time'] = timestamp
 
                except Exception as e:
                    print(f"处理手部数据错误: {e}")
                    import traceback
                    print(traceback.format_exc())           

    def start_recording(self, metadata=None):
        """开始记录数据"""
        with self.lock:
            self.is_recording = True
            self.recording_start_time = time.perf_counter()
            self.recording_metadata = metadata or {}
            self.recording_metadata['start_time'] = self.recording_start_time

        self.emg_buffer = {
            'timestamps' : [], 
            'raw_data' : [], 
            'filtered_data' : []
            }
        self.hand_buffer={
            'timestamps' : [], 
            'joint_angles' : []
            }

        self.stats = {
            'emg_rate' : 0,
            'hand_rate' : 0,
            'emg_samples' : 0,
            'hand_samples' : 0,
            'last_emg_time' : 0,
            'last_hand_time' : 0
        }

    def stop_recording(self):
        """停止记录数据"""
        with self.lock:
            if not self.is_recording:
                return None
            
            self.is_recording = False
            recording_end_time = time.perf_counter()
            self.recording_metadata['end_time'] = recording_end_time
            self.recording_metadata['duration'] = recording_end_time - self.recording_start_time

            # 调用保存数据的方法
            filepath = self.save_recording()
            return filepath


    def save_recording(self):
        """保存记录数据"""
        try:
            # 构建文件名
            # timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            # subject_id = self.recording_metadata.get('subject_id', 'unknown')
            # action = self.recording_metadata.get('action', 'unknown')
            # filename = f"{subject_id}_{action}_{timestamp}.h5"
            # filepath = os.path.join('data', 'raw', subject_id, filename)


            # # 创建目录
            # os.makedirs(os.path.dirname(filepath), exist_ok=True)

            # 简化的文件名，只使用时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"recording_{timestamp}.h5"
            
            # 简化的保存路径，直接保存在 data 目录下
            if not os.path.exists('data'):
                os.makedirs('data')
                
            filepath = os.path.join('data', filename)
            
            print(f"正在保存数据到: {filepath}")  # 添加日志输出

            with h5py.File(filepath, 'w') as f:

                # 保存EMG数据
                if len(self.emg_buffer['timestamps']) > 0:
                    emg_group = f.create_group('emg')
                    emg_group.create_dataset('timestamps', data=np.array(self.emg_buffer['timestamps']))
                    emg_group.create_dataset('raw_data', data=np.array(self.emg_buffer['raw_data']))
                    emg_group.create_dataset('filtered_data', data=np.array(self.emg_buffer['filtered_data']))
                    emg_group.attrs['sample_rate'] = self.stats['emg_rate']
                    emg_group.attrs['total_samples'] = len(self.emg_buffer['timestamps'])

                if len(self.hand_buffer['timestamps']) > 0:
                    hand_group = f.create_group('hand')
                    hand_group.create_dataset('timestamps', data=np.array(self.hand_buffer['timestamps']))
                    hand_group.create_dataset('joint_angles', data=np.array(self.hand_buffer['joint_angles']))

                    # 添加关节角度的描述信息
                    joint_names = []
                    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                        joint_names.extend([
                            f"{finger}_mcp_flexion",
                            f"{finger}_mcp_abduction",
                            f"{finger}_pip_flexion",
                            f"{finger}_dip_flexion"
                        ])
                    hand_group.attrs['joint_names'] = np.array(joint_names, dtype='S')
                    hand_group.attrs['sample_rate'] = self.stats['hand_rate']
                    hand_group.attrs['total_samples'] = len(self.hand_buffer['timestamps'])

                 # 保存统计信息
                stats_group = f.create_group('stats')
                for key, value in self.stats.items():
                    stats_group.attrs[key] = value

                print(f"数据保存成功: {filepath}")
                print(f"EMG平均采样率: {self.stats['emg_rate']:.2f} Hz")
                print(f"手部数据平均采样率: {self.stats['hand_rate']:.2f} Hz")
            
            return filepath
        
        except Exception as e:
            print(f"保存数据错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            return {
                'is_recording': self.is_recording,
                'duration': time.perf_counter() - self.recording_start_time if self.is_recording else 0,
                'emg_rate': self.stats['emg_rate'],
                'hand_rate': self.stats['hand_rate'],
                'emg_samples': self.stats['emg_samples'],
                'hand_samples': self.stats['hand_samples']
            }           
                
                
                



            
            
            
