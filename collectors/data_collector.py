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

        # 添加独立的数据采集线程
        self.collection_thread = None
        self.collection_running = False

                # 上次打印统计信息的时间
        self.last_stats_print_time = 0
        self.stats_print_interval = 5  # 每5秒打印一次统计信息

    def start_collection_thread(self):
        """启动独立的数据采集线程"""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            print("数据采集线程已在运行")
            return
            
        self.collection_running = True
        self.collection_thread = threading.Thread(target=self._collect_data_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        print("数据采集线程已启动")

    def stop_collection_thread(self):
        """停止数据采集线程"""
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
        print("数据采集线程已停止")

    
    def _collect_data_loop(self):
        """数据采集循环"""
        emg_interval = 1.0 / 200  # 目标EMG采样率：200Hz
        hand_interval = 1.0 / 40   # 提高目标手部数据采样率至40Hz
        
        last_emg_time = 0
        last_hand_time = 0
        last_sample_count_log = 0
        sample_count_log_interval = 100  # 每增加100个样本记录一次日志
        
        # 添加手部数据采样统计
        hand_sample_times = []
        max_sample_times = 100
        
        while self.collection_running:
            current_time = time.perf_counter()
            loop_start_time = current_time

            # 采集EMG数据
            if current_time - last_emg_time >= emg_interval:
                try:
                    # 获取所有缓冲的EMG数据
                    emg_data_batch = self.myo_manager.get_buffered_data()
                    
                    if emg_data_batch and self.is_recording:
                        for emg_data in emg_data_batch:
                            self.add_emg_data(emg_data['timestamp'], {
                                'raw_emg': emg_data['raw_emg'],
                                'filtered_emg': emg_data['filtered_emg']
                            })
                        
                        # 定期记录样本数量
                        current_samples = len(self.emg_buffer['timestamps'])
                        if current_samples - last_sample_count_log >= sample_count_log_interval:
                            print(f"当前EMG样本数: {current_samples}")
                            last_sample_count_log = current_samples
                            
                    last_emg_time = current_time
                except Exception as e:
                    print(f"EMG数据采集错误: {e}")
                    import traceback
                    print(traceback.format_exc())

                    
            # 采集手部数据
            if current_time - last_hand_time >= hand_interval:
                hand_sample_start = time.perf_counter()
                try:
                    hand_data = self.realsense_collector.get_hand_data()
                    if hand_data and self.is_recording:
                        self.add_hand_data(current_time, hand_data)
                        
                        # 记录采样时间
                        hand_sample_time = time.perf_counter() - hand_sample_start
                        hand_sample_times.append(hand_sample_time)
                        if len(hand_sample_times) > max_sample_times:
                            hand_sample_times.pop(0)
                            
                    last_hand_time = current_time
                except Exception as e:
                    print(f"手部数据采集错误: {e}")
            
            # 定期打印统计信息
            if self.is_recording and current_time - self.last_stats_print_time >= self.stats_print_interval:
                self._print_stats()

                # 打印Myo采样统计信息
                sampling_stats = self.myo_manager.get_sampling_stats()
                print(f"Myo采样统计 - 采样率: {sampling_stats['rate']:.1f} Hz | "
                    f"平均间隔: {sampling_stats['mean_interval']*1000:.2f} ms | "
                    f"标准差: {sampling_stats['std_interval']*1000:.2f} ms")
                
                # 打印手部数据采样统计
                if hand_sample_times:
                    avg_sample_time = sum(hand_sample_times) / len(hand_sample_times)
                    print(f"手部数据采样统计 - 平均采样时间: {avg_sample_time*1000:.2f} ms | "
                          f"目标采样率: {1.0/hand_interval:.1f} Hz")
                
                self.last_stats_print_time = current_time
                
            # 计算循环耗时
            loop_time = time.perf_counter() - loop_start_time
            
            # 动态调整休眠时间
            sleep_time = max(0, min(emg_interval, hand_interval) / 2 - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)
            else:
                # 如果处理时间已经超过了采样间隔的一半，只进行最小休眠
                time.sleep(0.0001)  # 0.1ms

    def _print_stats(self):
        """打印当前统计信息"""
        if not self.is_recording:
            return
            
        # 计算EMG采样率
        if len(self.emg_buffer['timestamps']) > 1:
            first_ts = self.emg_buffer['timestamps'][0]
            last_ts = self.emg_buffer['timestamps'][-1]
            duration = last_ts - first_ts
            if duration > 0:
                emg_rate = (len(self.emg_buffer['timestamps']) - 1) / duration
                print(f"EMG采样率: {emg_rate:.2f} Hz, 样本数: {len(self.emg_buffer['timestamps'])}")
        
        # 计算手部数据采样率
        if len(self.hand_buffer['timestamps']) > 1:
            first_ts = self.hand_buffer['timestamps'][0]
            last_ts = self.hand_buffer['timestamps'][-1]
            duration = last_ts - first_ts
            if duration > 0:
                hand_rate = (len(self.hand_buffer['timestamps']) - 1) / duration
                print(f"手部数据采样率: {hand_rate:.2f} Hz, 样本数: {len(self.hand_buffer['timestamps'])}")

    def add_emg_data(self, timestamp, data):
        """添加EMG数据"""
        with self.lock:
            if self.is_recording:
                self.emg_buffer['timestamps'].append(timestamp)
                self.emg_buffer['raw_data'].append(data['raw_emg'])
                self.emg_buffer['filtered_data'].append(data['filtered_emg'])


                self.stats['emg_samples'] += 1
                # 计算平均采样率（基于整个记录过程）
                if len(self.emg_buffer['timestamps']) > 1:
                    first_timestamp = self.emg_buffer['timestamps'][0]
                    duration = timestamp - first_timestamp
                    if duration > 0:
                        self.stats['emg_rate'] = (len(self.emg_buffer['timestamps']) - 1) / duration
                
                # 保存最后一次时间戳
                self.stats['last_emg_time'] = timestamp
        

    def add_hand_data(self, timestamp, data):
        """添加手部数据"""
        with self.lock:
            if self.is_recording:
                try:
                    if data and isinstance(data, dict):
                        # 检查是否有重复时间戳
                        if self.hand_buffer['timestamps'] and abs(timestamp - self.hand_buffer['timestamps'][-1]) < 0.001:
                            # 时间戳太接近，跳过
                            return
                            
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
                        # 计算平均采样率（基于整个记录过程）
                        if len(self.hand_buffer['timestamps']) > 1:
                            first_timestamp = self.hand_buffer['timestamps'][0]
                            duration = timestamp - first_timestamp
                            if duration > 0:
                                self.stats['hand_rate'] = (len(self.hand_buffer['timestamps']) - 1) / duration
                        
                        # 保存最后一次时间戳
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

        # 确保数据采集线程正在运行
        if not self.collection_running:
            self.start_collection_thread()
        
        self.last_stats_print_time = time.perf_counter()  # 重置统计信息打印时间

    def stop_recording(self):
        """停止记录数据"""
        with self.lock:
            if not self.is_recording:
                return None
            
            self.is_recording = False
            # 立即调用保存方法
            filepath = self.save_recording()
            print(f"停止记录，文件保存在: {filepath}")  # 添加日志输出
            return filepath


    def save_recording(self):
        """保存记录数据"""
        try:
            # 简化的文件名，只使用时间戳
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"recording_{timestamp}.h5"
            
            # 简化的保存路径，直接保存在 data 目录下
            if not os.path.exists('data'):
                os.makedirs('data')
                
            filepath = os.path.join('data', filename)
            
            print(f"正在保存数据到: {filepath}, EMG样本数: {len(self.emg_buffer['timestamps'])}, 手部样本数: {len(self.hand_buffer['timestamps'])}")

            # 分批保存大型数据集
            batch_size = 500  # 每批处理的样本数
            
            with h5py.File(filepath, 'w') as f:
                # 创建EMG数据集
                emg_group = f.create_group('emg')
                
                # 获取EMG数据的形状
                emg_timestamps_shape = (len(self.emg_buffer['timestamps']),)
                emg_raw_shape = (len(self.emg_buffer['raw_data']), len(self.emg_buffer['raw_data'][0]) if self.emg_buffer['raw_data'] else 8)
                emg_filtered_shape = (len(self.emg_buffer['filtered_data']), len(self.emg_buffer['filtered_data'][0]) if self.emg_buffer['filtered_data'] else 8)
                
                # 创建数据集
                emg_timestamps = emg_group.create_dataset('timestamps', shape=emg_timestamps_shape, dtype=np.float64)
                emg_raw = emg_group.create_dataset('raw_data', shape=emg_raw_shape, dtype=np.float64)
                emg_filtered = emg_group.create_dataset('filtered_data', shape=emg_filtered_shape, dtype=np.float64)
                
                # 分批写入EMG数据
                for i in range(0, len(self.emg_buffer['timestamps']), batch_size):
                    end = min(i + batch_size, len(self.emg_buffer['timestamps']))
                    emg_timestamps[i:end] = self.emg_buffer['timestamps'][i:end]
                    emg_raw[i:end] = self.emg_buffer['raw_data'][i:end]
                    emg_filtered[i:end] = self.emg_buffer['filtered_data'][i:end]
                    print(f"已保存EMG数据批次: {i//batch_size + 1}/{(len(self.emg_buffer['timestamps'])-1)//batch_size + 1}, 样本: {i}-{end}")
                
                # 保存手部数据
                hand_group = f.create_group('hand')
                hand_group.create_dataset('timestamps', data=self.hand_buffer['timestamps'])
                hand_group.create_dataset('joint_angles', data=self.hand_buffer['joint_angles'])

                # 保存基本的统计信息
                stats = f.create_group('stats')
                stats.attrs['emg_samples'] = len(self.emg_buffer['timestamps'])
                stats.attrs['hand_samples'] = len(self.hand_buffer['timestamps'])
                
            print(f"数据保存成功: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"保存数据错误: {e}")
            import traceback
            print(traceback.format_exc())
            return None

    def get_stats(self):
        """获取统计信息"""
        with self.lock:
            try:
                return {
                    'is_recording': self.is_recording,
                    'duration': time.perf_counter() - self.recording_start_time if self.is_recording else 0,
                    'emg_rate': self.stats['emg_rate'],
                    'hand_rate': self.stats['hand_rate'],
                    'emg_samples': self.stats['emg_samples'],
                    'hand_samples': self.stats['hand_samples']
                }
            except Exception as e:
                print(f"获取统计信息错误: {e}")
                import traceback
                print(traceback.format_exc())
                return {
                    'is_recording': False,
                    'duration': 0,
                    'emg_rate': 0,
                    'hand_rate': 0,
                    'emg_samples': 0,
                    'hand_samples': 0
                }
                
                
                



            
            
            
