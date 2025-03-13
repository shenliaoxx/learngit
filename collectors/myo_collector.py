import myo
from threading import Lock
import time
from collections import deque
from datetime import datetime
from data_processing.emg_filter import EMGFilter
import numpy as np

class MyoCollector(myo.DeviceListener):
    def __init__(self):
        super().__init__()
        self.lock = Lock()
        self.connected = False
        self.synced = False
        
        # 初始化滤波器
        self.emg_filter = EMGFilter()
        
        # 初始化数据
        self.raw_emg = [0] * 8
        self.filtered_emg = [0] * 8

        # 添加数据缓冲区 - 增加缓冲区大小或不限制大小
        self.emg_buffer = deque() # 移除最大长度限制
        self.buffer_lock = Lock()
        
        # 只保留采样率和总样本数
        self.total_samples = 0
        self.emg_intervals = deque(maxlen=100)
        self.last_emg_time = 0
        self.current_sampling_rate = 0
        
    def on_connected(self, event):
        print("Myo已连接")
        self.connected = True
        event.device.stream_emg(True)
        event.device.vibrate(myo.VibrationType.short)
        
    def on_disconnected(self, event):
        print("Myo已断开连接")
        self.connected = False
        self.synced = False
        
    def on_arm_synced(self, event):
        print("Myo已同步")
        self.synced = True
        event.device.stream_emg(True)
        event.device.vibrate(myo.VibrationType.medium)
        
    def on_arm_unsynced(self, event):
        print("Myo失去同步")
        self.synced = False
    
    def on_emg(self, event):
        current_time = time.perf_counter()

        # 计算采样间隔和采样率
        if self.last_emg_time > 0:
            interval = current_time - self.last_emg_time
            self.emg_intervals.append(interval)
            
            # 更新采样率
            if len(self.emg_intervals) > 10:
                mean_interval = sum(self.emg_intervals) / len(self.emg_intervals)
                self.current_sampling_rate = 1.0 / mean_interval if mean_interval > 0 else 0
        
        self.last_emg_time = current_time
        self.total_samples += 1

        with self.lock:
            # 获取原始数据
            raw_data = list(event.emg)
            self.raw_emg = raw_data
    
            # 更新滤波器缓冲区
            self.emg_filter.update_buffer(self.raw_emg)
            
            # 应用滤波
            filtered_data = self.emg_filter.filter_data()
            self.filtered_emg = filtered_data.tolist()
        
        with self.buffer_lock:
            self.emg_buffer.append({
                'timestamp': current_time,
                'raw_emg': raw_data,
                'filtered_emg': self.filtered_emg
            })
            
    def get_data(self):
        with self.lock:
            try:
                return {
                    'raw_emg': self.raw_emg.copy(),
                    'filtered_emg': self.filtered_emg.copy()
                }
            except Exception as e:
                print(f"获取EMG数据错误: {e}")
                return {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        
    def get_buffered_data(self):
        """获取并清空缓冲区中的所有数据"""
        with self.buffer_lock:
            try:
                data = list(self.emg_buffer)
                self.emg_buffer.clear()
                
                # 添加日志，监控数据获取
                if len(data) > 0:
                    print(f"获取EMG数据: {len(data)}个样本")
                elif self.connected:
                    print("警告: Myo已连接但未获取到数据")
                
                return data
            except Exception as e:
                print(f"获取缓冲数据错误: {e}")
                return []
    
    def get_status(self):
        # 只返回连接状态、同步状态、采样率和总样本数
        try:
            return {
                'connected': self.connected,
                'synced': self.synced,
                'sampling_rate': self.current_sampling_rate,
                'total_samples': self.total_samples
            }
        except Exception as e:
            print(f"获取状态错误: {e}")
            return {
                'connected': False,
                'synced': False,
                'sampling_rate': 0,
                'total_samples': 0
            }
        
    def get_sampling_stats(self):
        """获取采样统计信息"""
        try:
            mean_interval = 0
            std_interval = 0
            rate = 0
            
            if len(self.emg_intervals) > 0:
                mean_interval = sum(self.emg_intervals) / len(self.emg_intervals)
                rate = 1.0 / mean_interval if mean_interval > 0 else 0
                
                # 计算标准差
                if len(self.emg_intervals) > 1:
                    variance = sum((x - mean_interval) ** 2 for x in self.emg_intervals) / len(self.emg_intervals)
                    std_interval = variance ** 0.5
            
            return {
                'mean_interval': mean_interval,
                'std_interval': std_interval,
                'rate': rate
            }
        except Exception as e:
            print(f"获取采样统计信息错误: {e}")
            return {
                'mean_interval': 0,
                'std_interval': 0,
                'rate': 0
            }

class MyoManager:
    def __init__(self):
        self.collector = None
        self.hub = None
        self.is_running = False
        self.retry_count = 0
        self.max_retries = 5
        self.last_status_print = time.time()
        self.status_print_interval = 5  # 每5秒打印一次状态
        
        try:
            myo.init()
            self.hub = myo.Hub()
            self.collector = MyoCollector()
            self.is_running = True
            print("Myo管理器初始化成功")
        except Exception as e:
            print(f"Myo初始化错误: {e}")
        
    def run_collection(self):
        if not self.hub or not self.collector:
            print("Myo未正确初始化")
            return
            
        while self.is_running:
            try:
                if not self.collector.connected:
                    self.retry_count += 1
                    print(f"尝试连接Myo... (尝试 {self.retry_count}/{self.max_retries})")
                    if self.retry_count >= self.max_retries:
                        print("达到最大重试次数，等待5秒后重试")
                        time.sleep(5)
                        self.retry_count = 0
                else:
                    self.retry_count = 0
                    
                    # 定期打印状态信息
                    current_time = time.time()
                    if current_time - self.last_status_print >= self.status_print_interval:
                        status = self.collector.get_status()
                        print(f"Myo状态 - 采样率: {status['sampling_rate']:.1f} Hz | "
                              f"总样本数: {status['total_samples']} | "
                              f"同步状态: {'已同步' if status['synced'] else '未同步'}")
                        self.last_status_print = current_time
                
                self.hub.run(self.collector, 10)  # 降低轮询间隔以提高采样率
                time.sleep(0.001)  # 减少睡眠时间以提高响应性
            except Exception as e:
                print(f"采集错误: {e}")
                time.sleep(1)
    
    def get_buffered_data(self):
        """获取并清空缓冲区中的所有数据"""
        if not self.collector:
            return []
        
        try:
            data = self.collector.get_buffered_data()
            
            # 添加日志，监控数据获取
            if len(data) > 0:
                print(f"获取EMG数据: {len(data)}个样本")
            elif self.collector.connected:
                print("警告: Myo已连接但未获取到数据")
            
            return data
        except Exception as e:
            print(f"获取缓冲数据错误: {e}")
            return []
    
    def get_latest_data(self):
        if not self.collector:
            return {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        
        try:
            return self.collector.get_data()
        except Exception as e:
            print(f"获取最新数据错误: {e}")
            return {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        
    def get_status(self):
        if not self.collector:
            return {
                'connected': False, 
                'synced': False,
                'sampling_rate': 0,
                'total_samples': 0
            }
        
        try:
            return self.collector.get_status()
        except Exception as e:
            print(f"获取状态错误: {e}")
            return {
                'connected': False, 
                'synced': False,
                'sampling_rate': 0,
                'total_samples': 0
            }

    def get_sampling_stats(self):
        """获取采样统计信息"""
        if not self.collector:
            return {'mean_interval': 0, 'std_interval': 0, 'rate': 0}
        
        try:
            return self.collector.get_sampling_stats()
        except Exception as e:
            print(f"获取采样统计信息错误: {e}")
            return {'mean_interval': 0, 'std_interval': 0, 'rate': 0}
