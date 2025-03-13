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
        # 设备状态
        self.connected = False
        self.synced = False
        
        # 数据处理
        self.emg_filter = EMGFilter()
        self.raw_emg = [0] * 8
        self.filtered_emg = [0] * 8

        # 添加数据缓冲区 - 增加缓冲区大小或不限制大小
        self.emg_buffer = deque(maxlen=5000) 

        # 线程锁
        self.buffer_lock = Lock()
        self.lock = Lock()
        
        # 采样统计
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
        """处理EMG数据事件"""
        current_time = time.perf_counter()
        self._update_sampling_rate(current_time)
        self._process_emg_data(event.emg, current_time)
    
    def _update_sampling_rate(self, current_time):
        """更新采样率统计"""
        if self.last_emg_time > 0:
            interval = current_time - self.last_emg_time
            self.emg_intervals.append(interval)
            
            if len(self.emg_intervals) > 10:
                mean_interval = sum(self.emg_intervals) / len(self.emg_intervals)
                self.current_sampling_rate = 1.0 / mean_interval if mean_interval > 0 else 0
        
        self.last_emg_time = current_time
        self.total_samples += 1
    
    def _process_emg_data(self, emg_data, timestamp):
        """处理EMG数据"""
        with self.lock:

            self.raw_emg = list(emg_data)
    
            # 应用滤波
            self.emg_filter.update_buffer(self.raw_emg)
            self.filtered_emg = self.emg_filter.filter_data().tolist()

        with self.buffer_lock:
            self.emg_buffer.append({
                'timestamp': timestamp,
                'raw_emg': self.raw_emg,
                'filtered_emg': self.filtered_emg
            })
            
    def get_data(self):
        """获取最新的EMG数据"""
        with self.lock:
            return {
                'raw_emg': self.raw_emg.copy(),
                'filtered_emg': self.filtered_emg.copy()
            }

        
    def get_buffered_data(self):
        """获取并清空缓冲区中的所有数据"""
        with self.buffer_lock:
            data = list(self.emg_buffer)
            self.emg_buffer.clear()
            return data

    
    def get_status(self):
        # 只返回连接状态、同步状态、采样率和总样本数
        return {
            'connected': self.connected,
            'synced': self.synced,
            'sampling_rate': self.current_sampling_rate,
            'total_samples': self.total_samples
        }
 
        
    def get_sampling_stats(self):
        """获取采样统计信息"""
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


class MyoManager:
    def __init__(self):
        self.collector = None
        self.hub = None
        self.is_running = False
        self.retry_count = 0
        self.max_retries = 5
        self.last_status_print = time.time()
        self.status_print_interval = 5  # 每5秒打印一次状态
        self._initialize_myo()


    def _initialize_myo(self):
        """初始化Myo设备"""
        try:
            myo.init()
            self.hub = myo.Hub()
            self.collector = MyoCollector()
            self.is_running = True
            print("Myo管理器初始化成功")
        except Exception as e:
            print(f"Myo初始化错误: {e}")

    def run_collection(self):
        """运行数据采集循环"""
        if not self.hub or not self.collector:
            print("Myo未正确初始化")
            return
            
        while self.is_running:
            try:
                self._handle_connection()
                self._print_status()
                
                # 运行Myo Hub
                self.hub.run(self.collector, 10)
                time.sleep(0.001)
            except Exception as e:
                print(f"采集错误: {e}")
                time.sleep(1)


    def _handle_connection(self):
        """处理连接状态"""
        if not self.collector.connected:
            self.retry_count += 1
            print(f"尝试连接Myo... (尝试 {self.retry_count}/{self.max_retries})")
            if self.retry_count >= self.max_retries:
                print("达到最大重试次数，等待5秒后重试")
                time.sleep(5)
                self.retry_count = 0
        else:
            self.retry_count = 0
    
    def _print_status(self):
        """打印状态信息"""
        current_time = time.time()
        if (current_time - self.last_status_print >= self.status_print_interval and 
            self.collector.connected):
            status = self.collector.get_status()
            print(f"Myo状态 - 采样率: {status['sampling_rate']:.1f} Hz | "
                  f"总样本数: {status['total_samples']} | "
                  f"同步状态: {'已同步' if status['synced'] else '未同步'}")
            self.last_status_print = current_time

    def get_buffered_data(self):
        """获取并清空缓冲区中的所有数据"""
        data = self.collector.get_buffered_data()
        return data

    
    def get_latest_data(self):
        """获取最新的EMG数据"""
        return self.collector.get_data()
        
    def get_status(self):
        """获取设备状态信息"""
        return self.collector.get_status()


    def get_sampling_stats(self):
        """获取采样统计信息"""
        return self.collector.get_sampling_stats()
