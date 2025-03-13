import myo
from threading import Lock
import time
from collections import deque
from datetime import datetime
from data_processing.emg_filter import EMGFilter

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
        
        # 帧率计算
        self.timestamp_history = deque(maxlen=100)  # 存储最近100帧的时间戳
        self.frame_count = 0
        self.total_frames = 0
        self.last_frame_time = time.time()
        self.current_fps = 0
        

        
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
        with self.lock:
            # 更新帧率计算
            current_time = time.time()
            self.timestamp_history.append(current_time)
            self.frame_count += 1
            self.total_frames += 1
            
            # 每秒更新一次帧率
            if current_time - self.last_frame_time >= 1.0:
                self.current_fps = self.frame_count / (current_time - self.last_frame_time)
                self.frame_count = 0
                self.last_frame_time = current_time
            
            # 获取原始数据
            raw_data = list(event.emg)
            self.raw_emg = raw_data
    
            # 更新滤波器缓冲区
            self.emg_filter.update_buffer(self.raw_emg)
            
            # 应用滤波
            filtered_data = self.emg_filter.filter_data()
            self.filtered_emg = filtered_data.tolist()
            
    def get_data(self):
        with self.lock:
            return {
                'raw_emg': self.raw_emg.copy(),
                'filtered_emg': self.filtered_emg.copy()
            }
    
    def calculate_frame_rate(self):
        """计算当前帧率"""
        if len(self.timestamp_history) < 2:
            return 0
            
        # 计算最近100帧的平均帧率
        time_diff = self.timestamp_history[-1] - self.timestamp_history[0]
        if time_diff <= 0:
            return 0
        return (len(self.timestamp_history) - 1) / time_diff
            
    def get_status(self):
        return {
            'connected': self.connected,
            'synced': self.synced,
            'frame_rate': self.current_fps,
            'total_frames': self.total_frames,
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
                        print(f"Myo状态 - 帧率: {status['frame_rate']:.1f} FPS | "
                              f"同步状态: {'已同步' if status['synced'] else '未同步'}")
                        self.last_status_print = current_time
                
                self.hub.run(self.collector, 50)  # 降低轮询间隔以提高采样率
                time.sleep(0.005)  # 减少睡眠时间以提高响应性
            except Exception as e:
                print(f"采集错误: {e}")
                time.sleep(1)
    
    def get_latest_data(self):
        if not self.collector:
            return {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        return self.collector.get_data()
        
    def get_status(self):
        if not self.collector:
            return {
                'connected': False, 
                'synced': False,
                'frame_rate': 0,
                'total_frames': 0,
            }
        return self.collector.get_status()
