# import myo
# from threading import Lock
# import time

# class MyoCollector(myo.DeviceListener):
#     def __init__(self):
#         super().__init__()
#         self.emg_data = [0] * 8
#         self.lock = Lock()
#         self.connected = False
        
#     def on_connected(self, event):
#         print("Myo已连接")
#         self.connected = True
#         event.device.stream_emg(True)
        
#     def on_disconnected(self, event):
#         print("Myo已断开连接")
#         self.connected = False
        
#     def on_emg(self, event):
#         with self.lock:
#             self.emg_data = list(event.emg)
            
#     def get_data(self):
#         with self.lock:
#             return self.emg_data.copy()

# class MyoManager:
#     def __init__(self):
#         self.collector = None
#         self.hub = None
#         self.is_running = False
#         try:
#             myo.init()
#             self.hub = myo.Hub()
#             self.collector = MyoCollector()
#             self.is_running = True
#             print("Myo管理器初始化成功")
#         except Exception as e:
#             print(f"Myo初始化错误: {e}")
        
#     def run_collection(self):
#         if not self.hub or not self.collector:
#             print("Myo未正确初始化")
#             return
            
#         while self.is_running:
#             try:
#                 if not self.collector.connected:
#                     print("尝试连接Myo...")
#                 self.hub.run(self.collector, 100)
#                 time.sleep(0.01)
#             except Exception as e:
#                 print(f"采集错误: {e}")
#                 time.sleep(1)
    
#     def get_latest_data(self):
#         if not self.collector:
#             return [0] * 8
#         return self.collector.get_data()
    
import myo
from threading import Lock
import time

class MyoCollector(myo.DeviceListener):
    def __init__(self):
        super().__init__()
        self.emg_data = [0] * 8
        self.lock = Lock()
        self.connected = False
        self.synced = False
        
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
            self.emg_data = list(event.emg)
            
    def get_data(self):
        with self.lock:
            return self.emg_data.copy()
            
    def get_status(self):
        return {
            'connected': self.connected,
            'synced': self.synced
        }

class MyoManager:
    def __init__(self):
        self.collector = None
        self.hub = None
        self.is_running = False
        self.retry_count = 0
        self.max_retries = 5
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
                
                self.hub.run(self.collector, 100)
                time.sleep(0.01)
            except Exception as e:
                print(f"采集错误: {e}")
                time.sleep(1)
    
    def get_latest_data(self):
        if not self.collector:
            return [0] * 8
        return self.collector.get_data()
        
    def get_status(self):
        if not self.collector:
            return {'connected': False, 'synced': False}
        return self.collector.get_status()
