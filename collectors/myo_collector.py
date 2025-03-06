import myo
from threading import Lock
import time

class MyoCollector(myo.DeviceListener):
    def __init__(self):
        super().__init__()
        self.emg_data = [0] * 8
        self.lock = Lock()
        self.connected = False
        
    def on_connected(self, event):
        print("Myo已连接")
        self.connected = True
        event.device.stream_emg(True)
        
    def on_disconnected(self, event):
        print("Myo已断开连接")
        self.connected = False
        
    def on_emg(self, event):
        with self.lock:
            self.emg_data = list(event.emg)
            
    def get_data(self):
        with self.lock:
            return self.emg_data.copy()

class MyoManager:
    def __init__(self):
        self.collector = None
        self.hub = None
        self.is_running = False
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
                    print("尝试连接Myo...")
                self.hub.run(self.collector, 100)
                time.sleep(0.01)
            except Exception as e:
                print(f"采集错误: {e}")
                time.sleep(1)
    
    def get_latest_data(self):
        if not self.collector:
            return [0] * 8
        return self.collector.get_data()
    

