import numpy as np
from scipy import signal
from collections import deque


class EMGFilter:
    def __init__(self, sampling_rate=200):
        self.sampling_rate = sampling_rate
        
        # 初始化缓冲区
        self.buffer_size = 50
        self.emg_buffers = [deque(maxlen=self.buffer_size) for _ in range(8)]
        
        # 设计滤波器
        nyquist_freq = sampling_rate / 2
        self.b_bandpass, self.a_bandpass = signal.butter(2, 
            [20/nyquist_freq, 95/nyquist_freq], 
            btype='band')
        self.b_notch, self.a_notch = signal.iirnotch(
            50/nyquist_freq, 30, sampling_rate)
            
    def update_buffer(self, emg_data):
        """更新数据缓冲区"""
        for i, value in enumerate(emg_data):
            self.emg_buffers[i].append(value)
            
    def filter_data(self):
        """应用滤波器处理数据"""
        filtered_data = np.zeros(8)
        
        for i in range(8):
            if len(self.emg_buffers[i]) >= 10:  # 确保有足够的数据点
                data = np.array(list(self.emg_buffers[i]))
                
                try:
                    # 应用带通滤波
                    filtered = signal.lfilter(self.b_bandpass, self.a_bandpass, data)
                    # 应用陷波滤波
                    filtered = signal.lfilter(self.b_notch, self.a_notch, filtered)
                    # 获取最新的滤波结果
                    filtered_data[i] = filtered[-1]
                except Exception as e:
                    print(f"滤波错误: {e}")
                    filtered_data[i] = data[-1]  # 出错时使用原始数据
            else:
                # 数据不足时使用原始数据
                filtered_data[i] = self.emg_buffers[i][-1] if self.emg_buffers[i] else 0
                
        return filtered_data