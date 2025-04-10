# -*- coding: utf-8 -*-
import threading
import time
import os
import sys
import io
from datetime import datetime
import h5py
import numpy as np
import logging
from logging.handlers import RotatingFileHandler

# 解决Windows控制台编码问题
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

class DataCollector:
    """多模态数据采集器"""
    def __init__(self, myo_manager, realsense_collector):
        # 初始化日志系统
        self._setup_logging()
        
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

        # 数据采集线程
        self.collection_thread = None
        self.collection_running = False

        # 统计信息打印
        self.last_stats_print_time = 0
        self.stats_print_interval = 5  # 每5秒打印一次统计信息

    def _setup_logging(self):
        """配置日志系统，解决中文乱码问题"""
        self.logger = logging.getLogger('DataCollector')
        self.logger.setLevel(logging.INFO)
        
        # 清除可能存在的旧handler
        for handler in self.logger.handlers[:]:
            self.logger.removeHandler(handler)
        
        # 控制台Handler（UTF-8编码）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # 文件Handler（UTF-8编码，自动轮转）
        file_handler = RotatingFileHandler(
            'data_collector.log', 
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5,
            encoding='utf-8'
        )
        file_handler.setLevel(logging.INFO)
        
        # 统一的日志格式
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        
        console_handler.setFormatter(formatter)
        file_handler.setFormatter(formatter)
        
        self.logger.addHandler(console_handler)
        self.logger.addHandler(file_handler)

    def start_collection_thread(self):
        """启动独立的数据采集线程"""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            self.logger.warning("数据采集线程已在运行")
            return
            
        self.collection_running = True
        self.collection_thread = threading.Thread(target=self._collect_data_loop)
        self.collection_thread.daemon = True
        self.collection_thread.start()
        self.logger.info("数据采集线程已启动")

    def stop_collection_thread(self):
        """停止数据采集线程"""
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
        self.logger.info("数据采集线程已停止")

    def _collect_data_loop(self):
        """数据采集循环"""
        emg_interval = 1.0 / 200  # 目标EMG采样率：200Hz
        hand_interval = 1.0 / 40   # 目标手部数据采样率：40Hz
        
        last_emg_time = 0
        last_hand_time = 0
        
        while self.collection_running:
            current_time = time.perf_counter()

            # 采集EMG数据
            if current_time - last_emg_time >= emg_interval:
                try:
                    emg_data_batch = self.myo_manager.get_buffered_data()
                    
                    if emg_data_batch and self.is_recording:
                        for emg_data in emg_data_batch:
                            self.add_emg_data(emg_data['timestamp'], {
                                'raw_emg': emg_data['raw_emg'],
                                'filtered_emg': emg_data['filtered_emg']
                            })
                    last_emg_time = current_time
                except Exception as e:
                    self.logger.error("EMG数据采集错误: %s", str(e))
                    self.logger.exception(e)

            # 采集手部数据
            if current_time - last_hand_time >= hand_interval:
                try:
                    hand_data = self.realsense_collector.get_hand_data()
                    if hand_data and self.is_recording:
                        self.add_hand_data(current_time, hand_data)
                    last_hand_time = current_time
                except Exception as e:
                    self.logger.error("手部数据采集错误: %s", str(e))
                    self.logger.exception(e)
            
            # 定期打印统计信息
            if self.is_recording and current_time - self.last_stats_print_time >= self.stats_print_interval:
                self._print_stats()
                self.last_stats_print_time = current_time
                
            # 动态调整休眠时间
            sleep_time = max(0, min(emg_interval, hand_interval) / 2 - (time.perf_counter() - current_time))
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _print_stats(self):
        """打印当前统计信息"""
        if not self.is_recording:
            return
            
        stats = {
            'emg_rate': self.stats['emg_rate'],
            'emg_samples': self.stats['emg_samples'],
            'hand_rate': self.stats['hand_rate'],
            'hand_samples': self.stats['hand_samples'],
            'duration': time.perf_counter() - self.recording_start_time
        }
        
        self.logger.info(
            "采集统计 - EMG: %.1fHz (%d样本) | 手部: %.1fHz (%d样本) | 持续时间: %.1fs", 
            stats['emg_rate'], stats['emg_samples'],
            stats['hand_rate'], stats['hand_samples'],
            stats['duration']
        )
        
        sampling_stats = self.myo_manager.get_sampling_stats()
        self.logger.info(
            "Myo采样统计 - 采样率: %.1fHz | 平均间隔: %.2fms | 标准差: %.2fms",
            sampling_stats['rate'],
            sampling_stats['mean_interval']*1000,
            sampling_stats['std_interval']*1000
        )

    def add_emg_data(self, timestamp, data):
        """添加EMG数据"""
        with self.lock:
            if self.is_recording:
                self.emg_buffer['timestamps'].append(timestamp)
                self.emg_buffer['raw_data'].append(data['raw_emg'])
                self.emg_buffer['filtered_data'].append(data['filtered_emg'])

                self.stats['emg_samples'] += 1
                if len(self.emg_buffer['timestamps']) > 1:
                    first_timestamp = self.emg_buffer['timestamps'][0]
                    duration = timestamp - first_timestamp
                    if duration > 0:
                        self.stats['emg_rate'] = (len(self.emg_buffer['timestamps']) - 1) / duration
                self.stats['last_emg_time'] = timestamp

    def add_hand_data(self, timestamp, data):
        """添加手部数据"""
        with self.lock:
            if self.is_recording and data and isinstance(data, dict):
                # 检查重复时间戳
                if self.hand_buffer['timestamps'] and abs(timestamp - self.hand_buffer['timestamps'][-1]) < 0.001:
                    return
                    
                self.hand_buffer['timestamps'].append(timestamp)
                angles_data = [
                    data.get("thumb_cmc_flexion", 0.0),
                    data.get("thumb_cmc_abduction", 0.0),
                    data.get("thumb_mcp_flexion", 0.0),
                    data.get("thumb_ip_flexion", 0.0),
                    *[data.get(f"{finger}_{joint}", 0.0) 
                      for finger in ['index', 'middle', 'ring', 'pinky'] 
                      for joint in ['mcp_abduction', 'mcp_flexion', 'pip_flexion', 'dip_flexion']]
                ]
                self.hand_buffer['joint_angles'].append(angles_data)

                self.stats['hand_samples'] += 1
                if len(self.hand_buffer['timestamps']) > 1:
                    first_timestamp = self.hand_buffer['timestamps'][0]
                    duration = timestamp - first_timestamp
                    if duration > 0:
                        self.stats['hand_rate'] = (len(self.hand_buffer['timestamps']) - 1) / duration
                self.stats['last_hand_time'] = timestamp

    def start_recording(self, metadata=None):
        """开始记录数据"""
        with self.lock:
            self.is_recording = True
            self.recording_start_time = time.perf_counter()
            self.recording_metadata = metadata or {}
            self.recording_metadata['start_time'] = self.recording_start_time

            # 重置缓冲区
            self.emg_buffer = {'timestamps': [], 'raw_data': [], 'filtered_data': []}
            self.hand_buffer = {'timestamps': [], 'joint_angles': []}
            self.stats = {
                'emg_rate': 0, 'hand_rate': 0,
                'emg_samples': 0, 'hand_samples': 0,
                'last_emg_time': 0, 'last_hand_time': 0
            }

            if not self.collection_running:
                self.start_collection_thread()
            
            self.last_stats_print_time = time.perf_counter()
            self.logger.info("开始记录数据")

    def stop_recording(self):
        """停止记录数据"""
        with self.lock:
            if not self.is_recording:
                return None
            
            self.is_recording = False
            filepath = self.save_recording()
            self.logger.info("停止记录，文件保存在: %s", filepath)
            return filepath

    def save_recording(self, hand_info='right', recording_number=2, repeat_times=6):
        """保存记录数据"""
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{recording_number}_{hand_info}_{repeat_times}.h5"
            
            if not os.path.exists('data'):
                os.makedirs('data')
            
            filepath = os.path.join('data', filename)
            self.logger.info("正在保存数据到: %s", filepath)

            with h5py.File(filepath, 'w') as f:
                # 保存EMG数据
                emg_group = f.create_group('emg')
                emg_group.create_dataset('timestamps', data=self.emg_buffer['timestamps'])
                emg_group.create_dataset('raw_data', data=self.emg_buffer['raw_data'])
                emg_group.create_dataset('filtered_data', data=self.emg_buffer['filtered_data'])
                
                # 保存手部数据
                hand_group = f.create_group('hand')
                hand_group.create_dataset('timestamps', data=self.hand_buffer['timestamps'])
                joint_angles = hand_group.create_dataset('joint_angles', data=self.hand_buffer['joint_angles'])
                
                # 关节角度标签
                angle_labels = [
                    'thumb_cmc_flexion', 'thumb_cmc_abduction', 'thumb_mcp_flexion', 'thumb_ip_flexion',
                    *[f"{finger}_{joint}" 
                      for finger in ['index', 'middle', 'ring', 'pinky'] 
                      for joint in ['mcp_abduction', 'mcp_flexion', 'pip_flexion', 'dip_flexion']]
                ]
                joint_angles.attrs['angle_labels'] = angle_labels
                
                # 保存统计信息
                stats = f.create_group('stats')
                stats.attrs.update({
                    'emg_samples': len(self.emg_buffer['timestamps']),
                    'hand_samples': len(self.hand_buffer['timestamps']),
                    'repeat_times': repeat_times,
                    'hand_info': hand_info,
                    'recording_number': recording_number
                })
            
            self.logger.info("数据保存成功: %s", filepath)
            return filepath
        
        except Exception as e:
            self.logger.error("保存数据错误: %s", str(e))
            self.logger.exception(e)
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