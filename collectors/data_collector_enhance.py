# -*- coding: utf-8 -*-
import threading
import time
import os
from datetime import datetime
import h5py
import numpy as np


class DataCollector:
    """多模态数据采集器 - 优化版本"""
    
    def __init__(self, myo_manager, realsense_collector):
        """
        初始化数据采集器
        
        参数:
            myo_manager: Myo管理器对象
            realsense_collector: RealSense采集器对象
        """
        self.myo_manager = myo_manager
        self.realsense_collector = realsense_collector

        # 初始化数据缓冲区
        self._init_buffers()
        
        # 记录状态
        self.is_recording = False
        self.recording_start_time = None
        self.recording_metadata = {}

        # 统计信息
        self.stats = {
            'emg_rate': 0,
            'hand_rate': 0,
            'emg_samples': 0,
            'hand_samples': 0,
            'last_emg_time': 0,
            'last_hand_time': 0
        }

        # 线程安全锁
        self.lock = threading.Lock()

        # 数据采集线程控制
        self.collection_thread = None
        self.collection_running = False

        # 统计信息打印间隔(秒)
        self.stats_print_interval = 10
        self.last_stats_print_time = 0

    def _init_buffers(self):
        """初始化数据缓冲区"""
        self.emg_buffer = {
            'timestamps': [],
            'raw_data': [],
            'filtered_data': []
        }
        self.hand_buffer = {
            'timestamps': [],
            'joint_angles': [],
            'keypoints': []
        }

    def start_collection_thread(self):
        """启动独立的数据采集线程"""
        if self.collection_thread is not None and self.collection_thread.is_alive():
            print("[警告] 数据采集线程已在运行")
            return
            
        self.collection_running = True
        self.collection_thread = threading.Thread(
            target=self._collect_data_loop,
            name="DataCollectionThread"
        )
        self.collection_thread.daemon = True
        self.collection_thread.start()
        print("[信息] 数据采集线程已启动")

    def stop_collection_thread(self):
        """停止数据采集线程"""
        if not self.collection_running:
            return
            
        self.collection_running = False
        if self.collection_thread:
            self.collection_thread.join(timeout=1.0)
            if self.collection_thread.is_alive():
                print("[警告] 数据采集线程未能正常停止")
        print("[信息] 数据采集线程已停止")

    def _collect_data_loop(self):
        """数据采集主循环"""
        # 目标采样间隔(秒)
        emg_interval = 1.0 / 200  # EMG: 200Hz
        hand_interval = 1.0 / 60   # 手部数据: 60Hz
        
        last_emg_time = 0
        last_hand_time = 0
        
        while self.collection_running:
            loop_start_time = time.perf_counter()
            
            try:
                # 采集EMG数据
                current_time = time.perf_counter()
                if current_time - last_emg_time >= emg_interval:
                    self._collect_emg_data()
                    last_emg_time = current_time

                # 采集手部数据
                current_time = time.perf_counter()
                if current_time - last_hand_time >= hand_interval:
                    self._collect_hand_data()
                    last_hand_time = current_time

                # 定期打印统计信息
                if self.is_recording and current_time - self.last_stats_print_time >= self.stats_print_interval:
                    self._print_stats()
                    self.last_stats_print_time = current_time
                    
            except Exception as e:
                print(f"[错误] 数据采集循环异常: {str(e)}")
                
            # 动态调整休眠时间以保持采样率
            loop_time = time.perf_counter() - loop_start_time
            sleep_time = max(0, min(emg_interval, hand_interval) / 2 - loop_time)
            if sleep_time > 0:
                time.sleep(sleep_time)

    def _collect_emg_data(self):
        """采集EMG数据"""
        try:
            emg_data_batch = self.myo_manager.get_buffered_data()
            
            if emg_data_batch and self.is_recording:
                for emg_data in emg_data_batch:
                    self.add_emg_data(
                        emg_data['timestamp'], 
                        {
                            'raw_emg': emg_data['raw_emg'],
                            'filtered_emg': emg_data['filtered_emg']
                        }
                    )
        except Exception as e:
            print(f"[错误] EMG数据采集异常: {str(e)}")

    def _collect_hand_data(self):
        """采集手部数据"""
        try:
            hand_data = self.realsense_collector.get_hand_data()
            if hand_data and self.is_recording:
                self.add_hand_data(time.perf_counter(), hand_data)
        except Exception as e:
            print(f"[错误] 手部数据采集异常: {str(e)}")

    def _print_stats(self):
        """打印当前统计信息"""
        if not self.is_recording:
            return
            
        stats = self.get_stats()
        print(
            f"[统计] EMG: {stats['emg_rate']:.1f}Hz ({stats['emg_samples']}样本) | "
            f"手部: {stats['hand_rate']:.1f}Hz ({stats['hand_samples']}样本) | "
            f"持续时间: {stats['duration']:.1f}s"
        )

    def add_emg_data(self, timestamp, data):
        """添加EMG数据到缓冲区"""
        with self.lock:
            if not self.is_recording:
                return
                
            # 添加数据到缓冲区
            self.emg_buffer['timestamps'].append(timestamp)
            self.emg_buffer['raw_data'].append(data['raw_emg'])
            self.emg_buffer['filtered_data'].append(data['filtered_emg'])

            # 更新统计信息
            self.stats['emg_samples'] += 1
            if len(self.emg_buffer['timestamps']) > 1:
                duration = timestamp - self.emg_buffer['timestamps'][0]
                if duration > 0:
                    self.stats['emg_rate'] = (len(self.emg_buffer['timestamps']) - 1) / duration
            self.stats['last_emg_time'] = timestamp

    def add_hand_data(self, timestamp, data):
        """添加手部数据到缓冲区"""
        with self.lock:
            if not (self.is_recording and data and isinstance(data, dict)):
                return
                
            # 检查重复时间戳(1ms内视为重复)
            if (self.hand_buffer['timestamps'] and 
                abs(timestamp - self.hand_buffer['timestamps'][-1]) < 0.001):
                return
                
            self.hand_buffer['timestamps'].append(timestamp)
            
            # 处理关节角度数据
            angles_data = data.get('angles', {})
            angles = self._process_joint_angles(angles_data)
            self.hand_buffer['joint_angles'].append(angles)
            
            # 处理关键点数据
            if 'keypoints' in data:
                points = self._process_keypoints(data['keypoints'])
                self.hand_buffer['keypoints'].append(points)

            # 更新统计信息
            self.stats['hand_samples'] += 1
            if len(self.hand_buffer['timestamps']) > 1:
                duration = timestamp - self.hand_buffer['timestamps'][0]
                if duration > 0:
                    self.stats['hand_rate'] = (len(self.hand_buffer['timestamps']) - 1) / duration
            self.stats['last_hand_time'] = timestamp

    def _process_joint_angles(self, angles_data):
        """处理关节角度数据"""
        return [
            angles_data.get("thumb_cmc_flexion", 0.0),
            angles_data.get("thumb_cmc_abduction", 0.0),
            angles_data.get("thumb_mcp_flexion", 0.0),
            angles_data.get("thumb_ip_flexion", 0.0),
            *[angles_data.get(f"{finger}_{joint}", 0.0) 
              for finger in ['index', 'middle', 'ring', 'pinky'] 
              for joint in ['mcp_abduction', 'mcp_flexion', 'pip_flexion', 'dip_flexion']]
        ]

    def _process_keypoints(self, keypoints):
        """处理关键点数据"""
        return [
            [point['x'], point['y'], point['z'], point.get('visibility', 1.0)] 
            for point in keypoints
        ]

    def start_recording(self, metadata=None):
        """开始记录数据"""
        with self.lock:
            if self.is_recording:
                print("[警告] 已经在记录中")
                return
                
            self.is_recording = True
            self.recording_start_time = time.perf_counter()
            self.recording_metadata = metadata or {}
            self.recording_metadata['start_time'] = datetime.now().isoformat()

            # 重置缓冲区和统计信息
            self._init_buffers()
            self.stats = {
                'emg_rate': 0, 'hand_rate': 0,
                'emg_samples': 0, 'hand_samples': 0,
                'last_emg_time': 0, 'last_hand_time': 0
            }

            # 确保采集线程运行
            if not self.collection_running:
                self.start_collection_thread()
            
            self.last_stats_print_time = time.perf_counter()
            print("[信息] 开始记录数据")

    def stop_recording(self):
        """停止记录数据并保存"""
        with self.lock:
            if not self.is_recording:
                print("[警告] 当前没有在记录")
                return None
                
            self.is_recording = False
            filepath = self._save_recording()
            print(f"[信息] 停止记录，文件保存在: {filepath}")
            return filepath

    def _save_recording(self, hand_info='right', recording_number=1, repeat_times=1):
        """保存记录数据到HDF5文件"""
        try:
            # 创建数据目录
            os.makedirs('data', exist_ok=True)
            
            # 生成文件名
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            filename = f"{timestamp}_{recording_number}_{hand_info}_{repeat_times}.h5"
            filepath = os.path.join('data', filename)
            
            print(f"[信息] 正在保存数据到: {filepath}")

            # 压缩存储选项
            compression_opts = {
                'compression': 'gzip',
                'compression_opts': 4,
                'shuffle': True,
                'chunks': True
            }

            with h5py.File(filepath, 'w') as f:
                # 保存EMG数据
                self._save_emg_data(f, compression_opts)
                
                # 保存手部数据
                self._save_hand_data(f, compression_opts)
                
                # 保存元数据和统计信息
                self._save_metadata(f, hand_info, recording_number, repeat_times)
            
            print(f"[信息] 数据保存成功: {filepath}")
            return filepath
        
        except Exception as e:
            print(f"[错误] 保存数据失败: {str(e)}")
            return None

    def _save_emg_data(self, h5file, compression_opts):
        """保存EMG数据到HDF5文件"""
        emg_group = h5file.create_group('emg')
        emg_group.create_dataset('timestamps', 
                                data=np.array(self.emg_buffer['timestamps']), 
                                **compression_opts)
        emg_group.create_dataset('raw_data', 
                                data=np.array(self.emg_buffer['raw_data']), 
                                **compression_opts)
        emg_group.create_dataset('filtered_data', 
                                data=np.array(self.emg_buffer['filtered_data']), 
                                **compression_opts)

    def _save_hand_data(self, h5file, compression_opts):
        """保存手部数据到HDF5文件"""
        hand_group = h5file.create_group('hand')
        hand_group.create_dataset('timestamps', 
                                data=np.array(self.hand_buffer['timestamps']), 
                                **compression_opts)
        
        # 保存关节角度数据
        joint_angles = hand_group.create_dataset(
            'joint_angles', 
            data=np.array(self.hand_buffer['joint_angles']), 
            **compression_opts
        )
        joint_angles.attrs['angle_labels'] = [
            'thumb_cmc_flexion', 'thumb_cmc_abduction', 'thumb_mcp_flexion', 'thumb_ip_flexion',
            *[f"{finger}_{joint}" 
              for finger in ['index', 'middle', 'ring', 'pinky'] 
              for joint in ['mcp_abduction', 'mcp_flexion', 'pip_flexion', 'dip_flexion']]
        ]
        
        # 保存关键点数据
        if self.hand_buffer['keypoints']:
            keypoints = hand_group.create_dataset(
                'keypoints', 
                data=np.array(self.hand_buffer['keypoints']), 
                **compression_opts
            )
            keypoints.attrs['keypoint_labels'] = ['x', 'y', 'z', 'visibility']

    def _save_metadata(self, h5file, hand_info, recording_number, repeat_times):
        """保存元数据和统计信息"""
        # 保存统计信息
        stats_group = h5file.create_group('stats')
        stats_group.attrs.update({
            'emg_rate': self.stats['emg_rate'],
            'hand_rate': self.stats['hand_rate'],
            'repeat_times': repeat_times,
            'hand_info': hand_info,
            'recording_number': recording_number,
            'duration': time.perf_counter() - self.recording_start_time
        })
        
        # 保存用户元数据
        if self.recording_metadata:
            metadata_group = h5file.create_group('metadata')
            for key, value in self.recording_metadata.items():
                try:
                    metadata_group.attrs[key] = value
                except TypeError:
                    print(f"[警告] 无法保存元数据项 '{key}': 类型不支持")

    def get_stats(self):
        """获取当前统计信息"""
        with self.lock:
            return {
                'is_recording': self.is_recording,
                'duration': time.perf_counter() - self.recording_start_time if self.is_recording else 0,
                'emg_rate': self.stats['emg_rate'],
                'hand_rate': self.stats['hand_rate'],
                'emg_samples': self.stats['emg_samples'],
                'hand_samples': self.stats['hand_samples']
            }
