from flask import Flask, render_template, jsonify, request
from collectors.myo_collector import MyoManager
from collectors.realsense_collector import RealSenseCollector
from threading import Thread
import time
import base64
import cv2
from collectors.data_collector import DataCollector


app = Flask(__name__)

# 全局变量
myo_manager = None
realsense_collector = None
data_collector = None

def process_realsense():
    global realsense_collector
    print("开始处理RealSense数据流")
    while realsense_collector.is_running:
        try:
            realsense_collector.process_frame()
            # time.sleep(0.01)
        except Exception as e:
            print(f"RealSense处理错误: {e}")
            time.sleep(1)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    global myo_manager, realsense_collector, data_collector
    try:
        current_time = time.perf_counter()

        # 检查组件是否初始化
        if not myo_manager:
            return jsonify({
                'status': 'error',
                'message': 'Myo管理器未初始化',
                'emg_data': {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8},
                'hand_data': {},
                'frame': '',
                'camera_fps': 0,
                'camera_total_frames': 0,
                'stats': {}
            })
            
        if not realsense_collector:
            return jsonify({
                'status': 'error',
                'message': 'RealSense采集器未初始化',
                'emg_data': {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8},
                'hand_data': {},
                'frame': '',
                'camera_fps': 0,
                'camera_total_frames': 0,
                'stats': {}
            })
            
        if not data_collector:
            return jsonify({
                'status': 'error',
                'message': '数据采集器未初始化',
                'emg_data': {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8},
                'hand_data': {},
                'frame': '',
                'camera_fps': 0,
                'camera_total_frames': 0,
                'stats': {}
            })

        # 获取EMG数据
        try:
            emg_data = myo_manager.get_latest_data()
        except Exception as e:
            print(f"获取EMG数据错误: {e}")
            emg_data = {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8}
        
        # 获取手部数据和相机帧
        try:
            hand_data = realsense_collector.get_hand_data() or {}
        except Exception as e:
            print(f"获取手部数据错误: {e}")
            hand_data = {}
            
        try:
            frame = realsense_collector.get_frame()
        except Exception as e:
            print(f"获取相机帧错误: {e}")
            frame = None

        # 获取相机统计信息
        try:
            camera_stats = realsense_collector.get_camera_stats()
        except Exception as e:
            print(f"获取相机统计信息错误: {e}")
            camera_stats = {'camera_fps': 0, 'camera_total_frames': 0}

        # 转换相机帧为base64
        frame_base64 = ''
        if frame is not None:
            try:
                _, buffer = cv2.imencode('.jpg', frame)
                frame_base64 = base64.b64encode(buffer).decode('utf-8')
            except Exception as e:
                print(f"转换相机帧错误: {e}")

        # 获取统计信息
        try:
            stats = data_collector.get_stats()
        except Exception as e:
            print(f"获取统计信息错误: {e}")
            stats = {}

        response_data = {
            'status': 'success',
            'emg_data': emg_data,
            'hand_data': hand_data,
            'frame': frame_base64,
            'timestamp': time.time(),
            'camera_fps': camera_stats.get('camera_fps', 0),
            'camera_total_frames': camera_stats.get('camera_total_frames', 0),
            'stats': stats
        }
        
        return jsonify(response_data)
    
    except Exception as e:
        print(f"获取数据错误: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': f"获取数据错误: {str(e)}",
            'emg_data': {'raw_emg': [0] * 8, 'filtered_emg': [0] * 8},
            'hand_data': {},
            'frame': '',
            'camera_fps': 0,
            'camera_total_frames': 0,
            'stats': {}
        })


@app.route('/record_action', methods=['POST'])
def record_action():
    global data_collector, myo_manager
    try:
        data = request.json
        
        if not data_collector:
            return jsonify({
                'status': 'error',
                'message': '数据采集器未初始化'
            })
        
        # 检查Myo连接状态
        myo_status = myo_manager.get_status()
        if not myo_status['connected']:
            print("警告: Myo设备未连接，尝试重新初始化...")
            # 尝试重新初始化Myo
            try:
                myo_manager = MyoManager()
                myo_thread = Thread(target=myo_manager.run_collection)
                myo_thread.daemon = True
                myo_thread.start()
                
                # 更新数据采集器中的Myo管理器
                data_collector.myo_manager = myo_manager
                
                time.sleep(2)  # 等待初始化
                print("Myo设备重新初始化完成")
            except Exception as e:
                print(f"重新初始化Myo失败: {e}")
                import traceback
                print(traceback.format_exc())
        
        # 准备元数据
        metadata = {
            'subject_id': data.get('subject_id', 'unknown'),
            'action': data.get('action', 'manual_recording'),
            'trial': data.get('trial', int(time.time())),
            'dominant_hand': data.get('dominant_hand', 'right')
        }
        
        # 开始记录
        data_collector.start_recording(metadata)
        
        return jsonify({
            'status': 'success',
            'message': '开始记录'
        })
        
    except Exception as e:
        print(f"记录操作错误: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    global data_collector
    try:
        if not data_collector:
            return jsonify({
                'status': 'error',
                'message': '数据采集器未初始化'
            })
        
        # 停止记录并保存数据
        filepath = data_collector.stop_recording()
        
        if filepath:
            return jsonify({
                'status': 'success',
                'message': '记录已停止并保存',
                'filepath': filepath
            })
        else:
            return jsonify({
                'status': 'error',
                'message': '数据保存失败'
            })
        
    except Exception as e:
        print(f"停止记录错误: {e}")
        return jsonify({
            'status': 'error',
            'message': str(e)
        })
    
@app.route('/myo_status')
def myo_status():
    global myo_manager
    if myo_manager:
        status = myo_manager.get_status()
        return jsonify({
            'status': 'success',
            'connected': status['connected'],
            'synced': status['synced'],
            'sampling_rate': status['sampling_rate'],
            'total_samples': status['total_samples']
        })
    return jsonify({
        'status': 'error',
        'message': 'Myo管理器未初始化'
    })

@app.route('/reset_myo', methods=['POST'])
def reset_myo():
    global myo_manager, data_collector
    try:
        # 创建新的Myo管理器
        old_myo = myo_manager
        myo_manager = MyoManager()
        
        # 启动新的Myo线程
        myo_thread = Thread(target=myo_manager.run_collection)
        myo_thread.daemon = True
        myo_thread.start()
        
        # 更新数据采集器中的Myo管理器
        if data_collector:
            data_collector.myo_manager = myo_manager
        
        # 等待初始化
        time.sleep(2)
        
        # 获取新的状态
        status = myo_manager.get_status()
        
        # 尝试关闭旧的Myo管理器
        if old_myo:
            try:
                old_myo.is_running = False
                print("旧的Myo管理器已停止")
            except:
                pass
        
        return jsonify({
            'status': 'success',
            'message': 'Myo设备已重置',
            'connected': status['connected'],
            'synced': status['synced']
        })
    except Exception as e:
        print(f"重置Myo错误: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

if __name__ == '__main__':
    try:
        # 初始化RealSense
        realsense_collector = RealSenseCollector()
        realsense_collector.start()
        
        # 启动RealSense处理线程
        realsense_thread = Thread(target=process_realsense)
        realsense_thread.daemon = True
        realsense_thread.start()
        
        # 初始化并启动Myo线程
        myo_manager = MyoManager()
        myo_thread = Thread(target=myo_manager.run_collection)
        myo_thread.daemon = True
        myo_thread.start()

        # 等待Myo初始化完成
        print("等待Myo设备初始化...")
        time.sleep(3)
        
        # 检查Myo状态
        myo_status = myo_manager.get_status()
        print(f"Myo初始化状态: 连接={myo_status['connected']}, 同步={myo_status['synced']}")

        data_collector = DataCollector(myo_manager, realsense_collector)
        data_collector.start_collection_thread()
        
        # 启动Flask服务器
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"启动错误: {e}")
        import traceback
        print(traceback.format_exc())
    finally:
        # 停止数据采集线程
        if data_collector:
            data_collector.stop_collection_thread()
        # 停止RealSense
        if realsense_collector:
            realsense_collector.stop()
