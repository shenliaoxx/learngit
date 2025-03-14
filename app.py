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
    frame_time = 1.0 / 35  # 目标处理帧率略高于相机帧率，设为35FPS
    last_process_time = time.perf_counter()
    
    while realsense_collector.is_running:
        try:
            current_time = time.perf_counter()
            elapsed = current_time - last_process_time
            
            # 控制处理频率
            if elapsed >= frame_time:
                realsense_collector.process_frame()
                last_process_time = current_time
                
                # 动态调整处理间隔
                sleep_time = max(0, frame_time - (time.perf_counter() - current_time))
                if sleep_time > 0:
                    time.sleep(sleep_time / 2)  # 减少一半休眠时间，确保不会错过帧
            else:
                # 短暂休眠，避免CPU占用过高
                time.sleep(0.001)
        except Exception as e:
            print(f"RealSense处理错误: {e}")
            import traceback
            print(traceback.format_exc())
            time.sleep(0.1)  # 错误后短暂休眠

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_data')
def get_data():
    """获取所有传感器数据"""
    global myo_manager, realsense_collector, data_collector
    
    if not all([myo_manager, realsense_collector, data_collector]):
        return jsonify({
            'status': 'error',
            'message': '系统组件未完全初始化'
        })
    
    try:
        # 获取EMG数据
        emg_data = myo_manager.get_latest_data()
        
        # 获取手部数据和相机帧
        hand_data = realsense_collector.get_hand_data() or {}
        frame = realsense_collector.get_frame()
        camera_stats = realsense_collector.get_camera_stats()
        
        # 获取统计信息
        stats = data_collector.get_stats()
        
        # 转换相机帧为base64
        frame_base64 = ''
        if frame is not None:
            _, buffer = cv2.imencode('.jpg', frame)
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
        
        return jsonify({
            'status': 'success',
            'emg_data': emg_data,
            'hand_data': hand_data,
            'frame': frame_base64,
            'timestamp': time.time(),
            'camera_fps': camera_stats.get('camera_fps', 0),
            'camera_total_frames': camera_stats.get('camera_total_frames', 0),
            'stats': stats
        })
    
    except Exception as e:
        print(f"获取数据错误: {e}")
        return jsonify({
            'status': 'error',
            'message': f"获取数据错误: {str(e)}"
        })
    


@app.route('/record_action', methods=['POST'])
def record_action():
    """开始记录数据"""
    global data_collector, myo_manager
    
    if not data_collector:
        return jsonify({
            'status': 'error',
            'message': '数据采集器未初始化'
        })
    
    try:
        data = request.json
        
        # 检查Myo连接状态并尝试重连
        if not _check_myo_connection():
            return jsonify({
                'status': 'error',
                'message': 'Myo设备连接失败'
            })
        
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
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

def _check_myo_connection():
    """检查Myo连接状态并尝试重连"""
    global myo_manager, data_collector
    
    myo_status = myo_manager.get_status()
    if not myo_status['connected']:
        print("警告: Myo设备未连接，尝试重新初始化...")
        try:
            # 创建新的Myo管理器
            myo_manager = MyoManager()
            myo_thread = Thread(target=myo_manager.run_collection)
            myo_thread.daemon = True
            myo_thread.start()
            
            # 更新数据采集器中的Myo管理器
            data_collector.myo_manager = myo_manager
            
            time.sleep(2)  # 等待初始化
            print("Myo设备重新初始化完成")
            return myo_manager.get_status()['connected']
        except Exception as e:
            print(f"重新初始化Myo失败: {e}")
            return False
    return True

@app.route('/stop_recording', methods=['POST'])
def stop_recording():
    """停止记录数据"""
    global data_collector
    
    if not data_collector:
        return jsonify({
            'status': 'error',
            'message': '数据采集器未初始化'
        })
    
    try:
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
    """获取Myo设备状态"""
    global myo_manager
    
    if not myo_manager:
        return jsonify({
            'status': 'error',
            'message': 'Myo管理器未初始化'
        })
    
    status = myo_manager.get_status()
    return jsonify({
        'status': 'success',
        'connected': status['connected'],
        'synced': status['synced'],
        'sampling_rate': status['sampling_rate'],
        'total_samples': status['total_samples']
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


def initialize_system():
    """初始化系统组件"""
    global realsense_collector, myo_manager, data_collector
    
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

    # 初始化数据采集器
    data_collector = DataCollector(myo_manager, realsense_collector)
    data_collector.start_collection_thread()
    
    print("系统初始化完成")


if __name__ == '__main__':
    try:
        initialize_system()
        
        # 启动Flask服务器
        app.run(host='0.0.0.0', port=5000, debug=False)
        
    except Exception as e:
        print(f"启动错误: {e}")
    finally:
        # 清理资源
        if data_collector:
            data_collector.stop_collection_thread()
        if realsense_collector:
            realsense_collector.stop()

       
