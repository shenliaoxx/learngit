from flask import Flask, render_template, jsonify, request, send_from_directory
from collectors.myo_collector import MyoManager
from collectors.realsense_collector_enhance import RealSenseCollector
from threading import Thread
import time
import base64
import cv2
from collectors.data_collector_enhance import DataCollector
from utils.motion_lib import MotionLibrary
import h5py
import numpy as np
import os
from werkzeug.utils import secure_filename

# ===================== 初始化应用 =====================
app = Flask(__name__)

# ===================== 全局变量 =====================
myo_manager = None
realsense_collector = None
data_collector = None
motion_library = MotionLibrary()  # 初始化动作库
collection_sessions = {}  # 采集会话状态

# ===================== 辅助函数 =====================
def process_realsense():
    """RealSense数据处理线程函数"""
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

# ===================== 系统初始化函数 =====================
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

# ===================== 静态资源路由 =====================

@app.route('/videos/<path:filename>')
def serve_video(filename):
    """提供视频文件服务"""
    return send_from_directory(r'E:/multimodel-acquisition/static/videos', filename)

# ===================== 页面路由 =====================
@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

@app.route('/angle_comparison')
def angle_comparison():
    """角度比较页面"""
    return render_template('angle_comparison.html')

@app.route('/motion_demo')
def motion_demo():
    """动作演示系统主页"""
    return render_template('motion_demo/index.html')



# ===================== 数据采集API =====================
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
                'message': 'Myo设备连接失败，请检查设备是否已连接'
            })
        
        # 检查数据采集器状态
        if data_collector.is_recording:
            return jsonify({
                'status': 'error',
                'message': '已经在记录中，请先停止当前记录'
            })
        
        # 准备元数据 - 与DataCollector期望的字段对齐
        metadata = {
            'subject_id': data.get('subject_id', 'unknown'),
            'session_id': data.get('session_id', str(int(time.time()))),
            'task_type': data.get('task_type', 'manual_recording'),
            'dominant_hand': data.get('dominant_hand', 'right'),
            'action': data.get('action', 'unknown'),
            'trial': data.get('trial', 0)
        }
        
        # 开始记录
        data_collector.start_recording(metadata)
        
        # 等待确保记录已开始
        time.sleep(0.1)
        
        return jsonify({
            'status': 'success',
            'message': '开始记录',
            'metadata': metadata
        })
        
    except Exception as e:
        print(f"记录操作错误: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'status': 'error',
            'message': f"开始记录失败: {str(e)}"
        })

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
    
# ===================== 设备状态API =====================
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
    """重置Myo设备连接"""
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

# ===================== 数据分析API =====================
@app.route('/get_angle_comparison')
def get_angle_comparison():
    """获取原始和滤波后的角度数据"""
    global realsense_collector
    
    if not realsense_collector:
        return jsonify({
            'status': 'error',
            'message': 'RealSense采集器未初始化'
        })
    
    try:
        calculator = realsense_collector.calculator
        
        # 使用更高效的数据处理方式
        max_points = 50  # 减少数据点数量以提高性能
        
        # 使用列表推导式和切片操作优化数据处理
        raw_angles = {
            k: [float(x) for x in v[-max_points:]]
            for k, v in calculator.raw_angles.items()
            if v  # 只处理非空列表
        }
        
        filtered_angles = {
            k: [float(x) for x in v[-max_points:]]
            for k, v in calculator.filtered_angles.items()
            if v  # 只处理非空列表
        }
        
        # 优化运动状态计算
        motion_states = {
            k: calculator.detect_motion_state(k, v[-1])
            for k, v in filtered_angles.items()
            if v  # 只处理非空列表
        }
        
        return jsonify({
            'status': 'success',
            'raw_angles': raw_angles,
            'filtered_angles': filtered_angles,
            'motion_states': motion_states
        })
        
    except Exception as e:
        print(f"获取角度比较数据错误: {e}")
        import traceback
        print(traceback.format_exc())
        return jsonify({
            'status': 'error',
            'message': str(e)
        })

# ===================== 动作演示系统API =====================
@app.route('/api/sequences')
def get_sequences():
    """获取所有动作序列"""
    return jsonify({
        'status': 'success',
        'sequences': motion_library.get_all_sequences()
    })

@app.route('/api/sequence/<sequence_id>')
def get_sequence(sequence_id):
    """获取指定动作序列的详细信息"""
    sequence = motion_library.get_sequence(sequence_id)
    if sequence:
        return jsonify({
            'status': 'success',
            'sequence': sequence
        })
    else:
        return jsonify({
            'status': 'error',
            'message': f'未找到序列: {sequence_id}'
        }), 404

@app.route('/api/collection/prepare', methods=['POST'])
def prepare_collection():
    """准备采集 - 从动作演示到准备采集状态"""
    data = request.json
    motion_id = data.get('motion_id')
    
    if not motion_id:
        return jsonify({
            'status': 'error',
            'message': '未指定动作ID'
        }), 400
    
    motion = motion_library.get_motion(motion_id)
    if not motion:
        return jsonify({
            'status': 'error',
            'message': f'未找到动作: {motion_id}'
        }), 404
    
    # 创建采集会话
    session_id = f"{motion_id}_{int(time.time())}"
    collection_sessions[session_id] = {
        'motion_id': motion_id,
        'motion_name': motion['name'],
        'state': 'preparing',
        'repeat_times': motion['collection']['repeat_times'],
        'current_repeat': 0,
        'duration': motion['collection']['duration'],
        'rest_time': motion['collection']['rest_time'],
        'preparation_time': motion['collection']['preparation_time'],
        'start_time': None,
        'data_files': []
    }
    
    return jsonify({
        'status': 'success',
        'message': f'准备采集 {motion["name"]}',
        'session_id': session_id,
        'collection_config': motion['collection']
    })

@app.route('/api/collection/start', methods=['POST'])
def start_collection():
    """开始采集数据"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in collection_sessions:
        return jsonify({
            'status': 'error',
            'message': '无效的采集会话ID'
        }), 400
    
    session = collection_sessions[session_id]
    
    # 更新会话状态
    session['state'] = 'collecting'
    session['current_repeat'] += 1
    session['start_time'] = time.time()
    
    # 这里应该有实际的数据采集启动代码
    # 例如启动传感器数据记录等
    
    return jsonify({
        'status': 'success',
        'message': f'开始采集 {session["motion_name"]} (第 {session["current_repeat"]}/{session["repeat_times"]} 次)',
        'repeat': session['current_repeat'],
        'total_repeats': session['repeat_times'],
        'duration': session['duration']
    })

@app.route('/api/collection/stop', methods=['POST'])
def stop_collection():
    """停止当前采集"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in collection_sessions:
        return jsonify({
            'status': 'error',
            'message': '无效的采集会话ID'
        }), 400
    
    session = collection_sessions[session_id]
    
    # 停止数据采集
    # 这里应该有实际的数据采集停止代码
    
    # 生成数据文件名（实际应用中应该是真实保存的文件）
    filename = f"{session['motion_id']}_{session['current_repeat']}_{int(time.time())}.h5"
    session['data_files'].append(filename)
    
    # 更新会话状态
    if session['current_repeat'] >= session['repeat_times']:
        session['state'] = 'completed'
        message = f'采集完成: {session["motion_name"]}'
    else:
        session['state'] = 'resting'
        message = f'休息中: {session["rest_time"]}秒'
    
    return jsonify({
        'status': 'success',
        'message': message,
        'state': session['state'],
        'repeat': session['current_repeat'],
        'total_repeats': session['repeat_times'],
        'rest_time': session['rest_time'] if session['state'] == 'resting' else 0
    })

@app.route('/api/collection/status', methods=['GET'])
def collection_status():
    """获取采集状态"""
    session_id = request.args.get('session_id')
    
    if not session_id or session_id not in collection_sessions:
        return jsonify({
            'status': 'error',
            'message': '无效的采集会话ID'
        }), 400
    
    session = collection_sessions[session_id]
    
    return jsonify({
        'status': 'success',
        'collection_status': {
            'state': session['state'],
            'motion_name': session['motion_name'],
            'repeat': session['current_repeat'],
            'total_repeats': session['repeat_times']
        }
    })

@app.route('/api/collection/finish', methods=['POST'])
def finish_collection():
    """完成整个采集过程"""
    data = request.json
    session_id = data.get('session_id')
    
    if not session_id or session_id not in collection_sessions:
        return jsonify({
            'status': 'error',
            'message': '无效的采集会话ID'
        }), 400
    
    session = collection_sessions[session_id]
    
    # 清理会话数据（实际应用中可能需要保存会话记录）
    result = {
        'motion_name': session['motion_name'],
        'repeat_times': session['repeat_times']
    }
    
    # 从活动会话中移除
    del collection_sessions[session_id]
    
    return jsonify({
        'status': 'success',
        'message': f'采集会话已完成: {session_id}',
        'result': result
    })


# ===================== 主程序入口 =====================
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




       
