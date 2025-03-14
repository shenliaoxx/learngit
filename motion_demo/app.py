from flask import Flask, render_template, jsonify, request, session
from motion_lib import MotionLibrary
import time
import os
import json

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # 用于 session 加密

# 初始化动作库
motion_library = MotionLibrary()

# 采集会话状态
collection_sessions = {}

@app.route('/')
def index():
    """主页"""
    return render_template('index.html')

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

if __name__ == '__main__':
    app.run(debug=True)