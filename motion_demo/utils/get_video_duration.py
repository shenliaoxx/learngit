import cv2
import os

def get_video_duration(video_path):
    """获取视频时长（秒）"""
    if not os.path.exists(video_path):
        print(f"错误: 文件不存在 - {video_path}")
        return 0
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"错误: 无法打开视频 - {video_path}")
        return 0
    
    # 获取帧数和帧率
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # 计算时长
    duration = frame_count / fps
    
    cap.release()
    return duration

# 视频文件路径
videos = {
    "pinch": "static/videos/pinch_20250314_122804.mp4",
    "grasp": "static/videos/grasp_20250314_122825.mp4",
    "flex": "static/videos/flex_20250314_122903.mp4",
    "extend": "static/videos/extend_20250314_122926.mp4"
}

# 获取并打印每个视频的时长
for name, path in videos.items():
    duration = get_video_duration(path)
    print(f"{name} 视频时长: {duration:.2f} 秒") 