import cv2
import pyrealsense2 as rs
import numpy as np
import mediapipe as mp
import time
import os
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont

class BatchVideoRecorder:
    def __init__(self, output_dir="videos", width=640, height=480, fps=30):
        # 创建输出目录
        self.output_dir = output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        # 初始化RealSense
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 视频参数
        self.width = width
        self.height = height
        self.fps = fps
        
        # 初始化MediaPipe手部追踪
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            model_complexity=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # 录制状态
        self.is_recording = False
        self.video_writer = None
        self.start_time = 0
        self.duration = 0
        self.current_filename = ""
        
        # 显示选项
        self.show_skeleton = True
        
        # 预定义动作
        self.motions = [
            {"name": "pinch", "description": "捏取动作", "duration": 5},
            {"name": "grasp", "description": "抓握动作", "duration": 5},
            {"name": "flex", "description": "屈腕动作", "duration": 5},
            {"name": "extend", "description": "伸腕动作", "duration": 5}
        ]
        self.current_motion_index = -1
        
        # 添加姿态引导相关属性
        self.pose_guide_active = True
        self.pose_confirmed = False
        self.target_zone_size = 200  # 目标区域大小
        
        # 添加中文字体支持
        try:
            # 尝试加载系统中的中文字体
            self.font_path = '/usr/share/fonts/truetype/droid/DroidSansFallbackFull.ttf'  # Ubuntu系统
            self.font = ImageFont.truetype(self.font_path, 32)
            self.small_font = ImageFont.truetype(self.font_path, 24)
        except:
            try:
                # Windows系统
                self.font_path = "C:/Windows/Fonts/simhei.ttf"
                self.font = ImageFont.truetype(self.font_path, 32)
                self.small_font = ImageFont.truetype(self.font_path, 24)
            except:
                print("警告：未找到中文字体文件，将使用默认字体")
                self.font = None
                self.small_font = None

    def put_chinese_text(self, img, text, position, font_size=32, color=(255, 255, 255)):
        """使用PIL添加中文文字到图片上"""
        if isinstance(img, np.ndarray):
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        
        draw = ImageDraw.Draw(img)
        font = ImageFont.truetype(self.font_path, font_size)
        draw.text(position, text, font=font, fill=color)
        
        result = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        return result

    def draw_pose_guide(self, frame):
        """绘制姿态引导标记"""
        height, width = frame.shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # 创建半透明遮罩
        overlay = frame.copy()
        
        # 绘制目标区域
        cv2.rectangle(overlay,
                     (center_x - self.target_zone_size//2, center_y - self.target_zone_size//2),
                     (center_x + self.target_zone_size//2, center_y + self.target_zone_size//2),
                     (0, 255, 0), 2)
        
        # 绘制中心十字线
        cv2.line(overlay, (center_x, center_y - 50), (center_x, center_y + 50), (0, 255, 0), 1)
        cv2.line(overlay, (center_x - 50, center_y), (center_x + 50, center_y), (0, 255, 0), 1)
        
        # 添加指导文字
        instructions = [
            "手心朝向相机",
            "手指自然伸展",
            "保持手掌在绿框内",
            "按空格键确认姿势"
        ]
        
        y_offset = 50
        for instruction in instructions:
            overlay = self.put_chinese_text(overlay, instruction, (10, y_offset), 24, (255, 255, 255))
            y_offset += 30
        
        # 合并原始帧和引导层
        alpha = 0.7
        return cv2.addWeighted(frame, alpha, overlay, 1-alpha, 0)
    
    def check_hand_position(self, landmarks, frame_shape):
        """检查手部位置是否正确"""
        if landmarks is None:
            return False, "未检测到手部"
            
        height, width = frame_shape[:2]
        center_x = width // 2
        center_y = height // 2
        
        # 计算手掌中心点
        palm_landmarks = [landmarks.landmark[0],  # 手腕
                         landmarks.landmark[5],   # 食指根部
                         landmarks.landmark[17]]  # 小指根部
        palm_center = np.mean([[lm.x * width, lm.y * height] 
                             for lm in palm_landmarks], axis=0)
        
        # 检查位置是否在目标区域内
        target_radius = self.target_zone_size // 2
        distance = np.sqrt(((palm_center[0] - center_x) ** 2) + 
                         ((palm_center[1] - center_y) ** 2))
        
        if distance > target_radius:
            return False, "请将手掌移动到画面中心"
            
        return True, "位置正确"
    
    def start(self):
        """启动相机和主循环"""
        self.pipeline.start(self.config)
        print("批量录制工具已启动")
        print("按 'n' 录制下一个动作，按 'r' 重新录制当前动作，按 'q' 退出")
        
        try:
            self.main_loop()
        finally:
            self.pipeline.stop()
            if self.video_writer:
                self.video_writer.release()
            cv2.destroyAllWindows()
            print("程序已退出")
    
    def next_motion(self):
        """准备录制下一个动作"""
        # 如果当前正在录制，先停止录制
        if self.is_recording:
            self.stop_recording()
        
        # 如果是重新开始，重置索引
        if self.current_motion_index >= len(self.motions) - 1:
            print("所有动作已录制完成，重新开始...")
            self.current_motion_index = -1
        
        self.current_motion_index += 1
        motion = self.motions[self.current_motion_index]
        print(f"\n准备录制: {motion['name']} - {motion['description']}")
        print("请调整手部姿态...")
        
        # 重置姿态确认状态
        self.pose_confirmed = False
        self.pose_guide_active = True
        return True
    
    def restart_current_motion(self):
        """重新录制当前动作"""
        if self.current_motion_index < 0:
            print("没有可重新录制的动作")
            return False
        
        # 如果当前正在录制，先停止录制
        if self.is_recording:
            self.stop_recording()
        
        motion = self.motions[self.current_motion_index]
        print(f"\n重新录制: {motion['name']} - {motion['description']}")
        print("请调整手部姿态...")
        
        # 重置状态
        self.pose_confirmed = False
        self.pose_guide_active = True
        return True
    
    def start_recording(self, duration=5, motion_name="motion"):
        """开始录制视频"""
        if self.is_recording:
            return
            
        # 创建文件名
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{motion_name}_{timestamp}.mp4"
        filepath = os.path.join(self.output_dir, filename)
        
        # 创建视频写入器
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        self.video_writer = cv2.VideoWriter(filepath, fourcc, self.fps, (self.width, self.height))
        
        self.is_recording = True
        self.start_time = time.time()
        self.duration = duration
        self.current_filename = filename
        
        print(f"开始录制: {filename}, 持续时间: {duration}秒")
        
    def stop_recording(self):
        """停止录制视频"""
        if not self.is_recording:
            return
            
        self.is_recording = False
        if self.video_writer:
            self.video_writer.release()
            self.video_writer = None
            
        print(f"录制完成: {self.current_filename}")
        
        # 重置状态，等待下一步操作
        self.pose_guide_active = False
        print("\n请选择:")
        print("按 'n' 继续下一个动作")
        print("按 'r' 重新录制当前动作")
        print("按 'q' 退出程序")

    def main_loop(self):
        """主循环"""
        while True:
            # 获取帧
            frames = self.pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
                
            # 转换为numpy数组
            color_image = np.asanyarray(color_frame.get_data())
            
            # 处理手部追踪
            image_rgb = cv2.cvtColor(color_image, cv2.COLOR_BGR2RGB)
            results = self.hands.process(image_rgb)
            
            # 创建显示帧
            display_image = color_image.copy()
            
            # 绘制手部关键点
            if self.show_skeleton and results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    self.mp_drawing.draw_landmarks(
                        display_image,
                        hand_landmarks,
                        self.mp_hands.HAND_CONNECTIONS,
                        self.mp_drawing_styles.get_default_hand_landmarks_style(),
                        self.mp_drawing_styles.get_default_hand_connections_style()
                    )
            
            if not self.is_recording:
                if self.pose_guide_active:
                    # 添加姿态引导
                    display_image = self.draw_pose_guide(display_image)
                    
                    if results.multi_hand_landmarks:
                        is_correct, message = self.check_hand_position(
                            results.multi_hand_landmarks[0], display_image.shape)
                        
                        # 显示位置反馈
                        color = (0, 255, 0) if is_correct else (0, 0, 255)
                        display_image = self.put_chinese_text(
                            display_image, message,
                            (10, display_image.shape[0] - 40),
                            32, color
                        )
                else:
                    # 显示常规提示信息
                    if self.current_motion_index == -1:
                        status_text = "按 'n' 开始录制下一个动作"
                    else:
                        status_text = "按 'n' 继续下一个动作，按 'r' 重新录制当前动作"
                    
                    display_image = self.put_chinese_text(
                        display_image, status_text,
                        (10, 30), 32, (0, 255, 0)
                    )
                    
                    if self.current_motion_index >= 0:
                        motion = self.motions[self.current_motion_index]
                        motion_text = f"当前动作: {motion['name']} - {motion['description']}"
                        display_image = self.put_chinese_text(
                            display_image, motion_text,
                            (10, 70), 24, (255, 255, 255)
                        )
                    
                    progress_text = f"进度: {self.current_motion_index + 1}/{len(self.motions)}"
                    display_image = self.put_chinese_text(
                        display_image, progress_text,
                        (10, 110), 24, (255, 0, 0)
                    )
            else:
                # 显示录制状态
                motion = self.motions[self.current_motion_index]
                info_text = f"{motion['name']}: {motion['description']}"
                display_image = self.put_chinese_text(
                    display_image, info_text,
                    (10, 30), 32, (0, 255, 0)
                )
                
                elapsed = time.time() - self.start_time
                remaining = max(0, self.duration - elapsed)
                countdown_text = f"剩余时间: {remaining:.1f}秒"
                display_image = self.put_chinese_text(
                    display_image, countdown_text,
                    (10, 70), 24, (0, 0, 255)
                )
                
                # 录制帧
                self.video_writer.write(color_image)
                
                # 检查是否需要停止录制
                if elapsed >= self.duration:
                    self.stop_recording()
            
            # 显示图像
            cv2.imshow('RealSense Batch Recorder', display_image)
            
            # 处理键盘输入
            key = cv2.waitKey(1)
            if key == ord('q'):  # 退出
                break
            elif key == ord('n') and not self.is_recording:  # 下一个动作
                self.next_motion()
            elif key == ord('r') and not self.is_recording:  # 重新录制当前动作
                self.restart_current_motion()
            elif key == ord('s'):  # 切换骨架显示
                self.show_skeleton = not self.show_skeleton
                print(f"骨架显示: {'开启' if self.show_skeleton else '关闭'}")
            elif key == ord(' '):  # 空格键确认姿态
                if self.pose_guide_active and results.multi_hand_landmarks:
                    is_correct, _ = self.check_hand_position(
                        results.multi_hand_landmarks[0], display_image.shape)
                    if is_correct:
                        self.pose_guide_active = False
                        print("姿态已确认，3秒后开始录制...")
                        for i in range(3, 0, -1):
                            print(f"{i}...")
                            time.sleep(1)
                        motion = self.motions[self.current_motion_index]
                        self.start_recording(motion["duration"], motion["name"])

if __name__ == "__main__":
    recorder = BatchVideoRecorder()
    recorder.start()