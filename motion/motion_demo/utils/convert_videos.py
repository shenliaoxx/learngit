import os
import argparse
from pathlib import Path
from moviepy.editor import VideoFileClip

def convert_video_for_web(input_file, output_dir=None, format="mp4"):
    """
    将视频转换为 Web 友好的格式，使用 moviepy 库
    
    参数:
        input_file: 输入视频文件路径
        output_dir: 输出目录，默认为与输入文件相同的目录下的 web_videos 子目录
        format: 输出格式，默认为 mp4
    
    返回:
        输出文件路径
    """
    input_path = Path(input_file)
    
    # 如果未指定输出目录，则在输入文件所在目录下创建 web_videos 子目录
    if output_dir is None:
        output_dir = input_path.parent / "web_videos"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 构建输出文件路径
    output_filename = f"{input_path.stem}_web.{format}"
    output_path = Path(output_dir) / output_filename
    
    print(f"正在转换: {input_path}")
    
    try:
        # 加载视频
        video = VideoFileClip(str(input_path))
        
        # 转换视频 - 使用 H.264 编码和 AAC 音频
        video.write_videofile(
            str(output_path),
            codec='libx264',
            audio_codec='aac',
            bitrate='5000k',
            preset='medium',
            ffmpeg_params=["-crf", "23", "-movflags", "+faststart"]
        )
        
        # 关闭视频对象
        video.close()
        
        print(f"转换成功: {output_path}")
        return str(output_path)
    except Exception as e:
        print(f"转换失败: {e}")
        return None

def batch_convert_videos(input_dir, output_dir=None, format="mp4"):
    """
    批量转换目录中的所有视频文件
    
    参数:
        input_dir: 输入目录
        output_dir: 输出目录
        format: 输出格式
    
    返回:
        成功转换的文件列表
    """
    input_path = Path(input_dir)
    
    # 如果未指定输出目录，则在输入目录下创建 web_videos 子目录
    if output_dir is None:
        output_dir = input_path / "web_videos"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有视频文件
    video_files = [f for f in input_path.glob("*.mp4")]
    
    if not video_files:
        print(f"在 {input_dir} 中未找到 MP4 文件")
        return []
    
    # 转换所有视频文件
    converted_files = []
    for video_file in video_files:
        output_file = convert_video_for_web(video_file, output_dir, format)
        if output_file:
            converted_files.append(output_file)
    
    return converted_files

def create_test_html(video_files, output_path="test_videos.html"):
    """
    创建一个 HTML 文件来测试视频播放
    
    参数:
        video_files: 视频文件路径列表
        output_path: 输出 HTML 文件路径
    """
    html_content = """
    <!DOCTYPE html>
    <html lang="zh-CN">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>视频播放测试</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
            }
            h1 {
                text-align: center;
                margin-bottom: 30px;
            }
            .video-container {
                margin-bottom: 40px;
                padding: 20px;
                border: 1px solid #ddd;
                border-radius: 8px;
                background-color: #f9f9f9;
            }
            h2 {
                margin-top: 0;
                color: #333;
            }
            video {
                width: 100%;
                max-height: 400px;
                background-color: #000;
            }
            .controls {
                margin-top: 10px;
            }
            button {
                padding: 8px 16px;
                margin-right: 10px;
                background-color: #4CAF50;
                color: white;
                border: none;
                border-radius: 4px;
                cursor: pointer;
            }
            button:hover {
                background-color: #45a049;
            }
        </style>
    </head>
    <body>
        <h1>视频播放测试</h1>
    """
    
    for i, video_file in enumerate(video_files):
        video_path = Path(video_file)
        video_name = video_path.stem
        
        # 获取相对路径
        rel_path = os.path.relpath(video_file, os.path.dirname(output_path))
        
        html_content += f"""
        <div class="video-container">
            <h2>{i+1}. {video_name}</h2>
            <video id="video-{i}" controls>
                <source src="{rel_path}" type="video/mp4">
                您的浏览器不支持 HTML5 视频。
            </video>
            <div class="controls">
                <button onclick="document.getElementById('video-{i}').play()">播放</button>
                <button onclick="document.getElementById('video-{i}').pause()">暂停</button>
                <button onclick="document.getElementById('video-{i}').currentTime=0">重置</button>
            </div>
        </div>
        """
    
    html_content += """
    </body>
    </html>
    """
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    
    print(f"测试 HTML 文件已创建: {output_path}")
    return output_path

def main():
    parser = argparse.ArgumentParser(description="将视频转换为 Web 友好的格式")
    parser.add_argument("input", help="输入视频文件或目录")
    parser.add_argument("-o", "--output-dir", help="输出目录")
    parser.add_argument("-f", "--format", default="mp4", help="输出格式 (默认: mp4)")
    parser.add_argument("--batch", action="store_true", help="批量处理目录中的所有视频")
    
    args = parser.parse_args()
    
    converted_files = []
    
    if args.batch or os.path.isdir(args.input):
        print(f"批量转换目录: {args.input}")
        converted_files = batch_convert_videos(args.input, args.output_dir, args.format)
        print(f"成功转换 {len(converted_files)} 个文件")
    else:
        print(f"转换单个文件: {args.input}")
        output_file = convert_video_for_web(args.input, args.output_dir, args.format)
        if output_file:
            converted_files.append(output_file)
    
    if converted_files:
        test_html = create_test_html(converted_files)
        print(f"请在浏览器中打开 {test_html} 测试视频播放")

if __name__ == "__main__":
    main()