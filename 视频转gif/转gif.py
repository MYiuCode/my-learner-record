import os
import tkinter as tk
from tkinter import filedialog, messagebox


def convert_video_to_gif():
    """将视频转换为GIF格式（严格控制文件大小）"""
    from moviepy import VideoFileClip
    
    # 选择输入视频文件
    root = tk.Tk()
    root.withdraw()

    input_path = filedialog.askopenfilename(
        title="选择视频文件",
        filetypes=[("视频文件", "*.mp4 *.avi *.mov *.mkv *.flv *.wmv")]
    )

    if not input_path:
        return

    # 检查文件大小（不超过500MB）
    file_size = os.path.getsize(input_path) / (1024 * 1024)
    if file_size > 500:
        messagebox.showerror("错误", f"视频文件大小超过500MB限制（当前大小：{file_size:.2f}MB）")
        return

    # 选择输出路径
    output_path = filedialog.asksaveasfilename(
        title="保存GIF文件",
        defaultextension=".gif",
        filetypes=[("GIF文件", "*.gif")]
    )

    if not output_path:
        return

    video = None
    resized_video = None
    final_video = None
    
    try:
        print(f"开始转换...")
        print(f"输入: {input_path}")
        print(f"输出: {output_path}")
        
        # 加载视频
        video = VideoFileClip(input_path)
        original_width, original_height = video.size
        duration = video.duration
        original_fps = video.fps
        
        print(f"视频时长: {duration:.2f}秒")
        print(f"原始分辨率: {original_width}x{original_height}")
        print(f"原始帧率: {original_fps}fps")
        
        # 目标文件大小（字节）- 留有余量
        target_size_bytes = 18 * 1024 * 1024  # 18MB，留2MB余量
        
        # 保持原始分辨率（不降低清晰度）
        target_width = original_width
        target_height = original_height
        
        # 确保尺寸为偶数
        if target_width % 2 != 0:
            target_width -= 1
        if target_height % 2 != 0:
            target_height -= 1
        
        # 计算合适的帧率以控制文件大小
        pixels_per_frame = target_width * target_height
        estimated_bytes_per_frame = pixels_per_frame * 0.5
        
        # 最大帧数 = 目标大小 / 每帧大小
        max_frames = target_size_bytes / estimated_bytes_per_frame
        
        # 计算帧率
        calculated_fps = max_frames / duration
        
        # 限制帧率范围：最低5fps，最高15fps
        gif_fps = max(5, min(15, calculated_fps))
        
        # 如果是短视频（<5秒），可以适当提高帧率
        if duration < 5:
            gif_fps = min(15, gif_fps * 1.5)
        
        gif_fps = int(gif_fps)
        
        print(f"\n输出分辨率: {target_width}x{target_height} (保持原分辨率)")
        print(f"目标帧率: {gif_fps} fps")
        print(f"预计总帧数: {int(duration * gif_fps)}")
        print(f"目标文件大小: < 20 MB")
        
        # 调整大小和帧率
        resized_video = video.resized(width=target_width, height=target_height)
        final_video = resized_video.with_fps(gif_fps)
        
        # 导出为GIF
        final_video.write_gif(
            output_path,
            fps=gif_fps,
            logger='bar'
        )
        
        # 释放资源
        video.close()
        resized_video.close()
        final_video.close()
        video = None
        resized_video = None
        final_video = None
        
        # 检查文件大小
        output_size = os.path.getsize(output_path) / (1024 * 1024)
        print(f"\n转换完成！")
        print(f"原始视频: {file_size:.2f} MB")
        print(f"GIF文件: {output_size:.2f} MB")
        
        if output_size > 20:
            print(f"警告: 文件超过20MB，尝试进一步压缩...")
            os.remove(output_path)
            
            retry_fps = max(3, gif_fps - 3)
            print(f"重试帧率: {retry_fps} fps")
            
            # 重新加载视频进行重试
            video = VideoFileClip(input_path)
            resized_video = video.resized(width=target_width, height=target_height)
            final_video = resized_video.with_fps(retry_fps)
            
            final_video.write_gif(
                output_path,
                fps=retry_fps,
                logger='bar'
            )
            
            video.close()
            resized_video.close()
            final_video.close()
            video = None
            resized_video = None
            final_video = None
            
            output_size = os.path.getsize(output_path) / (1024 * 1024)
            print(f"重试后GIF文件: {output_size:.2f} MB")
        
        messagebox.showinfo("成功", 
            f"GIF已保存到：\n{output_path}\n\n"
            f"原始视频: {file_size:.2f} MB\n"
            f"GIF文件: {output_size:.2f} MB\n"
            f"分辨率: {target_width}x{target_height}\n"
            f"帧率: {gif_fps} fps"
        )

    except Exception as e:
        import traceback
        error_msg = f"转换失败：{str(e)}\n\n详细错误:\n{traceback.format_exc()}"
        print(error_msg)
        messagebox.showerror("错误", error_msg)
    finally:
        # 确保资源释放
        if final_video is not None:
            try:
                final_video.close()
            except:
                pass
        if resized_video is not None:
            try:
                resized_video.close()
            except:
                pass
        if video is not None:
            try:
                video.close()
            except:
                pass


if __name__ == "__main__":
    convert_video_to_gif()
