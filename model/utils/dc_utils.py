# This file is originally from DepthCrafter/depthcrafter/utils.py at main · Tencent/DepthCrafter
# SPDX-License-Identifier: MIT License license
#
# This file may have been modified by ByteDance Ltd. and/or its affiliates on [date of modification]
# Original file is released under [ MIT License license], with the full license text available at [https://github.com/Tencent/DepthCrafter?tab=License-1-ov-file].
import numpy as np
import matplotlib.cm as cm
import imageio
try:
    from decord import VideoReader, cpu
    DECORD_AVAILABLE = True
except:
    import cv2
    DECORD_AVAILABLE = False

def ensure_even(value):
    return value if value % 2 == 0 else value + 1

def read_video_frames(video_path, process_length, target_fps=-1, max_res=-1):
    if DECORD_AVAILABLE:
        vid = VideoReader(video_path, ctx=cpu(0))
        original_height, original_width = vid.get_batch([0]).shape[1:3]
        height = original_height
        width = original_width
        if max_res > 0 and max(height, width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = ensure_even(round(original_height * scale))
            width = ensure_even(round(original_width * scale))

        vid = VideoReader(video_path, ctx=cpu(0), width=width, height=height)

        fps = vid.get_avg_fps() if target_fps == -1 else target_fps
        stride = round(vid.get_avg_fps() / fps)
        stride = max(stride, 1)
        frames_idx = list(range(0, len(vid), stride))
        if process_length != -1 and process_length < len(frames_idx):
            frames_idx = frames_idx[:process_length]
        frames = vid.get_batch(frames_idx).asnumpy()
    else:
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

        if max_res > 0 and max(original_height, original_width) > max_res:
            scale = max_res / max(original_height, original_width)
            height = round(original_height * scale)
            width = round(original_width * scale)

        fps = original_fps if target_fps < 0 else target_fps

        stride = max(round(original_fps / fps), 1)

        frames = []
        frame_count = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret or (process_length > 0 and frame_count >= process_length):
                break
            if frame_count % stride == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # Convert BGR to RGB
                if max_res > 0 and max(original_height, original_width) > max_res:
                    frame = cv2.resize(frame, (width, height))  # Resize frame
                frames.append(frame)
            frame_count += 1
        cap.release()
        frames = np.stack(frames, axis=0)

    return frames, fps


def save_video(frames, output_video_path, fps=10, is_depths=False, grayscale=False):
    import imageio
    import numpy as np
    from pathlib import Path
    
    # 确保文件扩展名是视频格式
    output_path = Path(output_video_path)
    
    # 视频格式扩展名
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.wmv']
    
    # 如果扩展名不是视频格式，改为.mp4
    if output_path.suffix.lower() not in video_extensions:
        output_video_path = str(output_path.with_suffix('.mp4'))
        print(f"注意: 输出路径扩展名已改为 .mp4")
    
    # 确保fps参数是整数
    fps = int(fps)
    
    # 创建写入器 - 使用正确的参数
    writer = imageio.get_writer(
        output_video_path, 
        fps=fps, 
        macro_block_size=1, 
        codec='libx264', 
        ffmpeg_params=['-crf', '18', '-pix_fmt', 'yuv420p']
    )
    
    if is_depths:
        # 导入颜色映射
        from matplotlib import cm
        colormap = np.array(cm.get_cmap("inferno").colors)
        
        # 计算全局最小最大值
        d_min, d_max = frames.min(), frames.max()
        
        for i in range(frames.shape[0]):
            depth = frames[i]
            # 归一化到0-255
            depth_norm = ((depth - d_min) / (d_max - d_min) * 255).astype(np.uint8)
            
            if not grayscale:
                # 应用颜色映射
                depth_vis = (colormap[depth_norm] * 255).astype(np.uint8)
                # 确保是3通道
                if len(depth_vis.shape) == 2:
                    depth_vis = np.stack([depth_vis, depth_vis, depth_vis], axis=2)
            else:
                # 灰度图
                depth_vis = np.stack([depth_norm, depth_norm, depth_norm], axis=2)
            
            writer.append_data(depth_vis)
    else:
        for i in range(frames.shape[0]):
            frame = frames[i]
            
            # 确保帧格式正确
            if frame.dtype != np.uint8:
                if frame.max() <= 1.0:  # 如果是0-1范围的浮点数
                    frame = (frame * 255).astype(np.uint8)
                else:
                    frame = frame.astype(np.uint8)
            
            # 确保是3通道
            if len(frame.shape) == 2:  # 灰度图
                frame = np.stack([frame, frame, frame], axis=2)
            elif frame.shape[2] == 4:  # RGBA
                frame = frame[:, :, :3]  # 去掉alpha通道
            elif frame.shape[2] == 1:  # 单通道
                frame = np.repeat(frame, 3, axis=2)
            
            writer.append_data(frame)
    
    writer.close()
    print(f"视频已保存: {output_video_path} (FPS: {fps})")
