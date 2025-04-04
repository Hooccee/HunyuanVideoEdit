import os
from pathlib import Path
from einops import rearrange
import cv2
import torch
import torchvision
import numpy as np
import imageio

CODE_SUFFIXES = {
    ".py",  # Python codes
    ".sh",  # Shell scripts
    ".yaml",
    ".yml",  # Configuration files
}


def safe_dir(path):
    """
    Create a directory (or the parent directory of a file) if it does not exist.

    Args:
        path (str or Path): Path to the directory.

    Returns:
        path (Path): Path object of the directory.
    """
    path = Path(path)
    path.mkdir(exist_ok=True, parents=True)
    return path


def safe_file(path):
    """
    Create the parent directory of a file if it does not exist.

    Args:
        path (str or Path): Path to the file.

    Returns:
        path (Path): Path object of the file.
    """
    path = Path(path)
    path.parent.mkdir(exist_ok=True, parents=True)
    return path

def save_videos_grid(videos: torch.Tensor, path: str, rescale=False, n_rows=1, fps=24):
    """save videos by video tensor
       copy from https://github.com/guoyww/AnimateDiff/blob/e92bd5671ba62c0d774a32951453e328018b7c5b/animatediff/utils/util.py#L61

    Args:
        videos (torch.Tensor): video tensor predicted by the model
        path (str): path to save video
        rescale (bool, optional): rescale the video tensor from [-1, 1] to  . Defaults to False.
        n_rows (int, optional): Defaults to 1.
        fps (int, optional): video save fps. Defaults to 8.
    """
    # 将视频张量的维度从 (batch, channel, time, height, width) 转换为 (time, batch, channel, height, width)
    videos = rearrange(videos, "b c t h w -> t b c h w")
    outputs = []
    
    # 遍历每一帧
    for x in videos:
        # 将每一帧的多个视频拼接成一个网格图像
        x = torchvision.utils.make_grid(x, nrow=n_rows)
        # 调整维度顺序以便后续处理
        x = x.transpose(0, 1).transpose(1, 2).squeeze(-1)
        
        # 如果需要重新缩放，将张量从 [-1, 1] 缩放到 [0, 1]
        if rescale:
            x = (x + 1.0) / 2.0  # -1,1 -> 0,1
        
        # 将张量的值限制在 [0, 1] 范围内
        x = torch.clamp(x, 0, 1)
        # 将张量转换为 numpy 数组并转换为 uint8 类型
        x = (x * 255).numpy().astype(np.uint8)
        outputs.append(x)

    # 创建保存视频的目录（如果不存在）
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # 使用 imageio 库将帧列表保存为视频文件
    imageio.mimsave(path, outputs, fps=fps)


def video_to_tensor(video_path: str, video_size: list = None, rescale=True, video_length: int = None) -> torch.Tensor:
    """
    Convert a video file to a tensor.使用OpenCV读取视频并返回tensor和原始fps

    Args:
        video_path (str): Path to the input video file.
        video_size (list, optional): Target video size as [height, width]. If provided, resize video to this size.
        rescale (bool, optional): Rescale the video frames from [0, 255] to [-1, 1]. Defaults to False.
        video_length (int, optional): Target video length in frames. If provided, truncate or pad video to this length.

    Returns:
        torch.Tensor: Video tensor with shape (batch, channel, time, height, width).
        float: Original frames per second (fps) of the video.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"无法打开视频文件: {video_path}")
    
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # 转换BGR到RGB
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        # 调整尺寸
        if video_size is not None:
            frame = cv2.resize(frame, (video_size[1], video_size[0]))  # OpenCV尺寸顺序(width, height)
        frames.append(frame)
    
    cap.release()
    
    # 处理视频长度
    if video_length is not None:
        if len(frames) > video_length:
            frames = frames[:video_length]
        elif len(frames) < video_length:
            last_frame = frames[-1] if frames else np.zeros((video_size[0], video_size[1], 3), dtype=np.uint8)
            frames += [last_frame.copy()] * (video_length - len(frames))
    
    # 转换为tensor
    frames = np.array(frames)
    video_tensor = torch.from_numpy(frames).permute(3, 0, 1, 2).float()  # (T, H, W, C) -> (C, T, H, W)
    
    if rescale:
        video_tensor = (video_tensor / 127.5) - 1.0  # [0, 255] -> [-1, 1]
    
    return video_tensor, fps