import os
import yaml
import torch
from torch.utils.data import Dataset
from torchvision.io import read_video
from hyvideo.utils.file_utils import video_to_tensor

default_config_path = "/data/chx/V2VBench/config.yaml"
default_videos_dir = "/data/chx/V2VBench/videos"

class V2VBench_dataset(Dataset):
    def __init__(self, transform=None, config_path=default_config_path, videos_dir=default_videos_dir,
                 video_size=None, video_length=None, rescale=True):
        """
        Args:
            video_size (list): [height, width] 目标视频尺寸
            video_length (int): 目标视频长度（帧数）
            rescale (bool): 是否归一化到[-1, 1]
        """
        self.videos_dir = videos_dir
        self.video_size = video_size
        self.video_length = video_length
        self.rescale = rescale
        
        # Load config file
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # 扁平化数据结构
        self.samples = []
        for video_data in self.config['data']:
            video_id = video_data['video_id']
            source_prompt = video_data['prompt']
            video_path = os.path.join(self.videos_dir, f"{video_id}.mp4")
            
            for edit in video_data['edit']:
                self.samples.append({
                    'video_path': video_path,
                    'video_id': video_id,
                    'source_prompt': source_prompt,
                    'target_prompt': edit['prompt'],
                    'src_words': edit.get('src_words', ''),
                    'tgt_words': edit.get('tgt_words', ''),
                    'edit_type': edit.get('type', '')
                })

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        video_path = sample['video_path']
        
        # 获取视频tensor和原始fps
        video_tensor, fps = video_to_tensor(
            video_path=video_path,
            video_size=self.video_size,
            video_length=self.video_length,
            rescale=self.rescale
        )
        
        return {
            'video': video_tensor,
            'fps': fps,  # 新增fps字段
            'video_id': sample['video_id'],
            'source_prompt': sample['source_prompt'],
            'target_prompt': sample['target_prompt'],
            'src_words': sample['src_words'],
            'tgt_words': sample['tgt_words'],
            'edit_type': sample['edit_type']
        }