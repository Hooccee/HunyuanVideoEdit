import os
os.environ["CUDA_VISIBLE_DEVICES"] = '1'
import cv2
import time
import torch
from pathlib import Path
from loguru import logger
from datetime import datetime
from tqdm import tqdm

from hyvideo.utils.file_utils import save_videos_grid, video_to_tensor
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import pprint
from datasets import get_dataloader
from contextlib import contextmanager


def main():
    args = parse_args()
    pp = pprint.PrettyPrinter(width=80, compact=True)
    pp.pprint(vars(args))
    
    # 初始化模型路径
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # 创建保存目录
    save_path = args.save_path if args.save_path_suffix == "" else f'{args.save_path}_{args.save_path_suffix}'
    os.makedirs(save_path, exist_ok=True)

    # 加载模型
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    args = hunyuan_video_sampler.args  # 获取更新后的参数
    print("Updated args:")
    pp.pprint(vars(args))

    # 初始化数据加载模式
    processing_mode = None

    # 视频处理参数组
    video_params = {
        'video_size': args.video_size,
        'video_length': args.video_length,
        'rescale': True  # 保持与单视频处理一致
    }
    if args.dataset:
        # ===================== 数据集模式 =====================
        processing_mode = "dataset"
        dataset = get_dataloader(args.dataset,**video_params)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,  # 使用参数中的batch_size
            shuffle=False,
            num_workers=4
        )
        logger.info(f"Loaded {len(dataset)} samples from {args.dataset} dataset")
        
    elif args.inverse_video_path:
        # ==================== 单视频模式 ====================
        processing_mode = "single_video"
        assert args.prompt and args.target_prompt, "必须同时提供prompt和target_prompt"
        
        # 转换为tensor并获取FPS
        video_tensor, fps = video_to_tensor(
            args.inverse_video_path,
            args.video_size,
            video_length=args.video_length,
            rescale=True
        )
        # 添加batch维度以统一接口
        video_tensor = video_tensor.unsqueeze(0)  
        logger.info(f"Loaded single video tensor with shape: {video_tensor.shape}")
        
    else:
        raise ValueError("必须指定--dataset或--inverse-video-path")

    # ====================== 统一处理逻辑 ======================
    def process_sample(sample_data, meta=None):
        """处理batch样本的统一函数"""
        # 获取基础参数
        video_tensor = sample_data['video']
        batch_size = video_tensor.shape[0]
        
        # 遍历batch中的每个样本
        for batch_idx in range(batch_size):
            # 处理单个样本 --------------------------------------------------
            # 提取当前样本数据
            current_video = video_tensor[batch_idx]  # [C,T,H,W]
            current_meta = {
                'fps': sample_data['fps'][batch_idx].item() if 'fps' in sample_data else fps,
                'video_id': meta['video_id'][batch_idx] if meta else Path(args.inverse_video_path).stem,
                'edit_type': meta['edit_type'][batch_idx].replace(' ', '_') if meta else "single_video"
            }
            
            # 获取prompt
            if processing_mode == "dataset":
                source_prompt = sample_data['source_prompt'][batch_idx]
                target_prompt = sample_data['target_prompt'][batch_idx]
            else:
                source_prompt = args.prompt
                target_prompt = args.target_prompt

            # 模型推理（处理单个样本）
            outputs = hunyuan_video_sampler.predict(
                prompt=source_prompt, 
                target_prompt=target_prompt,
                inject=args.inject,
                feature_path=args.feature_path,
                video_tensor=current_video,
                height=args.video_size[0],
                width=args.video_size[1],
                video_length=args.video_length,
                seed=args.seed,
                negative_prompt=args.neg_prompt,
                infer_steps=args.infer_steps,
                guidance_scale=args.cfg_scale,
                num_videos_per_prompt=args.num_videos,
                flow_shift=args.flow_shift,
                batch_size=args.batch_size,
                embedded_guidance_scale=args.embedded_cfg_scale,
                gamma = args.gamma,  # 添加gamma参数用于控制编辑强度
                start_timestep=args.start_timestep,  # Start time for editing (0 to 1),
                stop_timestep=args.stop_timestep,  # Stop time for editing (0 to 1),
                eta_reverse=args.eta_reverse,  # rf_inv parameter for reverse process,
                decay_eta=args.decay_eta,  # Whether to decay eta over steps,
                eta_decay_power=args.eta_decay_power,  # Power for eta decay,          
            )
            
            # 保存结果 -----------------------------------------------------
            if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
                if processing_mode == "dataset" and args.dataset == "V2VBench":
                    print
                    # V2Vbench数据集专用保存逻辑
                    # 创建video_id专属目录
                    video_dir = os.path.join(save_path, str(current_meta['video_id']))
                    os.makedirs(video_dir, exist_ok=True)
                    
                    # 确保每个edit index有唯一文件名
                    existing_files = set(os.listdir(video_dir))
                    
                    for i, sample in enumerate(outputs['samples']):
                        # 生成符合要求的文件名
                        file_index = 0
                        while True:
                            filename = f"{file_index}.mp4"
                            if filename not in existing_files:
                                break
                            file_index += 1
                        existing_files.add(filename)
                        
                        # 构建完整保存路径
                        save_full_path = os.path.join(video_dir, filename)
                        
                        # 保存视频（添加batch维度）
                        save_videos_grid(
                            sample.unsqueeze(0),
                            save_full_path,
                            fps=current_meta['fps'],
                            #rescale=True
                        )
                        logger.info(f"保存编辑视频到: {save_full_path}")
                else:
                    # 默认保存逻辑（单视频模式和其他数据集）
                    for i, sample in enumerate(outputs['samples']):
                        time_flag = datetime.now().strftime("%Y%m%d-%H%M%S")
                        seed = outputs['seeds'][i]
                        
                        # 生成文件名
                        filename = f"{current_meta['video_id']}_{current_meta['edit_type']}_{time_flag}_s{seed}.mp4"
                        save_full_path = os.path.join(save_path, filename)
                        
                        # 保存视频（添加batch维度）
                        save_videos_grid(
                            sample.unsqueeze(0),  # 恢复batch维度
                            save_full_path,
                            fps=current_meta['fps'],
                            #rescale=True
                        )
                        logger.info(f"保存单次结果到: {save_full_path}")

    # ====================== 执行处理逻辑 ======================
    if processing_mode == "dataset":
        with tqdm(dataloader, desc="处理数据集批次", position=0) as pbar_outer:
            for batch in pbar_outer:
            # 处理每个批次时临时修改内部进度条配置
                
                    # 构造元数据（假设数据集中包含这些字段）
                meta = {
                    "video_id": batch['video_id'],       # [B,]
                    "edit_type": batch['edit_type'],     # [B,]
                }
                process_sample(batch, meta)
                pbar_outer.update(1) 

    elif processing_mode == "single_video":
        # 构造与数据集模式一致的输入结构
        sample_data = {
            'video': video_tensor,  # [1,C,T,H,W]
            'source_prompt': [args.prompt],
            'target_prompt': [args.target_prompt],
            'fps': torch.tensor([fps])  # 保持tensor格式
        }
        process_sample(sample_data)
        logger.info("单视频处理完成")

if __name__ == "__main__":
    main()