import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0'
import cv2
import time
from pathlib import Path
from loguru import logger
from datetime import datetime

from hyvideo.utils.file_utils import save_videos_grid,video_to_tensor
from hyvideo.config import parse_args
from hyvideo.inference import HunyuanVideoSampler
import pprint



def main():
    args = parse_args()
    pp = pprint.PrettyPrinter(width=80, compact=True)
    pp.pprint(vars(args))
    models_root_path = Path(args.model_base)
    if not models_root_path.exists():
        raise ValueError(f"`models_root` not exists: {models_root_path}")
    
    # Create save folder to save the samples
    save_path = args.save_path if args.save_path_suffix=="" else f'{args.save_path}_{args.save_path_suffix}'
    if not os.path.exists(args.save_path):
        os.makedirs(save_path, exist_ok=True)

    # Load models
    hunyuan_video_sampler = HunyuanVideoSampler.from_pretrained(models_root_path, args=args)
    
    # Get the updated args
    args = hunyuan_video_sampler.args
    print("Updated args:")
    pp.pprint(vars(args))


    video_capture = cv2.VideoCapture(args.inverse_video_path)

    # 检查视频是否成功打开
    if not video_capture.isOpened():
        print(f"Error: Could not open video file {args.inverse_video_path}")
    else:
        # 获取帧率
        fps = video_capture.get(cv2.CAP_PROP_FPS)
        print(f"FPS of the video: {fps}")

    # 释放视频捕获对象
    video_capture.release()

    video_tensor = video_to_tensor(args.inverse_video_path,args.video_size,video_length=args.video_length, rescale=True)





    # Start sampling
    # TODO: batch inference check
    outputs = hunyuan_video_sampler.predict(
        prompt=args.prompt, 
        target_prompt=args.target_prompt,
        inject=args.inject,
        feature_path=args.feature_path,
        video_tensor=video_tensor,
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
        embedded_guidance_scale=args.embedded_cfg_scale
    )
    samples = outputs['samples']
    # samples = [video_tensor]
    # Save samples
    if 'LOCAL_RANK' not in os.environ or int(os.environ['LOCAL_RANK']) == 0:
        for i, sample in enumerate(samples):
            sample = samples[i].unsqueeze(0)
            time_flag = datetime.fromtimestamp(time.time()).strftime("%Y-%m-%d-%H:%M:%S")
            save_path = f"{save_path}/{time_flag}_seed{outputs['seeds'][i]}_{outputs['prompts'][i][:100].replace('/','')}.mp4"
            # save_path = f"{save_path}/{time_flag}.mp4"
            save_videos_grid(sample, save_path, fps=fps,
                            #  rescale=True
                             )
            logger.info(f'Sample save to: {save_path}')

if __name__ == "__main__":
    main()
