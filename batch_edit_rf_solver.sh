python edit_video.py \
    --video-size 512 512 \
    --video-length 25 \
    --infer-steps 45 \
    --inject 7 \
    --seed 42 \
    --embedded-cfg-scale 3.5 \
    --flow-shift 7.0 \
    --flow-reverse \
    --use-cpu-offload \
    --use-fp8 \
    --save-path ./results/V2VBench/3op \
    --dataset "V2VBench"