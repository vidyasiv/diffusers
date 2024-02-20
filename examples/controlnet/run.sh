#!/bin/bash
cd /workspace/diffusers/examples/controlnet
export MODEL_DIR="runwayml/stable-diffusion-v1-5"
export OUTPUT_DIR="/localdisk/output"
accelerate launch train_controlnet.py \
        --pretrained_model_name_or_path=$MODEL_DIR  \
        --output_dir=$OUTPUT_DIR  \
        --dataset_name=fusing/fill50k  \
        --resolution=512 \
        --learning_rate=1e-5  \
        --validation_image "./conditioning_image_1.png" "./conditioning_image_2.png"  \
        --validation_prompt "red circle with blue background" "cyan circle with brown floral background" \
        --train_batch_size=16 \
        --mixed_precision=bf16 \
        --throughput_warmup_steps=3 \
        --tracker_project_name="a100-v0.26.3-seed102" --report_to="tensorboard" --seed 102

