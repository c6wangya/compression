python bls2017.py \
    --verbose \
    --num_filters 256 \
    train \
    --train_glob "../../dataset/ImageNet-Datasets-Downloader/filtered_images/*JPEG" \
    --checkpoint_dir ../results/17-imgnet9250-256f-2M-15e-3 \
    --pretrain_checkpoint_dir ../results/bf-new-baseguide-pretrain-load \
    --gpu_device 0 \
    --lambda 0.015 \
    --last_step 2000000 \
    --y_guidance_weight 0.1 \
    --guidance_type baseline_pretrain 

: '
    --train_manually \
    --channel_out 256 \
    --blk_type seq \
    --flow_loss_weight 0.1 \
    --kernel_size 3 \
    --y_guidance_weight 0.1 \
    --num_data 9250 \
'

