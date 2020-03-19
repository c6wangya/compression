python bls2017.py \
    --verbose \
    --num_filters 128 \
    inv_train \
    --train_glob "../../dataset/ImageNet-Datasets-Downloader/filtered_images/*JPEG" \
    --checkpoint_dir ../results/bf-new-gradclip-clamp-lkhdguide \
    --gpu_device 3 \
    --lambda 0.015 \
    --last_step 2000000 \
    --channel_out 256 \
    --blk_type seq \
    --flow_loss_weight 0.1 \
    --kernel_size 3 \
    --y_guidance_weight 0.1 \
    --guidance_type likelihood \
    --grad_clipping 5 \
    --clamp
