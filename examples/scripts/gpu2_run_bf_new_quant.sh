python bls2017.py \
    --verbose \
    --num_filters 128 \
    inv_train \
    --train_glob "../../dataset/ImageNet-Datasets-Downloader/filtered_images/*JPEG" \
    --checkpoint_dir ../results/bf-new-quant \
    --gpu_device 2 \
    --lambda 0.015 \
    --last_step 2000000 \
    --channel_out 256 \
    --blk_type seq \
    --flow_loss_weight 0.1 \
    --kernel_size 3 \
    --quant_grad 
