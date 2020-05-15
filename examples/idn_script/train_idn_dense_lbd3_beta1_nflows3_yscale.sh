python bls2017.py \
    --verbose \
    --num_filters 128 \
    int_train \
    --train_glob "../../dataset/filtered_images/*JPEG" \
    --val_glob "../../dataset/kodak24/kodim01.png" \
    --checkpoint_dir ../results/baseguide-val/idn-z0-dense-lbd3-beta1-nflows3-yscale\
    --pretrain_checkpoint_dir ../results/baseguide/bf-new-baseguide-pretrain \
    --gpu_device 0 \
    --lambda 30 \
    --beta 1 \
    --last_step 1000000 \
    --channel_out 256 \
    --blk_type dense \
    --flow_loss_weight 1 \
    --kernel_size 3 \
    --val_gap 50000 \
    --int_discrete_net \
    --downsample_type sqz \
    --zero_z \
    --prepos_ste \
    --n_flows 3 \
    --y_scale_up \
    --optimize_separately


: '
    --n_ops 4 \
    --y_guidance_weight 5e-4 \
    --guidance_type baseline \
    --df_iter 100000 \
    --end_iter 150000 \
    --int_flow \
    --lr_scheduler scheduled \
    --lr_warmup_steps 40000 \
    --lr_decay 0.999995 \
    --norm an
    --main_lr 1e-3 \
    --aux_lr 1e-3
    --train_manually \
    --num_data 9250 \
    --freeze_aux \
    --no_aux
'