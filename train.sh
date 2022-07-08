python3 train.py --dataroot /media/data2/dataset/ARkit_LMK_dataset/train/list.txt \
    --val_dataroot /media/data2/dataset/ARkit_LMK_dataset/val/list.txt \
    --snapshot /media/data1/checkpoints/pfld/220704/snapshot \
    --log_file /media/data1/checkpoints/pfld/220704/train.logs \
    --tensorboard /media/data1/checkpoints/pfld/220704/tensorboard \
    --train_batchsize 256 \
    -j 8