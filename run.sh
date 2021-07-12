#!/usr/bin/bash

read -p "Enter repo name: " repo

if [ -d $repo ]
then
    cd $repo
    git pull
    cd ..
else
    git clone https://huggingface.co/flax-community/$repo
fi

token=$(cat /home/chhablani_gunjan/.huggingface/token)

echo "Token found $token"
./run_image_text_mlm.py\
 --output_dir $repo\
 --data_dir /home/user/data/CC12M/images\
 --train_file /home/user/data/CC12M/val_file.tsv\
 --validation_file /home/user/data/CC12M/val_file.tsv\
 --save_steps 2000\
 --num_train_epochs 5\
 --eval_steps 1000\
 --logging_steps 1000\
 --save_total_limit 2\
 --push_to_hub\
 --push_to_hub_organization flax-community\
 --push_to_hub_token $token\
 --per_device_train_batch_size 8\
 --per_device_eval_batch_size 8\
 --warmup_ratio 0.1\
 --learning_rate 1e-5\
 --weight_decay 0.01\


# --resume_from_checkpoint test-vqa/ckpt-11
# save total limit
# learning_rate
# resume_from_checkpoint
# weight decay?
