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
 --train_file /home/user/data/CC12M/train_file_batch.tsv\
 --validation_file /home/user/data/CC12M/val_file_batch.tsv\
 --save_steps 10000\
 --num_train_epochs 5\
 --eval_steps 5000\
 --logging_steps 5000\
 --save_total_limit 5\
 --push_to_hub\
 --push_to_hub_organization flax-community\
 --push_to_hub_token $token\
 --per_device_train_batch_size 64\
 --per_device_eval_batch_size 64\
 --warmup_steps 5000\
 --learning_rate 5e-5


#  --weight_decay 0.01\
#  --max_train_samples 5000\
#  --max_eval_samples 500\
# --resume_from_checkpoint test-vqa/ckpt-11
# save total limit
# learning_rate
# resume_from_checkpoint
# weight decay?
