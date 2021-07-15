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
./run_image_text_classification.py\
 --output_dir $repo\
 --data_dir ~/VQAV2\
 --train_file ./train_file_trans.tsv\
 --validation_file ./val_file_trans.tsv\
 --save_steps 5000\
 --num_train_epochs 10\
 --eval_steps 2500\
 --logging_steps 2500\
 --save_total_limit 5\
 --push_to_hub\
 --push_to_hub_organization flax-community\
 --push_to_hub_token $token\
 --per_device_train_batch_size 64\
 --per_device_eval_batch_size 64\
 --warmup_steps 5000\
 --learning_rate 1e-4
#  --max_train_samples 5000\
#  --max_eval_samples 500\


#  --weight_decay 0.01\

# --resume_from_checkpoint test-vqa/ckpt-11
# save total limit
# learning_rate
# resume_from_checkpoint
# weight decay?
