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
 --push_to_hub\
 --push_to_hub_organization flax-community\
 --save_steps 1\
 --eval_steps 1\
 --logging_steps 1\
 --push_to_hub_token $token