#!/bin/bash
#SBATCH -J z1ksl8bl
#SBATCH -N1 --gres=gpu:V100:1 --ntasks-per-node=1 --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 16:00:00
#SBATCH -o logs/Report-%j.out

source ~/.bashrc

# export WANDB_API_KEY='<insert_key_here>'
# export HF_TOKEN='<insert_token_here>'

conda activate dlt_env

DATASET='amazon_review_1000'
# DATASET='amazon_review_250'
# DATASET='yahoo_answers_2000'
# DATASET='yahoo_answers_500'
# DATASET='yelp_review'
# DATASET='aclImdb'
# DATASET='ag_news'

# ALGORITHM='uda'
ALGORITHM='simmatch'
# ALGORITHM='remixmatch'

# MODEL=''
# MODEL='_roberta_base'
# MODEL='_roberta_base_sentiment'
# MODEL='_deberta_base'
# MODEL='_t5_base'
# MODEL='_xlnet_base_cased'
# MODEL='_llama_8b'
MODEL='_llama_8b_lora'

python train.py --c "config/usb_nlp/${ALGORITHM}/${ALGORITHM}_${DATASET}_0${MODEL}.yaml"
