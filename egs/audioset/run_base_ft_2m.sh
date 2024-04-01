#!/bin/bash
#SBATCH --job-name AS2M-FT
#SBATCH --nodes=1
#SBATCH --gpus-per-task=4 # number of GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --output=/mount/opr/yblin/cav-pt/egs/audioset/log/%j.out

eval "$(conda shell.bash hook)"
conda activate cav

model=cav-mae-ft

ftmode=mm_grad


# you can replace with any checkpoint you want, but by default, we use cav-mae-scale++
cur_dir=$(pwd)
pretrain_path=/mnt/opr/yblin/cav-pt/egs/audioset/exp/CL-base-AS-MMix-Multimodal-0050_5-overide-audioset-cav-mae-balNone-lr2e-4-epoch25-bs96-normTrue-c1-p0-tpFalse-mr-unstructured-0.75-a5/models/audio_model.10.pth
freeze_base=False


bal=bal



head_lr=100 # newly initialized ft layers uses 100 times larger than the base lr
lr=1e-5

epoch=15
lrscheduler_start=5
lrscheduler_decay=0.9
lrscheduler_step=1


batch_size=64
wa=True
wa_start=10
wa_end=15
lr_adapt=False
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
freqm=48
timem=192
mixup=0.5
label_smooth=0.1

dataset=audioset_2m
tr_data=/data/yanbo/uavm/egs/audioset/datafile/audioset_2m_cleaned.json
te_data=/data/yanbo/uavm/egs/audioset/datafile/audioset_eval_cleaned.json
label_csv=/data/yanbo/uavm/egs/audioset/datafile/class_labels_indices.csv



export PYTHONWARNINGS="ignore"

# exp_dir=./exp/CL-test2-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
exp_dir=./ft_base

mkdir -p $exp_dir


torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=8 \
    ../../src/run_cavmae_ft_base.py --model ${model} --dataset ${dataset} \
    --data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
    --label_csv ${label_csv} --n_class 527 \
    --lr $lr --n_epochs ${epoch} --batch_size $batch_size --save_model True \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --label_smooth ${label_smooth} \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --loss BCE --metrics mAP --warmup True \
    --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
    --pretrain_path ${pretrain_path} --ftmode ${ftmode} \
    --freeze_base ${freeze_base} --head_lr ${head_lr} \
    --num_workers 6 --skip_frame_agg False --wandb 1 --model_name as2m-CL-test2 --dis_w 0.0 --dis_w_2 0.0

