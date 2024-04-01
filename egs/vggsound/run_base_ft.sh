#!/bin/bash
#SBATCH --job-name VGG-FT
#SBATCH --nodes=1
#SBATCH --gpus-per-task=8 # number of GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --output=/mnt/opr/yblin/cav-pt/egs/vggsound/log/%j.out

eval "$(conda shell.bash hook)"
conda activate cav



model=cav-mae-ft
# ftmode=mm_grad
pretrain_path=/mnt/opr/yblin/cav-pt/egs/audioset/exp/CL-base-AS-0050-5+MAE-v2-overide-audioset-cav-mae-balNone-lr2e-4-epoch25-bs96-normTrue-c1-p0-tpFalse-mr-unstructured-0.25-a5/models/audio_model.20.pth
freeze_base=False


bal=bal



head_lr=10 # newly initialized ft layers uses 100 times larger than the base lr
lr=5e-05

epoch=20
lrscheduler_start=2
lrscheduler_decay=0.75
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

dataset=vggsound
tr_data=./vgg_train_cleaned.json
te_data=./vgg_test_cleaned.json
label_csv=./class_labels_indices_vgg.csv



export PYTHONWARNINGS="ignore"

# exp_dir=./exp/CL-test2-${model}-${lr}-${lrscheduler_start}-${lrscheduler_decay}-${lrscheduler_step}-bs${batch_size}-lda${lr_adapt}-${ftmode}-fz${freeze_base}-h${head_lr}-a5
exp_dir=./ft_base

mkdir -p $exp_dir



#  torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=1 \
#     ../../src/run_retrieval.py --model ${model} --dataset ${dataset} \
#     --data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
#     --label_csv ${label_csv} --n_class 309 \
#     --lr $lr --n_epochs ${epoch} --batch_size $batch_size --save_model True \
#     --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
#     --label_smooth ${label_smooth} \
#     --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
#     --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
#     --loss CE --metrics acc --warmup True \
#     --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
#     --pretrain_path ${pretrain_path} --ftmode ${ftmode} \
#     --freeze_base ${freeze_base} --head_lr ${head_lr} \
#     --mm_lr 1 \
#     --num_workers 6 --skip_frame_agg False --wandb 0 --model_name vgg-b_robust_2mean --dis_w 0.0 --dis_w_2 0.0



torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0 --nproc_per_node=8 \
    ../../src/run_cavmae_ft_base.py --model ${model} --dataset ${dataset} \
    --data_train ${tr_data} --data_val ${te_data} --exp_dir $exp_dir \
    --label_csv ${label_csv} --n_class 309 \
    --lr $lr --n_epochs ${epoch} --batch_size $batch_size --save_model True \
    --freqm $freqm --timem $timem --mixup ${mixup} --bal ${bal} \
    --label_smooth ${label_smooth} \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} \
    --loss CE --metrics acc --warmup True \
    --wa ${wa} --wa_start ${wa_start} --wa_end ${wa_end} --lr_adapt ${lr_adapt} \
    --pretrain_path ${pretrain_path} --ftmode ${ftmode} \
    --freeze_base ${freeze_base} --head_lr ${head_lr} \
    --mm_lr 1 \
    --num_workers 6 --skip_frame_agg False --wandb 0 --model_name vgg-b_robust_2mean_acav --dis_w 0.0 --dis_w_2 0.0


