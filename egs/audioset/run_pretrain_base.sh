#!/bin/bash
#SBATCH --job-name AS-PT-B
#SBATCH --nodes=2
#SBATCH --gpus-per-task=8 # number of GPUs
#SBATCH --ntasks-per-node=1
#SBATCH --output=/mnt/opr/yblin/cav-pt/egs/audioset/log/%j.out
#####SBATCH --exclude=node001


# ACTIVATE ANACONDA
eval "$(conda shell.bash hook)"
conda activate cav

export TORCH_HOME=../../pretrained_models

model=cav-mae
masking_ratio=0.25
masking_ratio_a=0.25
mask_mode=unstructured # or time, or freq, or tf
contrast_loss_weight=1
mae_loss_weight=0
tr_pos=False
norm_pix_loss=True


pretrain_path=None


bal=None # balanced sampling, should be false for pretraining
lr=2e-4
epoch=25
lrscheduler_start=10
lrscheduler_decay=0.5
# lrscheduler_decay=0.9
lrscheduler_step=5
dataset_mean=-5.081
dataset_std=4.4849
target_length=1024
noise=True
mixup=0.0
batch_size=4
lr_adapt=False

dataset=audioset
tr_data=/mnt/opr/yblin/audioset_sun/audioset_2m_cleaned.json
te_data=/mnt/opr/yblin/audioset_sun/audioset_eval_cleaned.json
label_csv=/mnt/opr/yblin/audioset_sun/class_labels_indices.csv

exp_dir=./exp/CL-base-AS-0050-5+MAE-ACAV-overide-${dataset}-${model}-bal${bal}-lr${lr}-epoch${epoch}-bs${batch_size}-norm${norm_pix_loss}-c${contrast_loss_weight}-p${mae_loss_weight}-tp${tr_pos}-mr-${mask_mode}-${masking_ratio}-a5
# mkdir -p $exp_dir



master_node=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)



# torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:0   --nnodes=1 --nproc_per_node=4 \
#     ../../src/run_cavmae_pretrain_base.py --model ${model} --dataset ${dataset} \
#     --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
#     --label-csv ${label_csv} --n_class 527 \
#     --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
#     --mixup ${mixup} --bal ${bal} \
#     --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
#     --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
#     --lr_adapt ${lr_adapt} \
#     --norm_pix_loss ${norm_pix_loss} \
#     --pretrain_path ${pretrain_path} \
#     --mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} --num_workers 6 \
#     --tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --masking_ratio_a ${masking_ratio_a}\
#     --mask_mode ${mask_mode} --wandb 0 --model_name ddp_CL_B-050-Kmean-128_SQL_overide



srun torchrun --rdzv_backend=c10d --rdzv_endpoint=${master_node}:9527   --nnodes=2 --nproc_per_node=8 \
    ../../src/run_cavmae_pretrain_base.py --model ${model} --dataset ${dataset} \
    --data-train ${tr_data} --data-val ${te_data} --exp-dir $exp_dir \
    --label-csv ${label_csv} --n_class 527 \
    --lr $lr --n-epochs ${epoch} --batch-size $batch_size --save_model True \
    --mixup ${mixup} --bal ${bal} \
    --lrscheduler_start ${lrscheduler_start} --lrscheduler_decay ${lrscheduler_decay} --lrscheduler_step ${lrscheduler_step} \
    --dataset_mean ${dataset_mean} --dataset_std ${dataset_std} --target_length ${target_length} --noise ${noise} --warmup True \
    --lr_adapt ${lr_adapt} \
    --norm_pix_loss ${norm_pix_loss} \
    --pretrain_path ${pretrain_path} \
    --mae_loss_weight ${mae_loss_weight} --contrast_loss_weight ${contrast_loss_weight} --num_workers 6 \
    --tr_pos ${tr_pos} --masking_ratio ${masking_ratio} --masking_ratio_a ${masking_ratio_a}\
    --mask_mode ${mask_mode} --wandb 1 --model_name ddp-A5000_ACAV-Mixed0050-5+MAE_ratio_SQL