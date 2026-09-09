PYTHON='python'

export CUDA_VISIBLE_DEVICES=0
export WANDB_MODE=offline

# Output directory for logs
SAVE_DIR='./outputs/'
mkdir -p ${SAVE_DIR}

# Dataset: gastrovision
for DATASET in 'gastrovision'
do
    for MODE in 1 2 3
    do
        # Generate log file number
        EXP_NUM=$(ls ${SAVE_DIR} | wc -l)
        EXP_NUM=$((${EXP_NUM}+1))
        echo "Running experiment ${EXP_NUM} dataset ${DATASET} mode ${MODE}"

        ${PYTHON} -m methods.contrastive_training.contrastive_training_wandb \
            --dataset_name ${DATASET} \
            --batch_size 128 \
            --grad_from_block 10 \
            --epochs 200 \
            --base_model vit_dino \
            --num_workers 8 \
            --use_ssb_splits True \
            --sup_con_weight 0.35 \
            --weight_decay 5e-5 \
            --unsupervised_smoothing 1.0 \
            --contrast_unlabel_only False \
            --transform imagenet \
            --lr 0.1 \
            --eval_funcs v1 v2 \
            --use_wandb \
            --wandb_project_prefix GBD \
            --unbalanced True \
            --num_hyperedges 16 \
            --use_gru_update True \
            --seed 42 \
            --mode ${MODE} \
            --wandb_run_name GBD_${DATASET}_mode${MODE} \
        > ${SAVE_DIR}logfile_${EXP_NUM}_${DATASET}_mode${MODE}.out
    done
done
