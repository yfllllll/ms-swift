export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
NPROC_PER_NODE=2 \
CUDA_VISIBLE_DEVICES=6,7 \
swift sft \
    --model /mnt/disk/lyf/IDEA-Research/Rex-Omni \
    --model_type qwen2_5_vl \
    --dataset labelme_dataset1 labelme_dataset2 vqa_dataset vqa_dataset2 image_caption_dataset, objvqa_dataset,objvqa_dataset1,objvqa_dataset2 \
    --custom_register_path mytools/processing/labelme_preprocessor.py \
    --train_type full \
    --torch_dtype bfloat16 \
    --attn_impl flash_attn \
    --learning_rate 5e-6 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 4 \
    --gradient_checkpointing true \
    --vit_gradient_checkpointing true \
    --freeze_vit false \
    --freeze_aligner false \
    --deepspeed zero2 \
    --warmup_ratio 0.05 \
    --save_strategy steps \
    --save_steps 1000 \
    --eval_steps 1000 \
    --save_total_limit 2 \
    --save_only_model true \
    --load_from_cache_file true \
    --dataset_num_proc 4 \
    --dataloader_num_workers 4 \
    --num_train_epochs 4 \
    --output_dir output/rex_omni_labelme_full_data1_data2_support_neg_imgcap_improved