export MODEL_NAME="/home/ec2-user/GenAI-Project/src/baseline/instruct-pix2pix/checkpoints/instruct-pix2pix-diffusers"
export DATA_DIR="/home/ec2-user/GenAI-Project/data/stylebooth_dataset"
export OUTPUT_DIR="/home/ec2-user/GenAI-Project/model/instructp2p"

cd /home/ec2-user/GenAI-Project/src/baseline/diffusers/examples/instruct_pix2pix
accelerate launch train_instruct_pix2pix.py \
  --pretrained_model_name_or_path=$MODEL_NAME \
  --train_data_dir=$DATA_DIR \
  --original_image_column=input_image \
  --edit_prompt_column=edit_prompt \
  --edited_image_column=edited_image \
  --resolution=512 \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --use_8bit_adam \
  --enable_xformers_memory_efficient_attention \
  --mixed_precision=fp16 \
  --learning_rate=1e-5 \
  --max_train_steps=10000 \
  --checkpointing_steps=1000 \
  --checkpoints_total_limit=3 \
  --output_dir=$OUTPUT_DIR \
  --report_to=wandb \
  --seed=42