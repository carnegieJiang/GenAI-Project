python3 ./train.py \
    --data_path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
    --output_dir /home/ec2-user/GenAI-Project/model/diffusion_dit_outputs/hptune_test/frozen \
    --resolution 512 \
    --batch_size 4 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --num_epochs 20 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every_steps 10000 \
    --sample_every_steps 50 \
    --num_sample_images 2 \
    --mixed_precision no \
    --seed 42 --freeze_text_encoder --freeze_vae --use_dit

# python3 ./train.py \
#     --data_path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --output_dir /home/ec2-user/GenAI-Project/model/diffusion_dit_outputs/hptune_test/frozen \
#     --resolution 512 \
#     --batch_size 4 \
#     --lr 1e-5 \
#     --weight_decay 1e-2 \
#     --num_epochs 10 \
#     --grad_accum_steps 1 \
#     --max_grad_norm 1.0 \
#     --save_every_steps 10000 \
#     --sample_every_steps 50 \
#     --num_sample_images 2 \
#     --mixed_precision no \
#     --seed 42 --freeze_text_encoder --freeze_vae --use_dit

# python3 ./train.py \
#     --data_path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --output_dir /home/ec2-user/GenAI-Project/model/diffusion_outputs/subset/heat \
#     --resolution 256 \
#     --batch_size 1 \
#     --lr 1e-5 \
#     --weight_decay 1e-2 \
#     --num_epochs 10 \
#     --grad_accum_steps 4 \
#     --max_grad_norm 1.0 \
#     --save_every_steps 10000 \
#     --sample_every_steps 50 \
#     --num_sample_images 2 \
#     --mixed_precision no \
#     --seed 42 




# python3 ./train.py \
#     --data_path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --output_dir /home/ec2-user/GenAI-Project/model/flow_outputs/subset/frozen \
#     --resolution 512 \
#     --batch_size 4 \
#     --lr 1e-5 \
#     --weight_decay 1e-2 \
#     --num_epochs 10 \
#     --grad_accum_steps 1 \
#     --max_grad_norm 1.0 \
#     --save_every_steps 10000 \
#     --sample_every_steps 50 \
#     --num_sample_images 2 \
#     --mixed_precision no \
#     --seed 42 --freeze_text_encoder --freeze_vae --model_type flow   


# python3 ./train.py \
#     --data_path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --output_dir /home/ec2-user/GenAI-Project/model/flow_outputs/subset/heat \
#     --resolution 256 \
#     --batch_size 1 \
#     --lr 1e-5 \
#     --weight_decay 1e-2 \
#     --num_epochs 10 \
#     --grad_accum_steps 4 \
#     --max_grad_norm 1.0 \
#     --save_every_steps 10000 \
#     --sample_every_steps 50 \
#     --num_sample_images 2 \
#     --mixed_precision no \
#     --seed 42 --model_type flow   