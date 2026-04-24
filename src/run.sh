# python3 ./train.py \
#     --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --output_dir /home/chealisa/Desktop/genAI/model/diffusion_outputs/hptune_test/frozen \
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
#     --seed 42 --freeze_text_encoder --freeze_vae --use_dit --model_type diffusion --run_name diffusion_dit_frozen

# python3 ./train.py \
#     --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --output_dir /home/chealisa/Desktop/genAI/model/diffusion_outputs/hptune_test/frozen_t5 \
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
#     --seed 42 --freeze_text_encoder --freeze_vae --use_dit --use_t5 --model_type diffusion --run_name diffusion_dit_frozen_t5

# python3 ./train.py \
#     --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --output_dir /home/chealisa/Desktop/genAI/model/flow_outputs/hptune_test/frozen \
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
#     --seed 42 --freeze_text_encoder --freeze_vae --use_dit --model_type flow --run_name flow_dit_frozen

# python3 ./train.py \
#     --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --output_dir /home/chealisa/Desktop/genAI/model/flow_outputs/hptune_test/frozen_t5 \
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
#     --seed 42 --freeze_text_encoder --freeze_vae --use_dit --use_t5 --model_type flow --run_name flow_dit_frozen_t5


python3 ./train.py \
    --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
    --output_dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen \
    --resolution 512 \
    --batch_size 4 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --num_epochs 10 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every_steps 10000 \
    --sample_every_steps 50 \
    --num_sample_images 2 \
    --mixed_precision no \
    --seed 42 --freeze_text_encoder --freeze_vae --use_dit --model_type decouple --run_name decouple_dit_frozen

python3 ./train.py \
    --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
    --output_dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_t5 \
    --resolution 512 \
    --batch_size 4 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --num_epochs 10 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every_steps 10000 \
    --sample_every_steps 50 \
    --num_sample_images 2 \
    --mixed_precision no \
    --seed 42 --freeze_text_encoder --freeze_vae --use_dit --use_t5 --model_type decouple --run_name decouple_dit_frozen_t5

python3 ./train.py \
    --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
    --output_dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_noise \
    --resolution 512 \
    --batch_size 4 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --num_epochs 10 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every_steps 10000 \
    --sample_every_steps 50 \
    --num_sample_images 2 \
    --mixed_precision no \
    --seed 42 --freeze_text_encoder --freeze_vae --use_dit --model_type decouple --run_name decouple_dit_frozen_add_noise --use_noise

python3 ./train.py \
    --data_path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
    --output_dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_advloss \
    --resolution 512 \
    --batch_size 4 \
    --lr 1e-5 \
    --weight_decay 1e-2 \
    --num_epochs 10 \
    --grad_accum_steps 1 \
    --max_grad_norm 1.0 \
    --save_every_steps 10000 \
    --sample_every_steps 50 \
    --num_sample_images 2 \
    --mixed_precision no \
    --seed 42 --freeze_text_encoder --freeze_vae --use_dit --model_type decouple --run_name decouple_dit_frozen_advanced_loss --use_advanced_loss

