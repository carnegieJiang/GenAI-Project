
# python3 ./eval.py \
#     --model-id diffusion \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/diffusion_outputs/hptune_test/frozen/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/diffusion_dit \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit

# python3 ./eval.py \
#     --model-id diffusion \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/diffusion_outputs/hptune_test/frozen_t5/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/diffusion_dit_t5 \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit --use_t5



# python3 ./eval.py \
#     --model-id flow \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/flow_outputs/hptune_test/frozen/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/flow_dit \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit

# python3 ./eval.py \
#     --model-id flow \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/flow_outputs/hptune_test/frozen_t5/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/flow_dit_t5 \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit --use_t5


# python3 ./eval.py \
#     --model-id decouple \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_no_pretrain/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/decouple_dit_no_pretrain \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit

# python3 ./eval.py \
#     --model-id decouple \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_content_pretrained/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/decouple_dit_content_pretrained \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit 

# python3 ./eval.py \
#     --model-id decouple \
#     --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
#     --model-dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_both_pretrained/final_model.pt \
#     --output-dir /home/chealisa/Desktop/genAI/results/decouple_dit_both_pretrained \
#     --resolution 512 \
#     --num-samples 16 \
#     --steps 30 \
#     --seed 42 \
#     --recon-guidance-scale 0.0 --use_dit

python3 ./eval.py \
    --model-id decouple \
    --metadata-path /home/chealisa/Desktop/genAI/stylebooth_subset/metadata.csv \
    --model-dir /home/chealisa/Desktop/genAI/model/decouple_outputs/hptune_test/frozen_content_improved_clip/final_model.pt \
    --output-dir /home/chealisa/Desktop/genAI/results/decouple_dit_content_improved_clip \
    --resolution 512 \
    --num-samples 16 \
    --steps 30 \
    --seed 42 \
    --recon-guidance-scale 0.0 --use_dit --style_strength 1.0 