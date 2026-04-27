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