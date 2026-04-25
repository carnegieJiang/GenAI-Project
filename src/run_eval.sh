python3 ./eval.py \
    --model-id baseline \
    --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
    --model-dir /home/ec2-user/GenAI-Project/model/instructp2p \
    --output-dir /home/ec2-user/GenAI-Project/results/baseline \
    --resolution 512 \
    --num-samples 16 \
    --steps 30 \
    --seed 42 \
    --recon-guidance-scale 0.0 

python3 ./eval.py \
    --model-id diffusion \
    --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
    --model-dir /home/ec2-user/GenAI-Project/model/diffusion_outputs/hptune_test/frozen/final_model.pt \
    --output-dir /home/ec2-user/GenAI-Project/results/diffusion_unet \
    --resolution 512 \
    --num-samples 16 \
    --steps 30 \
    --seed 42 \
    --recon-guidance-scale 0.0 


python3 ./eval.py \
    --model-id flow \
    --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
    --model-dir /home/ec2-user/GenAI-Project/model/flow_outputs/hptune_test/frozen/final_model.pt \
    --output-dir /home/ec2-user/GenAI-Project/results/flow_unet \
    --resolution 512 \
    --num-samples 16 \
    --steps 30 \
    --seed 42 \
    --recon-guidance-scale 0.0 
