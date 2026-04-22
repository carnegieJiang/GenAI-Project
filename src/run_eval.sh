# python3 ./eval.py \
#     --model-id baseline \
#     --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --model-dir /home/ec2-user/GenAI-Project/model/baseline/instruct-pix2pix-diffusers \
#     --output-dir /home/ec2-user/GenAI-Project/results/baseline \
#     --resolution 512 \
#     --num-samples 8 \
#     --steps 30 \
#     --seed 42 \
#     --guidance-scale 7.5 \
#     --recon-guidance-scale 0.0 

# python3 ./eval.py \
#     --model-id diffusion \
#     --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --model-dir /home/ec2-user/GenAI-Project/model/diffusion_outputs/hptune_test/heat/final_model.pt \
#     --output-dir /home/ec2-user/GenAI-Project/results/diffusion_heat \
#     --resolution 512 \
#     --num-samples 8 \
#     --steps 30 \
#     --seed 42 \
#     --guidance-scale 7.5 \
#     --recon-guidance-scale 0.0 

# python3 ./eval.py \
#     --model-id diffusion \
#     --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
#     --model-dir /home/ec2-user/GenAI-Project/model/diffusion_outputs/hptune_test/frozen/final_model.pt \
#     --output-dir /home/ec2-user/GenAI-Project/results/diffusion_frozen \
#     --resolution 512 \
#     --num-samples 8 \
#     --steps 30 \
#     --seed 42 \
#     --guidance-scale 7.5 \
#     --recon-guidance-scale 0.0 

python3 ./eval.py \
    --model-id flow \
    --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
    --model-dir /home/ec2-user/GenAI-Project/model/flow_outputs/hptune_test/heat/final_model.pt \
    --output-dir /home/ec2-user/GenAI-Project/results/flow_heat \
    --resolution 512 \
    --num-samples 8 \
    --steps 30 \
    --seed 42 \
    --guidance-scale 7.5 \
    --recon-guidance-scale 0.0 

python3 ./eval.py \
    --model-id flow \
    --metadata-path /home/ec2-user/GenAI-Project/data/stylebooth_subset/metadata.csv \
    --model-dir /home/ec2-user/GenAI-Project/model/flow_outputs/hptune_test/frozen/final_model.pt \
    --output-dir /home/ec2-user/GenAI-Project/results/flow_frozen \
    --resolution 512 \
    --num-samples 8 \
    --steps 30 \
    --seed 42 \
    --guidance-scale 7.5 \
    --recon-guidance-scale 0.0 