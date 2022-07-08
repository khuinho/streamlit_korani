python3 camera.py --model_path /media/data1/checkpoints/pfld/220704/snapshot/checkpoint_epoch_259.pth.tar \
    --input_path ./facial_expression_example.mp4 \
    --output_path ./results/facial_expression_example_result.avi

python3 camera.py --model_path checkpoint/checkpoint_pfld.pth.tar \
    --input_path ./facial_expression_example.mp4 \
    --output_path ./results/facial_expression_example_result_original_weight.avi