python train.py \
--name mrf \
--model pix2pix \
--dataroot /home/jlg/carson/Desktop/mrf_dataset \
--align_data \
--which_model_netG unet_128 \
--which_direction AtoB \
--lambda_A 20 \
--no_lsgan \
--use_dropout \
--input_nc 100 \
--output_nc 3 \
--batchSize 16 \
--print_freq 10 \
--ganloss 5.0 \
--orthoregloss 0.0001 \
--laploss 1000.0 \
--print_freq 1
# --lr 0.0000000002

# --orthoregloss 10.0 \
# --lr 0.00002
#--orthoregloss 0.01

# Remember that I removed dropout from initial compression

# Laploss - pretty good (also dropout in network)
# python train.py \
# --name mrf \
# --model pix2pix \
# --dataroot /home/jlg/carson/Desktop/mrf_dataset \
# --align_data \
# --which_model_netG unet_128 \
# --which_direction AtoB \
# --lambda_A 20 \
# --no_lsgan \
# --use_dropout \
# --laploss 0.001 \
# --input_nc 498 \
# --output_nc 3 \
# --batchSize 16 \
# --print_freq 10 \
# --ganloss 1.0 \
# --orthoregloss 0.0
