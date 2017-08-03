python train.py \
--name mrf \
--model pix2pix \
--dataroot /home/jlg/carson/Desktop/retinproj_dataset \
--align_data \
--which_model_netG unet_128 \
--which_direction AtoB \
--lambda_A 1.0 \
--no_lsgan \
--use_dropout \
--input_nc 7 \
--output_nc 3 \
--batchSize 64 \
--print_freq 10 \
--ganloss 5.0 \
--orthoregloss 0.0 \
--laploss 0.0 \
--print_freq 1 \
--niter 1000 \
--lr 0.000015 \
--warp_to_square


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
