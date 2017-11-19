python train.py \
--name brain2pix \
--model pix2pix \
--dataroot /home/jlg/carson/Desktop/snmovie_retinproj_dataset_clips \
--align_data \
--which_model_netG unet_128 \
--which_model_netD n_layers \
--n_layers_D 1 \
--which_direction AtoB \
--lambda_A 5.0 \
--no_lsgan \
--use_dropout \
--input_nc 21 \
--output_nc 45 \
--batchSize 32 \
--print_freq 10 \
--ganloss 1.0 \
--print_freq 1 \
--niter 500 \
--lr 0.001 \
--warp_to_square
#--continue_train \
#--load_dir current_best \
