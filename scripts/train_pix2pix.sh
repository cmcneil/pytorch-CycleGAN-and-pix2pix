python train.py \
--name brain2pix \
--model pix2pix \
--dataroot /home/jlg/carson/Desktop/snmovie_retinproj_dataset_clips \
--align_data \
--which_model_netG unet_128 \
--which_model_netD n_layers \
--n_layers_D 3 \
--which_direction AtoB \
--lambda_A 1.0 \
--loadSize 128 \
--fineSize 64 \
--no_lsgan \
--use_dropout \
--input_nc 21 \
--output_nc 45 \
--batchSize 32 \
--print_freq 10 \
--ganloss 1.0 \
--print_freq 1 \
--niter 1000 \
--lr 0.001 \
--gpu_ids 0 \
--warp_to_square \
# --continue_train \
# --load_dir patch_halfway \
# --run_on_cpu
