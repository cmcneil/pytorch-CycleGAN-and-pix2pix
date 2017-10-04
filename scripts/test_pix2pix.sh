python test.py \
--dataroot /home/jlg/carson/Desktop/snmovie_retinproj_dataset \
--name mrf \
--model pix2pix \
--which_model_netG unet_128 \
--which_direction AtoB \
--align_data \
--input_nc 21 \
--batchSize 32 \
--output_nc 3 \
--warp_to_square \
--use_dropout \
--how_many 50
#--display_channel 0
# --load_dir current_best \
