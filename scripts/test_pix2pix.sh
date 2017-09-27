python test.py \
--dataroot /home/jlg/carson/Desktop/retinproj_dataset \
--name mrf \
--model pix2pix \
--which_model_netG unet_128 \
--which_direction AtoB \
--align_data \
--input_nc 7 \
--batchSize 64 \
--output_nc 3 \
--load_dir current_best \
--warp_to_square \
--use_dropout \
--how_many 1
#--display_channel 0
