python test.py \
--dataroot /home/jlg/carson/Desktop/retinproj_dataset \
--name mrf \
--model pix2pix \
--which_model_netG unet_128 \
--which_direction AtoB \
--align_data \
--input_nc 7 \
--output_nc 3 \
--use_dropout \
--load_dir smooth_best
#--display_channel 0
