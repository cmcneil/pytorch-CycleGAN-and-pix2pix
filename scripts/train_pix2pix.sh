python train.py --bucket_A carson_public_datasets --bucket_B carson_public_datasets --bucket_path_A /human_connectome_project/3T/{subject}/simulations/{slice}/{fname} --bucket_path_B /human_connectome_project/3T/{subject}/{fname} --fnames_A /home/jlg/christine/mrf_project/sim_fnames.npy --fnames_B /home/jlg/christine/mrf_project/quant_fnames.npy --sid /home/jlg/christine/mrf_project/sid.npy --slice_id /home/jlg/christine/mrf_project/slice_id.npy  --name mrf --model pix2pix --which_model_netG unet_256 --which_direction AtoB --lambda_A 100 --align_data --no_lsgan --use_dropout --serial_batches True
