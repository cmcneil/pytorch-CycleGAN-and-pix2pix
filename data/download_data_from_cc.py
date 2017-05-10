import cottoncandy as cc
import os
import numpy as np

# Data paths
bucket_name = "carson_public_datasets"
bucket_path = "/human_connectome_project/3T/{subject}/mrf_training/{slice}"
# Subject/slices
subslices = np.load("/auto/k1/carson/glab/mrf/train_data_list.npy")
# Output paths
outpath = "/datasets/christine/mrf/{subject}/{slice}"

cci = cc.get_interface(bucket_name)
icount = 0

print("Total # of slices: %d"%len(subslices))

for isubject, islice in subslices:
    fpath = bucket_path.format(subject=isubject, slice=islice)
    opath = outpath.format(subject=isubject, slice=islice)
    if not os.path.exists(opath):
        os.makedirs(opath)

    mrf = cci.download_npy_array(os.path.join(fpath, "mrf.npy"))
    t1_t2_pd = cci.download_npy_array(os.path.join(fpath, "t1_t2_pd.npy"))
    np.save(os.path.join(opath, "mrf.npy"), mrf)
    np.save(os.path.join(opath, "t1_t2_pd.npy"), t1_t2_pd)

    icount += 1
    if icount % 100 == 0:
        print("Finished %d slices"%icount)
