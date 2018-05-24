# py imports
import os
import sys
import glob

# third party
import tensorflow as tf
import scipy.io as sio
import numpy as np
from keras.backend.tensorflow_backend import set_session
from scipy.interpolate import interpn

# project
sys.path.append('../ext/medipy-lib')
import medipy
import networks
from medipy.metrics import dice
import datagenerators

import nibabel as nib


def write_image_to_disk(image, affine, path):
    nib_image = nib.Nifti1Image(dataobj=image, affine=affine)
    nib.save(nib_image, path)

def test(model_name, gpu_id, iter_num, vol_size=(256, 256, 64), nf_enc= [16,32,32,32], nf_dec=[32,32,32,32,8,8,3]):
    """
	test

	nf_enc and nf_dec
	#nf_dec = [32,32,32,32,32,16,16,3]
    # This needs to be changed. Ideally, we could just call load_model, and we wont have to
    # specify the # of channels here, but the load_model is not working with the custom loss...
    """
    gpu = '/gpu:' + str(gpu_id)
    ref_name = 'FTD096K7'
    test_dir = '/media/alican/no_backup01/DeformReg/resized_vol'
    test_results = '/media/alican/no_backup01/DeformReg/test_results_resized'
    ref_file = os.path.join(test_dir, ref_name + '.nii.gz')
    ref_nib = nib.load(ref_file)
    ref_vol = ref_nib.get_data()
    test_files = glob.glob(os.path.join(test_dir, '*.nii.gz'))

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # load weights of model
    with tf.device(gpu):
        net = networks.unet(vol_size, nf_enc, nf_dec)
        net.load_weights('../models/' + model_name +
                         '/' + str(iter_num) + '.h5')

    n_batches = len(test_files)
    xx = np.arange(vol_size[1])
    yy = np.arange(vol_size[0])
    zz = np.arange(vol_size[2])
    grid = np.rollaxis(np.array(np.meshgrid(xx, yy, zz)), 0, 4)

    np.random.seed(17)

    for k in range(0, n_batches):
        vol_name =test_files[k]
        if(ref_name in vol_name):
            continue
        X_vol = datagenerators.load_example_by_name(vol_name)
        X1, X2 = datagenerators.pad(X_vol, ref_vol, vol_size)
        X1 = X1/(pow(2,16)-1.0)
        X2 = X2/(pow(2,16)-1.0)
        basename = os.path.basename(vol_name)
        write_image_to_disk(np.float32(X1), ref_nib.affine,
							os.path.join(test_results, basename.split('.')[0] + '_X1.nii.gz'))
        write_image_to_disk(np.float32(X2), ref_nib.affine,
							os.path.join(test_results, basename.split('.')[0] + '_X2.nii.gz'))
        X1 = X1[np.newaxis,:,:,:, np.newaxis]
        X2 = X2[np.newaxis,:,:,:, np.newaxis]
        with tf.device(gpu):
            pred = net.predict([X1, X2])

        # Warp segments with flow
        flow = pred[1][0, :, :, :, :]
        flow_mag = np.sqrt(np.sum(flow*flow, axis=3))
        flow_mag = np.squeeze(flow_mag)
        write_image_to_disk(np.float32(flow_mag), ref_nib.affine, os.path.join(test_results, basename.split('_')[0] + '_flow.nii.gz'))
        warped_X1 = pred[0][0,:, :, :, :]
        write_image_to_disk(np.float32(warped_X1), ref_nib.affine,
					os.path.join(test_results, basename.split('_')[0] + '_warpedX1.nii.gz'))


if __name__ == "__main__":
    # test(sys.argv[1], sys.argv[2], sys.argv[3])
    #model_name = 'exp_fixRefPatch_lambda1.0'
    #iter_num = 16400
    model_name = 'exp_resized_lambda1.0'
    iter_num = 25000
    test(model_name=model_name, gpu_id=0, iter_num=iter_num, vol_size=(128, 256, 256))