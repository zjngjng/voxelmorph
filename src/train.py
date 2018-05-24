"""
train atlas-based alignment with voxelmorph
"""

# python imports
import os
import glob
import sys
import random
from argparse import ArgumentParser
import pickle
# third-party imports
import tensorflow as tf
import numpy as np
import scipy.io as sio
from keras.backend.tensorflow_backend import set_session
from keras.optimizers import Adam
from keras.models import load_model, Model

# project imports
sys.path.append('../ext/medipy-lib')
import medipy
import datagenerators
import networks
import losses



vol_size = (128, 256, 256)
base_data_dir = '/media/alican/no_backup01/DeformReg/resized_vol'
train_vol_names = glob.glob(os.path.join(base_data_dir, '*.nii.gz'))
random.shuffle(train_vol_names)

def train(model,save_name, gpu_id, lr, n_iterations, reg_param, model_save_iter):

    model_dir = '../models/' + save_name
    if not os.path.isdir(model_dir):
        os.mkdir(model_dir)
    fid = open(os.path.join(model_dir, 'log.txt'), 'w')

    gpu = '/gpu:' + str(gpu_id)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    config.allow_soft_placement = True
    set_session(tf.Session(config=config))

    # UNET filters
    nf_enc = [16,32,32,32]
    if(model == 'vm1'):
        nf_dec = [32,32,32,32,8,8,3]
    else:
        nf_dec = [32,32,32,32,32,16,16,3]

    with tf.device(gpu):
        model = networks.unet(vol_size, nf_enc, nf_dec)
        model.compile(optimizer=Adam(lr=lr), loss=[
                      losses.cc3D(), losses.gradientLoss('l2')], loss_weights=[1.0, reg_param])
        # model.load_weights('../models/udrnet2/udrnet1_1/120000.h5')

    train_example_gen = datagenerators.example_gen_resized(train_vol_names, vol_size, fid)
    zero_flow = np.zeros((1, vol_size[0], vol_size[1], vol_size[2], 3))
    tot_loss = []
    for step in range(0, n_iterations):

        X, atlas_vol = train_example_gen.__next__()
        train_loss = model.train_on_batch(
            [X, atlas_vol], [atlas_vol, zero_flow])

        if not isinstance(train_loss, list):
            train_loss = [train_loss]
        tot_loss.append(train_loss)
        printLoss(step, 1, train_loss, fid)

        if(step % model_save_iter == 0):
            model.save(model_dir + '/' + str(step) + '.h5')
        if(step%200 ==0):
            #tot_loss = np.asarray(tot_loss, dtype=np.float32)
            with open(os.path.join(model_dir, 'losses.pickle'), 'wb') as handle:
                pickle.dump(tot_loss, handle)

    fid.close()

def printLoss(step, training, train_loss, fid):
    s = str(step) + "," + str(training)

    if(isinstance(train_loss, list) or isinstance(train_loss, np.ndarray)):
        for i in range(len(train_loss)):
            s += "," + str(train_loss[i])
    else:
        s += "," + str(train_loss)
    print(s)
    fid.write(s+ '\n')
    fid.flush()
    sys.stdout.flush()


if __name__ == "__main__":

    parser = ArgumentParser()
    parser.add_argument("--model", type=str,dest="model", 
                        choices=['vm1','vm2'],default='vm2',
                        help="Voxelmorph-1 or 2")
    parser.add_argument("--save_name", type=str, required=True,
                        dest="save_name", help="Name of model when saving")
    parser.add_argument("--gpu", type=int, default=0,
                        dest="gpu_id", help="gpu id number")
    parser.add_argument("--lr", type=float, 
                        dest="lr", default=1e-4,help="learning rate") 
    parser.add_argument("--iters", type=int, 
                        dest="n_iterations", default=150000,
                        help="number of iterations")
    parser.add_argument("--lambda", type=float, 
                        dest="reg_param", default=1.0,
                        help="regularization parameter")
    parser.add_argument("--checkpoint_iter", type=int,
                        dest="model_save_iter", default=5000, 
                        help="frequency of model saves")

    args = parser.parse_args()
    train(**vars(args))
