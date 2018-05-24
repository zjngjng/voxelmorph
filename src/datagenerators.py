import os
import numpy as np
import nibabel as nib

def load_example_by_name(vol_name):
    X = nib.load(vol_name).get_data()
    return X

def example_gen_fixe_ref_patch(vol_names, vol_size, fid=None):
    """
    this function outputs a pair of patches (moving, reference)
    the reference part name is fixed
    the patch index is fixed 
    """
    ref_name = 'FTD096K7'
    ref_patch = 0
    dir_path = os.path.dirname(vol_names[0])
    part_names = [os.path.basename(file).split('_')[0] for file in vol_names]
    patch_indices = [(os.path.basename(file).split('_')[1]).split('.')[0] for file in vol_names]
    part_names = list(set(part_names) -set([ref_name]))
    patch_indices = list(set(patch_indices)-set([ref_patch]))
    print('part_name is ', part_names)
    while(True):
        name_idx = np.random.permutation(len(part_names))[0]
        patch_idx = np.random.permutation(len(patch_indices))[0]
        file1 = os.path.join(dir_path, part_names[name_idx] + '_' + str(ref_patch) + '.nii.gz')
        file2 = os.path.join(dir_path, ref_name + '_' + str(ref_patch) + '.nii.gz')
        print(file1)
        print(file2)
        if(fid is not None):
            fid.write(file1 + '\n')
            fid.write(file2 + '\n')
            fid.flush()
        X1 = nib.load(file1).get_data()
        X2 = nib.load(file2).get_data()
        X1, X2 = augment(X1, X2, vol_size)
        X1 = X1/(pow(2,16)-1.0)
        X2 = X2/(pow(2,16)-1.0)
        X1 = X1[np.newaxis, :,:,:, np.newaxis]
        X2 = X2[np.newaxis, :,:,:, np.newaxis]
        return_vals = (X1, X2, part_names[name_idx], ref_name)
        yield tuple(return_vals)


def example_gen_fixe_only_patch(vol_names, vol_size, fid=None):
    """
    this function outputs a pair of patches (moving, reference)
    the reference part name is varying
    the patch index is fixed 
    """
    ref_patch = 0
    dir_path = os.path.dirname(vol_names[0])
    part_names = [os.path.basename(file).split('_')[0] for file in vol_names]
    patch_indices = [(os.path.basename(file).split('_')[1]).split('.')[0] for file in vol_names]
    part_names = list(set(part_names))
    patch_indices = list(set(patch_indices)-set([ref_patch]))
    print('part_name is ', part_names)
    while(True):
        name_idx = np.random.permutation(len(part_names))[0:2]
        patch_idx = np.random.permutation(len(patch_indices))[0]
        file1 = os.path.join(dir_path, part_names[name_idx[0]] + '_' + str(ref_patch) + '.nii.gz')
        file2 = os.path.join(dir_path, part_names[name_idx[1]] + '_' + str(ref_patch) + '.nii.gz')
        print(file1)
        print(file2)
        if(fid is not None):
            fid.write(file1 + '\n')
            fid.write(file2 + '\n')
            fid.flush()
        X1 = nib.load(file1).get_data()
        X2 = nib.load(file2).get_data()
        X1, X2 = augment(X1, X2, vol_size)
        X1 = X1/(pow(2,16)-1.0)
        X2 = X2/(pow(2,16)-1.0)
        X1 = X1[np.newaxis, :,:,:, np.newaxis]
        X2 = X2[np.newaxis, :,:,:, np.newaxis]
        return_vals = (X1, X2, part_names[name_idx[0]], part_names[name_idx[1]])
        yield tuple(return_vals)

def example_gen(vol_names, vol_size, fid=None):
    """
    this function outputs a pair of patches (moving, reference)
    the reference part name is varying
    the patch index is varying 
    """
    dir_path = os.path.dirname(vol_names[0])
    part_names = [os.path.basename(file).split('_')[0] for file in vol_names]
    patch_indices = [(os.path.basename(file).split('_')[1]).split('.')[0] for file in vol_names]
    part_names = list(set(part_names))
    patch_indices = list(set(patch_indices))
    while(True):
        name_idx = np.random.permutation(len(part_names))[0:2]
        patch_idx = np.random.permutation(len(patch_indices))[0]
        file1 = os.path.join(dir_path, part_names[name_idx[0]] + '_' + str(patch_indices[patch_idx]) + '.nii.gz')
        file2 = os.path.join(dir_path, part_names[name_idx[1]] + '_' + str(patch_indices[patch_idx]) + '.nii.gz')
        if(fid is not None):
            fid.write(file1 + '\n')
            fid.write(file2 + '\n')
            fid.flush()
        X1 = nib.load(file1).get_data()
        X2 = nib.load(file2).get_data()
        X1, X2 = augment(X1, X2, vol_size)
        X1 = X1/(pow(2,16)-1.0)
        X2 = X2/(pow(2,16)-1.0)
        X1 = X1[np.newaxis, :,:,:, np.newaxis]
        X2 = X2[np.newaxis, :,:,:, np.newaxis]
        return_vals = (X1, X2, part_names[name_idx[0]], part_names[name_idx[1]])
        yield tuple(return_vals)

def example_gen_resized(vol_names, vol_size, fid=None):
    """
    this function outputs a pair of resized volumes (moving, reference)
    """
    dir_path = os.path.dirname(vol_names[0])
    part_names = [os.path.basename(file).split('.')[0] for file in vol_names]
    part_names = list(set(part_names))
    while(True):
        name_idx = np.random.permutation(len(part_names))[0:2]
        file1 = os.path.join(dir_path, part_names[name_idx[0]] + '.nii.gz')
        file2 = os.path.join(dir_path, part_names[name_idx[1]] + '.nii.gz')
        
        if(fid is not None):
            fid.write(file1 + '\n')
            fid.write(file2 + '\n')
            fid.flush()
        X1 = nib.load(file1).get_data()
        X2 = nib.load(file2).get_data()

        X1, X2 = pad(X1, X2, vol_size)
        X1 = X1/(pow(2,16)-1.0)
        X2 = X2/(pow(2,16)-1.0)
        X1 = X1[np.newaxis, :,:,:, np.newaxis]
        X2 = X2[np.newaxis, :,:,:, np.newaxis]
        return_vals = (X1, X2)
        yield tuple(return_vals)

def pad(image, truth, vol_size):
    [H, W, D] = image.shape #(115, 175, 148)
    [h, w, d] = vol_size # 128*256*256
    cx = int((h-H)/2)
    cy = int((w-W)/2)
    cz = int((d-D)/2)
    new_img = np.zeros(shape=vol_size, dtype=np.float32)
    new_truth = np.zeros(shape=vol_size, dtype=np.float32)
    new_img[cx:cx+H, cy:cy+W, cz:cz+D] = image
    new_truth[cx:cx+H, cy:cy+W, cz:cz+D] = truth
    return new_img, new_truth

def augment(image, truth, vol_size):
    [H, W, D] = image.shape
    [h, w, d] = vol_size
    cx = int(H/2)
    cy = int(W/2)
    cz = int(D/2)

    cx =  cx + np.random.randint(0, int((H-h)/2))
    cy =  cy + np.random.randint(0, int((W-w)/2))
    cz =  cz + np.random.randint(0, int((D-d)/2))
    aug_image = image[int(cx - h / 2):int(cx + h - h / 2), int(cy - w / 2):int(cy + w - w / 2), int(cz - d / 2):int(cz + d - d / 2)]
    aug_truth = truth[int(cx - h / 2):int(cx + h - h / 2), int(cy - w / 2):int(cy + w - w / 2), int(cz - d / 2):int(cz + d - d / 2)]
    return aug_image, aug_truth

'''
def example_gen(vol_names, return_segs=False, seg_dir=None):
    #idx = 0
    while(True):
        idx = np.random.randint(len(vol_names))
        X = np.load(vol_names[idx])['vol_data']
        X = np.reshape(X, (1,) + X.shape + (1,))

        return_vals = [X]

        if(return_segs):
            name = os.path.basename(vol_names[idx])
            X_seg = np.load(seg_dir + name[0:-8]+'aseg.npz')['vol_data']
            X_seg = np.reshape(X_seg, (1,) + X_seg.shape + (1,))
            return_vals.append(X_seg)

        # print vol_names[idx] + "," + seg_dir + name[0:-8]+'aseg.npz'

        yield tuple(return_vals)
'''
