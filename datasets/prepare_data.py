from os import listdir
import numpy as np
import h5py
import cv2

def get_paths():
    ''' prepares a list of paths '''
    source = '../data/train/'
    paths  = listdir(source)
    paths  = [source + path for path in paths]
    return paths

def make_arrays(paths):
    ''' converts jpgs to numpy arrays '''
    new_width = 32
    ratio = 1
    new_height = ratio*new_width
    images = []
    for path in paths:
        img = cv2.imread(path)
        small = cv2.resize(img, (new_width, new_height))
        images.append(small)
    images = np.array(images)
    return images

def make_groups(arrays):
    ''' divide data into training and validation set '''
    n = arrays.shape[0]
    perm = np.arange(n)
    np.random.shuffle(perm)
    t_img = arrays[perm[:n/2]]
    v_img = arrays[perm[n/2:]]
    return t_img, v_img

def store(t_img, v_img):
    ''' store data in HDF5 file'''
    filepath = '../data/dogs_and_cats.storage'
    with h5py.File(filepath, 'w') as f:
        trg = f.create_group('training')
        val = f.create_group('validation')
        trg.create_dataset('images', data = t_img)
        val.create_dataset('images', data = v_img)
    print('File {} was created'.format(filepath))

def main():
    paths        = get_paths()[:10]
    arrays       = make_arrays(paths)
    t_img, v_img = make_groups(arrays)
    store(t_img, v_img)

if __name__ == '__main__':
    main()
