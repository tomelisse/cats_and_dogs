import numpy as np
import glob
import h5py
import cv2

def get_paths():
    ''' prepares a list of paths '''
    source = 'data/train/*'
    paths  = glob.glob(source)
    # print(paths)
    # paths  = [source + path for path in paths]
    return paths

def make_arrays(paths):
    ''' converts jpgs to numpy arrays '''
    new_width = 32
    ratio = 1
    new_height = ratio*new_width
    images = []
    labels = []
    for path in paths:
        img = cv2.imread(path)
        small = cv2.resize(img, (new_width, new_height))
        images.append(small)
        if 'cat' in path:
            labels.append([1, 0])
        else:
            labels.append([0, 1])
    images = np.array(images)
    labels = np.array(labels)
    return images, labels

def make_groups(images, labels):
    ''' divide data into training and validation set '''
    n = images.shape[0]
    perm = np.arange(n)
    np.random.shuffle(perm)
    t_img = images[perm[:n/2]]
    t_lab = labels[perm[:n/2]]
    v_img = images[perm[n/2:]]
    v_lab = labels[perm[n/2:]]
    return t_img, t_lab, v_img, v_lab

def store(t_img, t_lab, v_img, v_lab):
    ''' store data in HDF5 file'''
    filepath = 'data/cats_and_dogs.storage'
    with h5py.File(filepath, 'w') as f:
        trg = f.create_group('training')
        val = f.create_group('validation')
        trg.create_dataset('images', data = t_img)
        trg.create_dataset('labels', data = t_lab)
        val.create_dataset('images', data = v_img)
        val.create_dataset('labels', data = v_lab)
    print('File {} was created'.format(filepath))

def make_data():
    paths  = get_paths()[:20]
    arrays = make_arrays(paths)
    groups = make_groups(*arrays)
    store(*groups)

if __name__ == '__main__':
    make_data()
