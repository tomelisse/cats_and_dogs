import h5py

class HdfDataset(object):
    ''' HDF5 dataset '''
    def __init__(self, filepath):
        self.filepath = filepath
        self.where_in_epoch = 0
        with h5py.File(filepath) as f:
            self.n_examples = f['training/images'].shape[0]
            self.n_classes  = f['training/labels'].shape[1]
            self.width    = f['training/images'].shape[1]
            self.height   = f['training/images'].shape[2]
            self.depth    = f['training/images'].shape[3]
        print 'Accessing file {} with {} examples and {} classes.'\
                .format(filepath, self.n_examples, self.n_classes) 

    def next_batch(self, batch_size):
        ''' return next batch of examples '''
        start = self.where_in_epoch
        self.where_in_epoch += batch_size
        if self.where_in_epoch > self.n_examples:
            start = 0
            self.where_in_epoch = batch_size
        stop = self.where_in_epoch
        with h5py.File(self.filepath) as f:
            images = f['training/images'][start:stop]
            labels = f['training/labels'][start:stop]
        return images, labels


