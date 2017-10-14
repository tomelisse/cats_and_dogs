from datasets import prepare_data as dp
from datasets import get_dataset as dg

def main():
    # dp.make_data()
    filepath = 'data/cats_and_dogs.storage'
    dataset = dg.HdfDataset(filepath)
    for _ in range(5):
        images, labels = dataset.next_batch(2)
        # print images.shape
        # print labels

if __name__ == '__main__':
    main()
