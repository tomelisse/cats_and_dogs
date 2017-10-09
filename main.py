from datasets import prepare_data as dp
from datasets import get_dataset as dg

def main():
    dp.make_data()
    filepath = 'data/cats_and_dogs.storage'
    dg.HdfDataset(filepath)

if __name__ == '__main__':
    main()
