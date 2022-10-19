from data import real_dataset
from data import synthetic_dataset

from utils.parameters import args_parser
args = args_parser()

if __name__ == '__main__':
    if args.dataset_name == 'synthetic':
        train_size = 100
        x_raw, y_raw = synthetic_dataset.create_synthetic_dataset()
        x, y = x_raw[:train_size], y_raw[:train_size]
        X_test, y_test = x_raw[train_size:], y_raw[train_size:]

    elif args.dataset_name == 'MNIST':
        train_loader, test_loader = real_dataset.download_dataset(args.dataset_name)



