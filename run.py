from data import real_dataset
from data import synthetic_dataset
from models.DShap import DShap

from utils.parameters import args_parser
args = args_parser()

if __name__ == '__main__':
    if args.dataset_name == 'synthetic':
        train_size = 100
        x_raw, y_raw = synthetic_dataset.create_synthetic_dataset()
        x, y = x_raw[:train_size], y_raw[:train_size]
        x_test, y_test = x_raw[train_size:], y_raw[train_size:]

    elif args.dataset_name == 'MNIST':
        train_loader, test_loader = real_dataset.download_dataset(args.dataset_name)

    model = 'logistic'
    problem = 'classification'
    num_test = 1000
    directory = './temp'
    dshap = DShap(x, y, x_test, y_test, num_test,
                  sources=None,
                  sample_weight=None,
                  model_family=model,
                  metric='accuracy',
                  overwrite=True,
                  directory=directory, seed=0)
    dshap.run(100, 0.1, g_run=False)



