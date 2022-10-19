import argparse

def args_parser():
    parser = argparse.ArgumentParser()

    # Training settings
    parser.add_argument('--batchsize', type=int, default=64, help='Dataset batch size')

    parser.add_argument('--model', type=str, default='logistic', help="the type of model's net")
    parser.add_argument('--metric', type=str, default='accuracy', help='the performance score metric')
    parser.add_argument('--dataset_name', type=str, default='synthetic', help='the type of the dataset')
    parser.add_argument('--strategy', type=str, default='LOO', help='the strategies of data valuation')

    parser.add_argument('--problem', type=str, default='classification', help='the type of problem')

    args = parser.parse_args()
    return args