Data Shapley: Equitable Valuation of Data for Machine Learning
=====================================

Reproduction work using Pytorch for implementation of  ["Data Shapley: Equitable Valuation of Data for Machine Learning"](https://arxiv.org/pdf/1904.02868.pdf).

**Please cite the following work if you use this benchmark or the provided tools or implementations:**

```
@inproceedings{ghorbani2019data,
  title={Data Shapley: Equitable Valuation of Data for Machine Learning},
  author={Ghorbani, Amirata and Zou, James},
  booktitle={International Conference on Machine Learning},
  pages={2242--2251},
  year={2019}
}
```

## Prerequisites

- Python, NumPy, Pytorch 1.11.0, Scikit-learn, Matplotlib

## Run
```angular2html
python run.py
```

## Basic Usage

To divide value fairly between individual train data points/sources given the learning algorithm and a meausre of performance for the trained model (test accuracy, etc)
