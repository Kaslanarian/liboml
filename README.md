# LIBOML:Library For Online Machine Learning

Here we implement representative online learning algorithms. We split these algorithms into 3 parts:

- Linear classifiers;
- Kernel classifiers;
- Deep classifiers.

The models we have implement:

|         Name          | Binary/Multiclass |                         Description                          |
| :-------------------: | :---------------: | :----------------------------------------------------------: |
| `OGDBinaryClassifier` |      binary       |    Online gradient descent algorithm with logistic loss.     |
|    `OGDClassifier`    |    multiclass     |  Online gradient descent algorithm with cross-entropy loss.  |
|    `PAClassifier`     |      binary       | Passive-aggresive algorithm with 3 types of step-size calculation. |
|     `Perceptron`      |      binary       |                    Perceptron algorithm.                     |
|      `KernelOGD`      |      binary       |  Online gradient descent with kernels using logistic loss.   |
|   `KernelPercetron`   |      binary       |                   Perceptron with kernels.                   |
|      `KernelSVM`      |      binary       |                   Online SVM with kernels.                   |
|    `MLPClassifier`    |    multiclass     |    Online multi-layer percetron with cross-entropy loss.     |
|    `ODLClassifier`    |    multiclass     |               Online deep learning framework.                |
|   `ODLAEClassifier`   |    multiclass     |           Online deep learning with Auto-Encoder.            |

## Install

```bash
pip install liboml
```

or

```bash
git clone https://github.com/Kaslanarian/liboml
cd liboml
python setup.py install
```

## How to use

Our implementation is based on sklearn, so you can easily use it just like this:

```python
from sklearn.dataset import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from liboml.linear import OGDClassifier

X, y = load_breast_cancer(return_X_y=True)
train_X, test_X, train_y, test_y = train_test_split(
    X, 
    y, 
    train_size=0.8,
	random_state=42,
)
stder = StandardScaler().fit(train_X)
train_X, test_X = stder.transform(train_X), stder.transform(test_X)

model = OGDClassifier(init_lr=0.1)
model.fit(train_X, train_y)
acc = model.score(test_X, test_y)
print(acc) # 0.9649122807017544
```

## Future

- More algorithms.
- Support GPU.

