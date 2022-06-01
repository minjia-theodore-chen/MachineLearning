import numpy as np
import matplotlib.pyplot as plt
from sklearn import gaussian_process
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import WhiteKernel, RBF
from modAL.models import ActiveLearner
import os

os.system("cls")

# # generate data, use once, use saved data later.
# X: np.ndarray = np.random.choice(np.linspace(0, 20, 10000), size=200, replace=False).reshape(-1, 1)
# np.savetxt('ActiveLearning/X.csv', X, delimiter=',')
# y: np.ndarray = np.sin(X) + np.random.normal(scale=0.3, size=X.shape)
# np.savetxt("ActiveLearning/y.csv", y, delimiter=',')

# read from saved csv
X: np.ndarray = np.genfromtxt("ActiveLearning/X.csv", delimiter=',').reshape(-1, 1)
y: np.ndarray = np.genfromtxt("ActiveLearning/y.csv", delimiter=',')

# with plt.style.context('seaborn-white'):
#     plt.figure(figsize=(10,5))
#     plt.scatter(X,y,c='k', s=20)
#     plt.title('sin(x) + noise')
#     plt.show()

# # choose initial data, use once, use saved data later.
# n_initial: int = 5
# initial_idx: np.ndarray = np.random.choice(range(len(X)), size=n_initial, replace=False)
# X_training, y_training = X[initial_idx], y[initial_idx]
# print("X_training:", X_training,"; y_training", y_training)
# np.savetxt("ActiveLearning/X_training.cvs", X_training, delimiter=',')
# np.savetxt("ActiveLearning/y_training.cvs", y_training, delimiter=',')

# read from saved csv
X_training = np.genfromtxt("ActiveLearning/X_training.cvs", delimiter=',')
y_training = np.genfromtxt("ActiveLearning/y_training.cvs", delimiter=',')

# define kernel function
kernel = RBF(length_scale=1.0, length_scale_bounds=(1e-2, 1e+3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-10, 1e+1))

# define query_strategy


def GP_regression_std(regressor, X):
    _, std = regressor.predict(X, return_std=True)
    query_idx = np.argmax(std)
    return query_idx, X[query_idx]


# define activelearner
regressor = ActiveLearner(estimator=GaussianProcessRegressor(kernel=kernel), query_strategy=GP_regression_std, X_training=X_training.reshape(-1, 1), y_training=y_training.reshape(-1, 1))

# visualize precision of initial model
X_grid = np.linspace(0, 20, 1000)
y_pred, y_std = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
y_pred, y_std = y_pred.ravel(), y_std.ravel()
with plt.style.context("seaborn-white"):
    plt.figure(figsize=(10, 5))
    plt.plot(X_grid, y_pred)
    plt.fill_between(X_grid, y_pred-y_std, y_pred+y_std, alpha=0.2)
    plt.scatter(X, y, c='k', s=20)
    plt.scatter(regressor.X_training, regressor.y_training, marker='^', c='g')
    plt.title("initial prediction")
    plt.savefig("ActiveLearning/fig/initialmodel.pdf")

n_queries: int = 20
for idx in range(n_queries):
    query_idx, query_instance = regressor.query(X)
    regressor.teach(X[query_idx].reshape(1, -1), y[query_idx].reshape(1, -1))

y_pred_final, y_std_final = regressor.predict(X_grid.reshape(-1, 1), return_std=True)
y_pred_final, y_std_final = y_pred_final.ravel(), y_std_final.ravel()
with plt.style.context("seaborn-white"):
    plt.figure(figsize=(10, 5))
    plt.plot(X_grid, y_pred_final)
    plt.fill_between(X_grid, y_pred_final-y_std_final, y_pred_final+y_std_final, alpha=0.2)
    plt.scatter(X, y, c='k', s=20)
    plt.scatter(regressor.X_training, regressor.y_training, marker='^', c='g')
    plt.title("final prediction")
    plt.savefig("ActiveLearning/fig/finalmodel.pdf")
