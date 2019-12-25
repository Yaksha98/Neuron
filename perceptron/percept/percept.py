import inline as inline
import numpy as np
import matplotlib.pyplot as plt


X = np.array([
    [3,4,-1],
    [2,1,-1],
    [2, 21, -1],
    [-2, 5, -1],
    [1, 5, -1],

])

y = np.array([-1,-1,1,-1,1])


def perceptron_sgd_plot(X, Y):
    '''
    train perceptron and plot the total loss in each epoch.

    :param X: data samples
    :param Y: data labels
    :return: weight vector as a numpy array
    '''
    w = np.zeros(len(X[0]))
    eta = 1
    n = 30
    errors = []

    for t in range(n):
        total_error = 0
        for i, x in enumerate(X):
            if (np.dot(X[i], w) * Y[i]) <= 0:
                total_error += (np.dot(X[i], w) * Y[i])
                w = w + eta * X[i] * Y[i]
        errors.append(total_error * -1)

    plt.plot(errors)
    plt.xlabel('Epoch')
    plt.ylabel('Total Loss')
    yr = np.dot(X, w)
    for i, x in enumerate(yr):
        if yr[i] > 0:
            yr[i] = 1
        else:
            yr[i] = -1
    print(yr)

    return w
print(perceptron_sgd_plot(X, y))
plt.show()
