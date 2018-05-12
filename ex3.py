
import numpy as np


# Example back propagation code for binary classification with 2-layer
# neural network (single hidden layer)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()


def forward_propagation(x, y, params):
    # Follows procedure given in notes
    W1, b1, W2, b2 = [params[key] for key in ('W1', 'b1', 'W2', 'b2')]
    z1 = np.dot(W1, x.T) + b1
    h1 = softmax(z1)
    z2 = np.dot(W2, h1) + b2
    h2 = softmax(z2)
    loss = -(y * np.log(h2) + (1 - y) * np.log(1 - h2))
    ret = {'x': x, 'y': y, 'z1': z1, 'h1': h1, 'z2': z2, 'h2': h2, 'loss': loss}
    for key in params:
        ret[key] = params[key]
    return ret


def backward_propagation(x, y, params, active_function):
    # Follows procedure given in notes
    x, y, z1, h1, z2, h2, loss = [params[key] for key in ('x', 'y', 'z1', 'h1', 'z2', 'h2', 'loss')]
    d_z2 = (h2 - y)  # dL/d_z2
    d_w2 = np.dot(d_z2, h1.T)  # dL/d_z2 * d_z2/dw2
    db2 = d_z2  # dL/d_z2 * d_z2/db2
    dz1 = np.dot(params['W2'].T, (h2 - y)) * sigmoid(z1) * (1 - sigmoid(z1))  # dL/d_z2 * d_z2/dh1 * dh1/dz1
    d_w1 = np.dot(dz1, x.T)  # dL/d_z2 * d_z2/dh1 * dh1/dz1 * dz1/dw1
    db1 = dz1  # dL/d_z2 * d_z2/dh1 * dh1/dz1 * dz1/db1
    return {'b1': db1, 'W1': d_w1, 'b2': db2, 'W2': d_w2}


def loss_func(y_hat, y):
    return np.sum(y * np.log(y_hat))


def train(params, epochs, active_func, lr, train_x, train_y, dev_x, dev_y):
    for i in range(epochs):
        sum_loss = 0.0

        train_size = train_x.shape[0]
        # Shuffles arrays
        s = np.arange(train_size)
        np.random.shuffle(s)
        train_x = train_x[s]
        train_y = train_y[s]

        for x, y in zip(train_x, train_y):
            # get probabilities vector as result, where index y is the probability that x is classified as tag y
            out = forward_propagation(train_x, train_y, params)
            loss = loss_func(out, y)

            sum_loss += loss
            gradients = backward_propagation(train_x, train_y, params, active_func)


if __name__ == '__main__':

    # train_x = np.loadtxt("train_x")
    # train_y = np.loadtxt("train_y")
    # test_x = np.loadtxt("test_x")
    train_x = np.load("train_x.bin.npy")
    train_y = np.load("train_y.bin.npy")
    test_x = np.load("test_x.bin.npy")

    train_size = train_x.shape[0]

    dev_size = int(train_size * 0.2)
    dev_x, dev_y = train_x[-dev_size:, :], train_y[-dev_size:]
    train_x, train_y = train_x[:-dev_size, :], train_y[:-dev_size]

    # Initializes random parameters and inputs
    hidden_size = 100
    W1 = np.random.uniform(-0.08, 0.08, (hidden_size, 784))
    b1 = np.random.uniform(-0.08, 0.08, (hidden_size, train_size - dev_size))
    W2 = np.random.uniform(-0.08, 0.08, (10, hidden_size))
    b2 = np.random.uniform(-0.08, 0.08, (10, train_size - dev_size))
    params = {'W1': W1, 'b1': b1, 'W2': W2, 'b2': b2}

    lr = 0.1
    train(params, 10, None, lr, train_x, train_y, dev_x, dev_y)

    # Numerical gradient checking
    # Note how slow this is! Thus we want to use the back propagation algorithm instead.
    eps = 1e-6
    ng_cache = {}

    # For every single parameter (W, b)
    for key in params:
        param = params[key]

    # Compare numerical gradients to those computed using back propagation algorithm
    for key in params:
        print(key)

        # These should be the same
        print(ng_cache[key])
