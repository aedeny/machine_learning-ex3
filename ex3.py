import numpy as np


# Example back propagation code for binary classification with 3-layer neural network (single hidden layer)

class NeuralNetwork:
    def __init__(self, train_x, train_y, hidden_layer_size, train_validation_ratio):
        # Divides data
        self.train_validation_ratio = train_validation_ratio
        self.train_x = train_x[:int(train_size * self.train_validation_ratio), :]
        self.vali_x = train_x[int(train_size * self.train_validation_ratio):, :]
        self.train_y = train_y[:int(train_size * self.train_validation_ratio)]
        self.vali_y = train_y[int(train_size * self.train_validation_ratio):]

        self.w1 = np.random.uniform(-0.08, 0.08, (hidden_size, 784))
        self.b1 = np.random.uniform(-0.08, 0.08, (hidden_size, 1))
        self.w2 = np.random.uniform(-0.08, 0.08, (10, hidden_size))
        self.b2 = np.random.uniform(-0.08, 0.08, (10, 1))

    def train(self, num_of_epochs, learning_rate):
        for epoch in range(num_of_epochs):

            # Shuffles arrays
            s = np.arange(self.train_x.shape[0])
            np.random.shuffle(s)
            self.train_x = self.train_x[s]
            self.train_y = self.train_y[s]

            loss = 0
            validation_loss = 0
            validation_scc = 0

            # Trains model
            for x, y in zip(self.train_x, self.train_y):
                y = int(y)
                # Normalize
                x = np.ndarray.astype(x, float)
                x /= 255
                x = np.expand_dims(x, 1)

                # Forward propagation
                back_params = {}
                y_hat = forward_propagation(x, params, back_params)
                h1, z1 = back_params['h1'], back_params['z1']

                # Computes loss
                loss -= np.log(y_hat[y])

                # Backward propagation
                params2 = {'y_hat': y_hat, 'h1': h1, 'w2': params['w2'], 'z1': z1}
                dw1, db1, dw2, db2 = backward_propagation(x, y, params2)

                # update
                derivatives = {'dW1': dw1, 'db1': db1, 'dW2': dw2, 'db2': db2}
                params = update(params, derivatives, learning_rate)

            # Validates
            for x, y in zip(self.vali_x, self.vali_y):
                y = int(y)

                # Normalizes the vector
                x = np.ndarray.astype(x, dtype=float)
                x /= 255
                x = np.expand_dims(x, axis=1)

                # Applies forward propagation
                y_hat = forward_propagation(x, params)

                # Computes success
                if y == np.argmax(y_hat):
                    validation_scc += 1

                # Computes loss
                validation_loss -= np.log(y_hat[y])

            avg_loss = loss / (train_size * self.train_validation_ratio)
            avg_vali_loss = validation_loss / (train_size * (1 - self.train_validation_ratio))
            vali_acc = validation_scc / (train_size * (1 - self.train_validation_ratio))

            print ('#', epoch, ' acc ', vali_acc, ' loss ', avg_loss[0], ' vali loss ', avg_vali_loss[0])


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(z):
    e_z = np.exp(z - np.max(z))
    return e_z / e_z.sum()


def leakyrelu(x):
    """
    leakyReLU function
    :param x: vector to activate function on
    :return: the vector after activation of function
    """
    for i in range(len(x)):
        x[i] = max(0.01 * x[i], x[i])
    return x


def d_leaky_relu(x):
    """
    leakyReLU function derivative
    :param x: vector to activate function derivative on
    :return: the vector after activation of function derivative
    """
    for i in range(len(x)):
        x[i] = max(0.01 * np.sign(x[i]), np.sign(x[i]))
    return x


def forward_propagation(x, model, params={}):
    # Follows procedure given in notes
    w1, b1, w2, b2 = [model[key] for key in ('w1', 'b1', 'w2', 'b2')]
    activation_func = active['func']

    z1 = np.dot(w1, x) + b1
    h1 = activation_func(z1)
    z2 = np.dot(w2, h1) + b2
    y_hat = softmax(z2)

    params['z1'], params['h1'], params['z2'], params['h2'] = z1, h1, z2, y_hat
    return y_hat


def backward_propagation(x, y, params):
    y_hat, h1, w2, z1 = [params[key] for key in ('y_hat', 'h1', 'w2', 'z1')]
    active_derivative = active['derivative']

    dz2 = y_hat
    dz2[y] -= 1
    dw2 = np.dot(dz2, h1.T)
    db2 = dz2

    dz1 = np.dot(y_hat.T, w2).T * active_derivative(z1)
    dw1 = np.dot(dz1, x.T)
    db1 = dz1

    return dw1, db1, dw2, db2


def update(model, derivatives, lr):
    """
    Update network parameters by the computed back propagation derivatives
    :param model: network parameters
    :param derivatives: derivatives computed in back propagation phase
    :param lr: learning rate
    :return: updated model
    """
    w1, b1, w2, b2 = [model[key] for key in ('w1', 'b1', 'w2', 'b2')]
    dw1, db1, dw2, db2 = [derivatives[key] for key in ('dW1', 'db1', 'dW2', 'db2')]

    w2 -= lr * dw2
    b2 -= lr * db2
    w1 -= lr * dw1
    b1 -= lr * db1

    model = {'W1': w1, 'b1': b1, 'W2': w2, 'b2': b2}
    return model


def loss_func(y_hat, y):
    return np.sum(y * np.log(y_hat))


# activation functions
activation_funcs = {'leakyReLU': {'func': leakyrelu, 'derivative': d_leaky_relu}}
active = activation_funcs['leakyReLU']

if __name__ == '__main__':
    lr = 0.001
    hidden_size = 100
    num_of_epochs = 10

    # Reads data
    train_x = np.load("train_x.bin.npy")
    train_y = np.load("train_y.bin.npy")
    test_x = np.load("test_x.bin.npy")

    train_size = train_x.shape[0]

    # Initializes random parameters and inputs
    w1 = np.random.uniform(-0.08, 0.08, (hidden_size, 784))
    b1 = np.random.uniform(-0.08, 0.08, (hidden_size, 1))
    w2 = np.random.uniform(-0.08, 0.08, (10, hidden_size))
    b2 = np.random.uniform(-0.08, 0.08, (10, 1))
    params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    # divide data - 80% train, 20% validation
    train_x, vali_x = train_x[:int(train_size * 0.8), :], train_x[int(train_size * 0.8):, :]
    train_y, vali_y = train_y[:int(train_size * 0.8)], train_y[int(train_size * 0.8):]

    for epoch in range(num_of_epochs):

        # Shuffles arrays
        s = np.arange(train_x.shape[0])
        np.random.shuffle(s)
        train_x = train_x[s]
        train_y = train_y[s]

        loss = 0
        validation_loss = 0
        validation_scc = 0

        # trains model
        for x, y in zip(train_x, train_y):
            y = int(y)
            # Normalize
            x = np.ndarray.astype(x, float)
            x /= 255
            x = np.expand_dims(x, 1)

            # Forward propagation
            back_params = {}
            y_hat = forward_propagation(x, params, back_params)
            h1, z1 = back_params['h1'], back_params['z1']

            # Computes loss
            loss -= np.log(y_hat[y])

            # Backward propagation
            params2 = {'y_hat': y_hat, 'h1': h1, 'w2': params['w2'], 'z1': z1}
            dw1, db1, dw2, db2 = backward_propagation(x, y, params2)

            # update
            deri = {'dW1': dw1, 'db1': db1, 'dW2': dw2, 'db2': db2}
            params = update(params, deri, lr)

        # validation
        for x, y in zip(vali_x, vali_y):
            y = int(y)

            # normalize the vector
            x = np.ndarray.astype(x, dtype=float)
            x /= 255
            x = np.expand_dims(x, axis=1)
            # forward
            y_hat = forward_propagation(x, params)

            # compute success
            if y == np.argmax(y_hat):
                validation_scc += 1

            # compute loss
            validation_loss -= np.log(y_hat[y])

        avg_loss = loss / (train_size * 0.8)
        avg_vali_loss = validation_loss / (train_size * 0.2)
        vali_acc = validation_scc / (train_size * 0.2)

        print ('#', epoch, ' acc ', vali_acc, ' loss ', avg_loss[0], ' vali loss ', avg_vali_loss[0])
