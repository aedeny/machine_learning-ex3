import os

import numpy as np
import pickle


class NeuralNetwork:
    def __init__(self, train_x, train_y, hidden_layer_size, train_validation_ratio):
        self.train_size = train_x.shape[0]

        # shuffle samples
        s = np.arange(self.train_size)
        np.random.shuffle(s)
        train_x = train_x[s]
        train_y = train_y[s]

        # Divides data
        self.train_validation_ratio = train_validation_ratio
        self.train_x = train_x[:int(self.train_size * self.train_validation_ratio), :]
        self.validation_x = train_x[int(self.train_size * self.train_validation_ratio):, :]
        self.train_y = train_y[:int(self.train_size * self.train_validation_ratio)]
        self.validation_y = train_y[int(self.train_size * self.train_validation_ratio):]

        self.w1 = np.random.uniform(-0.08, 0.08, (hidden_layer_size, 784))
        self.b1 = np.random.uniform(-0.08, 0.08, (hidden_layer_size, 1))
        self.w2 = np.random.uniform(-0.08, 0.08, (10, hidden_layer_size))
        self.b2 = np.random.uniform(-0.08, 0.08, (10, 1))

        self.params = {'w1': self.w1, 'b1': self.b1, 'w2': self.w2, 'b2': self.b2}

    def train(self, num_of_epochs, learning_rate, should_print=True):
        for epoch in range(num_of_epochs):

            # Shuffles arrays
            s = np.arange(self.train_x.shape[0])
            np.random.shuffle(s)
            self.train_x = self.train_x[s]
            self.train_y = self.train_y[s]

            loss_sum = 0
            validation_loss = 0
            validation_success = 0

            # Trains model
            for x, y in zip(self.train_x, self.train_y):
                y = int(y)

                # Normalizes vector
                x = np.ndarray.astype(x, dtype=float)
                x /= 255.0
                x = np.expand_dims(x, axis=1)

                # Forward propagation
                back_params = {}
                y_hat = self.forward_propagation(x, back_params)
                h1, z1 = back_params['h1'], back_params['z1']

                # Computes loss_sum
                loss_sum -= np.log(y_hat[y])

                # Backward propagation
                backward_params = {'y_hat': y_hat, 'h1': h1, 'w2': self.params['w2'], 'z1': z1}
                dw1, db1, dw2, db2 = self.backward_propagation(x, y, backward_params)

                # update
                derivatives = {'dw1': dw1, 'db1': db1, 'dw2': dw2, 'db2': db2}
                self._update(derivatives, learning_rate)

            # Validates
            for x, y in zip(self.validation_x, self.validation_y):
                y = int(y)

                # Normalizes vector
                x = np.ndarray.astype(x, dtype=float)
                x /= 255.0
                x = np.expand_dims(x, axis=1)

                # Applies forward propagation
                y_hat = self.forward_propagation(x)

                # Computes success
                if y == np.argmax(y_hat):
                    validation_success += 1

                # Computes loss_sum
                validation_loss -= np.log(y_hat[y])

            avg_loss = loss_sum / (self.train_size * self.train_validation_ratio)
            avg_validation_loss = validation_loss / (self.train_size * (1 - self.train_validation_ratio))
            validation_accuracy = validation_success / (self.train_size * (1 - self.train_validation_ratio))

            if should_print:
                print('Epoch #' + str(epoch) + ', Validation accuracy: ' + str(
                    validation_accuracy) + ', Loss sum: ' + str(avg_loss[0]) + ', Validation loss sum: ' + str(
                    avg_validation_loss[0]))

    def _update(self, derivatives, learning_rate):
        w1, b1, w2, b2 = [self.params[key] for key in ('w1', 'b1', 'w2', 'b2')]
        dw1, db1, dw2, db2 = [derivatives[key] for key in ('dw1', 'db1', 'dw2', 'db2')]

        w2 -= learning_rate * dw2
        b2 -= learning_rate * db2
        w1 -= learning_rate * dw1
        b1 -= learning_rate * db1

        self.params = {'w1': w1, 'b1': b1, 'w2': w2, 'b2': b2}

    def predict(self, sample):
        sample = np.asmatrix(sample).T
        return np.argmax(self.forward_propagation(sample))

    def forward_propagation(self, sample, back_params=None):
        # Follows procedure given in notes
        if back_params is None:
            back_params = {}
        w1, b1, w2, b2 = [self.params[key] for key in ('w1', 'b1', 'w2', 'b2')]
        activation_func = active['func']

        z1 = np.dot(w1, sample) + b1
        h1 = activation_func(z1)
        z2 = np.dot(w2, h1) + b2
        y_hat = softmax(z2)

        back_params['z1'], back_params['h1'], back_params['z2'], back_params['h2'] = z1, h1, z2, y_hat
        return y_hat

    @staticmethod
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


def softmax(x):
    e_z = np.exp(x - np.max(x))
    return e_z / e_z.sum()


def leaky_relu(x):
    for i in range(len(x)):
        x[i] = max(0.01 * x[i], x[i])
    return x


def d_leaky_relu(x):
    for i in range(len(x)):
        x[i] = max(0.01 * np.sign(x[i]), np.sign(x[i]))
    return x


# activation functions
activation_funcs = {'leakyReLU': {'func': leaky_relu, 'derivative': d_leaky_relu}}
active = activation_funcs['leakyReLU']


def read_resources(train_x, train_y, test):
    x, y, t = None, None, None
    if os.path.exists(train_x + ".npy"):
        x = np.load(train_x + ".npy")
    else:
        x = np.loadtxt(train_x)
        np.save(train_x, x)

    if os.path.exists(train_y + ".npy"):
        y = np.load(train_y + ".npy")
    else:
        y = np.loadtxt(train_y)
        np.save(train_y, y)

    if os.path.exists(test + ".npy"):
        t = np.load(test + ".npy")
    else:
        t = np.loadtxt(test)
        np.save(test, t)

    return x, y, t


if __name__ == '__main__':
    file_path = "my_neural_network.pickle"
    # Reads data
    x, y, test = read_resources("train_x", "train_y", "test_x")

    nn = None
    if os.path.exists(file_path):
        with open(file_path, 'rb') as f:
            nn = pickle.load(f)
    else:
        nn = NeuralNetwork(x, y, 100, 0.8)
        nn.train(10, 0.001)
        with open(file_path, 'wb') as f:
            pickle.dump(nn, f)

    # Writes predictions of given tests
    with open("test.pred", "w") as f:
        for t in test:
            f.write(str(nn.predict(t)) + '\n')
