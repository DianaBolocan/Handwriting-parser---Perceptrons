import numpy as np, _pickle as cPickle, gzip


class DNN:
    class __InputLayer:
        def __init__(self, input_size: int, output_size: int):
            self.weights = np.random.random((input_size, output_size))

        def total_net(self, data: np.array):
            return self.weights.dot(data)

    class __HiddenLayer:
        def __init__(self, input_size: int, output_size: int):
            self.weights = np.random.random((input_size, output_size))
            self.biases = np.random.random((input_size, 1))

        def total_net(self, data: np.array):
            return self.weights.dot(data) + self.biases

        def sigmoid(self, x: float):
            return 1 / (1 + np.exp(x))

    class __OutputLayer:
        def __init__(self, input_size: int):
            self.biases = np.random.random((input_size, 1))

        def softmax(self, x: float):
            return np.exp(x) / np.sum(np.exp(x))

        def cross_entropy(self, results: np.array, labels: np.array):
            return results - labels

    def __init__(self, sizes: list, learning_rate: float):
        if len(sizes) != 3 or type(sizes[0]) != int or type(sizes[1]) != int or type(sizes[2]) != int:
            raise Exception("Expecting 3 int values got {} values: {}.".format(len(sizes), sizes))
        self.input = self.__InputLayer(sizes[0], sizes[1])
        self.hidden = self.__HiddenLayer(sizes[1], sizes[2])
        self.output = self.__OutputLayer(sizes[2])
        self.learning_rate = learning_rate

    def train(self):
        return

    def predict(self):
        return


def train(number_of_iterations: int, learning_rate: float, weights: np.array, biases: np.array, train_data,
          batch_size: int):
    delta = np.zeros((10, 784))
    beta = np.zeros((10, 1))
    while number_of_iterations:
        for batch in range(0, len(train_data[0]) - batch_size + 1, batch_size):
            for index in range(batch, batch + 100):
                data, label = train_data[0][index], train_data[1][index]
                t = np.zeros((10, 1))
                t[label] = 1
                # weights shape => (10, 784)
                z = weights.dot(data).reshape((10, 1)) + biases
                y = (z >= 0.0).astype(float)
                delta += (t - y) * data * learning_rate  # (10, 784)
                beta += (t - y) * learning_rate  # (10, 1)
            weights += delta
            biases += beta
        # shuffle
        indeces = np.arange(len(train_data[0]))
        np.random.shuffle(indeces)
        train_data = (train_data[0][indeces], train_data[1][indeces])
        number_of_iterations -= 1
    return weights, biases


def predict(test_set, weights: np.array, biases: np.array):
    results = np.zeros(test_set[1].shape)
    for index in range(len(test_set[0])):
        # print(weights.dot(test_set[0][index]).shape)
        values = weights.dot(test_set[0][index]).reshape(10, 1) - biases
        results[index] = np.argmax(values, axis=0)
    values, counts = np.unique(results == test_set[1], return_counts=True)
    statistics = dict(zip(values, counts))
    print("Accuracy: {}/{}={}".format(statistics[True], statistics[True] + statistics[False],
                                      statistics[True] / (statistics[True] + statistics[False])))
    return results


if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()

    # weights, biases = initialize_model(10, 784, 10)
    # weights, biases = train(20, 0.05, weights, biases, train_set, 5000)
    # predict(train_set, weights, biases)
    # results = predict(test_set, weights, biases)
    # print(results)
    # print(test_set[1])

    DNN([784, 100, 10], 1)
