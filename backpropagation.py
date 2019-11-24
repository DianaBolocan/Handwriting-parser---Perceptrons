import numpy as np, _pickle as cPickle, gzip


class DNN:
    class __InputLayer:
        def __init__(self, input_size: int, output_size: int):
            self.weights = np.random.random((input_size, output_size))
            self.biases = np.random.random(output_size)
            self.activated = None

        def total_net(self, data: np.array):
            return data.dot(self.weights) + self.biases

        def sigmoid(self, results: np.array):
            self.activated = 1 / (1 + np.exp(-results))
            return self.activated

    class __HiddenLayer:
        def __init__(self, input_size: int, output_size: int):
            self.weights = np.random.random((input_size, output_size))
            self.biases = np.random.random(output_size)
            self.activated = None

        def total_net(self, data: np.array):
            return data.dot(self.weights) + self.biases

        def sigmoid(self, results: np.array):
            self.activated = 1 / (1 + np.exp(-results))
            return self.activated

        def softmax(self, results: np.array):
            self.activated = np.exp(results) / np.sum(np.exp(results))
            return self.activated

    class __OutputLayer:
        def __init__(self):
            self.results = None

        def compute_results(self, data: np.array):
            self.results = np.zeros(data.shape)
            highest_values_indexes = data.argmax(axis=1)
            for index in range(len(highest_values_indexes)):
                self.results[index][highest_values_indexes[index]] = 1
            return self.results

    def __init__(self, sizes: list, learning_rate: float):
        if len(sizes) != 3 or type(sizes[0]) != int or type(sizes[1]) != int or type(sizes[2]) != int:
            raise Exception("Expecting 3 int values got {} values: {}.".format(len(sizes), sizes))
        self.input = self.__InputLayer(sizes[0], sizes[1])
        self.hidden = self.__HiddenLayer(sizes[1], sizes[2])
        self.output = self.__OutputLayer()
        self.learning_rate = learning_rate
        self.total_error = None
        self.errors = None

    def feed_forward(self, data: np.array):
        self.input.sigmoid(self.input.total_net(data))
        # self.hidden.sigmoid(self.hidden.total_net(self.input.activated))
        self.hidden.softmax(self.hidden.total_net(self.input.activated))
        self.output.compute_results(self.hidden.activated)
        return self.output.results

    def cross_entropy(self, labels: np.array):
        epsilon = 1e-15
        targets = np.zeros(self.output.results.shape)
        for index in range(len(labels)):
            targets[index][labels[index]] = 1
        # print(np.log(self.output.results + epsilon) * targets)
        self.errors = np.log(self.output.results + epsilon) * targets
        self.total_error = -np.sum(np.log(self.output.results + epsilon) * targets, axis=1)/self.errors.shape[-1]
        return self.total_error, self.errors

    def backpropagation(self, labels: np.array):
        # self.cross_entropy(labels)
        targets = np.zeros(self.output.results.shape)
        for index in range(len(labels)):
            targets[index][labels[index]] = 1
        # update weights and biases from output layer
        lambda_output = self.output.results - targets
        for line in lambda_output * self.hidden.activated:
            self.hidden.weights -= self.learning_rate * line
        # TODO: check for biases weights
        for line in lambda_output:
            self.hidden.biases -= line
        # update from hidden layer
        temp = np.zeros(self.hidden.weights.shape)
        for line in lambda_output:
            temp += self.hidden.weights * line
        print(temp.shape)
        print((self.input.activated*(1 - self.input.activated)).shape)
        # for line in self.input.activated*(1 - self.input.activated)*temp:
        #     print(line)
        return

    def train(self, data: np.array):
        return

    def predict(self, data: np.array):
        return


# def train(number_of_iterations: int, learning_rate: float, weights: np.array, biases: np.array, train_data,
#           batch_size: int):
#     delta = np.zeros((10, 784))
#     beta = np.zeros((10, 1))
#     while number_of_iterations:
#         for batch in range(0, len(train_data[0]) - batch_size + 1, batch_size):
#             for index in range(batch, batch + 100):
#                 data, label = train_data[0][index], train_data[1][index]
#                 t = np.zeros((10, 1))
#                 t[label] = 1
#                 # weights shape => (10, 784)
#                 z = weights.dot(data).reshape((10, 1)) + biases
#                 y = (z >= 0.0).astype(float)
#                 delta += (t - y) * data * learning_rate  # (10, 784)
#                 beta += (t - y) * learning_rate  # (10, 1)
#             weights += delta
#             biases += beta
#         # shuffle
#         indeces = np.arange(len(train_data[0]))
#         np.random.shuffle(indeces)
#         train_data = (train_data[0][indeces], train_data[1][indeces])
#         number_of_iterations -= 1
#     return weights, biases
#
#
# def predict(test_set, weights: np.array, biases: np.array):
#     results = np.zeros(test_set[1].shape)
#     for index in range(len(test_set[0])):
#         # print(weights.dot(test_set[0][index]).shape)
#         values = weights.dot(test_set[0][index]).reshape(10, 1) - biases
#         results[index] = np.argmax(values, axis=0)
#     values, counts = np.unique(results == test_set[1], return_counts=True)
#     statistics = dict(zip(values, counts))
#     print("Accuracy: {}/{}={}".format(statistics[True], statistics[True] + statistics[False],
#                                       statistics[True] / (statistics[True] + statistics[False])))
#     return results


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

    dnn = DNN([784, 100, 10], 0.05)
    dnn.feed_forward(train_set[0][:20])
    total, all = dnn.cross_entropy(train_set[1][:20])
    # print(total)
    # print(all)
    dnn.backpropagation(train_set[1][:20])

