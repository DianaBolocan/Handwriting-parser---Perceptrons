import numpy as np, _pickle as cPickle, gzip
from tqdm import tqdm


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

    def __init__(self, sizes: list, learning_rate: float, batch_size: int, iterations: int):
        if len(sizes) != 3 or type(sizes[0]) != int or type(sizes[1]) != int or type(sizes[2]) != int:
            raise Exception("Expecting 3 int values got {} values: {}.".format(len(sizes), sizes))
        self.input = self.__InputLayer(sizes[0], sizes[1])
        self.hidden = self.__HiddenLayer(sizes[1], sizes[2])
        self.output = self.__OutputLayer()
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.iterations = iterations

    def feed_forward(self, data: np.array):
        self.input.sigmoid(self.input.total_net(data))
        # self.hidden.sigmoid(self.hidden.total_net(self.input.activated))
        self.hidden.softmax(self.hidden.total_net(self.input.activated))
        self.output.compute_results(self.hidden.activated)
        return self.output.results

    def backpropagation(self, labels: np.array):
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
        dot = self.hidden.weights.dot(lambda_output.T).T
        lambda_hidden = self.input.activated * (1 - self.input.activated) * dot
        for line in lambda_hidden * self.input.activated:
            self.input.weights -= self.learning_rate * line
        for line in lambda_hidden:
            self.input.biases -= line
        return

    def train(self, data: np.array):
        while self.iterations:
            count_batch = 0
            for batch in tqdm(range(0, len(data[0]) - self.batch_size + 1, self.batch_size)):
                input_data, label = data[0][batch: batch + self.batch_size], data[1][batch:batch + self.batch_size]
                self.feed_forward(input_data)
                self.backpropagation(label)
                count_batch += 1
            # shuffle
            indeces = np.arange(len(data[0]))
            np.random.shuffle(indeces)
            data = (data[0][indeces], data[1][indeces])
            self.iterations -= 1
        return

    def predict(self, data: np.array):
        results = np.argmax(self.feed_forward(data[0]), axis=1)
        values, counts = np.unique(results == data[1], return_counts=True)
        statistics = dict(zip(values, counts))
        print("Accuracy: {}/{}={}".format(statistics[True], statistics[True] + statistics[False],
                                          statistics[True] / (statistics[True] + statistics[False])))
        return results


if __name__ == '__main__':
    f = gzip.open('mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
    f.close()

    dnn = DNN([784, 100, 10], 0.05, 5000, 10)
    dnn.train(train_set)
    dnn.predict(train_set)
    dnn.predict(valid_set)
    dnn.predict(test_set)
