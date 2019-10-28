import numpy as np, _pickle as cPickle, gzip


# noIterations = 10
# rate = 0.01
#
# weights = np.random.rand((784, 10))
# biases = np.random.rand((1, 784))
#
# # n iterari pe fiecare perceptron
# for j in range(noIterations):
#     result = []
#     newWeights = []
#     newBiases = []
#
#     print("Iteration:", j)
#     for i in range(10):
#         result = perceptronTraining(i, train_set, weights[i], biases[i], rate)
#         newWeights.append(result[0])
#         newBiases.append(result[1])
#
#     newWeights = np.array(newWeights)
#     newBiases = np.array(newBiases)
#     biases = newBiases
#     weights = newWeights
#
#     # do the test
#     digits = []
#     for input in test_set[0]:
#         max = 0
#         digit = 0
#         for perc in range(10):
#             compute = perceptronCompute(perc, input, newWeights[perc], newBiases[perc])
#             if max < compute:
#                 max = compute
#                 digit = perc
#         digits.append(digit)
#
#     match = 0
#     for i in range(len(digits)):
#         if digits[i] == test_set[1][i]:
#             match += 1
#
#     print(match / len(digits) * 100)
#
#     # shuffle the train_set
#     indexes = []
#     train = []
#     results = []
#     for index in range(len(train_set[0])):
#         indexes.append(index)
#     random.shuffle(indexes)
#     for i in indexes:
#         train.append(train_set[0][i])
#         results.append(train_set[1][i])
#     train = np.array(train)
#     results = np.array(results)
#     train_set = [train, results]


def initialize_model(number_of_perceptrons: int, weights_size: int, biases_size: int):
    weights = np.random.random((number_of_perceptrons, weights_size))
    biases = np.random.random((biases_size, 1))
    return weights, biases


def train(number_of_iterations: int, learning_rate: float, weights: np.array, biases: np.array, train_data, batch_size: int):
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

    weights, biases = initialize_model(10, 784, 10)
    weights, biases = train(20, 0.05, weights, biases, train_set, 5000)
    predict(train_set, weights, biases)
    results = predict(test_set, weights, biases)
    print(results)
    print(test_set[1])
