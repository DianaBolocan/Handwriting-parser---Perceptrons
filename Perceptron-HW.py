import numpy as np, _pickle as cPickle, gzip, random

# daca poza seamana cel putin 50% cu ceea ce verific
def activation(y):
    if y >= 0.5:
        return 1
    return 0

# initializare lista cu 784 x 0
def initialization(list):
    for i in range(784):
        list.append(0)
    return list

# weight, bias initialization pt toate perceptroanele (cu 0)
def initializationWeight():
    weights = [[], [], [], [], [], [], [], [], [], []]
    for list in weights:
        initialization(list)
    return weights

def initializationBias():
    bias = []
    initialization(bias)
    return bias

# perceptronul primeste: vectorul poza, vectorul weight, bias, rata de training, nr de iteratii
def perceptronTraining(perceptronType, set, weight, bias, rate):
    for i in range(len(set[0])):
        y = weight.dot(set[0][i]) + bias
        z = activation(y)
        if set[1][i] == perceptronType:
            t = 1
        else:
            t = 0
        weight = weight + (t - z) * set[0][i] * rate
        bias = bias + (t - z) * rate
    return [weight, bias]

# returneaza un numar intre 0 si 1 care reprezinta procentul de match pentru cifra setata
def perceptronCompute(perceptronType, input, weight, bias):
    return weight.dot(input) + bias


f = gzip.open('mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f, encoding='latin1')
f.close()

noIterations = 10
rate = 0.01

# initializeaza weight-urile si bias-urile pentru fiecare perceptron (np.array)
weights = np.array(initializationWeight())
biases = np.array(initializationBias())

# n iterari pe fiecare perceptron
for j in range(noIterations):
    result = []
    newWeights = []
    newBiases = []

    print("Iteration:",j)
    for i in range(10):
        result = perceptronTraining(i, train_set, weights[i], biases[i], rate)
        newWeights.append(result[0])
        newBiases.append(result[1])

    newWeights = np.array(newWeights)
    newBiases = np.array(newBiases)
    biases = newBiases
    weights = newWeights

    # do the test
    digits = []
    for input in test_set[0]:
        max = 0
        digit = 0
        for perc in range(10):
            compute = perceptronCompute(perc, input, newWeights[perc], newBiases[perc])
            if max < compute:
                max = compute
                digit = perc
        digits.append(digit)

    match = 0
    for i in range(len(digits)):
        if digits[i] == test_set[1][i]:
            match += 1

    print(match/len(digits)*100)

    # shuffle the train_set
    indexes = []
    train = []
    results = []
    for index in range(len(train_set[0])):
        indexes.append(index)
    random.shuffle(indexes)
    for i in indexes:
        train.append(train_set[0][i])
        results.append(train_set[1][i])
    train = np.array(train)
    results = np.array(results)
    train_set = [train,results]

    # zip
