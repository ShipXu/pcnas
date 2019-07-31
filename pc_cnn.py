import keras
import keras.backend as K
import numpy as np
import tensorflow as tf

from mnist import get_mnist
from evaluator import evaluate_flops, evaluate_params
from model import generate_keras_model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli
from coenvolve import coSelect, coEnvolve

import os
def remove_files():
    files = ['train_history_pcnas.csv']

    for file in files:
        if os.path.exists(file):
            os.remove(file)

remove_files()

STAGES = np.array(["s1", "s2"])  # S
NUM_NODES = np.array([3, 5])  # K

L = 0  # genome length
BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0  # to keep track of bits for each stage S
for nn in NUM_NODES:
    t = nn * (nn - 1)
    BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
    l_bpi = int(0.5 * t)
    L += t
L = int(0.5 * L)

TRAINING_EPOCHS = 20
BATCH_SIZE = 20

import csv
def evaluateModel(individual, dataset):
    print(individual)
    score = 0.0
    ((x_train, y_train), (x_test, y_test)) = dataset
    with tf.Session(graph=tf.Graph()) as network_sess:
        K.set_session(network_sess)
        model = generate_keras_model(individual, STAGES, NUM_NODES, BITS_INDICES)
        graph = network_sess.graph
        params = evaluate_params(graph)
        flops = evaluate_flops(graph)
        model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])

        # train the model using Keras methods
        model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS)

        # evaluate the model
        loss, acc = model.evaluate(x_test, y_test)
        print('Accuracy {0}, FLOPS {1}, PARAMS {2}'.format(acc, flops, params))
        with open('train_history_pcnas.csv', mode='a+') as f:
            data = [acc, flops, params]
            data.append(str(individual))
            writer = csv.writer(f)
            writer.writerow(data)
        return acc, flops, params

population_size = 20
num_generations = 3
PRESIZE=20
NOBJ=3
CXPB=0.9
MUTPB=0.9

dataset = get_mnist()
creator.create("FitnessMax", base.Fitness, weights=(1.0, -1.0, -1.0))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("binary", bernoulli.rvs, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n=L)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.8)
toolbox.register("select", coSelect)
toolbox.register("evaluate", lambda individual : evaluateModel(individual, dataset=dataset))

pop = toolbox.population(n=population_size)
pop, stats = coEnvolve(pop, toolbox, ngen=num_generations, npreference=PRESIZE, nobj=NOBJ, cxpb=CXPB, mutpb=MUTPB, seed=None)
print(stats)
