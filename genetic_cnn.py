import keras.backend as K
import tensorflow as tf
import numpy as np

from mnist import get_mnist
from evaluator import  evaluate_flops, evaluate_params
from model import generate_keras_model

from deap import base, creator, tools, algorithms
from scipy.stats import bernoulli

import os
def if_exists(filepath):
    return os.path.exists(filepath)

metrics = ['flops', 'params', 'acc']
def remove_files():
    files = ['experiment_data/geneticnn/genetic_{0}.csv'.format(metric) for metric in metrics]
    for file in files:
        if if_exists(file):
            os.remove(file)

def create_dirs():
    dirs = ['experiment_data', 'experiment_data/geneticnn', 'model_visulization', 'model_visulization/geneticnn']
    for dir in dirs:
        if not if_exists(dir):
            os.makedirs(dir)

remove_files()
create_dirs()

STAGES = np.array(["s1", "s2"])  # S
NUM_NODES = np.array([3, 4])  # K

L = 0  # genome length
BITS_INDICES, l_bpi = np.empty((0, 2), dtype=np.int32), 0  # to keep track of bits for each stage S
print(BITS_INDICES, l_bpi)
for nn in NUM_NODES:
    t = nn * (nn - 1)
    BITS_INDICES = np.vstack([BITS_INDICES, [l_bpi, l_bpi + int(0.5 * t)]])
    l_bpi += int(0.5 * t)
    L += t

L = int(0.5 * L)
print('L', L)
print('BITS_INDICES', BITS_INDICES)
TRAINING_EPOCHS = 20
BATCH_SIZE = 20

import csv
from keras.utils import plot_model
def acc_evaluate(graph, model):
    ((x_train, y_train), (x_test, y_test)) = dataset
    model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])
    # train the model using Keras methods
    model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=TRAINING_EPOCHS)
    # evaluate the model
    loss, acc = model.evaluate(x_test, y_test)
    return acc

def flops_evaluate(graph, model):
    return evaluate_params(graph)

def params_evaluate(graph, model):
    return evaluate_flops(graph)

evaluate_fuc = {
    'flops' : flops_evaluate,
    'params' : params_evaluate,
    'acc' : acc_evaluate
}

def evaluate(metric):
    def fuc(individual):
        score = 0.0
        with tf.Session(graph=tf.Graph()) as network_sess:
            K.set_session(network_sess)
            graph = network_sess.graph
            model = generate_keras_model(individual, STAGES, NUM_NODES, BITS_INDICES)
            plot_model(model,
                       to_file='model_visulization/geneticnn/model_{0}.png'.format(str(individual)),
                       show_shapes=True, show_layer_names=True, rankdir='TB')
            if metric in evaluate_fuc:
                score = evaluate_fuc[metric](graph, model)
            else:
                raise EnvironmentError('fuc not implemented')

            with open('experiment_data/geneticnn/genetic_{0}.csv'.format(metric), mode='a+') as f:
                data = [score]
                data.append(str(individual))
                writer = csv.writer(f)
                writer.writerow(data)
            return score,
    return fuc

population_size = 20
num_generations = 3
dataset = get_mnist()
creator.create("FitnessMax", base.Fitness, weights = (1.0,))
creator.create("Individual", list , fitness = creator.FitnessMax)

toolbox = base.Toolbox()
toolbox.register("binary", bernoulli.rvs, 0.5)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.binary, n = L)
toolbox.register("population", tools.initRepeat, list , toolbox.individual)

toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb = 0.8)
toolbox.register("select", tools.selRoulette)

for metric in metrics:
    toolbox.register("evaluate", evaluate(metric))

    popl = toolbox.population(n = population_size)

    # Initialize statistics object
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean, axis=0)
    stats.register("std", np.std, axis=0)
    stats.register("min", np.min, axis=0)
    stats.register("max", np.max, axis=0)

    from time import time
    start = time()
    result, logbook = algorithms.eaSimple(popl, toolbox, stats=stats, cxpb = 0.4, mutpb = 0.05, ngen = num_generations, verbose = True)
    stop = time()
    print("search Time = {}".format(stop -start))

    with open('experiment_data/genetic_cnn_result_{}.txt'.format(metric), 'w') as f:
        for ind in result:
            f.write(str(ind))
            f.write('\n')
        f.write("search Time = {}\n".format(stop -start))

    print(logbook)