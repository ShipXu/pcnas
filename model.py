import keras
import numpy as np
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Dense, Conv2D, MaxPooling2D, Flatten, Dropout
from dag import DAG

def linear_layer(x, n_hidden_units, layer_name):
    return Dense(units=n_hidden_units, name=layer_name)(x)


def apply_convolution(x, kernel_height, kernel_width, in_channels, out_chanels, layer_name):
    tensor = Conv2D(out_chanels, kernel_height, strides=2, padding='same', activation='relu', bias_initializer='ones',
                    name=layer_name)(x)
    return tensor


def apply_pool(x, kernel_height, kernel_width, stride_size):
    return MaxPooling2D(pool_size=(kernel_height, kernel_width), strides=2, padding='same')(x)


def add_node(node_name, connector_node_name, h=5, w=5, ic=1, oc=1):
    with tf.name_scope(node_name) as scope:
        conv = apply_convolution(tf.get_default_graph().get_tensor_by_name(connector_node_name),
                                 kernel_height=h, kernel_width=w, in_channels=ic, out_chanels=oc,
                                 layer_name=''.join(["conv_", node_name]))
        print('conv_name', conv.name)


def sum_tensors(tensor_a, tensor_b):
    return keras.layers.add([tf.get_default_graph().get_tensor_by_name(tensor_a),
                             tf.get_default_graph().get_tensor_by_name(tensor_b)])


def get_node_name(node_name, activation_function_pattern):
    layer_name = ''.join(["/conv_", node_name])
    return ''.join([node_name, layer_name, activation_function_pattern])

def get_sum_name(node_name, activation_function_pattern):
    if not node_name.startswith("Add"):
        node_name = get_node_name(node_name, activation_function_pattern)
    return node_name

def has_same_elements(x):
    #     return len(set(x)) <= 1
    return sum(x) > 0


def generate_dag(optimal_indvidual, stage_name, num_nodes):
    # create nodes for the graph
    nodes = np.empty((0), dtype=np.str)
    for n in range(1, (num_nodes + 1)):
        nodes = np.append(nodes, ''.join([stage_name, "_", str(n)]))

    # initialize directed asyclic graph (DAG) and add nodes to it
    dag = DAG()
    for n in nodes:
        dag.add_node(n)

    # split best indvidual found via GA to identify vertices connections and connect them in DAG
    edges = np.split(optimal_indvidual, np.cumsum(range(num_nodes - 1)))[1:]
    v2 = 2
    for e in edges:
        v1 = 1
        for i in e:
            if i:
                dag.add_edge(''.join([stage_name, "_", str(v1)]), ''.join([stage_name, "_", str(v2)]))
            v1 += 1
        v2 += 1

    # delete nodes not connected to anyother node from DAG
    for n in nodes:
        if len(dag.predecessors(n)) == 0 and len(dag.downstream(n)) == 0:
            dag.delete_node(n)
            nodes = np.delete(nodes, np.where(nodes == n)[0][0])

    return dag, nodes


def generate_keras_model(individual, stages, num_nodes, bits_indices):
    activation_function_pattern = "/Relu:0"

    #     tf.reset_default_graph()
    ip = Input(shape=(28, 28, 1), name="X")

    d_node = ip
    for stage_index, stage_name, num_node, bpi in zip(range(0, len(stages)), stages, num_nodes, bits_indices):
        indv = individual[bpi[0]:bpi[1]]

        ic = 1
        oc = 1
        if stage_index == 0:
            add_node(''.join([stage_name, "_input"]), d_node.name, ic=1, oc=20)
            ic = 20
            oc = 20
        elif stage_index == 1:
            add_node(''.join([stage_name, "_input"]), d_node.name, ic=20, oc=50)
            ic = 50
            oc = 50

        pooling_layer_name = get_node_name(''.join([stage_name, "_input"]), activation_function_pattern)
        print('pooling_layer_name:', pooling_layer_name)
        if not has_same_elements(indv):
            # ------------------- Temporary DAG to hold all connections implied by GA solution ------------- #

            # get DAG and nodes in the graph
            dag, nodes = generate_dag(indv, stage_name, num_node)
            # get nodes without any predecessor, these will be connected to input node
            without_predecessors = dag.ind_nodes()
            # get nodes without any successor, these will be connected to output node
            without_successors = dag.all_leaves()

            # ----------------------------------------------------------------------------------------------- #

            # --------------------------- Initialize tensforflow graph based on DAG ------------------------- #

            for wop in without_predecessors:
                add_node(wop, get_node_name(''.join([stage_name, "_input"]), activation_function_pattern),
                         ic=ic, oc=oc)

            for n in nodes:
                predecessors = dag.predecessors(n)
                if len(predecessors) == 0:
                    continue
                elif len(predecessors) >= 1:
                    first_predecessor = predecessors[0]
                    for prd in range(1, len(predecessors)):
                        first_predecessor = get_sum_name(first_predecessor, activation_function_pattern)
                        predecessor = get_node_name(predecessors[prd], activation_function_pattern)
                        t = sum_tensors(first_predecessor, predecessor)
                        first_predecessor = t.name
                    add_node(n,
                             get_node_name(''.join([stage_name, "_input"]), activation_function_pattern),
                             ic=ic, oc=oc)

            if len(without_successors) >= 1:
                first_successor = without_successors[0]
                for suc in range(1, len(without_successors)):
                    first_successor = get_sum_name(first_successor, activation_function_pattern)
                    without_successor = get_node_name(without_successors[suc], activation_function_pattern)
                    t = sum_tensors(first_successor, without_successor)
                    first_successor = t.name
                add_node(''.join([stage_name, "_output"]), first_successor, ic=ic, oc=oc)
            elif nodes:
                last_node = nodes[-1]
                add_node(''.join([stage_name, "_output"]), last_node, ic=ic, oc=oc)
            else:
                add_node(''.join([stage_name, "_output"]),
                         get_node_name(''.join([stage_name, "_input"]), activation_function_pattern),
                         ic=ic, oc=oc)

            pooling_layer_name = get_node_name(''.join([stage_name, "_output"]), activation_function_pattern)
            # ------------------------------------------------------------------------------------------ #

        d_node = apply_pool(tf.get_default_graph().get_tensor_by_name(pooling_layer_name),
                            kernel_height=2, kernel_width=2, stride_size=2)

    flat = Flatten()(d_node)
    logits = Dense(units=500, name="logits")(flat)
    output = Dense(10, activation='softmax', name="output")(logits)
    model = Model(ip, output)
    return model