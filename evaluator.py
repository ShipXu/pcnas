import tensorflow as tf


def stats_graph(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    print('FLOPs: {};    Trainable params: {}'.format(flops.total_float_ops, params.total_parameters))
    return flops, params

def evaluate_flops(graph):
    flops = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.float_operation())
    return flops.total_float_ops

def evaluate_params(graph):
    params = tf.profiler.profile(graph, options=tf.profiler.ProfileOptionBuilder.trainable_variables_parameter())
    return params.total_parameters

# with tf.Graph().as_default() as graph:
#     A = tf.get_variable(initializer=tf.constant_initializer(value=1, dtype=tf.float32), shape=(25, 16), name='A')
#     B = tf.get_variable(initializer=tf.zeros_initializer(dtype=tf.float32), shape=(16, 9), name='B')
#     C = tf.matmul(A, B, name='ouput')
#
#     flops = evaluate_flops(graph)
#     params = evaluate_params(graph)
#     print('flops:', flops)
#     print('params:', params)



