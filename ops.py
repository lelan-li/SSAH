import tensorflow as tf
from setting import *

def conv_layer(input, weights, bias, pad, stride, i, net):
    pad = pad[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[1, stride[0], stride[1], 1], padding='VALID', name='conv' + str(i))
    return tf.nn.bias_add(conv, b, name='add' + str(i))


def full_conv(input, weights, bias, i, net):
    w = tf.Variable(weights, name='w' + str(i), dtype='float32')
    b = tf.Variable(bias, name='bias' + str(i), dtype='float32')
    net['weights' + str(i)] = w
    net['b' + str(i)] = b
    conv = tf.nn.conv2d(input, w, strides=[1, 1, 1, 1], padding='VALID', name='fc' + str(i))
    return tf.nn.bias_add(conv, b, name='add' + str(i))


def pool_layer(input, stride, pad, area):
    pad = pad[0]
    area = area[0]
    stride = stride[0]
    input = tf.pad(input, [[0, 0], [pad[0], pad[1]], [pad[2], pad[3]], [0, 0]], "CONSTANT")
    return tf.nn.max_pool(input, ksize=[1, area[0], area[1], 1], strides=[1, stride[0], stride[1], 1], padding='VALID')


def conv2d(input, kernel, strides, padding, init_rate, name):
    with tf.variable_scope(name):
        W = tf.Variable(tf.random_normal(kernel, stddev=init_rate) * 0.01)
        b = tf.Variable(tf.random_normal([kernel[-1]], stddev=init_rate) * 0.01)
        conv = tf.nn.conv2d(input, W, strides=strides, padding=padding)
        out = tf.nn.bias_add(conv, b)
    return out

# activate function
def lrelu(x, leak=0.2):
    return tf.maximum(x, leak*x)

def relu(x):
    return tf.nn.relu(x)

def sigmoid(x):
    return tf.nn.sigmoid(x)

def tanh(x):
    return tf.nn.tanh(x)

# loss
def mse_criterion(in_, target):
    return tf.reduce_mean(tf.nn.l2_loss(in_ - target))

def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))

#
def ClipIfNotNone(grad):
    if grad is None:
        return grad
    return tf.clip_by_value(grad, -1., 1.)

def calc_neighbor(label_1, label_2):
    Sim = (np.dot(label_1, label_2.transpose()) > 0).astype(int)*0.999
    return Sim

# normalization
def local_norm(x):
    return tf.nn.local_response_normalization(x, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

def batch_norm(x, name="batch_norm"):
    return tf.contrib.layers.batch_norm(x, decay=0.9, updates_collections=None, epsilon=1e-5, scale=True)

def interp_block(text_input, level):
    shape = [1, 1, 5 * level, 1]
    stride = [1, 1, 5 * level, 1]
    prev_layer = tf.nn.avg_pool(text_input, ksize=shape, strides=stride, padding='VALID')
    W_fc1 = tf.random_normal([1, 1, 1, 1], stddev=1.0) * 0.01
    fc1W = tf.Variable(W_fc1)
    prev_layer = tf.nn.conv2d(prev_layer, fc1W, strides=[1, 1, 1, 1], padding='VALID')
    prev_layer = tf.nn.relu(prev_layer)
    prev_layer = tf.image.resize_images(prev_layer, [1, dimTxt])
    return prev_layer

def MultiScaleTxt(text_input, input):
    interp_block1 = interp_block(input, 10)
    interp_block2 = interp_block(input, 6)
    interp_block3 = interp_block(input, 3)
    interp_block6 = interp_block(input, 2)
    interp_block10 = interp_block(input, 1)
    output = tf.concat([text_input,
                        interp_block10,
                        interp_block6,
                        interp_block3,
                        interp_block2,
                        interp_block1], axis=-1)
    return output
