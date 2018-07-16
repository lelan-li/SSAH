import scipy.misc
import scipy.io
from ops import *
from setting import *

def img_net(inputs, bit, numclass):
    data = scipy.io.loadmat(MODEL_DIR)
    layers = (
        'conv1', 'relu1', 'norm1', 'pool1', 'conv2', 'relu2', 'norm2', 'pool2', 'conv3', 'relu3', 'conv4', 'relu4',
        'conv5', 'relu5', 'pool5', 'fc6', 'relu6', 'fc7', 'relu7')
    weights = data['layers'][0]

    labnet = {}
    current = tf.convert_to_tensor(inputs, dtype='float32')
    for i, name in enumerate(layers):
        if name.startswith('conv'):
            kernels, bias = weights[i][0][0][0][0]

            bias = bias.reshape(-1)
            pad = weights[i][0][0][1]
            stride = weights[i][0][0][4]
            current = conv_layer(current, kernels, bias, pad, stride, i, labnet)
        elif name.startswith('relu'):
            current = tf.nn.relu(current)
        elif name.startswith('pool'):
            stride = weights[i][0][0][1]
            pad = weights[i][0][0][2]
            area = weights[i][0][0][5]
            current = pool_layer(current, stride, pad, area)
        elif name.startswith('fc'):
            kernels, bias = weights[i][0][0][0][0]
            bias = bias.reshape(-1)
            current = full_conv(current, kernels, bias, i, labnet)
        elif name.startswith('norm'):
            current = tf.nn.local_response_normalization(current, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)
        labnet[name] = current

    W_fc8 = tf.random_normal([1, 1, 4096, SEMANTIC_EMBED], stddev=1.0) * 0.01
    b_fc8 = tf.random_normal([SEMANTIC_EMBED], stddev=1.0) * 0.01
    w_fc8 = tf.Variable(W_fc8, name='w' + str(20))
    b_fc8 = tf.Variable(b_fc8, name='bias' + str(20))

    fc8 = tf.nn.conv2d(current, w_fc8, strides=[1, 1, 1, 1], padding='VALID')
    fc8 = tf.nn.bias_add(fc8, b_fc8)
    relu8 = tf.nn.relu(fc8)
    labnet['feature'] = relu8

    W_fc9 = tf.random_normal([1, 1, SEMANTIC_EMBED, bit], stddev=1.0) * 0.01 
    b_fc9 = tf.random_normal([bit], stddev=1.0) * 0.01
    w_fc9 = tf.Variable(W_fc9, name='w' + str(21))
    b_fc9 = tf.Variable(b_fc9, name='bias' + str(21))

    # fc9 = tf.nn.conv2d(fc8, w_fc9, strides=[1, 1, 1, 1], padding='VALID')
    fc9 = tf.nn.conv2d(relu8, w_fc9, strides=[1, 1, 1, 1], padding='VALID')
    fc9 = tf.nn.bias_add(fc9, b_fc9)
    labnet['hash'] = tf.nn.tanh(fc9)

    W_fc10 = tf.random_normal([1, 1, SEMANTIC_EMBED, numclass], stddev=1.0) * 0.01
    b_fc10 = tf.random_normal([numclass], stddev=1.0) * 0.01
    w_fc10 = tf.Variable(W_fc10, name='w' + str(22))
    b_fc10 = tf.Variable(b_fc10, name='bias' + str(22))

    # fc10 = tf.nn.conv2d(fc8, w_fc10, strides=[1, 1, 1, 1], padding='VALID')
    fc10 = tf.nn.conv2d(relu8, w_fc10, strides=[1, 1, 1, 1], padding='VALID')
    fc10 = tf.nn.bias_add(fc10, b_fc10)
    labnet['label'] = tf.nn.sigmoid(fc10)

    return tf.squeeze(labnet['hash']), tf.squeeze(labnet['feature']), tf.squeeze(labnet['label'])

def lab_net(imput_label, bit, numClass):
    LAYER1_NODE = 4096
    labnet = {}
    W_fc1 = tf.random_normal([1, numClass, 1, LAYER1_NODE], stddev=1.0) * 0.01
    b_fc1 = tf.random_normal([1, LAYER1_NODE], stddev=1.0) * 0.01
    labnet['fc1W'] = tf.Variable(W_fc1)
    labnet['fc1b'] = tf.Variable(b_fc1)
    labnet['conv1'] = tf.nn.conv2d(imput_label, labnet['fc1W'], strides=[1, 1, 1, 1], padding='VALID')
    W1_plus_b1 = tf.nn.bias_add(labnet['conv1'], tf.squeeze(labnet['fc1b']))
    relu1 = tf.nn.relu(W1_plus_b1)

    norm1 = tf.nn.local_response_normalization(relu1, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

    W_fc2 = tf.random_normal([1, 1, LAYER1_NODE, SEMANTIC_EMBED], stddev=1.0) * 0.01
    b_fc2 = tf.random_normal([1, SEMANTIC_EMBED], stddev=1.0) * 0.01
    labnet['fc2W'] = tf.Variable(W_fc2)
    labnet['fc2b'] = tf.Variable(b_fc2)
    labnet['conv2'] = tf.nn.conv2d(norm1, labnet['fc2W'], strides=[1, 1, 1, 1], padding='VALID')
    fc2 = tf.nn.bias_add(labnet['conv2'], tf.squeeze(labnet['fc2b']))
    relu2 = tf.nn.relu(fc2)
    labnet['feature'] = relu2

    #norm2 = tf.nn.local_response_normalization(relu2, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

    W_fc3 = tf.random_normal([1, 1, SEMANTIC_EMBED, bit], stddev=1.0) * 0.01
    b_fc3 = tf.random_normal([1, bit], stddev=1.0) * 0.01
    labnet['fc3W'] = tf.Variable(W_fc3)
    labnet['fc3b'] = tf.Variable(b_fc3)
    labnet['conv3'] = tf.nn.conv2d(relu2, labnet['fc3W'], strides=[1, 1, 1, 1], padding='VALID')
    output_h = tf.nn.bias_add(labnet['conv3'], tf.squeeze(labnet['fc3b']))
    labnet['hash'] = tf.nn.tanh(output_h)

    W_fc4 = tf.random_normal([1, 1, SEMANTIC_EMBED, numClass], stddev=1.0) * 0.01
    b_fc4 = tf.random_normal([1, numClass], stddev=1.0) * 0.01
    labnet['fc4W'] = tf.Variable(W_fc4)
    labnet['fc4b'] = tf.Variable(b_fc4)
    labnet['conv4'] = tf.nn.conv2d(relu2, labnet['fc4W'], strides=[1, 1, 1, 1], padding='VALID')
    label_ = tf.nn.bias_add(labnet['conv4'], tf.squeeze(labnet['fc4b']))
    labnet['label'] = tf.nn.sigmoid(label_)

    return tf.squeeze(labnet['hash']), tf.squeeze(labnet['feature']), tf.squeeze(labnet['label'])


def dis_net_IL(feature, keep_prob, reuse=False, name="disnet_IL"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        disnet = {}
        relu1 = relu(conv2d(feature, [1,SEMANTIC_EMBED,1,512], [1,1,1,1], 'VALID', 1.0, "disnet_IL_fc1"))
        dropout1 = tf.nn.dropout(relu1, keep_prob)
        relu2 = relu(conv2d(dropout1, [1,1,512,256], [1,1,1,1], 'VALID', 1.0, "disnet_IL_fc2"))
        dropout2 = tf.nn.dropout(relu2, keep_prob)
        disnet['output'] = conv2d(dropout2, [1, 1, 256, 1], [1, 1, 1, 1], 'VALID', 1.0, "disnet_IL_out")

        # relu1 = relu(batch_norm(conv2d(feature, [1, 1, SEMANTIC_EMBED, 512], [1, 1, 1, 1], 'VALID', 1.0, "disnet_IL_fc1")))
        # dropout1 = tf.nn.dropout(relu1, keep_prob)
        # relu2 = relu(batch_norm(conv2d(dropout1, [1, 1, 512, 256], [1, 1, 1, 1], 'VALID', 1.0, "disnet_IL_fc2")))
        # dropout2 = tf.nn.dropout(relu2, keep_prob)
        # disnet['output'] = conv2d(dropout2, [1, 1, 256, 1], [1, 1, 1, 1], 'VALID', 1.0, "disnet_IL_out")
    return tf.squeeze(disnet['output'])

def dis_net_TL(feature, keep_prob, reuse=False, name="disnet_TL"):
    with tf.variable_scope(name):
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        disnet = {}
        relu1 = relu(conv2d(feature, [1, SEMANTIC_EMBED, 1, 512], [1, 1, 1, 1], 'VALID', 1.0, "disnet_TL_fc1"))
        dropout1 = tf.nn.dropout(relu1, keep_prob)
        relu2 = relu(conv2d(dropout1, [1, 1, 512, 256], [1, 1, 1, 1], 'VALID', 1.0, "disnet_TL_fc2"))
        dropout2 = tf.nn.dropout(relu2, keep_prob)
        disnet['output'] = conv2d(dropout2, [1, 1, 256, 1], [1, 1, 1, 1], 'VALID', 1.0, "disnet_TL_out")

        # relu1 = relu(batch_norm(conv2d(feature, [1, 1, SEMANTIC_EMBED, 512], [1, 1, 1, 1], 'VALID', 1.0, "disnet_TL_fc1")))
        # dropout1 = tf.nn.dropout(relu1, keep_prob)
        # relu2 = relu(batch_norm(conv2d(dropout1, [1, 1, 512, 256], [1, 1, 1, 1], 'VALID', 1.0, "disnet_TL_fc2")))
        # dropout2 = tf.nn.dropout(relu2, keep_prob)
        # disnet['output'] = conv2d(dropout2, [1, 1, 256, 1], [1, 1, 1, 1], 'VALID', 1.0, "disnet_TL_out")
        return tf.squeeze(disnet['output'])

def txt_net(text_input, dimy, bit, numclass):
    txtnet={}
    MultiScal = MultiScaleTxt(text_input)
    W_fc1 = tf.random_normal([1, dimy, 6, 4096], stddev=1.0) * 0.01
    b_fc1 = tf.random_normal([1, 4096], stddev=1.0) * 0.01
    fc1W = tf.Variable(W_fc1)
    fc1b = tf.Variable(b_fc1)
    txtnet['conv1'] = tf.nn.conv2d(MultiScal, fc1W, strides=[1, 1, 1, 1], padding='VALID')

    W1_plus_b1 = tf.nn.bias_add(txtnet['conv1'], tf.squeeze(fc1b))
    txtnet['fc1'] = tf.nn.relu(W1_plus_b1)

    txtnet['norm1'] = tf.nn.local_response_normalization(txtnet['fc1'], depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

    W_fc2 = tf.random_normal([1, 1, 4096, SEMANTIC_EMBED], stddev=1.0) * 0.01
    b_fc2 = tf.random_normal([1, SEMANTIC_EMBED], stddev=1.0) * 0.01
    fc2W = tf.Variable(W_fc2)
    fc2b = tf.Variable(b_fc2)
    txtnet['conv2'] = tf.nn.conv2d(txtnet['norm1'], fc2W, strides=[1, 1, 1, 1], padding='VALID')
    W2_plus_b2 = tf.nn.bias_add(txtnet['conv2'], tf.squeeze(fc2b))
    relu2 = tf.nn.relu(W2_plus_b2)
    txtnet['feature'] = relu2
    
    txtnet['norm2'] = tf.nn.local_response_normalization(relu2, depth_radius=2, bias=2.000, alpha=0.0001, beta=0.75)

    W_fc3 = tf.random_normal([1, 1, SEMANTIC_EMBED, bit], stddev=1.0) * 0.01
    b_fc3 = tf.random_normal([bit], stddev=1.0) * 0.01
    fc3W = tf.Variable(W_fc3)
    fc3b = tf.Variable(b_fc3)

    txtnet['conv3'] = tf.nn.conv2d(txtnet['norm2'], fc3W, strides=[1, 1, 1, 1], padding='VALID')
    W3_plus_b3 = tf.nn.bias_add(txtnet['conv3'], tf.squeeze(fc3b))
    txtnet['hash'] = tf.nn.tanh(W3_plus_b3)

    W_fc4 = tf.random_normal([1, 1, SEMANTIC_EMBED, numclass], stddev=1.0) * 0.01
    b_fc4 = tf.random_normal([numclass], stddev=1.0) * 0.01
    fc4W = tf.Variable(W_fc4)
    fc4b = tf.Variable(b_fc4)

    txtnet['conv4'] = tf.nn.conv2d(txtnet['norm2'], fc4W, strides=[1, 1, 1, 1], padding='VALID')
    W4_plus_b4 = tf.nn.bias_add(txtnet['conv4'], tf.squeeze(fc4b))
    txtnet['label'] = tf.nn.sigmoid(W4_plus_b4)

    return tf.squeeze(txtnet['hash']), tf.squeeze(txtnet['feature']), tf.squeeze(txtnet['label'])

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

def MultiScaleTxt(input):
    interp_block1  = interp_block(input, 10)
    interp_block2  = interp_block(input, 6)
    interp_block3  = interp_block(input, 3)
    interp_block6  = interp_block(input, 2)
    interp_block10 = interp_block(input, 1)
    output = tf.concat([input,
                         interp_block10,
                         interp_block6,
                         interp_block3,
                         interp_block2,
                         interp_block1], axis = -1)
    return output
