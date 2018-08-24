import logging
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Lambda, Dense, RepeatVector, GlobalAveragePooling2D,Activation
from tensorflow.keras.layers import Reshape
from tensorflow.keras.layers import BatchNormalization
from tensorflow.keras.models import Model

import config

layers_dict = dict()


def get(obj):
    layer_name = obj.name
    if layer_name not in layers_dict:
        logging.info('-> Creating layer [{}]'.format(layer_name))
        # create it
        layers_dict[layer_name] = obj
    else:
        logging.info('-> Using layer [{}]'.format(layer_name))
    return layers_dict[layer_name]


# no need to share the weights here because it does not exist.
def clipped_relu(inputs):
    return get(Lambda(lambda y: K.minimum(K.maximum(y, 0), 20), name='clipped_relu'))(inputs)


def identity_block(input_tensor, kernel_size, filters, stage, block):
    conv_name_base = 'res{}_{}_branch'.format(stage, block)

    x = get(Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2a'))(input_tensor)
    x = get(BatchNormalization(name=conv_name_base + '_2a_bn'))(x)
    x = clipped_relu(x)

    x = get(Conv2D(filters,
                   kernel_size=kernel_size,
                   strides=1,
                   activation=None,
                   padding='same',
                   kernel_initializer='glorot_uniform',
                   kernel_regularizer=regularizers.l2(l=0.0001),
                   name=conv_name_base + '_2b'))(x)
    x = get(BatchNormalization(name=conv_name_base + '_2b_bn'))(x)

    x = layers.add([x, input_tensor])
    x = clipped_relu(x)
    return x


def convolutional_model(input):
    # http://cs231n.github.io/convolutional-networks/
    # conv weights
    # #params = ks * ks * nb_filters * num_channels_input

    # Conv128-s
    # 5*5*128*128/2+128
    # ks*ks*nb_filters*channels/strides+bias(=nb_filters)

    # take 100 ms -> 4 frames.
    # if signal is 3 seconds, then take 100ms per 100ms and average out this network.
    # 8*8 = 64 features.

    # used to share all the layers across the inputs

    # num_frames = K.shape() - do it dynamically after.

    def conv_and_res_block(inp, filters, stage):
        conv_name = 'conv{}-s'.format(filters)
        o = get(Conv2D(filters,
                       kernel_size=5,
                       strides=2,
                       padding='same',
                       kernel_initializer='glorot_uniform',
                       kernel_regularizer=regularizers.l2(l=0.0001), name=conv_name))(inp)
        o = get(BatchNormalization(name=conv_name + '_bn'))(o)
        o = clipped_relu(o)
        for i in range(3):
            o = identity_block(o, kernel_size=3, filters=filters, stage=stage, block=i)
        return o

    def cnn_component(inp):
        x_ = conv_and_res_block(inp, 64, stage=1)
        x_ = conv_and_res_block(x_, 128, stage=2)
        x_ = conv_and_res_block(x_, 256, stage=3)
        # x_ = conv_and_res_block(x_, 512, stage=4)
        return x_

    # inputs = Input(batch_shape=batch_input_shape)  # TODO the network should be definable without explicit batch shape
    x = cnn_component(input)
    x = Conv2D(2048,(1,1))(x)
    x = GlobalAveragePooling2D()(x)
    embedding = Dense(512, name='affine',activation="relu")(x)
    # embedding = Lambda(lambda y: K.l2_normalize(y, axis=1), name='embedding')(x)


    # x = Reshape((2048,))(x)
    # x = Lambda(lambda y: K.reshape(y, (batch_size, num_frames, 2048)), name='reshape')(x)
    # x = Lambda(lambda y: K.mean(y, axis=1), name='average')(x)
    # x = Dense(512, name='affine')(x)  # .shape = (BATCH_SIZE * NUM_FRAMES, 512)
    # x = Lambda(lambda y: K.l2_normalize(y, axis=1), name='ln')(x)
    return embedding

def sigmoid_loss(embedding_1,embedding_2,y):

    diff = tf.abs(embedding_1 - embedding_2)

    logit = Dense(1,)(diff)

    loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logit,labels=y),name="sigmoid_loss")
    return loss
    