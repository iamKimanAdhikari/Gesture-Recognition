import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, BatchNormalization,
    Activation, Add, GlobalAveragePooling1D, Dense
)
from keras.regularizers import l2

def residual_block(x, filters, kernel_size, dilation_rate, weight_decay=1e-4):
    shortcut = x
    x = Conv1D(
        filters, kernel_size,
        padding='same', dilation_rate=dilation_rate,
        kernel_regularizer=l2(weight_decay)
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Conv1D(
        filters, kernel_size,
        padding='same', dilation_rate=dilation_rate,
        kernel_regularizer=l2(weight_decay)
    )(x)
    x = BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    return x

def build_tcn_model(
    input_shape,
    num_classes,
    num_blocks=3,
    filters=32,
    kernel_size=3,
    weight_decay=1e-4
):
    inputs = Input(shape=input_shape)
    x = Conv1D(
        filters, kernel_size,
        padding='same',
        kernel_regularizer=l2(weight_decay)
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    for rate in [2**i for i in range(num_blocks)]:
        x = residual_block(x, filters, kernel_size, rate, weight_decay)
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)
    return Model(inputs=inputs, outputs=outputs)
