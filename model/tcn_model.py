import tensorflow as tf
from keras.models import Model
from keras.layers import (
    Input, Conv1D, BatchNormalization,
    Activation, Add, GlobalAveragePooling1D, Dense
)
from keras.regularizers import l2

def residual_block(x, filters, kernel_size, dilation_rate, weight_decay=1e-4):
    shortcut = x

    # First conv -> BN -> ReLU
    x = Conv1D(
        filters, kernel_size,
        padding='same', dilation_rate=dilation_rate,
        kernel_regularizer=l2(weight_decay)
    )(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Second conv -> BN
    x = Conv1D(
        filters, kernel_size,
        padding='same', dilation_rate=dilation_rate,
        kernel_regularizer=l2(weight_decay)
    )(x)
    x = BatchNormalization()(x)

    # Adjust shortcut if channel dims differ
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)

    # Add & final activation
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

    # Initial conv
    x = Conv1D(
        filters, kernel_size,
        padding='same',
        kernel_regularizer=l2(weight_decay)
    )(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)

    # Residual blocks with increasing dilation
    for rate in [2**i for i in range(num_blocks)]:
        x = residual_block(x, filters, kernel_size, rate, weight_decay)

    # Global pooling + classifier
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(num_classes, activation='softmax')(x)

    return Model(inputs=inputs, outputs=outputs)
