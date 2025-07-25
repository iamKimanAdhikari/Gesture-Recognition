import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv1D, BatchNormalization, Activation, Add, Flatten, Dense, GlobalAveragePooling1D
from keras.regularizers import l2

def residual_block(x, filters, kernel_size, dilation_rate, weight_decay=1e-4):
    # Shortcut connection
    shortcut = x
    
    # First convolution
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Second convolution
    x = Conv1D(filters, kernel_size, padding='same', dilation_rate=dilation_rate,
               kernel_regularizer=l2(weight_decay))(x)
    x = BatchNormalization()(x)
    
    # Adjust shortcut if needed
    if shortcut.shape[-1] != filters:
        shortcut = Conv1D(filters, 1, padding='same')(shortcut)
    
    # Add shortcut and apply activation
    x = Add()([x, shortcut])
    x = Activation('relu')(x)
    
    return x

def build_tcn_model(input_shape, num_classes, num_blocks=3, filters=32, kernel_size=3, weight_decay=1e-4):
    inputs = Input(shape=input_shape)
    
    # Initial convolution
    x = Conv1D(filters, kernel_size, padding='same', kernel_regularizer=l2(weight_decay))(inputs)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    
    # Stack residual blocks with increasing dilation rates
    dilation_rates = [2**i for i in range(num_blocks)]
    for rate in dilation_rates:
        x = residual_block(x, filters, kernel_size, rate, weight_decay)
    
    # Global average pooling - FIXED: Use Keras layer instead of tf.reduce_mean
    x = GlobalAveragePooling1D()(x)
    
    # Classification layer
    outputs = Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs=inputs, outputs=outputs)
    return model