""" AlexNet Model for Keras.
Partially based on https://github.com/heuritech/convnets-keras/blob/master/convnetskeras/convnets.py
# Reference

- [Imagenet classification with deep con- volutional neural networks] (dl.acm.org/citation.cfm?doid=3098997.3065386)

"""
import os 
import warnings

from keras import backend as K
from keras.models import Model
from keras import layers
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import Lambda
from keras.layers import MaxPooling2D
from keras.layers import ZeroPadding2D
from keras.layers import Concatenate
from keras.layers import Flatten
from keras.layers import Dropout
from keras.engine.topology import get_source_inputs
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.data_utils import get_file

import tensorflow as tf

TF_WEIGHTS_PATH = 'https://www.dropbox.com/s/lvczhq0fw0497ft/alexnet_weights.h5?dl=1'

def AlexNet(include_top=True, weights='imagenet',
             input_tensor=None, input_shape=None,
             pooling=None,
             classes=1000):
    """Instantiates the Alex architecture.
        
    Default image size is 227x227x3.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization),
            'imagenet' (pre-training on ImageNet),
            or the path to the weights file to be loaded.
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(227, 227, 3)`.
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 71.
            E.g. `(150, 150, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.
    """
    if not (weights in {'imagenet', None} or os.path.exists(weights)):
        raise ValueError('The `weights` argument should be either '
                         '`None` (random initialization), `imagenet` '
                         '(pre-training on ImageNet), '
                         'or the path to the weights file to be loaded.')
    if weights == 'imagenet' and include_top and classes != 1000:
        raise ValueError('If using `weights` as imagenet with `include_top`'
                         ' as true, `classes` should be 1000')
    if K.backend() != 'tensorflow':
        raise RuntimeError('The Xception model is only available with '
                           'the TensorFlow backend.')
    if K.image_data_format() != 'channels_last':
        warnings.warn('The CaffeNet model is only available for the '
                      'input data format "channels_last" '
                      '(width, height, channels). '
                      'However your settings specify the default '
                      'data format "channels_first" (channels, width, height). '
                      'You should set `image_data_format="channels_last"` in your Keras '
                      'config located at ~/.keras/keras.json. '
                      'The model being returned right now will expect inputs '
                      'to follow the "channels_last" data format.')
        K.set_image_data_format('channels_last')
        old_data_format = 'channels_first'
    else:
        old_data_format = None

    # Determine proper input shape
    input_shape = _obtain_input_shape(input_shape,
                                      default_size=227,
                                      min_size=71,
                                      data_format=K.image_data_format(),
                                      require_flatten=False,
                                      weights=None)

    if input_tensor is None:
        img_input = Input(shape=input_shape)
    else:
        if not K.is_keras_tensor(input_tensor):
            img_input = Input(tensor=input_tensor, shape=input_shape)
        else:
            img_input = input_tensor

    
    # Ensure that the model takes into account
    # any potential predecessors of `input_tensor`.
    if input_tensor is not None:
        inputs = get_source_inputs(input_tensor)
    else:
        inputs = img_input

    def ccn(alpha=1e-4, k=2, beta=0.75, n=5, **kwargs):
        """Cross Channel Normalisation used in the original paper."""
        def f(X):
            _, _, _, ch = X.shape
            half = n // 2
            square = K.square(X)
            extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 2, 3, 1)), padding=((0, 0), (half, half)))
            extra_channels = K.permute_dimensions(extra_channels, (0, 3, 1, 2))
            scale = k
            for i in range(n):
                scale += alpha * extra_channels[:, :, :, i:i + ch.value]
            scale = scale ** beta
            return X / scale

        return Lambda(f, output_shape=lambda input_shape: input_shape, **kwargs)

    def split_conv2D(x, filters, kernel_size, strides=(1, 1), activation='relu', name=None):
        x1 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[0])(x)
        x2 = Lambda(lambda x: tf.split(x, num_or_size_splits=2, axis=-1)[1])(x)
        x1 = Conv2D(filters, kernel_size, strides=strides, activation=activation, name=name+'_1')(x1)
        x2 = Conv2D(filters, kernel_size, strides=strides, activation=activation, name=name+'_2')(x2)
        return Concatenate(axis=-1, name=name)([x1, x2])

    # 1st Layer: Conv (w ReLu)
    x = Conv2D(96, (11, 11), strides=(4, 4), activation='relu', name='conv_1')(img_input)
    
    # 2nd Layer: Conv Pool -> Lrn -> Conv (w Relu)
    x = ccn()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = ZeroPadding2D(padding=(2,2))(x)
    x = split_conv2D(x, 128, (5, 5), name='conv_2')

    # 3rd Layer: Pool -> Lrn -> Conv (w Relu)
    x = ccn()(x)
    x = MaxPooling2D(pool_size=(3,3), strides=(2,2))(x)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = Conv2D(384, (3, 3), strides=(1, 1), name='conv3')(x)

    # 4th Layer: Conv (w ReLu)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = split_conv2D(x, 192, (3, 3), name='conv_4')

    # 5th Layer: Conv (w Relu)
    x = ZeroPadding2D(padding=(1,1))(x)
    x = split_conv2D(x, 128, (3, 3), name='conv_5')

    # 6th Layer: Conv (w ReLu) -> Pool -> Flatten -> Dense
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = Flatten(name='flatten')(x)
    x = Dense(4096, activation='relu', name='dense_1')(x)
    x = Dropout(0.5)(x)

    # 7th Later: Dense
    x = Dense(4096, activation='relu', name='dense_2')(x)
    x = Dropout(0.5)(x)

    # 8th Layer: Softmax
    x = Dense(classes, activation='softmax', name="output")(x)

    # Create model.
    model = Model(inputs=inputs, outputs=x, name='alexnet')

    # Load the weights.
    if weights == 'imagenet':
        weights_path = get_file('alexnet_weights.h5',
                                TF_WEIGHTS_PATH,
                                cache_subdir='models',
                                file_hash='6b7ccd8989354b8b2f30f442dc0c6a9e')

        model.load_weights(weights_path)

    # Remove top if required.
    if not include_top:
        model.layers.pop() # Remove Softmax
        model.layers.pop() # Remove Dropout
        model.layers.pop() # Remove Dense 2
        model.layers.pop() # Remove Dropout
        model.layers.pop() # Remove Dense 1
        model.outputs = [model.layers[-1].output]
        model.layers[-1].outbound_nodes = []

    # Restore old image format if it was different.
    if old_data_format:
        K.set_image_data_format(old_data_format)
    return model