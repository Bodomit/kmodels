"""Siamese model for keras. 
Takes an arbitray model and siamese-ifies it.
Partially based on https://github.com/NVIDIA/keras/blob/master/examples/mnist_siamese_graph.py
"""

from keras.models import Model
from keras.layers import Input, Lambda
from keras import backend as K

def Siamese(base_network, distance_metric='euclidian'):
    """Instanciates a siamese network.

    # Arguments
        base_network: the network that will form each branch of the siamese network.
        distance_metric: the distance metric that used to combine the two branches.
    """
    # Get teh distance metric.
    if distance_metric == 'euclidian':
        dist_func = _euclidean_distance
    else:
        raise ValueError("Distance metric not valid.")

    # Get input shape to base network and create new inputs for siamese network.
    input_shapes = [tuple([d.value for d in i.shape[1:]]) for i in base_network.inputs]

    inputs_A = [Input(shape=input_shape, name="InputA") for input_shape in input_shapes]
    inputs_B = [Input(shape=input_shape, name="InputB") for input_shape in input_shapes]

    # Pass inputs to each branch of the siamese net.
    network_a = base_network(inputs_A)
    network_b = base_network(inputs_B)

    # Get the distance between the two using the chosen metric.
    distance = Lambda(dist_func, output_shape=_euclidean_output_shape, name="dist")([network_a, network_b])

    # Create the model object and return.
    inputs = [item for sublist in [inputs_A, inputs_B] for item in sublist]
    return Model(inputs=inputs, outputs=distance)

# Loss function to be used with siamese network.
def contrastive_loss(y_true, y_pred):
    '''Contrastive loss from Hadsell-et-al.'06
    http://yann.lecun.com/exdb/publis/pdf/hadsell-chopra-lecun-06.pdf
    '''
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))

# Distance Metrics
def _euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.sum(K.square(x - y), axis=1, keepdims=True))

def _euclidean_output_shape(shapes):
    shape1, _ = shapes
    return (shape1[0], 1)

# Other metrics.
def accuracy(y_true, y_pred, threshold = 0.5):
    '''Compute classification accuracy with a fixed threshold on distances.
    '''
    return K.mean(K.equal(y_true, K.cast(y_pred < threshold, y_true.dtype)))


# Adhoc test script.
if __name__ == '__main__':
    from alexnet import AlexNet
    input_shape = (227,227,3)
    base = AlexNet(input_shape=input_shape, include_top=False)
    siamese = Siamese(base)
    siamese.summary()
