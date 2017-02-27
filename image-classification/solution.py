import problem_unittests as tests
from os.path import isfile, isdir
import helper
import numpy as np
import tensorflow as tf


cifar10_dataset_folder_path = 'cifar-10-batches-py'

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """

    # get x min (darkest pixel in the image) per each color channel
    x_min = np.min(x, axis=(0, 1))
    # get x max (brightest pixel in the image) per each color channel
    x_max = np.max(x, axis=(0, 1))
    delta = x_max - x_min
    return (x - x_min) / delta

def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector
    for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    batch_size = len(x)
    num_targets = 10
    output = np.zeros((batch_size, num_targets))
    output.flat[np.arange(batch_size)*num_targets + x] = 1
    return output

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    tensor_shape = (None, image_shape[0], image_shape[1], image_shape[2])
    return tf.placeholder(dtype=tf.float32, shape=tensor_shape)

def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    return tf.placeholder(dtype=tf.float32, shape=(None, n_classes))

def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    return tf.placeholder(dtype=tf.float32, name='keep_prob')

tf.reset_default_graph()

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides,
                   pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor of shape?
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    n_channels = int(x_tensor.shape[3])
    weights = tf.Variable(tf.truncated_normal(
        [conv_ksize[0], conv_ksize[1], n_channels, conv_num_outputs]))
    biases = tf.Variable(tf.truncated_normal([conv_num_outputs]))

    conv_output = tf.nn.conv2d(x_tensor, weights,
                               strides=[1, conv_strides[0], conv_strides[1], 1],
                               padding='SAME')
    conv_output = tf.nn.bias_add(conv_output, biases)
    conv_output = tf.nn.relu(conv_output)
    mp_output = tf.nn.max_pool(conv_output,
                               ksize=[1, pool_ksize[0], pool_ksize[1], 1],
                               strides=[1, pool_strides[0], pool_strides[1], 1],
                               padding='SAME'
                               )
    return mp_output

def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    #batch_size = int(x_tensor.shape[0])
    input_size = np.prod(x_tensor.get_shape().as_list()[1:])
    return tf.reshape(tensor=x_tensor, shape=[-1, input_size])

def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    input_width = int(x_tensor.shape[1])
    weights = tf.Variable(tf.truncated_normal(shape=(input_width, num_outputs)))
    biases = tf.Variable(tf.zeros(shape=num_outputs))
    fc_output = tf.matmul(x_tensor, weights) + biases
    # I assume non-linearity is required
    fc_output = tf.nn.relu(fc_output)
    return fc_output

def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    input_width = int(x_tensor.shape[1])
    weights = tf.Variable(tf.truncated_normal(shape=(input_width, num_outputs)))
    biases = tf.Variable(tf.zeros(shape=num_outputs))
    logits = tf.matmul(x_tensor, weights) + biases
    return logits

def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

    # output should be ~(16, 16, 16)
    layer_1 = conv2d_maxpool(x_tensor=x,
                             conv_num_outputs=16,
                             conv_ksize=(3, 3),
                             conv_strides=(1, 1),
                             pool_ksize=(2, 2),
                             pool_strides=(2, 2))

    # output should be ~(8, 8, 32)
    layer_2 = conv2d_maxpool(x_tensor=layer_1,
                             conv_num_outputs=32,
                             conv_ksize=(3, 3),
                             conv_strides=(1, 1),
                             pool_ksize=(2, 2),
                             pool_strides=(2, 2))

    # output should be ~(4, 4, 64)
    layer_3 = conv2d_maxpool(x_tensor=layer_2,
                             conv_num_outputs=64,
                             conv_ksize=(3, 3),
                             conv_strides=(1, 1),
                             pool_ksize=(2, 2),
                             pool_strides=(2, 2))

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)

    l3_flat = flatten(x_tensor=layer_3)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)

    fc_1 = fully_conn(x_tensor=l3_flat, num_outputs=32)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)


    # TODO: return output
    return output(x_tensor=fc_1, num_outputs=10)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    # TODO: Implement Function
    session.run(optimizer, feed_dict={x: feature_batch,
                                      y: label_batch,
                                      keep_prob: keep_probability})

def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """
    # TODO: Implement Function
    loss = session.run(cost, feed_dict={x: feature_batch,
                                        y: label_batch,
                                        keep_prob: 1.0})

    valid_acc = session.run(accuracy, feed_dict={x: valid_features,
                                                 y: valid_labels,
                                                 keep_prob: 1.0})

    print('Loss: {:.4f} Validation Accuracy: {:.6f}'.format(loss, valid_acc))

# TODO: Tune Parameters
epochs = 10
batch_size = 64
keep_probability = 0.75

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
print('Checking the Training on a Single Batch...')
with tf.Session() as sess:
    # Initializing the variables
    sess.run(tf.global_variables_initializer())

    # Training cycle
    for epoch in range(epochs):
        batch_i = 1
        for batch_features, batch_labels in helper.load_preprocess_training_batch(batch_i, batch_size):
            train_neural_network(sess, optimizer, keep_probability, batch_features, batch_labels)
        print('Epoch {:>2}, CIFAR-10 Batch {}:  '.format(epoch + 1, batch_i), end='')
        print_stats(sess, batch_features, batch_labels, cost, accuracy)
