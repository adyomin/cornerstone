# Dropout quiz

import tensorflow as tf

hidden_layer_weights = [
    [0.1, 0.2, 0.4],
    [0.4, 0.6, 0.6],
    [0.5, 0.9, 0.1],
    [0.8, 0.2, 0.8]]
out_weights = [
    [0.1, 0.6],
    [0.2, 0.1],
    [0.7, 0.9]]

# Weights and biases
weights = [
    tf.Variable(hidden_layer_weights),
    tf.Variable(out_weights)]
biases = [
    tf.Variable(tf.zeros(3)),
    tf.Variable(tf.zeros(2))]

# Input
features = tf.Variable([[0.0, 2.0, 3.0, 4.0],
                        [0.1, 0.2, 0.3, 0.4],
                        [11.0, 12.0, 13.0, 14.0]]
                       )

# TODO: Create Model with Dropout

keep_prob = tf.placeholder(dtype=tf.float32)

hidden_input = tf.add(tf.matmul(features, weights[0]), biases[0])
hidden_output = tf.nn.relu(features=hidden_input)
hidden_output = tf.nn.dropout(x=hidden_output, keep_prob=keep_prob)

output = tf.add(tf.matmul(hidden_output, weights[1]), biases[1])
# output = tf.nn.dropout(x=output, keep_prob=keep_prob)

# TODO: Print logits from a session

with tf.Session() as session:
    session.run(fetches=tf.global_variables_initializer())
    print(session.run(fetches=output, feed_dict={keep_prob: 0.5}))
