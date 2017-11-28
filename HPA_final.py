import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import time
import os
from pathlib import Path

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)




def weight_variable(shape,astype):
    initial = tf.truncated_normal(shape, stddev=0.01, dtype = astype)
    return tf.Variable(initial, dtype = astype)

def bias_variable(shape,astype):
    initial = tf.constant(0.0, shape=shape, dtype = astype)
    return tf.Variable(initial, dtype = astype)

def nn_example(e, b, data_type):
    learning_rate = 0.5
    epochs = e
    batch_size = b
    input_feature_count = 784
    out_classes = 10
    data_type = data_type

    # Neural network hidden layer variables
    h1 = 50
    h2 = 20

    # declare the training data placeholders
    # input x - for 28 x 28 pixels = 784
    x = tf.placeholder(data_type, [None, input_feature_count])
    # now declare the output data placeholder - 10 digits
    y = tf.placeholder(data_type, [None, out_classes])

    # build the network
    keep_prob_input = tf.placeholder(data_type)
    x_drop = tf.nn.dropout(x, keep_prob=keep_prob_input)

    W_fc1 = weight_variable([input_feature_count, h1], data_type)
    b_fc1 = bias_variable([h1], data_type)
    h_fc1 = tf.nn.relu(tf.matmul(x_drop, W_fc1) + b_fc1)

    keep_prob = tf.placeholder(data_type)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

    W_fc2 = weight_variable([h1, h2],data_type)
    b_fc2 = bias_variable([h2], data_type)
    h_fc2 = tf.nn.relu(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

    h_fc2_drop = tf.nn.dropout(h_fc2, keep_prob)

    W_fc3 = weight_variable([h2, out_classes],data_type)
    b_fc3 = bias_variable([out_classes], data_type)

    # now calculate the hidden layer output - in this case, let's use a softmax activated
    # output layer
    y_ = tf.nn.softmax(tf.matmul(h_fc2_drop, W_fc3) + b_fc3)

    # now let's define the cost function which we are going to train the model on
    y_clipped = tf.clip_by_value(y_, 1e-10, 0.9999999)
    cross_entropy = -tf.reduce_mean(tf.reduce_sum(y * tf.log(y_clipped)
                                                  + (1 - y) * tf.log(1 - y_clipped), axis=1))

    # add an optimiser
    optimiser = tf.train.GradientDescentOptimizer(learning_rate=learning_rate).minimize(cross_entropy)

    # finally setup the initialisation operator
    init_op = tf.global_variables_initializer()

    # define an accuracy assessment operation
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, data_type))

    # add a summary to store the accuracy
    accuracy_sum = tf.summary.scalar('accuracy', accuracy)

    saver = tf.train.Saver()

    merged = tf.summary.merge([accuracy_sum])
    writer = tf.summary.FileWriter(r'C:\Users\anant\dev\repos\HPA')

    test_accuracy = 0
    # start the session
    with tf.Session() as sess:
        # initialise the variables
        sess.run(init_op)
        total_batch = int(len(mnist.train.labels) / batch_size)
        start_time = time.time()
        for epoch in range(max(epochs)+1):
            avg_cost = 0
            train_acc = 0
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size=batch_size)
                train_acc,_, c = sess.run([accuracy, optimiser, cross_entropy], feed_dict={x: batch_x, y: batch_y, keep_prob_input: 1.0, keep_prob: 1.0})
                avg_cost += c / total_batch
                arr_accuracy = [];
                time_elapsed = [];
                nw_size = [];
            if(epoch in epochs):
                elapsed = time.time() - start_time
                filepath = "D:\\UCI\\HPA_Project\\weights_data\\epoch_"+ str(epoch)+ '_batch_' +str(batch_size)
                save_path = saver.save(sess, filepath)
                mfilepath = filepath + ".data-00000-of-00001"
                file = Path(mfilepath)
                size = file.stat().st_size
                test_accuracy = sess.run(accuracy, feed_dict={x: mnist.test.images, y: mnist.test.labels, keep_prob_input: 1.0, keep_prob: 1.0})
                arr_accuracy.append(test_accuracy)
                time_elapsed.append(elapsed)
                nw_size.append(size)
                print(epoch, b, test_accuracy, elapsed)
    return arr_accuracy, time_elapsed, nw_size

if __name__ == "__main__":
    data_type = tf.float32
    batch_size = [50, 100, 200, 500, 1000]
    epochs = [5,10, 15, 20] #, 30, 50] #, 80, 100, 150, 200, 300, 500, 700, 1000]
    num_epochs = len(epochs)
    num_bsize = len(batch_size)
    accuracy_mat = [[] for i in range(num_bsize)]
    elapsed_time = [[] for i in range(num_bsize)]
    size_mat = [[] for i in range(num_bsize)]
    for j, b in enumerate(batch_size):
        accuracy_mat[j], elapsed_time[j], size_mat[j] = nn_example(epochs, b, data_type)
        print()
