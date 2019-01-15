# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import datetime
from tensorflow.contrib.learn.python.learn.datasets.mnist import DataSet
from tensorflow.contrib.learn.python.learn.datasets import base
from tensorflow.python.framework import dtypes
from tensorflow.examples.tutorials.mnist import input_data
from OWMLayer_2Layers import OWMLayer
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # ignore warning
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # use gpu0,1

# Parameters
# ==================================================
tf.flags.DEFINE_integer("num_checkpoints", 3, "Number of checkpoints to store (default: 5)")
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")
tf.app.flags.DEFINE_string("buckets", "", "")
tf.app.flags.DEFINE_string("checkpointDir", "", "oss info")
tf.flags.DEFINE_integer("num_class", 10, "")
tf.flags.DEFINE_integer("batch_size", 40, "Batch Size (default: 64)")
tf.flags.DEFINE_integer("epoch", 20, "")
FLAGS = tf.flags.FLAGS
# ==================================================

mnist = input_data.read_data_sets("./data/MNIST_data/", one_hot=True)
def split_mnist(mnist, cond):
    sets = ["train", "validation", "test"]
    sets_list = []
    for set_name in sets:
        this_set = getattr(mnist, set_name)
        maxlabels = np.argmax(this_set.labels, 1)
        sets_list.append(DataSet(this_set.images[cond(maxlabels),:], this_set.labels[cond(maxlabels)],
                                 dtype=dtypes.uint8, reshape=False))
    return base.Datasets(train=sets_list[0], validation=sets_list[1], test=sets_list[2])


def train(mnist_list):
    # Training
    # ==================================================
    g1 = tf.Graph()
    middle = 800
    with g1.as_default():
        OWM = OWMLayer([[784 + 1, middle], [middle + 1, 10]], seed_num=79)

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    test_array = []
    with tf.Session(graph=g1, config=config) as sess1:
        # Initialize all variables
        init = [tf.global_variables_initializer(), tf.local_variables_initializer()]
        sess1.run(init)
        task_num = 10
        for j in range(0, task_num):
            print("Training Disjoint MNIST %d" % (j + 1))
            # Update the parameters

            # Update the parameters
            epoch_owm = FLAGS.epoch
            batch_size_owm = FLAGS.batch_size
            all_data = len(mnist_list[j].train.labels[:])
            all_step = all_data*epoch_owm//batch_size_owm
            for current_step in range(all_step):
                lamda = current_step/all_step
                current_step = current_step+1
                batch_xs, batch_ys = mnist_list[j].train.next_batch(batch_size_owm)
                feed_dict = {
                    OWM.input_x: batch_xs,
                    OWM.input_y: batch_ys,
                    OWM.lr_array: np.array([[0.2]]),
                    OWM.alpha_array: np.array([[0.9 * 0.001 ** lamda, 0.6]]),
                }
                acc, loss,  _, = sess1.run([OWM.accuracy, OWM.loss, OWM.back_forward], feed_dict,)
                if current_step % (all_step // 2) == 0:
                    print("Train->>>Task: [{:d}/{:d}] Step: {:d}/{:d} Train: loss: {:.2f}, acc: {:.2f}  %"
                          .format(j+1, task_num, current_step*epoch_owm // all_step+1,
                                  epoch_owm, loss, acc * 100))
            print("Test on Previous Datasets:")
            correct = 0
            total = 0
            for i_test in range(j + 1):
                feed_dict = {
                    OWM.input_x: mnist_list[i_test].test.images[:],
                    OWM.input_y: mnist_list[i_test].test.labels[:],
                }
                accu, = sess1.run([OWM.accuracy], feed_dict)
                total += np.shape(mnist_list[i_test].test.labels[:])[0]
                correct += round(np.shape(mnist_list[i_test].test.labels[:])[0] * accu)
            test_accu = 100 * correct / total
            test_array.append(test_accu)
            print("Test:->>>[{:d}/{:d}], acc: {:.2f} %".format(j + 1, task_num, test_accu))
        feed_dict = {
            OWM.input_x: mnist.test.images[:],
            OWM.input_y: mnist.test.labels[:],
        }
        accu, loss = sess1.run([OWM.accuracy, OWM.loss], feed_dict)
        print("accu_owm {:g} %\n".format(accu * 100))
        print(test_array)


def main(_):
    # Create 10 disjoint MNIST datasets
    mnist_list = [split_mnist(mnist, lambda x: x < 1)]
    for i in range(9):
        mnist_list.append(split_mnist(mnist, lambda x: (i < x) & (x < i+2)))
    train(mnist_list)


if __name__ == '__main__':
    tf.app.run()
