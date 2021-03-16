"""
Test the accuracy of the pre-trained model
"""

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tensorflow import keras
from utils import *
import tensorflow as tf
from layer import *
import time
import datetime
from tqdm import tqdm


class CustomSchedule(keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, d_model, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.d_model = d_model
        self.d_model = tf.cast(self.d_model, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)


def validate(lstmSort, dataset, eval_type='test'):
    print("*** Running {} step ***".format(eval_type))
    accuracy = keras.metrics.BinaryAccuracy(name='{}_accuracy'.format(eval_type.lower()))
    t = tqdm(dataset)
    for (batch, (tree1, tree2, tar)) in enumerate(t):

        predictions, ast1_state, ast2_state = lstmSort(tree1,
                                     tree2)
        # print("prediction:", predictions)
        # print("tar:", tar)
        accuracy(tar, predictions)
        if batch % 50 == 0:
            print('Batch {} Accuracy {:.4f}'.format(batch, accuracy.result()))

    print('\n{} Accuracy {}\n'.format(eval_type, accuracy.result()))


if __name__ == '__main__':
    batch_size = 256
    # dataName = 'Java'  # Default is Java dataset, you can also modify to Python
    dataName = 'Python'
    tst_ast1 = dataName + "/test/test_ast1_" + dataName + ".pkl"
    tst_ast2 = dataName + "/test/test_ast2_" + dataName + ".pkl"
    tst_label = dataName + "/test/test_label_" + dataName + ".pkl"
    tst_label = read_pickle(tst_label)

    ast_i2w = read_pickle(dataName + "/ast_i2w_" + dataName + ".pkl")
    ast_w2i = read_pickle(dataName + "/ast_w2i_" + dataName + ".pkl")
    d_model = 512
    lstmSort = LSTMSort(d_model, d_model, len(ast_i2w))
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                      beta_2=0.98, epsilon=1e-9)
    best_model = "./checkpoints_" + dataName +"/train/ckpt-1"    # could change
    # checkpoint_path = "./checkpoints/train"
    ckpt = tf.train.Checkpoint(model=lstmSort, optimizer=optimizer)
    ckpt.restore(best_model)
    Datagen = Datagen_tree
    tst_gen = Datagen(tst_ast1, tst_ast2, tst_label, batch_size, ast_w2i, train=False)
    validate(lstmSort, tst_gen(0))

