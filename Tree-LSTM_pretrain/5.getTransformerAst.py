"""
Put the constructed data set into the pre-training model to get the overall AST hidden state corresponding to each code
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


def get_pre_output(lstmSort, tree1, tree2):
    predictions, ast1_state, ast2_state = lstmSort(tree1,
                                                   tree2)
    avg_output = tf.reduce_mean(ast1_state, axis=0)
    avg_output = avg_output.numpy()
    res.append(avg_output)


if __name__ == '__main__':
    # dataName = 'Java'  # Default is Java dataset, you can also modify to Python
    dataName = 'Python'
    d_model = 512
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                      beta_2=0.98, epsilon=1e-9)
    ast_i2w = read_pickle(dataName + "/ast_i2w_" + dataName + '.pkl')
    ast_w2i = read_pickle(dataName + "/ast_w2i_" + dataName + '.pkl')
    lstmSort = LSTMSort(d_model, d_model, len(ast_i2w))
    best_model = "./checkpoints_" + dataName + "/train/ckpt-1"
    ckpt = tf.train.Checkpoint(model=lstmSort, optimizer=optimizer)
    ckpt.restore(best_model)
    # batch_size = 1024
    dataType = ['test', 'valid', 'train']
    for type in dataType:
        tst_ast = dataName + '/' + type + '/' + type + '_pre_ast_' + dataName + '.pkl'
        file = dataName + '/' + type + ".token.ast"
        res = []
        tst_ast = read_pickle(tst_ast)
        for i in tqdm(range(len(tst_ast))):
            cur_trees = tst_ast[i]
            ast1 = [consult_tree(n, ast_w2i) for n in cur_trees]
            tree1 = tree2tensor(ast1)
            tree2 = tree1
            get_pre_output(lstmSort, tree1, tree2)
        res = np.array(res)
        np.savetxt(file, res, delimiter=' ')
