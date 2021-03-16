import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
from tensorflow import keras
from utils import *
import tensorflow as tf
from layer import *
import time
import datetime
from tqdm import tqdm

BASE_DIR = os.path.dirname(__file__)


def loss_function(real, pred, loss_object):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)


def get_path(relative_path):
    path = os.path.join(BASE_DIR, relative_path)
    return path


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


def validate(lstmSort, dataset, eval_type='Val'):
    print("*** Running {} step ***".format(eval_type))
    accuracy = keras.metrics.BinaryAccuracy(name='{}_accuracy'.format(eval_type.lower()))
    t = tqdm(dataset)
    for (batch, (tree1, tree2, tar)) in enumerate(t):

        predictions, _, _ = lstmSort(tree1,
                                     tree2)
        # print("prediction:", predictions)
        # print("tar:", tar)
        accuracy(tar, predictions)
        if batch % 10 == 0:
            print('Batch {} Accuracy {:.4f}'.format(batch, accuracy.result()))

    print('\n{} Accuracy {}\n'.format(eval_type, accuracy.result()))


def train_step(tree1, tree2, tar):
    with tf.GradientTape() as tape:
        predictions, _, _ = lstmSort(tree1, tree2)
        loss = loss_function(tar, predictions, loss_object)
    gradients = tape.gradient(loss, lstmSort.trainable_variables)
    optimizer.apply_gradients(zip(gradients, lstmSort.trainable_variables))
    train_loss(loss)
    train_accuracy(tar, predictions)


if __name__ == '__main__':
    d_model = 512
    learning_rate = CustomSchedule(d_model)
    optimizer = keras.optimizers.Adam(learning_rate, beta_1=0.9,
                                      beta_2=0.98, epsilon=1e-9)

    loss_object = keras.losses.BinaryCrossentropy(from_logits=True, reduction='none')

    train_loss = keras.metrics.Mean(name='train_loss')
    train_accuracy = keras.metrics.BinaryAccuracy(name='train_accuracy')

    # dataName = 'Java'  # Default is Java dataset, you can also modify to Python
    dataName = 'Python'

    trn_ast1 = dataName + "/train/train_ast1_" + dataName + ".pkl"
    trn_ast2 = dataName + "/train/train_ast2_" + dataName + ".pkl"
    trn_label = dataName + "/train/train_label_" + dataName + ".pkl"

    vld_ast1 = dataName + "/valid/valid_ast1_" + dataName + ".pkl"
    vld_ast2 = dataName + "/valid/valid_ast2_" + dataName + ".pkl"
    vld_label = dataName + "/valid/valid_label_" + dataName + ".pkl"

    trn_label = read_pickle(trn_label)
    batch_size = 256
    epoch_num = 20

    vld_label = read_pickle(vld_label)

    ast_i2w = read_pickle(dataName + "/ast_i2w_" + dataName + ".pkl")
    ast_w2i = read_pickle(dataName + "/ast_w2i_" + dataName + ".pkl")
    lstmSort = LSTMSort(d_model, d_model, len(ast_i2w))

    checkpoint_path = "./checkpoints_" + dataName + "/train"
    ckpt = tf.train.Checkpoint(model=lstmSort, optimizer=optimizer)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=20)

    if ckpt_manager.latest_checkpoint:
        ckpt.restore(ckpt_manager.latest_checkpoint)
        print('Latest checkpoint restored!!')

    current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    train_log_dir = get_path('logs/gradient_tape/' + current_time + '/train')
    train_summary_writer = tf.summary.create_file_writer(train_log_dir)

    for epoch in range(epoch_num):
        print("epoch ", str(epoch+1), " begin training")
        Datagen = Datagen_tree
        trn_gen = Datagen(trn_ast1, trn_ast2, trn_label, batch_size, ast_w2i, train=True)
        vld_gen = Datagen(vld_ast1, vld_ast2, vld_label, batch_size, ast_w2i, train=False)
        start = time.time()
        train_loss.reset_states()
        train_accuracy.reset_states()

        t = tqdm(trn_gen(0))
        for (batch, (tree1, tree2, tar)) in enumerate(t):
            train_step(tree1, tree2, tar)

            with train_summary_writer.as_default():
                tf.summary.scalar('loss', train_loss.result(), step=epoch)
                tf.summary.scalar('accuracy', train_accuracy.result(), step=epoch)

            if batch % 500 == 0:
                print('Epochs {} Batch {} Loss {:.4f} Accuracy {:.4f}'.format(
                    epoch + 1, batch, train_loss.result(), train_accuracy.result()))

        if (epoch + 1) % 1 == 0:
            ckpt_save_path = ckpt_manager.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1, ckpt_save_path))

        print('Epoch {} Loss {:.4f} Accuracy {:.4f}'.format(epoch + 1, train_loss.result(), train_accuracy.result()))
        print('Time taken for 1 epoch: {} secs\n'.format(time.time() - start))
        validate(lstmSort, vld_gen(0))




