from tensorflow import keras
from utils import *

class TreeEmbeddingLayer(keras.layers.Layer):
    def __init__(self, dim_E, in_vocab):
        super(TreeEmbeddingLayer, self).__init__()
        self.E = tf.Variable(initial_value=tf.random.uniform([in_vocab, dim_E]), name="E", dtype=tf.float32)

    def call(self, x):
        '''x: list of [1,]'''
        x_len = [xx.shape[0] for xx in x]
        ex = tf.nn.embedding_lookup(self.E, tf.concat(x, axis=0))
        exs = tf.split(ex, x_len, 0)
        return exs


class ChildSumLSTMLayer(keras.layers.Layer):
    def __init__(self, dim_in, dim_out):
        super(ChildSumLSTMLayer, self).__init__()
        self.dim_in = dim_in
        self.dim_out = dim_out
        self.U_f = tf.keras.layers.Dense(dim_out, use_bias=False)
        self.U_iuo = tf.keras.layers.Dense(dim_out * 3, use_bias=False)
        self.W = tf.keras.layers.Dense(dim_out * 4)
        # self.h_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        # self.c_init = tfe.Variable(
        #     tf.get_variable("h_init", [1, dim_out], tf.float32, initializer=he_normal()))
        self.h_init = tf.zeros([1, dim_out], tf.float32)
        self.c_init = tf.zeros([1, dim_out], tf.float32)

    def call(self, tensor, indices):
        h_tensor = self.h_init
        c_tensor = self.c_init
        res_h, res_c = [], []
        for indice, x in zip(indices, tensor):
            h_tensor, c_tensor = self.apply(x, h_tensor, c_tensor, indice)
            h_tensor = tf.concat([self.h_init, h_tensor], 0)
            c_tensor = tf.concat([self.c_init, c_tensor], 0)
            res_h.append(h_tensor[1:, :])
            res_c.append(c_tensor[1:, :])
        return res_h, res_c

    def apply(self, x, h_tensor, c_tensor, indice):

        mask_bool = tf.not_equal(indice, -1)
        mask = tf.cast(mask_bool, tf.float32)  # [batch, child]

        h = tf.gather(h_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))  # [nodes, child, dim]
        c = tf.gather(c_tensor, tf.where(mask_bool,
                                         indice, tf.zeros_like(indice)))
        h_sum = tf.reduce_sum(h * tf.expand_dims(mask, -1), 1)  # [nodes, dim_out]

        W_x = self.W(x)  # [nodes, dim_out * 4]
        W_f_x = W_x[:, :self.dim_out * 1]  # [nodes, dim_out]
        W_i_x = W_x[:, self.dim_out * 1:self.dim_out * 2]
        W_u_x = W_x[:, self.dim_out * 2:self.dim_out * 3]
        W_o_x = W_x[:, self.dim_out * 3:]

        branch_f_k = tf.reshape(self.U_f(tf.reshape(h, [-1, h.shape[-1]])), h.shape)
        branch_f_k = tf.sigmoid(tf.expand_dims(W_f_x, 1) + branch_f_k)
        branch_f = tf.reduce_sum(branch_f_k * c * tf.expand_dims(mask, -1), 1)  # [node, dim_out]

        branch_iuo = self.U_iuo(h_sum)  # [nodes, dim_out * 3]
        branch_i = tf.sigmoid(branch_iuo[:, :self.dim_out * 1] + W_i_x)   # [nodes, dim_out]
        branch_u = tf.tanh(branch_iuo[:, self.dim_out * 1:self.dim_out * 2] + W_u_x)
        branch_o = tf.sigmoid(branch_iuo[:, self.dim_out * 2:] + W_o_x)

        new_c = branch_i * branch_u + branch_f  # [node, dim_out]
        new_h = branch_o * tf.tanh(new_c)  # [node, dim_out]

        return new_h, new_c


class ChildsumLayer(keras.layers.Layer):
    def __init__(self, dim_E, dim_rep, in_vocab, layer=1):
        super(ChildsumLayer, self).__init__()
        self.layer = layer
        self.in_vocab = in_vocab
        self.E = TreeEmbeddingLayer(dim_E, in_vocab)
        self.dim_rep = dim_rep
        for i in range(layer):
            self.__setattr__("layer{}".format(i), ChildSumLSTMLayer(dim_E, dim_rep))
        print("I am Child-sum model, dim is {} and {} layered".format(
            str(self.dim_rep), str(self.layer)))

    def call(self, x):
        tensor, indice, tree_num = x
        tensor = self.E(tensor)
        # tensor = [tf.nn.dropout(t, 1. - self.dropout) for t in tensor]
        for i in range(self.layer):
            skip = tensor
            tensor, c = getattr(self, "layer{}".format(i))(tensor, indice)
            tensor = [t + s for t, s in zip(tensor, skip)]

        hx = tensor[-1]
        return hx


class LSTMSort(keras.Model):
    def __init__(self, dim_E, dim_rep, in_vocab):
        super(LSTMSort, self).__init__()

        self.childSumLayer1 = ChildsumLayer(dim_E, dim_rep, in_vocab, layer=1)
        self.childSumLayer2 = ChildsumLayer(dim_E, dim_rep, in_vocab, layer=1)
        self.sortLayer = keras.layers.Dense(1, activation='sigmoid')

    def call(self, tree1, tree2):
        tree_state1 = self.childSumLayer1(tree1)
        tree_state2 = self.childSumLayer2(tree2)
        predict = self.sortLayer(tf.concat([tree_state1,tree_state2], 1))

        return predict, tree_state1, tree_state2