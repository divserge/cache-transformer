import numpy as np
import tensorflow as tf

class LRUCache:
    def __init__(self, hidden_size, max_size=10, batch_size=1):
        '''
        hidden_size: size of the embeddings
        '''
        self.attention_tensor_ = tf.Variable(tf.zeros((batch_size, max_size, hidden_size)))
        self.state_tensor_ = tf.Variable(tf.zeros((batch_size, max_size, hidden_size)))
        self.max_size_ = max_size
        self.batch_size_ = batch_size
        self.mapping_ = []
        for _ in range(batch_size):
            self.mapping_.append(dict())

        self.lru_state_ = [[] for i in range(batch_size)]

    def Query(self, matching_vectors):
        '''
        matching_vector: tensor of size (batch_size x hidden_size) 

        return: vector of size (batch_size x hidden_size)
        '''
        #print(matching_vectors.shape)
        #print(self.state_tensor_.shape)

        query = tf.einsum("ijk,iqk->iqj", self.state_tensor_, matching_vectors)
        weights = tf.nn.softmax(query, dim=2)
        return tf.einsum("iqj,ijk->iqk", weights, self.attention_tensor_)

    def Add(self, tokens, state_vectors, attention_vectors):
        '''
        Function adds the vectors to cachce
        state_vectors: tensor of size (batch_size x ... x hidden_size)
        attention_vectors: tensor of size (batch_size x ... x hidden_size)
        tokens: tensor of size (batch_size x ...)

        return: tf.float32(0)
        '''

        indices, alphas = tf.py_func(self._AddPy, [tokens], (tf.int32, tf.float32))


        indices.set_shape((state_vectors.shape[0] * state_vectors.shape[1], 2))
        alphas.set_shape((state_vectors.shape[0] * state_vectors.shape[1],))

        updates = tf.multiply(alphas[:, None], tf.gather_nd(self.state_tensor_, indices)) + \
            tf.multiply(1 - alphas[:, None], tf.reshape(state_vectors, (-1, tf.shape(state_vectors)[-1])))

        self.state_tensor_ = tf.scatter_nd_update(
            self.state_tensor_,
            indices,
            updates
        )

        #self.state_tensor_[tf.range(self.batch_size_)[:, None], indeces] = \
        #    state_vectors * alphas[:, :, None] + \
        #    self.state_tensor_[tf.range(self.batch_size_)[
        #       :, None], indeces] * (1 - alphas[:, :, None])

        #self.attention_tensor_[tf.range(self.batch_size_)[:, None], indeces] = \
        #    attention_vectors * alphas[:, :, None] + \
        #    self.attention_tensor_[tf.range(self.batch_size_)[
        #        :, None], indeces] * (1 - alphas[:, :, None])
        return tf.cast(0 * tf.reduce_sum(self.state_tensor_), tf.float32)

    def _AddEntry(self, num_batch, token):
        '''
        Returns tensor of size (2, ) with index and alpha params
        '''
        if token == 0:
            pass
        lru_index = len(self.mapping_[num_batch])
        if len(self.mapping_[num_batch]) == self.max_size_:
            lru_key = self.lru_state_[num_batch].pop()
            lru_index = self.mapping_[num_batch].pop(lru_key)

        if token not in self.mapping_[num_batch]:
            self.mapping_[num_batch][token] = lru_index
            index = lru_index
            alpha = 1.
        else:
            index = self.mapping_[num_batch][token]
            alpha = 0.5
            self.lru_state_[num_batch].remove(token)

        self.lru_state_[num_batch].insert(0, token)
        return index, alpha

    def _AddPy(self, keys_batch):
        '''
        keys_batch: tensor with token of size (batch_size x sentence_size)

        return: tensor of size (batch_size x sentence_size)
        '''
        indeces = np.zeros((keys_batch.shape))
        alphas = np.zeros((keys_batch.shape))

        for i in range(self.batch_size_):
            for token_ind in range(len(keys_batch[i])):

                indeces[i, token_ind], alphas[i, token_ind] = self._AddEntry(
                    i, keys_batch[i][token_ind])

        indices = np.hstack((
            np.repeat(
                np.arange(keys_batch.shape[0], dtype=indeces.dtype),
                keys_batch.shape[1],
            ).reshape(-1, 1),
            indeces.reshape(-1, 1)
        ))

        alphas = alphas.reshape(-1)

        return indices.astype(np.int32), alphas.astype(np.float32)

    def Flush(self):
        self.attention_tensor_ = 0.0
        self.state_tensor_ = 0.0
        return tf.reduce_sum(self.attention_tensor_ + self.state_tensor_)