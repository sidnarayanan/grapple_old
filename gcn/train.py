from __future__ import division
from __future__ import print_function

import time
import tensorflow as tf
from glob import glob 
import numpy as np 
import scipy.sparse as sp 
import scipy.linalg 

from gcn.utils import *
from gcn.models import GCN, MLP

# Settings
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_string('dataset','/local/grapple_1/', 'Data directory') 
flags.DEFINE_string('model', 'gcn', 'Model string.')  # 'gcn', 'gcn_cheby', 'dense'
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('epochs', 20, 'Number of epochs to train.')
flags.DEFINE_integer('max_batch', 1000, 'Maximum number of batches per epoch.')
flags.DEFINE_integer('hidden1', 16, 'Number of units in hidden layer 1.')
flags.DEFINE_float('dropout', 0.5, 'Dropout rate (1 - keep probability).')
flags.DEFINE_float('weight_decay', 5e-4, 'Weight for L2 loss on embedding matrix.')
flags.DEFINE_integer('early_stopping', 10, 'Tolerance for early stopping (# of epochs).')
flags.DEFINE_integer('max_degree', 1, 'Maximum polynomial degree.')

# Load data
class Standardizer(object):
    def __init__(self):
        self._mu = None
        self._sigmainv = None
    def __call__(self, x):
        if self._mu is None:
            self._mu = x.mean(axis=0)
            self._sigmainv = np.divide(1, x.std(axis=0))
        return  (x - self._mu) * self._sigmainv 

if FLAGS.model == 'gcn':
    to_support = lambda x : [preprocess_adj(x)]
    num_supports = 1 
else:
    to_support = lambda x : chebyshev_polynomials(x, FLAGS.max_degree)
    num_supports = 1 + FLAGS.max_degree

class _Dataset(object):
    def __init__(self, *args, **kwargs):
        pass
    @staticmethod
    def _transform(x, A, y):
        x = sp.lil_matrix(x) 
        x = preprocess_features(x)

        supp = to_support(A)

        y = y.reshape(-1,1)

        return x, supp, y

class FakeDataset(_Dataset):
    def __init__(self, n_nodes=10):
        self.n_nodes = n_nodes
        super(FakeDataset, self).__init__()
    def gen(self, N=FLAGS.max_batch):
        for _ in range(N):
            split = np.random.randint(2, self.n_nodes-2) 
            bndrys = [(0, split), (split+1, self.n_nodes)]
            xs, ys, As = [], [], [] 
            for lo,hi in bndrys:
                n = hi - lo
                xs.append(np.ones((n, 1)))
                ys.append(np.ones(n)*len(ys))
                A = np.random.binomial(n=1, p=0.5, size=(n*n)).reshape(n,n).astype(bool)
                A += A.T 
                As.append(A)
            x = np.concatenate(xs).astype(np.float32)
            y = np.concatenate(ys).astype(np.int64)
            A = scipy.linalg.block_diag(*As)

            yield self._transform(x, A, y)

# Define placeholders
placeholders = {
    'support': [tf.sparse_placeholder(tf.float32) for _ in range(num_supports)],
    'features': tf.sparse_placeholder(tf.float32, shape=(None, 1)),
    'labels': tf.placeholder(tf.float32, shape=(None, 1)),
    'labels_mask': tf.placeholder(tf.int32),
    'dropout': tf.placeholder_with_default(0., shape=()),
    'num_features_nonzero': tf.placeholder(tf.int32)  # helper variable for sparse dropout
}

# Create model
model = GCN(placeholders, input_dim=1, logging=True)

# Initialize session
sess = tf.Session()


# Define model evaluation function
def evaluate(features, support, labels, mask, placeholders):
    t_test = time.time()
    feed_dict_val = construct_feed_dict(features, support, labels, mask, placeholders)
    outs_val = sess.run([model.loss, model.accuracy], feed_dict=feed_dict_val)
    return outs_val[0], outs_val[1], (time.time() - t_test)


# Init variables
sess.run(tf.global_variables_initializer())

cost_val = []

# Train model
for epoch in range(FLAGS.epochs):

    ds = FakeDataset()

    for features, support, y_train in ds.gen():
        t = time.time()
        train_mask = np.ones_like(y_train)
        # Construct feed dictionary
        feed_dict = construct_feed_dict(features, support, y_train, train_mask, placeholders)
        feed_dict.update({placeholders['dropout']: FLAGS.dropout})

        # Training step
        outs = sess.run([model.opt_op, model.loss, model.accuracy], feed_dict=feed_dict)
    # Print results
    print("Epoch:", '%04d' % (epoch + 1), "train_loss=", "{:.5f}".format(outs[1]),
          "train_acc=", "{:.5f}".format(outs[2]))

