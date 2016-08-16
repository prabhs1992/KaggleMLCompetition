from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from sklearn import preprocessing
from sklearn.decomposition import PCA, IncrementalPCA

import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
matplotlib.use("Agg") #Needed to save figures
from sklearn import cross_validation
import time


import numpy as np
import tensorflow as tf

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq

#the below code is a modification of the PTB word model tutorial on the tensor flow website.
#the tutorial has been modified in a way to run the rnn and train on the Santander data set.

flags = tf.flags
logging = tf.logging

flags.DEFINE_string(
    "model", "small",
    "A type of model. Possible options are: small, medium, large.")
flags.DEFINE_string("data_path", None, "data_path")

FLAGS = flags.FLAGS


class ProjModel(object):

  def __init__(self, is_training, config):
    self.batch_size = batch_size = config.batch_size
    self.num_steps = num_steps = config.num_steps
    size = config.hidden_size
    vocab_size = config.vocab_size

    self._input_data = tf.placeholder(tf.float32, [batch_size, num_steps, size])
    self._targets = tf.placeholder(tf.int32, [batch_size, num_steps])

    # Slightly better results can be obtained with forget gate biases
    # initialized to 1 but the hyperparameters of the model would need to be
    # different than reported in the paper.
    lstm_cell = rnn_cell.BasicLSTMCell(size, forget_bias=0.0)
    if is_training and config.keep_prob < 1:
      lstm_cell = rnn_cell.DropoutWrapper(
          lstm_cell, output_keep_prob=config.keep_prob)
    cell = rnn_cell.MultiRNNCell([lstm_cell] * config.num_layers)

    self._initial_state = cell.zero_state(batch_size, tf.float32)

    inputs = self._input_data
    outputs = []
    states = []
    state = self._initial_state
    with tf.variable_scope("RNN"):
      for time_step in range(num_steps):
        if time_step > 0: tf.get_variable_scope().reuse_variables()
        (cell_output, state) = cell(inputs[:, time_step, :], state)
        outputs.append(cell_output)
        states.append(state)

    output = tf.reshape(tf.concat(1, outputs), [-1, size])
    logits = tf.nn.xw_plus_b(output,
                             tf.get_variable("softmax_w", [size, vocab_size]),
                             tf.get_variable("softmax_b", [vocab_size]))
    loss = seq2seq.sequence_loss_by_example([logits],
                                            [tf.reshape(self._targets, [-1])],
                                            [tf.ones([batch_size * num_steps])],
                                            vocab_size)
    self._cost = cost = tf.reduce_sum(loss) / batch_size
    self._final_state = states[-1]
    self._output = output
    self._logits = logits

    if not is_training:
      return

    self._lr = tf.Variable(0.0, trainable=False)
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars),
                                      config.max_grad_norm)
    optimizer = tf.train.GradientDescentOptimizer(self.lr)
    self._train_op = optimizer.apply_gradients(zip(grads, tvars))

  def assign_lr(self, session, lr_value):
    session.run(tf.assign(self.lr, lr_value))

  @property
  def input_data(self):
    return self._input_data

  @property
  def targets(self):
    return self._targets

  @property
  def initial_state(self):
    return self._initial_state

  @property
  def cost(self):
    return self._cost

  @property
  def output(self):
    return self._output

  @property
  def logits(self):
    return self._logits

  @property
  def final_state(self):
    return self._final_state

  @property
  def lr(self):
    return self._lr

  @property
  def train_op(self):
    return self._train_op


class SmallConfig(object):
  """Small config."""
  init_scale = 0.1
  learning_rate = 1.0
  max_grad_norm = 5
  num_layers = 1
  num_steps = 20
  hidden_size = 142
  max_epoch = 6
  max_max_epoch = 13
  keep_prob = 1.0
  lr_decay = 0.5
  batch_size = 20
  vocab_size = 2


def run_epoch(bool_test, session, m, data, label, eval_op, verbose=False):
  """Runs the model on the given data."""
  epoch_size = ((len(data) // m.batch_size)) // m.num_steps
  start_time = time.time()
  costs = 0.0
  iters = 0
  state = m.initial_state.eval()

  ip_data = []
  ip_label = []

  for i in range(((len(data) // m.batch_size)) // m.num_steps):
    temp3d = []
    templ3d = []
    for j in range(m.num_steps):
        temp = data[i*(m.batch_size*m.num_steps)+j*(m.batch_size):i*(m.batch_size*m.num_steps)+(j+1)*m.batch_size,:]
        temp = temp[np.newaxis,:,:]
        if len(label) != 0:
            templ = label[i*(m.batch_size*m.num_steps)+j*(m.batch_size):i*(m.batch_size*m.num_steps)+(j+1)*m.batch_size]
            templ = np.transpose(templ)
        if len(temp3d) == 0:
            temp3d = temp
            if len(label) != 0:
                templ3d = templ
        else:
            temp3d = np.concatenate((temp3d,temp),axis=0)
            if len(label) != 0:
                templ3d = np.concatenate((templ3d,templ),axis=0)
    temp3d = temp3d[np.newaxis,:,:,:]
    if len(label) != 0:
        templ3d = templ3d[np.newaxis,:,:]
    if len(ip_data) == 0:
        ip_data = temp3d
        if len(label) != 0:
            ip_label = templ3d
    else:
        ip_data = np.concatenate((ip_data,temp3d),axis=0)
        if len(label) != 0:
            ip_label = np.concatenate((ip_label,templ3d),axis=0)

  all_logits = []
  for step in range(epoch_size):
    x = ip_data[step]
    if not bool_test:
        y = ip_label[step]
        cost, state, _, output, logits = session.run([m.cost, m.final_state, eval_op, m.output, m.logits],
                                                     {m.input_data: x,
                                                      m.targets: y,
                                                      m.initial_state: state})
        costs += cost
    else:
        state, _, output, logits = session.run([m.final_state, eval_op, m.output, m.logits],
                                                     {m.input_data: x,
                                                      m.initial_state: state})

    iters += m.num_steps
    if bool_test == 1:
      if len(all_logits) == 0:
        all_logits = logits
      else:
        all_logits = np.concatenate((all_logits,logits),axis = 0)
    if verbose and step % (epoch_size // 10) == 10:
      print("%.3f perplexity: %.3f speed: %.0f wps" %
            (step * 1.0 / epoch_size, np.exp(costs / iters),
             iters * m.batch_size / (time.time() - start_time)))

  if bool_test == 0:
    return np.exp(costs / iters)
  else:
    if len(label) > 0:
        exp_logits = np.exp(all_logits)
        sum_exp = np.sum(exp_logits,axis =1)
        prob = np.zeros((len(exp_logits),2))
        for i in range(len(exp_logits)):
            prob[i,0] = exp_logits[i,0] / sum_exp[i]
            prob[i,1] = exp_logits[i,1] / sum_exp[i]

        max = np.argmax(exp_logits,axis=1)
        #logical_arr = max == ip_label #takes a lot of time so removing this
        correct_pred = 0
        for i in range(len(max)):
            if max[i] == ip_label[i]:
                correct_pred += 1
        return (100*correct_pred) / len(ip_label)
    else:
        exp_logits = np.exp(all_logits)
        sum_exp = np.sum(exp_logits, axis=1)

        prob = np.zeros((len(exp_logits),2))
        for i in range(len(exp_logits)):
            prob[i,0] = exp_logits[i,0] / sum_exp[i]
            prob[i,1] = exp_logits[i,1] / sum_exp[i]

        return prob


def get_config():
  if FLAGS.model == "small":
    return SmallConfig()

def main(unused_args):
    data = pd.read_csv('/home/prabhanjan/Downloads/te/train.csv', sep=",")
    data = data.replace(-999999, 2)
    numCol = len(data.columns)
    numRow = len(data)
    target = []
    target = data["TARGET"]
    finData = []
    id = []
    id = data["ID"]
    finData = data.iloc[:, 1:numCol - 1]
    numCol = len(finData.columns)
    numRow = len(finData)

    data_test = pd.read_csv('/home/prabhanjan/Downloads/te/test.csv', sep=",")
    data_test = data.replace(-999999, 2)
    id_test = []
    id_test = data_test["ID"]
    numColt = len(data_test.columns)
    numRowt = len(data_test)
    finDatat = []
    finDatat = data_test.iloc[:, 1:numColt - 1]
    numColt = len(finDatat.columns)
    numRowt = len(finDatat)

    scaler = preprocessing.StandardScaler().fit(finData)
    train_scaled = scaler.transform(finData)
    test_scaled = scaler.transform(finDatat)

    p = PCA(n_components=train_scaled.shape[1])
    p.fit(train_scaled)
    trainX = p.transform(train_scaled)
    testX = p.transform(test_scaled)

    tr = trainX[:, 0:142]
    te = testX[:, 0:142]
    
    train_complete = tr

    X_train, X_test, y_train, y_test = \
       cross_validation.train_test_split(train_complete, target, random_state=1301, stratify=target, test_size=0.35)
    sel_test = te

    config = get_config()
    valid_config = get_config()
    valid_config.batch_size = 1
    valid_config.num_steps = 1

    with tf.Graph().as_default(), tf.Session() as session:

        initializer = tf.random_uniform_initializer(-config.init_scale,
                                                    config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            m = ProjModel(is_training=True, config=config)

        valid_config = config
        valid_config.batch_size = 1
        valid_config.num_steps = 1
        with tf.variable_scope("model", reuse=True, initializer=initializer):
            mvalid = ProjModel(is_training=False, config=valid_config)
            mtest = ProjModel(is_training=False, config=valid_config)

        tf.initialize_all_variables().run()

        for i in range(config.max_max_epoch):
            lr_decay = config.lr_decay ** max(i - config.max_epoch, 0.0)
            m.assign_lr(session, config.learning_rate * lr_decay)

            print("Epoch: %d Learning rate: %.3f" % (i + 1, session.run(m.lr)))
            train_perplexity = run_epoch(0, session, m, tr, target, m.train_op)
            print("Epoch: %d Train Perplexity: %.3f" % (i + 1, train_perplexity))
            #validation code has been commented out it looks as below:
            # valid_perplexity = run_epoch(session, mvalid, valid_data, tf.no_op())
            # print("Epoch: %d Valid Perplexity: %.3f" % (i + 1, valid_perplexity))
            # print(session.run(m._final_state))
        #v_acc = run_epoch(1, session, mvalid, npX_test, npy_test, tf.no_op())
        #print("Validation accuracy: %.3f" % v_acc)
        y_pred = run_epoch(1, session, mtest, sel_test, [], tf.no_op())
        submission = pd.DataFrame({"ID":id_test, "TARGET":y_pred[:,1]})
        submission.to_csv("/home/prabhanjan/Downloads/te/submission.csv", index=False)

if __name__ == "__main__":
  tf.app.run()

