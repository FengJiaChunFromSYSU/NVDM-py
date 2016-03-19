import time
import numpy as np
import tensorflow as tf

from base import Model

class NVDM(Model):
  """Neural Varational Document Model"""

  def __init__(self, sess, reader, dataset="ptb",
               batch_size=20, num_steps=3, embed_dim=500,
               h_dim=50, learning_rate=0.01, max_iter=450000,
               checkpoint_dir="checkpoint"):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: The name of dataset to use.
      h_dim: The dimension of document representations (h). [50, 200]
    """
    self.sess = sess
    self.reader = reader

    self.h_dim = h_dim
    self.embed_dim = embed_dim

    self.max_iter = max_iter
    self.batch_size = batch_size
    self.learning_rate = learning_rate
    self.checkpoint_dir = checkpoint_dir

    self.step = tf.Variable(0, trainable=False)  

    self.dataset = "ptb"
    self._attrs = ["h_dim", "embed_dim", "max_iter", "batch_size", "num_steps", "learning_rate"]

    self.build_model()

  def build_model(self):
    self.x = tf.placeholder(tf.float32, [self.reader.vocab_size], name="input")

    self.build_encoder()
    self.build_decoder()

    # Kullback Leibler divergence
    self.e_loss = -0.5 * tf.reduce_sum(1 + self.log_sigma_sq - tf.square(self.mu) - tf.exp(self.log_sigma_sq))

    # Log likelihood
    self.g_loss = tf.reduce_sum(tf.log(self.p_x_i + 1e-10))

    self.loss = tf.reduce_mean(self.e_loss + self.g_loss)
    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(-self.loss)

    _ = tf.scalar_summary("encoder loss", self.e_loss)
    _ = tf.scalar_summary("decoder loss", self.g_loss)
    _ = tf.scalar_summary("loss", self.loss)

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    with tf.variable_scope("encoder"):
      l1 = tf.nn.relu(tf.nn.rnn_cell.linear(tf.expand_dims(self.x, 0), self.embed_dim, bias=True, scope="l1"))
      l2 = tf.nn.relu(tf.nn.rnn_cell.linear(l1, self.embed_dim, bias=True, scope="l2"))

      self.mu = tf.nn.rnn_cell.linear(l2, self.h_dim, bias=True, scope="mu")
      self.log_sigma_sq = tf.nn.rnn_cell.linear(l2, self.h_dim, bias=True, scope="log_sigma_sq")

      eps = tf.random_normal((1, self.h_dim), 0, 1, dtype=tf.float32)
      sigma = tf.sqrt(tf.exp(self.log_sigma_sq))

      _ = tf.histogram_summary("mu", self.mu)
      _ = tf.histogram_summary("sigma", sigma)

      self.h = self.mu + sigma * eps

  def build_decoder(self):
    """Inference Network. p(X|h)"""
    with tf.variable_scope("decoder"):
      R = tf.get_variable("R", [self.reader.vocab_size, self.h_dim])
      b = tf.get_variable("b", [self.reader.vocab_size])

      x_i = tf.diag([1.]*self.reader.vocab_size)

      e = -tf.matmul(tf.matmul(self.h, R, transpose_b=True), x_i) + b
      self.p_x_i = tf.squeeze(tf.nn.softmax(e))

  def train(self, config):
    merged_sum = tf.merge_all_summaries()
    writer = tf.train.SummaryWriter("./logs/%s" % self.get_model_dir(), self.sess.graph_def)

    tf.initialize_all_variables().run()
    self.load(self.checkpoint_dir)

    start_time = time.time()
    start_iter = self.step.eval()

    iterator = self.reader.next_batch()
    for step in xrange(start_iter, start_iter + self.max_iter):
      x = iterator.next()

      _, loss, e_loss, g_loss, summary_str = self.sess.run(
          [self.optim, self.loss, self.e_loss, self.g_loss, merged_sum], feed_dict={self.x: x})

      if step % 2 == 0:
        writer.add_summary(summary_str, step)

      if step % 10 == 0:
        print("Step: [%4d/%4d] time: %4.4f, loss: %.8f, e_loss: %.8f, g_loss: %.8f" \
            % (step, self.max_iter, time.time() - start_time, loss, e_loss, g_loss))

      if step % 1000 == 0:
        self.save(self.checkpoint_dir, step)

  def sample(self):
    """Sample the documents."""
    p = 1
    x, word_idxs = self.reader.random()
    print " ".join([self.reader.idx2word[word_idx] for word_idx in word_idxs])

    for idx in xrange(20):
      cur_ps = self.sess.run([self.p_x_i], feed_dict={self.x: x})
      cur_p, word_idx = np.max(cur_ps), np.argmax(cur_ps)

      print self.reader.idx2word[word_idx], cur_p
      p *= cur_p

    print("perplexity : %8.f" % np.log(p))
