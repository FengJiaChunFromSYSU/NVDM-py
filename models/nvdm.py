import tensorflow as tf

from base import Model

class NVDM(Model):
  """Neural Varational Document Model"""

  def __init__(self, sess, reader, dataset="ptb",
               batch_size=10, num_steps=3, embed_dim=500,
               h_dim=50, learning_rate=0.01):
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
    self.learning_rate = learning_rate

    self.dataset="ptb"
    self._attrs=["batch_size", "num_steps", "embed_dim", "h_dim", "learning_rate"]

  def build_model(self):
    self.input_ = tf.placeholder(tf.float32, name="input")
    self.otuput = tf.placeholder(tf.float32, name="output")

    # Kullback Leibler divergence
    self.encoder_loss = -0.5 * tf.reduce_sum(self.h_dim + self.log_sigma - tf.square(self.mu) - tf.exp(self.log_sigma))

    # log likelihood
    self.decoder_loss = tf.reduce_sum(tf.log(self.p_x_i))

    self.cost = tf.reduce_mean(self.encoder_loss + self.decoder_loss)
    self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    with tf.variable_scope("encoder"):
      l1 = tf.nn.relu(tf.nn.rnn_cell.linear(self.input_, self.embed_dim, bias=True))
      l2 = tf.nn.relu(tf.nn.rnn_cell.linear(l1, self.embed_dim, bias=True))

      self.mu = tf.nn.rnn_cell.linear(l2, self.h_dim, bias=True)
      self.log_sigma = tf.rnn_cell.linear(l2, self.h_dim, bias=True)

      eps = tf.random_normal((self.batch_size, self.h_dim), 0, 1, dtype=tf.float32)
      self.h = self.mu + self.sigma * eps

  def build_decoder(self):
    """Inference Network. p(X|h)"""
    with tf.variable_scope("decoder"):
      R = tf.get_variable("R", [self.vocab_size, self.hidden_dim])
      b = tf.get_variable("b", [self.vocab_size])

      x_i = tf.diag([1]*self.vocab_size)

      e = -tf.matmul(tf.matmul(self.h, R), x_i) + b
      self.p_x_i = tf.nn.softmax(e)

  def train(self, config):
    pass

  def sample(self, sample_size=10):
    """Sample the documents."""
    sample_h = np.random.uniform(-1, 1, size=(self.sample_size , self.h_dim))

    p_X_h = p_x_h.run(feed_dict={self.h: sample_h})
