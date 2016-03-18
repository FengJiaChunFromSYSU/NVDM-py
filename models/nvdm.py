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

    self.loss = None

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    lambda_ = tf.nn.relu(tf.nn.rnn_cell.linear(self.input_, self.embed_dim, bias=True))
    phi = tf.nn.relu(tf.nn.rnn_cell.linear(lambda_, self.embed_dim, bias=True))

    self.mu = tf.nn.rnn_cell.linear(phi, self.h_dim, bias=True)
    self.sigma = tf.exp(tf.rnn_cell.linear(phi, self.h_dim, bias=True))
    eps = tf.random_normal((self.batch_size, self.h_dim), 0, 1, dtype=tf.float32)

    self.h = self.mu + self.sigma * eps

  def build_decoder(self):
    """Inference Network. p(X|h)"""
    R = tf.Variable()
    b = tf.Variable()
    e = tf.exp(-self.h * R * self.input_) + b

    p_x_h = tf.nn.softmax(e)

  def train(self, config):
    pass

  def sample(self, sample_size=10):
    """Sample the documents."""
    sample_h = np.random.uniform(-1, 1, size=(self.sample_size , self.h_dim))

    p_X_h = p_x_h.run(feed_dict={self.h: sample_h})
