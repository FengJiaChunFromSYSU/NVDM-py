import tensorflow as tf

from base import Model

class NVDM(Model):
  """Neural Varational Document Model"""

  def __init__(self, sess, reader, dataset="ptb",
               batch_cnt=10, num_steps=3, input_dim=100,
               h_dim=100, learning_rate=0.01):
    """Initialize Neural Varational Document Model.

    params:
      sess: TensorFlow Session object.
      reader: TextReader object for training and test.
      dataset: the name of dataset to use.
    """
    self.sess = sess
    self.reader = reader
    self.input_dim = input_dim
    self.h_dim = h_dim
    self.learning_rate = learning_rate

    self.dataset="ptb"
    self._attrs=["batch_cnt", "num_steps", "input_dim", "h_dim", "learning_rate"]

  def build_model(self):
    self.input_ = tf.placeholder(tf.float32, name="input")
    self.otuput = tf.placeholder(tf.float32, name="output")

    self.loss = None

  def build_encoder(self):
    """Inference Network. q(h|X)"""
    lambda_ = tf.nn.relu(tf.nn.linear(self.input_))
    phi = tf.nn.relu(tf.nn.linear(lambda_))

    mu = tf.nn.linear(phi)
    sigma = tf.exp(tf.linear(phi))

    self.h = None # sampler

  def build_decoder(self):
    """Inference Network. p(X|h)"""
    R = tf.Variable()
    b = tf.Variable()
    e = tf.exp(-self.h * R * self.input_) + b

    p_x_h = tf.nn.softmax(e)

  def train(self):
    pass

  def sample(self, sample_size=10):
    """Sample the documents."""
    sample_h = np.random.uniform(-1, 1, size=(self.sample_size , self.h_dim))

    p_X_h = p_x_h.run(feed_dict={self.h: sample_h})
