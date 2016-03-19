import os
import numpy as np
import tensorflow as tf

from utils import *
from collections import Counter

class TextReader(object):
  def __init__(self, data_path, batch_size, num_steps=1):
    self.batch_size = batch_size
    self.num_steps = num_steps

    train_path = os.path.join(data_path, "train.txt")
    valid_path = os.path.join(data_path, "valid.txt")
    test_path = os.path.join(data_path, "test.txt")
    vocab_path = os.path.join(data_path, "vocab.pkl")

    if os.path.exists(vocab_path):
      self._load(vocab_path, train_path, valid_path, test_path)
    else:
      self._build_vocab(train_path, vocab_path)
      self.train_data = self._file_to_data(train_path)
      self.valid_data = self._file_to_data(valid_path)
      self.test_data = self._file_to_data(test_path)

    self.vocab_size = len(self.vocab)

  def _read_text(self, file_path):
    with open(file_path) as f:
      return f.read().replace("\n", "<eos>").split()

  def _build_vocab(self, file_path, vocab_path):
    counter = Counter(self._read_text(file_path))

    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))
    words, _ = list(zip(*count_pairs))
    self.vocab = dict(zip(words, range(len(words))))

    save_pkl(vocab_path, self.vocab)

  def _file_to_data(self, file_path):
    text = self._read_text(file_path)
    data = np.array(map(self.vocab.get, text))

    save_npy(file_path + ".npy", data)
    return data

  def _load(self, vocab_path, train_path, valid_path, test_path):
    self.vocab = load_pkl(vocab_path)

    self.train_data = load_npy(train_path + ".npy")
    self.valid_data = load_npy(valid_path + ".npy")
    self.test_data = load_npy(test_path + ".npy")

  def next_batch(self, data_type="train"):
    if data_type == "train":
      raw_data = self.train_data
    elif data_type == "valid":
      raw_data = self.valid_data
    elif data_type == "test":
      raw_data = self.test_data
    else:
      raise Exception(" [!] Unkown data type %s: %s" % data_type)

    self.batch_cnt = len(raw_data) // self.batch_size

    for idx in range(self.batch_cnt):
      yield np.bincount(raw_data[self.batch_size*idx:self.batch_size*(idx+1)], minlength=self.vocab_size)
