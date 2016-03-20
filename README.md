Neural Variational Document Model
=================================

Tensorflow implementation of [Neural Variational Inference for Text Processing](http://arxiv.org/abs/1511.06038).

![model_demo](./assets/model.png)


Prerequisites
-------------

- Python 2.7 or Python 3.3+
- [NLTK](http://www.nltk.org/)
- [TensorFlow](https://www.tensorflow.org/)


Usage
-----

To train a model with Penn Tree Bank dataset:

    $ python main.py --dataset ptb

To test an existing model:

    $ python main.py --dataset ptb --forward_only True


Results
-------

Training details of NVDM. The best result can be achieved by onehost updates, not alternative updates.

![scalar](2016-03-20-final-scalar.png)

![histogram](2016-03-20-final-hist.png)

Author
------

Taehoon Kim / [@carpedm20](http://carpedm20.github.io/)
