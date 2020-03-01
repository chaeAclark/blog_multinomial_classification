# Multinomial Classification
This repo is the model and evaluation code related to the blog post [here](https://www.twosixlabs.com/an-all-vs-all-scheme-for-deep-learning/).

The evaluate_models.py file has code that will train the One-vs-All, All-vs-All, and Hierarchical models.

The multinomial_class.py contains the class code for the All-vs-All, and Hierarchical models.
The multinomial_class_orig.py contains similar class code for the All-vs-All, and Hierarchical models. The only difference between the two are that "..._orig.py" freezes every layer except the output layer, while "multinomial_class.py" trains the entire model.
