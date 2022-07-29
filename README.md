#### Repository description

This is an (unsuccessful) attempt at reproducing double descent, in particular Figure 4 from the paper ["Reconciling modern machine learning practice
and the bias-variance trade-off"](https://arxiv.org/pdf/1812.11118.pdf).

#### Double descent

One might expect that the test loss of a model decreases with increasing model capacity. However, some people observed the so-called "double descent phenomenon" where the test loss *increases* for a shot while, just to fall off again with even larger model capacity.

The squared loss should look like this (Figure 4 from paper):

![](https://raw.githubusercontent.com/99991/double-descent/main/figures/paper_double_descent.png)

#### Results

My unsuccessful attempt at reproducing this figure looks like this:

![](https://raw.githubusercontent.com/99991/double-descent/main/figures/double_descent.png)

Unfortunately, no double descent can be observed.

#### Usage

1. Install PyTorch, torchvision and matplotlib
2. Run `python double_descent_mnist.py` to generate `log.json`.
3. Run `plot_log.py` to generate `double_descent.png` and to show the resulting plot.

#### Notes

* The "squared loss" is probably not the same as the mean squared error over the one-hot-encoded label differences. Instead, the sum over the differences divided by the number of samples is more likely.
* The train loss in the author's Figure 4 is slightly lower than the train loss in this reproduction attempt.
* Extremely strong overfitting could possibly explain the bad test performance in the double descent region, but I was not able to find an optimizer which could overfit that much.
