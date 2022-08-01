#### Repository description

This is an (unsuccessful) attempt at testing double descent without changing the training regime, in particular Figure 4 from the paper ["Reconciling modern machine learning practice
and the bias-variance trade-off"](https://arxiv.org/pdf/1812.11118.pdf).

#### Double descent

One might expect that the test loss of a model decreases with increasing model capacity. However, some people observed the so-called "double descent phenomenon" where the test loss *increases* for a shot while, just to fall off again with even larger model capacity.

The squared loss should look like Figure 4 from paper with change in training regime:

![Figure 4](https://raw.githubusercontent.com/99991/double-descent/main/figures/paper_double_descent.png)

Or like Figure 9 (c) from paper without weight reuse:

![Figure 9 c](https://user-images.githubusercontent.com/18725165/182159231-19e10b99-996a-40e8-8e82-672c4baae95f.png)

#### Results

![](https://raw.githubusercontent.com/99991/double-descent/main/figures/double_descent.png)

Unfortunately, no double descent can be observed.

#### Usage

1. Install PyTorch, torchvision and matplotlib
2. Run `python double_descent_mnist.py` to generate `log.json`.
3. Run `plot_log.py` to generate `double_descent.png` and to show the resulting plot.
