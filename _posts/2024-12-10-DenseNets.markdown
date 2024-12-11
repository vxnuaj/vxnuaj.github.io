---
layout: post
title:  "DenseNets"
date:   2024-11-08 04:08:20 -0800
categories: deeplearning
---
## Gradient Flow $\in \mathcal{D}$

The forward pass for a DenseBlock can be given as:

$$
H_{\ell} = f_{\ell}([H_0, H_1, ..., H_{\ell - 1}]) \tag{1}
$$

where the argument passed into the current, $\ell$th layer, is a concatenation of all previous outputs with the current input within the given DenseBlock.

> Note that $\ell$ must be $0 < \ell < \mathcal{D}_L$ ($\mathcal{D}_L$ is the total number of layers in the DenseBlock)

To compute the gradient for any given $H_{\ell}$, you first can note that a given $H_{\ell}$, as given in the function above, depends on all $H_{\ell - n}$ (where $0 < n < \ell$).

Also, $H_{\ell}$ contributes to all $H_{i + m}$ (where $0 < m < (L - \ell)$).

$$
H_{i + m} = f_{\ell}([H_0, ... , H_{\ell}, ..., H_{\ell + m - 1}]) \tag{2}
$$

Say we have a loss, $\mathcal{L}$, and we want to take the gradient of the loss w.r.t. $H_{\ell}$.

$$
\frac{∂\mathcal{L}}{∂H_{\ell}}
$$

> *This will be denoted as $∂H_{\ell}$ for simplicity*

To compute $∂H_{\ell}$, we need to account for its gradient $\forall \hspace{1mm} \ell \in L$, in which $H_{\ell}$ contributed to an output.

> So using equation $(2)$, $\forall \hspace{1mm} m$

Then the total gradient (accumulated across all $\ell \in L$, where $H_{\ell}$ had a contribution) is simply a summation of all $∂H_{\ell}$, $\forall \ell \in L$, where $H_{\ell}$ had a contribution.

$$
\text{EQ. (3)}\\[3mm]
\frac{∂\mathcal{L}}{∂H_{\ell}} = \sum_{j = \ell + i}^L \left(\frac{∂\mathcal{L}}{∂[H_{0}, H_{1}, \dots, H_{\ell}, \dots, H_{j-1}]}\right) \left(\frac{[∂H_{0}, ∂H_{1}, \dots, ∂H_{\ell}, \dots, ∂H_{j-1}]}{∂H_{\ell}}\right)
$$

> $\ell$ is equal to the current layer.  
> We do $j = \ell + 1$ in the $\sum$ as $H_{\ell}$ only had a contribution in layers after $\ell$, and not $\ell$.  
> We do $j - 1$ in the denominator of the first factor as if we did $H_L$, it wouldn't make mathematical sense as for the last layer of the DenseBlock ($L$), its output can't contribute to itself.

Taking a look at the overall gradient flow, for $\ell = 3$ in a 5-layer DenseBlock:

$$
\frac{∂\mathcal{L}}{∂H_{\ell}} = \left(\frac{∂\mathcal{L}}{∂H_{\ell + 2}}\right) \left(\frac{∂H_{\ell+2}}{∂H_{\ell + 1}}\right) \left(\frac{∂H_{\ell+1}}{∂H_{\ell}}\right)
$$

It's easy to see how the vanishing gradient can be diminished, as we're accumulating gradients for the respective $\partial H_{\ell+2}, \partial H_{\ell+1}, \partial H_{\ell}$, through equation (3), meaning $\forall \, \ell \in \mathcal{D}_L$ that any given $H_{\ell}$ had a contribution, we include the respective $\partial H_{\ell}$ in the $\sum$, leading to a higher magnitude of gradients, that then gets propagated back **again** to compute every remaining $\partial H_{\ell - j}$, where $0 < j < \ell$.


Albeit, given that we're accumulating gradients, a common conception might be the inverse, exploding gradients.

BatchNormalization exists for mitigating this very issue, such that inputs to the $\text{ReLU}$, don't become so large that $∂$'s begin to explode as we backpropagate deeper, into earlier layers.

## Diversified Training

Coming alongside the improved gradient flow, via concatenation, the other benefit of DenseNet, over Residual Networks, includes their ability to learn more diversified features at every layer.

A given layer is being fed multiple sets of feature maps from previous layers, such that the $\mathcal{K}_{\ell}$ is able to extract correlated features across multiple previous convolutional layers $\in \mathcal{D}$.

The complexity of which a DenseNet hierarchically concatenates previous outputs to the current input can be given by the growth rate, $k$, where $k$ is the total count of output channels for a given $\ell \in \mathcal{D}$.

> *$\forall \ell \in \mathcal{D}$, the amount of channels to the input of layer $\ell$ grows by $k$.*

The hyperparameter $k$ is what allows you to easily control the complexity of the DenseNet, across its entire architecture.