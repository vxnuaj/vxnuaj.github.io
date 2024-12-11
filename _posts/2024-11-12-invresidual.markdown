---
layout: post
title: On the Effectiveness of Inverted Residual Blocks
date: 2024-11-12 04:08:20 -0800
categories: deeplearning
---

> Note: <br/>
> I'm referring to the channel-wise dimension. <br/>
> Also, $\hspace{1.5mm}\mathbb{R}^m < \mathbb{R}^n$

Considering a Neural Network, $\mathcal{F}$, as a Manifold Learner, for a set of input feature maps, $X_i \in \mathbb{R}^n$, to layer $\ell_i \in L$, where $L = \set{\ell_1, \ell_2, \dots, \ell_n}$, the non-linear activation, $H$, after the convolution operation in $\ell_i$, tends to embed the weighted sum, $Z_i \in \mathbb{R}^n$, into a lower dimensional space, $\mathbb{R}^m$, where $\mathbb{R}^m$ is a subspace of $\mathbb{R}^n$.  

> Intuitively, $H$ discards of irrelevant features and extracts relevant features.

Thereby, for a convolution with a kernel $\mathcal{K_i}$, applied onto $X \in \mathbb{R}^n$, we can easily reduce the dimensionality of $X$ to $\mathbb{R}^m$, to reduce the computational cost, while still retaining the important features. Such can be the case, given the prior that $H$ embeds features in the subspace of $\mathbb{R}^n$, as $\mathbb{R}^m$.

You could reduce the dimensionality, through the width multiplier $k$, introduced in MobileNetV1 -- that is until your depth is of the same representation as underlying manifold. But the non-linear activation, $H = \text{ReLU}$, can break down the effectiveness of doing so.

Assume $H = \text{ReLU}$, where $\text{ReLU} = \text{max}(0, z)$. <br/>For an input to $\text{ReLU}$, $Z_i$, if $Z_i > 0$, $\text{ReLU}(Z_i) = Z_i$, else $\text{ReLU}(Z_i) = 0$.

Therefore, after applying $H$ to the high dimensional structure $\in \mathbb{R}^n$, we'll lose information that is negative valued, them being $0$ed out, while the information that is $> 0$ will pass through as the $I$ transformation, remaining a linear transformation (given by the prior convolution).

If the input to $\text{ReLU}$ has the important features lying on a $m$-dimensional manifold in $\mathbb{R}^n$, $\text{ReLU}$ will extract meaningful features $> 0$, while discarding of the irrelevance ($< 0$), while still remaining in $\mathbb{R}^n$.

Therefore you'll still have feature redundancy in the channel dimension after $H$, even when applying a width multiplier $k$, given that $H$ is applied to the **outputs** of a convolution, which in the context of MobileNetV1, have widths of $k \cdot \text{ch}_{out}$.

We can use the fact that if the essential features, which are the positive valued outputs to the convolution, them essentially being a linear combination between $X$ and $\mathcal{K}$ at position of $X$ at $i, j$, given that $H$ performs the $I$ transformation for positive valued inputs while non-essential features are zeroed out, resulting in the retained information of the output to $H$ lying on a subspace $\in \mathbb{R}^n$, to then rationalize the use of linear bottleneck layers, meaning $1 \times 1$ convolutions as depthwise parametric pooling or projection layers, to embed the high dimensional features $\in \mathbb{R}^n \rightarrow \mathbb{R}^m$, without using $H$ after the projection.

> For the inverted Residual Block presented in MobileNetV2, this is used in the latter 2 layers of the 3, while the former is a $1 \times 1$ expansion layer.  
> 
> You (1) expand the low dimensional representation to a high dimensional space, (2) learn the important features via the depthwise 3x3 conv and ReLU6, (3) then project the learnt features in the high dimensional space to the low dimensional space.

### Inverted Residuals

A residual block goes as:

$$ 
\text{BottleNeck} \rightarrow 3 \times 3 \rightarrow \text{Expansion} \rightarrow \text{Output}
$$

where $\text{Output} = Z + X$, where $Z$ is the output of the expansion layer and $X$ is the input to the residual block.

In the case of MobileNet, their inverted residual blocks go as:

$$
\text{Expansion} \rightarrow 3 \times 3 \text{ Depthwise} \rightarrow \text{BottleNeck} \rightarrow \text{Output}
$$

where $\text{Output} = X + Z$ (as prior). 

The inverted residual connection connnects the 

$$\text{BottleNeck}_{\ell-1}$$ 

to the 

$$\text{BottleNeck}_{\ell}$$

via an element wise summation, instead of connecting 

$$\text{Expansion}_{\ell - 1}$$ 

to 

$$\text{Expansion}_{\ell}$$ 

as done in residual blocks.

This is inspired by the aformentioned, *"We can use the fact that if the essential features, which are the positive valued outputs to the convolution, them essentially being a linear combination between $X$ and $\mathcal{K}$ at position of $X$ at $i, j$, given that $H$ performs the $I$ transformation for positive valued inputs while non-essential features are zeroed out, resulting in the retained information of the output to $H$ lying on a subspace $\in \mathbb{R}^n$, to then rationalize the use of linear bottleneck layers, meaning $1 \times 1$ convolutions as depthwise parametric pooling or projection layers, to embed the high dimensional features $\in \mathbb{R}^n \rightarrow \mathbb{R}^m$, without using $H$ after the projection."*.

Thereby, for outputs to a given inverted residual block, you'll end up with a lower dimensional feature space, $\mathbb{R}^m$, rather than, $\mathbb{R}^n$.

The alternative scenario would be a $1 \times 1$ conv, acting as a means to learn representations across feature maps, without reduction into a lower dimensional feature space, and then applying a non-linearity, which was shown to hold redundant features across channels, given that $\text{ReLU}$ cancels out features, resulting in important features lying on a lower dimensional subspace.

This allows the future computations to focus on purely the important features $\in \mathbb{R}^m$, **without** the redundant parameters that focus on sparse representations $\in \mathbb{R}^n$ ultimately being computationally efficient, especially on edge devices where inference cost needs to me reduced drastically for proper functionality.