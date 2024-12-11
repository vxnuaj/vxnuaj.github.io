---
layout: post
title:  "Depthwise Separable Convolutions"
date:   2024-11-11 04:08:20 -0800
categories: deeplearning
---

Depthwise Convolutions are the key building block for efficient neural networks.
<br/>
You factorize a convolution as:
<br/>
1. A grouped Convolution, where the groups is equal to the number of total channels (1 $\mathcal{K}$ per channel). 
<br/>
2. Aftward, taking the independently learned features from each input channel, you learn the relationships between those through a pointwise, $1 \times 1$ convolution.
<br/>

<div align = 'center'>
<img src = 'https://ars.els-cdn.com/content/image/1-s2.0-S2214317322000026-gr2.jpg'/>
</div>

<br/>

Standard Convs have a computational complexity of 

$$
\mathcal{O}(X_{\text{h}} \times X_{\text{w}} \times X_{\text{ch}} \times \mathcal{K}_{\text{ch}} \times \mathcal{K}_h \times \mathcal{K}_w)
$$

while Depthwise Seperable Convs have a computational compelxity of:

$$
\mathcal{O}(X_{\text{h}} \times X_{\text{w}} \times X_{\text{ch}} \times \mathcal{K}_h \times \mathcal{K}_w) + \mathcal{O}(X_{\text{h}} \times X_{\text{w}} \times X_{\text{ch}} \times \mathcal{K}^{1 \times 1}_{\text{ch}})
$$

Considering an input of size $3 \times 7 \times 7$, a standard convolution with $\mathcal{K}$ of shape $3 \times 3 \times 3$ (assuming $\text{stride} = 1$ and $\text{padding} = 0$, $\rightarrow$ $3 \times 5 \times 5$) would yield:

$$
3 \times 7 \times 7 \times 3 \times 3 \times 3 = 3,969 \text{ multi-adds}
$$

while a depthwise seperable convolution, of same output size ($3 \times 5 \times 5$), would yield:

$$
(3 \times 7 \times 7 \times 3 \times 3) + (5 \times 5 \times 3 \times 3) = 1,323 + 225 = 1,548 \text{ multi-adds}
$$

Clearly the depthwise seperable convolutions have a lower computational cost than a regular full-sized convolution, being $\text{256}\%$ more efficient, and empirically showing near same accuracy, albeit with a tiny bit of decreased performance.