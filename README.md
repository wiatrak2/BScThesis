# Experiments with unsupervised domain adaptation using the gradient reversal layer

This repository contains my engineer's thesis, which I had been writing by February 2019, during the final term of my engineer's degree studies at The University of Wroclaw. It can be found [here](https://github.com/wiatrak2/BScThesis/blob/master/thesis.pdf).
## Unsupervised domain adaptation
Domain adaptation is a specific training process, which is used to obtain a well-performing model learned with a dataset other (but related) than the target one. For instance, we can adjust the network trained on SVHN images  to classify a new, unlabeled dataset of digits from football jerseys photos.
![digits](https://github.com/wiatrak2/BScThesis/blob/master/images/digits.png?raw=true)
![football digits](https://github.com/wiatrak2/BScThesis/blob/master/images/football.png?raw=true)
## Gradient Reversal Layer
The thesis is about dealing with unsupervised domain adaptation with some new modifications of the gradient reversal layer (GRL)  [introduced by Ganin and Lempitsky](https://arxiv.org/abs/1409.7495). The authors' network tries to remove as much domain-specific information as possible with *domain classifier*, which is a specific part of the model's architecture.
![GRL architecture](https://github.com/wiatrak2/BScThesis/blob/master/images/GRLarch.png?raw=true)
With a simple trick during backpropagation the model achieved some satisfying results. Within the thesis I firstly reproduced the paper and then verified some intuitions presented by the paper authors.
## Model's modifications
After the paper reproduction I introduced some modifications of the proposed architecture, such as plugging the *domain classifier* to the layers of the *label predictor*
![domain predictor plugging](https://github.com/wiatrak2/BScThesis/blob/master/images/domain_v.png?raw=true) This approach reached higher accuracy than the original one and also filtered out more domain features. Some other network's adjustments described in the thesis were trying to explain the core of the domain adaptation problem and the GRL mechanism.
## Visualizations
To get a better understanding of achieved results and the unsupervised domain adaptation I made some visualizations of the datasets transformed by a learned model. Few of them were really surprising, while others clearly confirm our intuitions. Here are some examples:

 - Source domain (MNIST) transformed by learned model

![MNIST with GRL](https://github.com/wiatrak2/BScThesis/blob/master/images/MNIST_2D_GRL.png?raw=true)

- Target domain (MNIST-M) transformed by a simple model, learned without GRL
![MNIST-M no GRL](https://github.com/wiatrak2/BScThesis/blob/master/images/MNIST_M_2D.png?raw=true)
- Target domain (MNIST-M) transformed by the model learned with GRL. The better performance of this network against the previous one is really extraordinary.
![MNIST-M 2D GRL](https://github.com/wiatrak2/BScThesis/blob/master/images/MNIST_M_2D_GRL.png?raw=true)
- Visualization of the samples that was predicted most accurately
![Best predictions](https://github.com/wiatrak2/BScThesis/blob/master/images/best_pred1.png?raw=true)
- Visualization of the target domain (MNIST dataset) modification by the model learned with SVHN images
![SVHN](https://github.com/wiatrak2/BScThesis/blob/master/images/SVHN_GRL2.png?raw=true)

To learn more about my experiments I highly encourage you to read the whole thesis :)
