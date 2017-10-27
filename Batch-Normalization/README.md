# Batch Normalization

Ankur (@ankur6ue) presented on [Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift](https://arxiv.org/abs/1502.03167) by Sergey Ioffe and Christian Szegedy.

The bullet points that Ankur went over included:

Here's what we'll go through the reading group meeting today:

1. Core Idea: We know that centering the input - i.e., making the input batch zero mean and unit variance, and whitening the input distribution (using PCA for example) is beneficial for training. Whatever transformation is applied to the input during training must also be applied during inference as well
2. Since each layer of a network can be considered as the beginning of a new network, the output of the previous layers should be normalized as well, so each layer sees a normalized input, not just the first layer
3. Normalization can also help address vanishing gradient problem by ensuring that the distribution of the input to the non-linear layer (softmax/ReLU) doesn't change much
4. Full whitening of the output of each layer is very expensive
5. Instead, lets make each component of all batch inputs zero mean and unit variance. The mean/variance is computed for the current minibatch, instead for the entire training set.
6. Add a scale factor and bias for each activation. These parameters are learnable parameters. The original bias parameter in the weights can now be removed. Looks like we increased the overall number of parameters though??
7. We'll go over the derivation of backprop equations
8. After the network is trained, scale the lambda/beta for each layer by aggregate (or running average) training data statistics

Additionally, we went over covariate shift and batch optimization with gradient descent.

## Open Questions
