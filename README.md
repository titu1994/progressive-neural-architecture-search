# Progressive Neural Architecture Search with Encoder RNN

Basic implementation of Encoder RNN from [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559).

- Uses Keras to define and train children / generated networks, which are found via sequential model-based optimization in Tensorflow, ranked by the Encoder RNN.
- Define a state space by using `StateSpace`, a manager which maintains input states and handles communication between the Encoder RNN and the user.
- `Encoder` manages the training and evaluation of the Encoder RNN
- `NetworkManager` handles the training and reward computation of the children Keras model

# Usage
At a high level : For full training details, please see `train.py`.
```python
# construct a state space (the default operators are from the paper)
state_space = StateSpace(B, operators=None)  # B = number of blocks in each cell

# create the managers
controller = Encoder(tf_session, state_space, B, K)  # K = number of children networks to train after initial step
manager = NetworkManager(dataset, epochs=max_epochs, batchsize=batchsize)

# For `B` number of trials
  actions = controller.get_actions(K)  # get all the children model to train in this trial

  For each `child` in action
    store reward = manager.get_reward(child) in `rewards` list

  encoder.train(rewards)  # train encoder RNN with a surrogate loss function
  encoder.update()  # build next set of children to train in next trial, and sort them
```

# Implementation details
This is a very limited project.
- It is not a faithful re-implementation of the original paper. There are several small details not incorporated (like bias initialization, actually using the Hc-2 - Hcb-1 values etc)
- It doesnt have support for skip connections via 'anchor points' etc. (though it may not be that hard to implement it as a special state)
- Learning rate, regularization strength etc are all random values (which make somewhat sense to me)
- Single GPU model only. There would need to be a **lot** of modifications to this for multi GPU training (and I have just 1)

# Result
I tried a toy CNN model with 2 CNN cells the default search space, train for just 1 epoch of training on CIFAR-10.

The top score was for the model `0.3766,-1,1x7-7x1 conv,-1,1x7-7x1 conv`, which obtained slightly higher score than `0.3663,-1,1x7-7x1 conv,-1,3x3 avgpool,0,3x3 avgpool,0,3x3 avgpool,-2,1x7-7x1 conv,-2,1x7-7x1 conv`
This may have just been due to training noise, since it was trained for just 1 epoch.

<img src="https://github.com/titu1994/progressive-neural-architecture-search/blob/master/images/losses.PNG?raw=true" height=100% width=100%>

# Requirements
- Keras >= 2.1.2
- Tensorflow-gpu >= 1.2

# Acknowledgements
Code somewhat inspired by [wallarm/nascell-automl](https://github.com/wallarm/nascell-automl)
