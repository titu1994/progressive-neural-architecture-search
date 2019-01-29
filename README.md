# Progressive Neural Architecture Search with ControllerManager RNN

Basic implementation of ControllerManager RNN from [Progressive Neural Architecture Search](https://arxiv.org/abs/1712.00559).

- Uses tf.keras to define and train children / generated networks, which are found via sequential model-based optimization in Tensorflow, ranked by the Controller RNN.
- Define a state space by using `StateSpace`, a manager which maintains input states and handles communication between the ControllerManager RNN and the user.
- `ControllerManager` manages the training and evaluation of the Controller RNN
- `NetworkManager` handles the training and reward computation of the children models

# Usage
At a high level : For full training details, please see `train.py`.
```python
# construct a state space (the default operators are from the paper)
state_space = StateSpace(B, # B = number of blocks in each cell
                         operators=None # whether to use custom operators or the default ones from the paper
                         input_lookback_depth=0, # limit number of combined inputs from previous cell
                         input_lookforward_depth=0, # limit number of combined inputs in same cell
                         )

# create the managers
controller = ControllerManager(state_space, B, K)  # K = number of children networks to train after initial step
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
- Learning rate, number of epochs to train per B_i, regularization strength etc are all random values (which make somewhat sense to me)
- Single GPU model only. There would need to be a **lot** of modifications to this for multi GPU training (and I have just 1)

# Result
I tried a toy CNN model with 2 CNN cells the a custom search space, train for just 5 epoch of training on CIFAR-10.

All models configuration strings can be ranked using `rank_architectures.py` script to parse train_history.csv, or you can use
`score_architectures.py` to pseudo-score all combinations of models for all values of B, and then pass these onto `rank_architectures.py` to approximate the scores that they would obtain.


# Requirements
- Tensorflow-gpu >= 1.12
- Scikit-learn (most recent available from pip)

# Acknowledgements
Code somewhat inspired by [wallarm/nascell-automl](https://github.com/wallarm/nascell-automl)
