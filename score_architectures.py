import numpy as np

import tensorflow as tf

from encoder import ControllerManager, StateSpace

tf.enable_eager_execution()

B = 5  # number of blocks in each cell
K = None  # number of children networks to train
INPUT_B = 3  # number of blocks in each cell during training

MAX_EPOCHS = 3  # maximum number of epochs to train
BATCHSIZE = 128  # batchsize
CHILD_MODEL_LR = 0.001  # learning rate for the child models.
REGULARIZATION = 0  # regularization strength
CONTROLLER_CELLS = 100  # number of cells in RNN controller
RNN_TRAINING_EPOCHS = 10  # number of epochs to train the controller
RESTORE_CONTROLLER = True  # restore controller to continue training

operators = ['3x3 dconv', '5x5 dconv', '7x7 dconv',
             '1x7-7x1 conv', '3x3 maxpool', '3x3 avgpool']  # use the default set of operators, minus identity and conv 3x3

operators = ['3x3 maxpool', '1x7-7x1 conv']  # mini search space

# construct a state space
state_space = StateSpace(B, input_lookback_depth=0, input_lookforward_depth=0,
                         operators=operators)

# print the state space being searched
state_space.print_state_space()

# create the ControllerManager and build the internal policy network
controller = ControllerManager(state_space, B=B, K=K,
                               train_iterations=RNN_TRAINING_EPOCHS,
                               reg_param=REGULARIZATION,
                               controller_cells=CONTROLLER_CELLS,
                               input_B=INPUT_B,
                               restore_controller=RESTORE_CONTROLLER)

# train for number of trails
for trial in range(B):
    if trial == 0:
        k = None
    else:
        k = K

    controller.update_step()
    print()

print("Finished")