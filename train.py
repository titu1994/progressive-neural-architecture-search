import numpy as np
import csv

import tensorflow as tf
from keras import backend as K
from keras.datasets import cifar10, cifar100
from keras.utils import to_categorical

from encoder import Encoder, StateSpace
from manager import NetworkManager
from model import model_fn

# create a shared session between Keras and Tensorflow
policy_sess = tf.Session()
K.set_session(policy_sess)

B = 5  # number of blocks in each cell
K_ = 64  # number of children networks to train

MAX_EPOCHS = 5  # maximum number of epochs to train
BATCHSIZE = 128  # batchsize
REGULARIZATION = 0  # regularization strength
CONTROLLER_CELLS = 100  # number of cells in RNN controller
RESTORE_CONTROLLER = True  # restore controller to continue training

operators = ['3x3 dconv', '5x5 dconv', '7x7 dconv',
             '1x7-7x1 conv', '3x3 maxpool', '3x3 avgpool']  # use the default set of operators, minus identity and conv 3x3

# construct a state space
state_space = StateSpace(B, input_lookback_depth=0, input_lookforward_depth=0,
                         operators=operators)

# print the state space being searched
state_space.print_state_space()
NUM_TRAILS = state_space.print_total_models(K_)

# prepare the training data for the NetworkManager
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
x_train = x_train.astype('float32') / 255.
x_test = x_test.astype('float32') / 255.
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)

dataset = [x_train, y_train, x_test, y_test]  # pack the dataset for the NetworkManager

with policy_sess.as_default():
    # create the Encoder and build the internal policy network
    controller = Encoder(policy_sess, state_space, B=B, K=K_,
                         reg_param=REGULARIZATION,
                         controller_cells=CONTROLLER_CELLS,
                         restore_controller=RESTORE_CONTROLLER)

# create the Network Manager
manager = NetworkManager(dataset, epochs=MAX_EPOCHS, batchsize=BATCHSIZE)
print()

# train for number of trails
for trial in range(B):
    with policy_sess.as_default():
        K.set_session(policy_sess)

        if trial == 0:
            k = None
        else:
            k = K_

        actions = controller.get_actions(top_k=k)  # get all actions for the previous state

    rewards = []
    for t, action in enumerate(actions):
        # print the action probabilities
        state_space.print_actions(action)
        print("Model #%d / #%d" % (t + 1, len(actions)))
        print("Predicted actions : ", state_space.parse_state_space_list(action))

        # build a model, train and get reward and accuracy from the network manager
        reward = manager.get_rewards(model_fn, state_space.parse_state_space_list(action))
        print("Final Accuracy : ", reward)

        rewards.append(reward)
        print("\nFinished %d out of %d models ! \n" % (t + 1, len(actions)))

        # write the results of this trial into a file
        with open('train_history.csv', mode='a+', newline='') as f:
            data = [reward]
            data.extend(state_space.parse_state_space_list(action))
            writer = csv.writer(f)
            writer.writerow(data)

    with policy_sess.as_default():
        K.set_session(policy_sess)
        # train the controller on the saved state and the discounted rewards
        loss = controller.train_step(rewards)
        print("Trial %d: Encoder loss : %0.6f" % (trial + 1, loss))

        controller.update_step()
        print()

print("Finished !")