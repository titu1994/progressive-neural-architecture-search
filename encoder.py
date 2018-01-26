import numpy as np
import pprint
import csv
from collections import OrderedDict

from keras import backend as K
import tensorflow as tf

import os
if not os.path.exists('weights/'):
    os.makedirs('weights/')


class StateSpace:
    '''
    State Space manager

    Provides utilit functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''
    def __init__(self, B, operators=None):
        self.states = OrderedDict()
        self.state_count_ = 0

        self.children = None
        self.intermediate_children = None

        self.B = B

        if operators is None:
            self.operators = ['identity', '3x3 dconv', '5x5 dconv', '7x7 dconv',
                              '1x7-7x1 conv', '3x3 conv', '3x3 maxpool', '3x3 avgpool']
        else:
            self.operators = operators

        input_values = list(range(-2, self.B))  # -2 = Hc-2, -1 = Hc-1, 0-(B-1) = Hci
        self._add_state('inputs', values=input_values)  # -2 = Hc-2, -1 = Hc-1
        self._add_state('ops', values=self.operators)
        self.prepare_initial_children()

    def _add_state(self, name, values):
        '''
        Adds a "state" to the state manager, along with some metadata for efficient
        packing and unpacking of information required by the RNN Encoder.

        Stores metadata such as:
        -   Global ID
        -   Name
        -   Valid Values
        -   Number of valid values possible
        -   Map from value ID to state value
        -   Map from state value to value ID

        Args:
            name: name of the state / action
            values: valid values that this state can take

        Returns:
            Global ID of the state. Can be used to refer to this state later.
        '''
        index_map = {}
        for i, val in enumerate(values):
            index_map[i] = val

        value_map = {}
        for i, val in enumerate(values):
            value_map[val] = i

        metadata = {
            'id': self.state_count_,
            'name': name,
            'values': values,
            'size': len(values),
            'index_map_': index_map,
            'value_map_': value_map,
        }
        self.states[self.state_count_] = metadata
        self.state_count_ += 1

        return self.state_count_ - 1

    def one_hot_encode(self, id, value):
        '''
        One hot encode the specific state value

        Args:
            id: global id of the state
            value: state value

        Returns:
            one hot encoded representation of the state value
        '''
        state = self[id]
        size = state['size']
        value_map = state['value_map_']
        value_idx = value_map[value]

        one_hot = np.zeros((1, size), dtype=np.float32)
        one_hot[np.arange(1), value_idx] = 1.0
        return one_hot

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        Returns:
            The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']
        value = index_map[index]
        return value

    def parse_state_space_list(self, state_list):
        '''
        Parses a list of one hot encoded states to retrieve a list of state values

        Args:
            state_list: list of one hot encoded states

        Returns:
            list of state values
        '''
        state_values = []
        for id, state_one_hot in enumerate(state_list):
            state_val_idx = np.argmax(state_one_hot, axis=-1)[0]
            value = self.get_state_value(id % 2, state_val_idx)
            state_values.append(value)

        return state_values

    def one_hot_encode_child(self, child):
        encoded_child = []
        for i, val in enumerate(child):
            encoded_child.append(self.one_hot_encode(i % 2, val))

        return encoded_child

    def prepare_initial_children(self):
        inputs = [-2, -1]
        ops = list(range(len(self.operators)))

        print()
        print("Obtaining search space for b = 1")
        print("Search space size : ", (4 * (len(self.operators) ** 2)))

        search_space = [inputs, ops, inputs, ops]
        self.children = list(self._construct_permutations(search_space))

    def prepare_intermediate_children(self, new_b):
        new_ip_values = list(range(-2, new_b))
        ops = list(range(len(self.operators)))

        child_count = ((2 + new_b - 1) ** 2) * (len(self.operators) ** 2)
        print()
        print("Obtaining search space for b = %d" % new_b)
        print("Search space size : ", child_count)

        search_space = [new_ip_values, ops, new_ip_values, ops]
        new_search_space = list(self._construct_permutations(search_space))

        for i, child in enumerate(self.children):
            for permutation in new_search_space:
                temp_child = list(child)
                temp_child.extend(permutation)
                yield temp_child

    def _construct_permutations(self, search_space):
        ''' state space is a 4-tuple (ip1, op1, ip2, op2) '''
        for input1 in search_space[0]:
            for operation1 in search_space[1]:
                for input2 in search_space[2]:
                    for operation2 in search_space[3]:
                        yield (input1, self.operators[operation1], input2, self.operators[operation2])

    def print_state_space(self):
        ''' Pretty print the state space '''
        print('*' * 40, 'STATE SPACE', '*' * 40)

        pp = pprint.PrettyPrinter(indent=2, width=100)
        for id, state in self.states.items():
            pp.pprint(state)
            print()

    def print_actions(self, actions):
        ''' Print the action space properly '''
        print('Actions :')

        for id, action in enumerate(actions):
            state = self[id]
            name = state['name']
            vals = [(n, p) for n, p in zip(state['values'], *action)]
            print("%s : " % name, vals)
        print()

    def update_children(self, children):
        self.children = children

    def __getitem__(self, id):
        return self.states[id % self.size]

    @property
    def size(self):
        return self.state_count_

    def print_total_models(self, K):
        level1 = 2 * 2 * (len(self.operators) ** 2)
        remainder = (self.B - 1) * K
        total = level1 + remainder

        print("Total number of models : ", total)
        print()
        return total



class Encoder:
    '''
    Utility class to manage the RNN Encoder
    '''
    def __init__(self, policy_session, state_space,
                 B=5, K=256,
                 reg_param=0.001,
                 controller_cells=32,
                 restore_controller=False):
        self.policy_session = policy_session  # type: tf.Session

        self.state_space = state_space  # type: StateSpace
        self.state_size = self.state_space.size

        self.b_ = 0
        self.B = B
        self.K = K

        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.restore_controller = restore_controller

        self.build_policy_network()

    def get_actions(self, top_k=None):
        '''
        Gets a one hot encoded action list, either from random sampling or from
        the Encoder RNN

        Args:
            top_k: Number of models to return

        Returns:
            A one hot encoded action list
        '''
        models = self.state_space.children

        if top_k is not None:
            models = models[:top_k]

        actions = []
        for model in models:
            encoded_model = self.state_space.one_hot_encode_child(model)
            actions.append(encoded_model)

        return actions

    def build_policy_network(self):
        with self.policy_session.as_default():
            K.set_session(self.policy_session)

            with tf.name_scope('controller'):
                with tf.variable_scope('policy_network'):

                    # state input is the first input fed into the controller RNN.
                    # the rest of the inputs are fed to the RNN internally
                    with tf.name_scope('state_input'):
                        state_input = tf.placeholder(dtype=tf.float32, shape=(1, None, 1), name='state_input')

                    self.state_input = state_input

                    # we can use LSTM as the controller as well
                    lstm_cell = tf.nn.rnn_cell.LSTMCell(self.controller_cells)
                    cell_state = lstm_cell.zero_state(batch_size=1, dtype=tf.float32)

                    # initially, cell input will be 1st state input
                    cell_input = state_input

                    # we provide a flat list of chained input-output to the RNN
                    with tf.name_scope('controller_input'):
                        # feed the ith layer input (i-1 layer output) to the RNN
                        outputs, final_state = tf.nn.dynamic_rnn(lstm_cell,
                                                                 cell_input,
                                                                 initial_state=cell_state,
                                                                 dtype=tf.float32)

                        # add a new classifier for each layers output
                        regressor = tf.layers.dense(outputs[:, -1, :], units=1, name='rnn_scorer')
                        self.rnn_score = tf.nn.sigmoid(regressor)

            with tf.name_scope('optimizer'):
                self.global_step = tf.Variable(0, trainable=False)
                starter_learning_rate = 0.1
                learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                           500, 0.95, staircase=True)

                tf.summary.scalar('learning_rate', learning_rate)
                self.optimizer = tf.train.RMSPropOptimizer(learning_rate=learning_rate)

            policy_net_variables = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='policy_network')

            with tf.name_scope('losses'):
                self.labels = tf.placeholder(dtype=tf.float32, shape=(None, 1), name='cell_label')

                l2_loss = tf.losses.mean_squared_error(self.labels, self.rnn_score)
                tf.summary.scalar('l2 loss', tf.reduce_mean(l2_loss))

                reg_loss = tf.reduce_sum([tf.reduce_sum(tf.square(x)) for x in policy_net_variables])  # Regularization

                # sum up policy gradient and regularization loss
                self.total_loss = l2_loss + self.reg_strength * reg_loss
                tf.summary.scalar('total_loss', self.total_loss)

                # training update
                with tf.name_scope("train_policy_network"):
                    # apply gradients to update policy network
                    self.train_op = self.optimizer.minimize(self.total_loss, global_step=self.global_step)

            self.summaries_op = tf.summary.merge_all()
            self.summary_writer = tf.summary.FileWriter('logs', graph=self.policy_session.graph)

            self.policy_session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=1)

            if self.restore_controller:
                path = tf.train.latest_checkpoint('weights/')

                if path is not None and tf.train.checkpoint_exists(path):
                    print("Loading Encoder Checkpoint !")
                    self.saver.restore(self.policy_session, path)

    def train_step(self, rewards, children_ids=[]):
        '''
        Perform a single train step on the Encoder RNN

        Returns:
            the training loss
        '''
        #assert len(rewards) == len(self.state_space.children)

        if children_ids is not None and len(children_ids) == 0:
            children = self.state_space.children  # take all the children
        else:
            children = []
            for id in range(len(self.state_space.children)):
                if id in children_ids:
                    children.append(self.state_space.children[id])

        for id, (child, score) in enumerate(zip(children, rewards)):
            state_list = self.state_space.one_hot_encode_child(child)
            state_list = np.concatenate(state_list, axis=-1)
            state_list = state_list.reshape((1, -1, 1))

            feed_dict = {
                self.state_input: state_list,
                self.labels: np.array([[score]]),
            }

            with self.policy_session.as_default():
                K.set_session(self.policy_session)

                _, loss, summary, global_step = self.policy_session.run(
                    [self.train_op, self.total_loss, self.summaries_op,
                     self.global_step],
                    feed_dict=feed_dict)

                self.summary_writer.add_summary(summary, global_step)
                self.saver.save(self.policy_session, save_path='weights/controller.ckpt', global_step=self.global_step)

        return loss

    def update_step(self):
        if self.b_ + 1 <= self.B:
            with self.policy_session.as_default():
                K.set_session(self.policy_session)

                self.b_ += 1
                models_scores = []

                for i, intermediate_child in enumerate(self.state_space.prepare_intermediate_children(self.b_)):
                    state_list = self.state_space.one_hot_encode_child(intermediate_child)
                    state_list = np.concatenate(state_list, axis=-1)
                    state_list = state_list.reshape((1, -1, 1))

                    feed_dict = {
                        self.state_input: state_list,
                    }

                    score = self.policy_session.run(self.rnn_score, feed_dict=feed_dict)
                    score = score[0, 0]

                    models_scores.append([intermediate_child, score])

                    with open('scores_%d.csv' % (self.b_), mode='a+', newline='') as f:
                        writer = csv.writer(f)
                        data = [score]
                        data.extend(intermediate_child)
                        writer.writerow(data)

                    if (i + 1) % 500 == 0:
                        print("Scored %d models. Current model score = %0.4f" % (i + 1, score))

                models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)

                children = []
                for i in range(self.K):
                    children.append(models_scores[i][0])

                self.state_space.update_children(children)
        else:
            print()
            print("No more updates necessary as max B has been reached !")

