import numpy as np
import time
import pprint
import csv
from collections import OrderedDict

from keras import backend as K
import tensorflow as tf

import os

if not os.path.exists('weights/'):
    os.makedirs('weights/')

if not os.path.exists('logs/'):
    os.makedirs('logs/')


class StateSpace:
    '''
    State Space manager

    Provides utilit functions for holding "states" / "actions" that the controller
    must use to train and predict.

    Also provides a more convenient way to define the search space
    '''

    def __init__(self, B, operators, input_lookback_depth=-1, input_lookforward_depth=None):
        '''
        Constructs a search space which models the NAS and PNAS papers

        A single block consists of the 4-tuple:
        (input 1, operation 1, input 2, operation 2)

        The merge operation can be a sum or a concat as required.

        The input operations are used for adding up intermediate values
        inside the same cell. See the NASNet and P-NASNet models to see
        how intermediate blocks connect based on input values.

        The default operation values are based on the P-NAS paper. They
        should be changed as needed.

        # Note:
        This only provides a convenient mechanism to train the networks.
        It is upto the model designer to interpret this block tuple
        information and construct the final model.

        # Args:
            B: Maximum number of blocks
            operators: a list of operations (can be anything, must be
                interpreted by the model designer when constructing the
                actual model. Usually a list of strings.
            input_lookback_depth: should be a negative number or 0.
                Describes how many cells the input should look behind.
                Can be used to tensor information from 0 or more cells from
                the current cell.

                The negative number describes how many cells to look back.
                Set to 0 if the lookback feature is not needed (flat cells).
            input_lookforward_depth: sets a limit on input depth that can be looked forward.
                This is useful for scenarios where "flat" models are preferred,
                wherein each cell is flat, though it may take input from deeper
                layers (if the designer so chooses)

                The default searches over cells which can have inter-connections.
                Setting it to 0 limits this to just the current input for that cell (flat cells).
        '''
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

        self.input_lookback_depth = input_lookback_depth
        self.input_lookforward_depth = input_lookforward_depth

        input_values = list(range(input_lookback_depth, self.B))  # -1 = Hc-1, 0-(B-1) = Hci

        self._add_state('inputs', values=input_values)
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
        '''
        Perform one hot encoding for all blocks in a cell

        Args:
            child: a list of blocks (which forms one cell / layer)

        Returns:
            list of one hot encoded blocks of the cell
        '''
        encoded_child = []
        for i, val in enumerate(child):
            encoded_child.append(self.one_hot_encode(i % 2, val))

        return encoded_child

    def prepare_initial_children(self):
        '''
        Prepare the initial set of child models which must
        all be trained to obtain the initial set of scores
        '''
        # set of all operations
        ops = list(range(len(self.operators)))
        inputs = list(range(self.input_lookback_depth, 0))

        # if input_lookback_depth == 0, then we need to adjust to have at least
        # one input (generally 0)
        if len(inputs) == 0:
            inputs = [0]

        print()
        print("Obtaining search space for b = 1")
        print("Search space size : ", (len(inputs) * (len(self.operators) ** 2)))

        search_space = [inputs, ops, inputs, ops]
        self.children = list(self._construct_permutations(search_space))

    def prepare_intermediate_children(self, new_b):
        '''
        Generates the intermediate product of the previous children
        and the current generation of children.

        Note: This is a very long step and can take an enormous amount
        of time !

        Args:
            new_b: the number of blocks in current stage

        Returns:
            a generator that produces a joint of the previous and current
            child models
        '''
        if self.input_lookforward_depth is not None:
            new_b = min(self.input_lookforward_depth, new_b)

        new_ip_values = list(range(self.input_lookback_depth, new_b))
        ops = list(range(len(self.operators)))

        # if input_lookback_depth == 0, then we need to adjust to have at least
        # one input (generally 0)
        if len(new_ip_values) == 0:
            new_ip_values = [0]

        new_child_count = ((len(new_ip_values)) ** 2) * (len(self.operators) ** 2)
        print()
        print("Obtaining search space for b = %d" % new_b)
        print("Search space size : ", new_child_count)

        print()
        print("Total models to evaluate : ", (len(self.children) * new_child_count))
        print()

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
        ''' Compute the total number of models to generate and train '''
        num_inputs = 1 if self.input_lookback_depth == 0 else abs(self.input_lookback_depth)
        level1 = (num_inputs ** 2) * (len(self.operators) ** 2)
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
                 train_iterations=10,
                 reg_param=0.001,
                 controller_cells=32,
                 restore_controller=False):
        self.policy_session = policy_session  # type: tf.Session

        self.state_space = state_space  # type: StateSpace
        self.state_size = self.state_space.size

        self.b_ = 1
        self.B = B
        self.K = K

        self.train_iterations = train_iterations
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
                starter_learning_rate = 0.001
                learning_rate = tf.train.exponential_decay(starter_learning_rate, self.global_step,
                                                           500, 0.98, staircase=True)

                tf.summary.scalar('learning_rate', learning_rate)
                self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

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

            timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
            filename = 'logs/%s' % timestr

            self.summary_writer = tf.summary.FileWriter(filename, graph=self.policy_session.graph)

            self.policy_session.run(tf.global_variables_initializer())
            self.saver = tf.train.Saver(max_to_keep=1)

            if self.restore_controller:
                path = tf.train.latest_checkpoint('weights/')

                if path is not None and tf.train.checkpoint_exists(path):
                    print("Loading Encoder Checkpoint !")
                    self.saver.restore(self.policy_session, path)

    def train_step(self, rewards):
        '''
        Perform a single train step on the Encoder RNN

        Returns:
            final training loss
        '''
        children = np.array(self.state_space.children, dtype=np.object)  # take all the children
        rewards = np.array(rewards, dtype=np.float32)
        loss = 0

        for _ in range(self.train_iterations):
            ids = np.array(list(range(len(rewards))))
            np.random.shuffle(ids)

            for id, (child, score) in enumerate(zip(children[ids], rewards[ids])):
                child = child.tolist()
                state_list = self.state_space.one_hot_encode_child(child)
                state_list = np.concatenate(state_list, axis=-1)
                state_list = state_list.reshape((1, -1, 1))

                # feed in the child model and the score
                feed_dict = {
                    self.state_input: state_list,
                    self.labels: score.reshape((1, 1)),
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
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        if self.b_ + 1 <= self.B:
            with self.policy_session.as_default():
                K.set_session(self.policy_session)

                self.b_ += 1
                models_scores = []

                # iterate through all the intermediate children
                for i, intermediate_child in enumerate(self.state_space.prepare_intermediate_children(self.b_)):
                    state_list = self.state_space.one_hot_encode_child(intermediate_child)
                    state_list = np.concatenate(state_list, axis=-1)
                    state_list = state_list.reshape((1, -1, 1))

                    # score the child
                    feed_dict = {
                        self.state_input: state_list,
                    }

                    score = self.policy_session.run(self.rnn_score, feed_dict=feed_dict)
                    score = score[0, 0]

                    # preserve the child and its score
                    models_scores.append([intermediate_child, score])

                    with open('scores_%d.csv' % (self.b_), mode='a+', newline='') as f:
                        writer = csv.writer(f)
                        data = [score]
                        data.extend(intermediate_child)
                        writer.writerow(data)

                    if (i + 1) % 500 == 0:
                        print("Scored %d models. Current model score = %0.4f" % (i + 1, score))

                # sort the children according to their score
                models_scores = sorted(models_scores, key=lambda x: x[1], reverse=True)

                # take only the K highest scoring children for next iteration
                children = []
                for i in range(self.K):
                    children.append(models_scores[i][0])

                # save these children for next round
                self.state_space.update_children(children)
        else:
            print()
            print("No more updates necessary as max B has been reached !")
