import numpy as np
import time
import pprint
import csv
from collections import OrderedDict

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
    def __init__(self, B, operators,
                 input_lookback_depth=-1,
                 input_lookforward_depth=None):
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

        self.inputs_embedding_max = len(input_values)
        self.operator_embedding_max = len(np.unique(operators))

        self._add_state('inputs', values=input_values)
        self._add_state('ops', values=self.operators)
        self.prepare_initial_children()

    def _add_state(self, name, values):
        '''
        Adds a "state" to the state manager, along with some metadata for efficient
        packing and unpacking of information required by the RNN ControllerManager.

        Stores metadata such as:
        -   Global ID
        -   Name
        -   Valid Values
        -   Number of valid values possible
        -   Map from value ID to state value
        -   Map from state value to value ID

        # Args:
            name: name of the state / action
            values: valid values that this state can take

        # Returns:
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

    def embedding_encode(self, id, value):
        '''
        Embedding index encode the specific state value

        # Args:
            id: global id of the state
            value: state value

        # Returns:
            embedding encoded representation of the state value
        '''
        state = self[id]
        value_map = state['value_map_']
        value_idx = value_map[value]

        encoding = np.zeros((1, 1), dtype=np.float32)
        encoding[0, 0] = value_idx
        return encoding

    def get_state_value(self, id, index):
        '''
        Retrieves the state value from the state value ID

        # Args:
            id: global id of the state
            index: index of the state value (usually from argmax)

        # Returns:
            The actual state value at given value index
        '''
        state = self[id]
        index_map = state['index_map_']
        value = index_map[index]
        return value

    def parse_state_space_list(self, state_list):
        '''
        Parses a list of one hot encoded states to retrieve a list of state values

        # Args:
            state_list: list of one hot encoded states

        # Returns:
            list of state values
        '''
        state_values = []
        for id, state_value in enumerate(state_list):
            state_val_idx = state_value[0, 0]
            value = self.get_state_value(id % 2, state_val_idx)
            state_values.append(value)

        return state_values

    def entity_encode_child(self, child):
        '''
        Perform encoding for all blocks in a cell

        # Args:
            child: a list of blocks (which forms one cell / layer)

        # Returns:
            list of entity encoded blocks of the cell
        '''
        encoded_child = []
        for i, val in enumerate(child):
            encoded_child.append(self.embedding_encode(i % 2, val))

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

        # Args:
            new_b: the number of blocks in current stage

        # Returns:
            a generator that produces a joint of the previous and current
            child models
        '''
        if self.input_lookforward_depth is not None:
            new_b_dash = min(self.input_lookforward_depth, new_b)
        else:
            new_b_dash = new_b

        new_ip_values = list(range(self.input_lookback_depth, new_b_dash))
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
            vals = [(self.get_state_value(id % 2, p), p) for n, p in zip(state['values'], *action)]
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


class Controller(tf.keras.Model):
    
    def __init__(self, controller_cells, embedding_dim,
                 input_embedding_max, operator_embedding_max):
        '''
        LSTM Controller model which accepts encoded sequence describing the
        architecture of the model and predicts a singular value describing
        its probably validation accuracy.
        
        # Args:
            controller_cells: number of cells of the Controller LSTM.
            embedding_dim: size of the embedding dimension.
            input_embedding_max: maximum input dimension of the input embedding.
            operator_embedding_max: maximum input dimension of the operator encoding.
        '''
        super(Controller, self).__init__(name='EncoderRNN')

        self.controller_cells = controller_cells
        self.embedding_dim = embedding_dim
        self.input_embedding_max = input_embedding_max
        self.operator_embedding_max = operator_embedding_max

        # Layers
        self.input_embedding = tf.keras.layers.Embedding(input_embedding_max + 1, embedding_dim)
        self.operators_embedding = tf.keras.layers.Embedding(operator_embedding_max + 1, embedding_dim)
        
        if tf.test.is_gpu_available():
            self.rnn = tf.keras.layers.CuDNNLSTM(controller_cells, return_state=True)
        else:
            self.rnn = tf.keras.layers.LSTM(controller_cells, return_state=True)
            
        self.rnn_score = tf.keras.layers.Dense(1, activation='sigmoid')

    def call(self, inputs_operators, states=None, training=None, mask=None):
        inputs, operators = self._get_inputs_and_operators(inputs_operators)  # extract the data

        if states is None:  # initialize the state vectors
            states = self.rnn.get_initial_state(inputs)
            states = [tf.to_float(state) for state in states]
        
        # map the sparse inputs and operators into dense embeddings
        embed_inputs = self.input_embedding(inputs)
        embed_ops = self.operators_embedding(operators)
        
        # concatenate the embeddings
        embed = tf.concat([embed_inputs, embed_ops], axis=-1)  # concatenate the embeddings
        
        # run over the LSTM
        out = self.rnn(embed, initial_state=states)
        out, h, c = out  # unpack the outputs and states
        
        # get the predicted validation accuracy
        score = self.rnn_score(out)

        return [score, [h, c]]

    def _get_inputs_and_operators(self, inputs_operators):
        '''
        Splits the joint inputs and operators into seperate inputs
        and operators list for convenience of the SearchSpace.
        
        # Args:
            inputs_operators: interleaved [input; operator] pairs.

        # Returns:
            list of inputs and list of operators.
        '''
        inputs = inputs_operators[:, 0::2]  # even place data
        operators = inputs_operators[:, 1::2]  # odd place data

        return inputs, operators


class ControllerManager:
    '''
    Utility class to manage the RNN Controller.
    
    Tasked with maintaining the state of the training schedule,
    keep track of the children models generated from cross-products,
    cull non-optimal children model configurations and resume
    training.
    '''
    def __init__(self, state_space,
                 B=5, K=256,
                 train_iterations=10,
                 reg_param=0.001,
                 controller_cells=32,
                 embedding_dim=20,
                 input_B=None,
                 restore_controller=False):
        '''
        Manages the Controller network training and prediction process.
        
        # Args:
            state_space: completely defined search space.
            B: depth of progression.
            K: maximum number of children model trained per level of depth.
            train_iterations: number of training epochs for the RNN per depth level.
            reg_param: strength of the L2 regularization loss.
            controller_cells: number of cells in the Controller LSTM.
            embedding_dim: embedding dimension for inputs and operators.
            input_B: override value of B, used only when we are restoring the controller.
                Determing the maximum input connectivity allowed to the RNN Controller,
                to maintain backward compatibility with trained models.
                
                Use it alongside `restore_controller` to evaluate model settings
                with larger depth `B` than allowed at training time.
            restore_controller: flag whether to restore a pre-trained RNN controller
                upon construction.
        '''
        
        self.state_space = state_space  # type: StateSpace
        self.state_size = self.state_space.size

        self.b_ = 1
        self.B = B
        self.K = K
        self.embedding_dim = embedding_dim

        self.train_iterations = train_iterations
        self.controller_cells = controller_cells
        self.reg_strength = reg_param
        self.input_B = input_B
        self.restore_controller = restore_controller

        self.children_history = None
        self.score_history = None
    
        # No support for summary statistics yet
        timestr = time.strftime("%Y-%m-%d-%H-%M-%S")
        self.logdir = 'logs/%s' % timestr
        # self.summary_writer = tf.contrib.summary.create_file_writer(self.logdir)
        # self.summary_write.set_as_default()

        self.build_policy_network()

    def get_actions(self, top_k=None):
        '''
        Gets a one hot encoded action list, either from random sampling or from
        the ControllerManager RNN

        # Args:
            top_k: Number of models to return

        # Returns:
            A one hot encoded action list
        '''
        models = self.state_space.children

        if top_k is not None:
            models = models[:top_k]

        actions = []
        for model in models:
            encoded_model = self.state_space.entity_encode_child(model)
            actions.append(encoded_model)

        return actions

    def build_policy_network(self):
        '''
        Construct the RNN controller network with the provided settings.
        
        Also constructs saver and restorer to the RNN controller if required.
        '''
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'
        self.device = device

        if self.restore_controller and self.input_B is not None:
            input_B = self.input_B
        else:
            input_B = self.state_space.inputs_embedding_max

        self.global_step = tf.train.get_or_create_global_step()
        learning_rate = tf.train.exponential_decay(0.001, self.global_step,
                                                   500, 0.98, staircase=True)

        with tf.device(device):
            self.controller = Controller(self.controller_cells,
                                         self.embedding_dim,
                                         input_B,
                                         self.state_space.operator_embedding_max)

            self.optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)

        self.saver = tf.train.Checkpoint(controller=self.controller,
                                         optimizer=self.optimizer,
                                         global_step=self.global_step)

        if self.restore_controller:
            path = tf.train.latest_checkpoint('weights/')

            if path is not None and tf.train.checkpoint_exists(path):
                print("Loading Controller Checkpoint !")
                self.saver.restore(path)

    def loss(self, real_acc, rnn_scores):
        '''
        Computes the surrogate losses to train the controller.
        
        - rnn score loss is the MSE between the real validation acc and the
        predicted acc of the rnn.
        
        - reg loss is the L2 regularization loss on the parameters of the controller.
        
        # Args:
            real_acc: actual validation accuracy obtained by child models.
            rnn_scores: predicted validation accuracy obtained by child models.

        # Returns:
            weighted sum of rnn score loss + reg loss.
        '''
        # RNN predicted accuracies
        rnn_score_loss = tf.losses.mean_squared_error(real_acc, rnn_scores)

        # Regularization of model
        params = self.controller.trainable_variables
        reg_loss = tf.reduce_sum([tf.nn.l2_loss(x) for x in params])

        total_loss = rnn_score_loss + self.reg_strength * reg_loss

        return total_loss

    def train_step(self, rewards):
        '''
        Perform a single train step on the Controller RNN

        # Returns:
            final training loss
        '''
        children = np.array(self.state_space.children, dtype=np.object)  # take all the children
        rewards = np.array(rewards, dtype=np.float32)
        loss = 0

        if self.children_history is None:
            self.children_history = [children]
            self.score_history = [rewards]
            batchsize = rewards.shape[0]
        else:
            self.children_history.append(children)
            self.score_history.append(rewards)
            batchsize = sum([data.shape[0] for data in self.score_history])

        train_size = batchsize * self.train_iterations
        print("Controller: Number of training steps required for this stage : %d" % (train_size))
        print()

        for current_epoch in range(self.train_iterations):
            print("Controller: Begin training epoch %d" % (current_epoch + 1))
            print()

            for dataset_id in range(self.b_):
                children = self.children_history[dataset_id]
                scores = self.score_history[dataset_id]

                ids = np.array(list(range(len(scores))))
                np.random.shuffle(ids)

                print("Controller: Begin training - B = %d" % (dataset_id + 1))

                for id, (child, score) in enumerate(zip(children[ids], scores[ids])):
                    child = child.tolist()
                    state_list = self.state_space.entity_encode_child(child)
                    state_list = np.concatenate(state_list, axis=-1).astype('int32')

                    with tf.device(self.device):
                        state_list = tf.convert_to_tensor(state_list)

                        with tf.GradientTape() as tape:
                            rnn_scores, states = self.controller(state_list, states=None)
                            acc_scores = score.reshape((1, 1))

                            total_loss = self.loss(acc_scores, rnn_scores)

                        grads = tape.gradient(total_loss, self.controller.trainable_variables)
                        grad_vars = zip(grads, self.controller.trainable_variables)

                        self.optimizer.apply_gradients(grad_vars, self.global_step)

                    loss += total_loss.numpy().sum()

                print("Controller: Finished training epoch %d / %d of B = %d / %d" % (
                    current_epoch + 1, self.train_iterations,
                    dataset_id + 1, self.b_))

        # save weights
        self.saver.save('weights/controller.ckpt')

        return loss.mean()

    def update_step(self):
        '''
        Updates the children from the intermediate products for the next generation
        of larger number of blocks in each cell
        '''
        if self.b_ + 1 <= self.B:

            self.b_ += 1
            models_scores = []

            # iterate through all the intermediate children
            for i, intermediate_child in enumerate(self.state_space.prepare_intermediate_children(self.b_)):
                state_list = self.state_space.entity_encode_child(intermediate_child)
                state_list = np.concatenate(state_list, axis=-1).astype('int32')

                state_list = tf.convert_to_tensor(state_list)

                # score the child
                score, _ = self.controller(state_list, states=None)
                score = score[0, 0].numpy()

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

            # account for case where there are fewer children than K
            if self.K is not None:
                children_count = min(self.K, len(models_scores))
            else:
                children_count = len(models_scores)

            # take only the K highest scoring children for next iteration
            children = []
            for i in range(children_count):
                children.append(models_scores[i][0])

            # save these children for next round
            self.state_space.update_children(children)
        else:
            print()
            print("No more updates necessary as max B has been reached !")
