import numpy as np
import shutil
import os
import tqdm

import tensorflow as tf
from tensorflow.contrib.eager.python import tfe
from tensorflow.python.keras.models import Model
from tensorflow.python.keras.callbacks import ModelCheckpoint


if not os.path.exists('temp_weights/'):
    os.makedirs('temp_weights/')
else:
    shutil.rmtree('temp_weights')
    os.makedirs('temp_weights/', exist_ok=True)


class NetworkManager:
    '''
    Helper class to manage the generation of subnetwork training given a dataset
    '''
    def __init__(self, dataset, epochs=5, batchsize=128, learning_rate=0.001):
        '''
        Manager which is tasked with creating subnetworks, training them on a dataset, and retrieving
        rewards in the term of accuracy, which is passed to the controller RNN.

        # Args:
            dataset: a tuple of 4 arrays (X_train, y_train, X_val, y_val)
            epochs: number of epochs to train the subnetworks
            batchsize: batchsize of training the subnetworks
            learning_rate: learning rate for the Optimizer.
        '''
        self.dataset = dataset
        self.epochs = epochs
        self.batchsize = batchsize
        self.lr = learning_rate

    def get_rewards(self, model_fn, actions, display_model_summary=True):
        '''
        Creates a subnetwork given the actions predicted by the controller RNN,
        trains it on the provided dataset, and then returns a reward.

        # Args:
            model_fn: a function which accepts one argument, a list of
                parsed actions, obtained via an inverse mapping from the
                StateSpace.

            actions: a list of parsed actions obtained via an inverse mapping
                from the StateSpace. It is in a specific order as given below:

                Consider 4 states were added to the StateSpace via the `add_state`
                method. Then the `actions` array will be of length 4, with the
                values of those states in the order that they were added.

                If number of layers is greater than one, then the `actions` array
                will be of length `4 * number of layers` (in the above scenario).
                The index from [0:4] will be for layer 0, from [4:8] for layer 1,
                etc for the number of layers.

                These action values are for direct use in the construction of models.

            display_model_summary: Display the child model summary at the end of training.

        # Returns:
            a reward for training a model with the given actions
        '''
        if tf.test.is_gpu_available():
            device = '/gpu:0'
        else:
            device = '/cpu:0'

        tf.keras.backend.reset_uids()

        # generate a submodel given predicted actions
        with tf.device(device):
            model = model_fn(actions)  # type: Model

            # build model shapes
            X_train, y_train, X_val, y_val = self.dataset

            # generate the dataset for training
            train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train))
            train_dataset = train_dataset.apply(tf.data.experimental.shuffle_and_repeat(10000, seed=0))
            train_dataset = train_dataset.batch(self.batchsize)
            train_dataset = train_dataset.apply(tf.data.experimental.prefetch_to_device(device))

            # generate the dataset for evaluation
            val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val))
            val_dataset = val_dataset.batch(self.batchsize)
            val_dataset = val_dataset.apply(tf.data.experimental.prefetch_to_device(device))

            num_train_batches = X_train.shape[0] // self.batchsize + 1

            global_step = tf.train.get_or_create_global_step()
            lr = tf.train.cosine_decay(self.lr, global_step, decay_steps=num_train_batches * self.epochs, alpha=0.1)

            # construct the optimizer and saver of the child model
            optimizer = tf.train.AdamOptimizer(lr)
            saver = tf.train.Checkpoint(model=model, optimizer=optimizer, global_step=global_step)

            best_val_acc = 0.0
            for epoch in range(self.epochs):
                # train child model
                with tqdm.tqdm(train_dataset,
                               desc='Train Epoch (%d / %d): ' % (epoch + 1, self.epochs),
                               total=num_train_batches) as iterator:

                    for i, (x, y) in enumerate(iterator):
                        # get gradients
                        with tf.GradientTape() as tape:
                            preds = model(x, training=True)
                            loss = tf.keras.losses.categorical_crossentropy(y, preds)

                        grad = tape.gradient(loss, model.variables)
                        grad_vars = zip(grad, model.variables)

                        # update weights of the child model
                        optimizer.apply_gradients(grad_vars, global_step)

                        if (i + 1) >= num_train_batches:
                            break

                print()

                # evaluate child model
                acc = tfe.metrics.CategoricalAccuracy()
                for j, (x, y) in enumerate(val_dataset):
                    preds = model(x, training=False)
                    acc(y, preds)

                acc = acc.result().numpy()

                print("Epoch %d: Val accuracy = %0.6f" % (epoch + 1, acc))

                # if acc improved, save the weights
                if acc > best_val_acc:
                    print("Val accuracy improved from %0.6f to %0.6f. Saving weights !" % (
                        best_val_acc, acc))

                    best_val_acc = acc
                    saver.save('temp_weights/temp_network')

                print()

            # load best weights of the child model
            path = tf.train.latest_checkpoint('temp_weights/')
            saver.restore(path)

            # display the structure of the child model
            if display_model_summary:
                model.summary()

            # evaluate the best weights of the child model
            acc = tfe.metrics.CategoricalAccuracy()

            for j, (x, y) in enumerate(val_dataset):
                preds = model(x, training=False)
                acc(y, preds)

            acc = acc.result().numpy()

        # compute the reward (validation accuracy)
        reward = acc

        print()
        print("Manager: Accuracy = ", reward)

        # clean up resources and GPU memory
        del model
        del optimizer
        del global_step

        return reward