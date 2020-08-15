# -*- coding: utf-8 -*-
import os
import time
import numpy as np
import tensorflow as tf
from tqdm import tqdm

from tensorboard.plugins.hparams import api as hp
from model.metric import calculate_auroc, calculate_aupr
from utils.utils import plot_loss_curve, plot_roc_curve, plot_pr_curve
from utils.utils import write2txt, write2csv

class EarlyStopTrainer(object):
    def __init__(self, model, loss_object, optimizer, hparams,
                 result_dir, checkpoint_dir, summary_dir,
                 epochs=100, patience=5, verbosity=0, tensorboard=True, max_to_keep=5):
        self.model = model
        self.loss_object = loss_object
        self.optimizer = optimizer
        self.hparams = hparams

        self.result_dir = result_dir
        self.summary_dir = summary_dir
        self.checkpoint_dir = checkpoint_dir

        self.epochs = epochs
        self.patience = patience
        self.verbosity = verbosity
        self.tensorboard = tensorboard
        self.max_to_keep = max_to_keep

        if self.verbosity == 0:
            self.dis_show_bar = True
        else:
            self.dis_show_bar = False

        # Initialize the Metrics.
        self.metric_tra_loss = tf.keras.metrics.Mean()
        self.metric_val_loss = tf.keras.metrics.Mean()

        # Initialize the SummaryWriter.
        self.writer = tf.summary.create_file_writer(
            logdir=self.summary_dir)

        # Initialize the CheckpointManager
        self.ckpt = tf.train.Checkpoint(
            step=tf.Variable(0, dtype=tf.int64),
            net=self.model,
            optimizer=self.optimizer)
        self.manager = tf.train.CheckpointManager(
            checkpoint=self.ckpt,
            directory=self.checkpoint_dir,
            max_to_keep=self.max_to_keep)

    # @tf.function(input_signature=[tf.TensorSpec([None, 1000, 4], tf.float32)])
    def train_step(self, x, y):
        with tf.GradientTape() as tape:
            predictions = self.model(inputs=x, training=True)
            loss = self.loss_object(y_true=y, y_pred=predictions)
            loss = loss + tf.reduce_sum(self.model.losses)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return loss, predictions

    def train(self, dataset_train, dataset_valid, train_steps, valid_steps):
        print('Begin to train the model.', flush=True)
        dataset_train = dataset_train
        dataset_valid = dataset_valid

        best_valid_loss = np.inf
        patience_temp = 0
        history = {'epoch': [], 'train_loss': [], 'valid_loss': []}

        for epoch in range(1, self.epochs+1):
            start_time = time.time()
            with tqdm(range(train_steps), ascii=True, disable=self.dis_show_bar) as pbar:
                for _, (batch_x, batch_y) in zip(pbar, dataset_train):
                    train_loss, predictions = self.train_step(batch_x, batch_y)
                    batch_size = tf.shape(batch_x)[0]
                    self.metric_tra_loss.update_state(train_loss, batch_size)
                    pbar.set_description('Train loss: {:.4f}'.format(train_loss))

            with tqdm(range(valid_steps), ascii=True, disable=self.dis_show_bar) as pbar:
                for _, (batch_x, batch_y) in zip(pbar, dataset_valid):
                    predictions = self.model(inputs=batch_x, training=False)
                    valid_loss = self.loss_object(y_true=batch_y, y_pred=predictions)
                    batch_size = tf.shape(batch_x)[0]
                    self.metric_val_loss.update_state(valid_loss, batch_size)
                    pbar.set_description('Valid loss: {:.4f}'.format(valid_loss))
            end_time = time.time()

            epoch_time = end_time - start_time
            real_epoch = self.ckpt.step.assign_add(1)
            epoch_train_loss = self.metric_tra_loss.result()
            epoch_valid_loss = self.metric_val_loss.result()
            history['epoch'].append(real_epoch.numpy())
            history['train_loss'].append(epoch_train_loss.numpy())
            history['valid_loss'].append(epoch_valid_loss.numpy())
            print("Epoch: {} | Train Loss: {:.5f}".format(real_epoch.numpy(), epoch_train_loss.numpy()), flush=True)
            print("Epoch: {} | Valid Loss: {:.5f}".format(real_epoch.numpy(), epoch_valid_loss.numpy()), flush=True)
            print("Epoch: {} | Cost time: {:.5f}: second".format(real_epoch.numpy(), epoch_time), flush=True)
            self.metric_tra_loss.reset_states()
            self.metric_val_loss.reset_states()

            # Save the checkpoint. (Only save the best performance checkpoints)
            if epoch_valid_loss < best_valid_loss:
                best_valid_loss = epoch_valid_loss
                patience_temp = 0
                save_path = self.manager.save(checkpoint_number=real_epoch)
                print("Saved checkpoint for epoch {}: {}".format(real_epoch.numpy(), save_path), flush=True)
            else:
                patience_temp += 1

            # Early Stop the training loop, if the validation loss didn't decrease for patience epochs.
            if patience_temp == self.patience:
                print('Validation dice has not improved in {} epochs. Stopped training.'
                      .format(self.patience), flush=True)
                break

        # Plot the loss curve of training and validation, and save the loss value of training and validation.
        print('History dict: ', history, flush=True)
        np.save(os.path.join(self.result_dir, 'history.npy'), history)

        return history

    def test(self, dataset_test, test_steps):
        if self.manager.latest_checkpoint:
            self.ckpt.restore(self.manager.latest_checkpoint).expect_partial()
            print("Restored from {}".format(self.manager.latest_checkpoint), flush=True)
        else:
            print("Initializing from scratch.", flush=True)

        results = []
        labels = []
        with tqdm(range(test_steps), ascii=True, disable=self.dis_show_bar, desc='Testing ... ') as pbar:
            for i, (batch_x, batch_y) in zip(pbar, dataset_test):
                predictions = self.model(batch_x, training=False)
                results.append(predictions)
                labels.append(batch_y)

        result = np.concatenate(results)
        result = np.mean((result[0:227512], result[227512:]), axis=0)
        label = np.concatenate(labels)
        label = label[0:227512]

        # Evaluate the result.
        result_shape = np.shape(result)
        fpr_list, tpr_list, auroc_list = [], [], []
        precision_list, recall_list, aupr_list = [], [], []
        with tqdm(range(result_shape[1]), ascii=True, disable=self.dis_show_bar, desc='Evaluating... ') as pbar:
            for i in pbar:
                fpr_temp, tpr_temp, auroc_temp = calculate_auroc(result[:, i], label[:, i])
                precision_temp, recall_temp, aupr_temp = calculate_aupr(result[:, i], label[:, i])

                fpr_list.append(fpr_temp)
                tpr_list.append(tpr_temp)
                precision_list.append(precision_temp)
                recall_list.append(recall_temp)
                auroc_list.append(auroc_temp)
                aupr_list.append(aupr_temp)

        header = np.array([['auroc', 'aupr']])
        content = np.stack((auroc_list, aupr_list), axis=1)
        content = np.concatenate((header, content), axis=0)
        write2csv(content, os.path.join(self.result_dir, 'result.csv'))

        avg_auroc = np.nanmean(auroc_list)
        avg_aupr = np.nanmean(aupr_list)
        message = 'AVG-AUROC:{:.5f}, AVG-AUPR:{:.5f}.'.format(avg_auroc, avg_aupr)
        write2txt([message], os.path.join(self.result_dir, 'result.txt'))
        print(message)

        return result, label