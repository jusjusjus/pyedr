"""
This file is part of pyedr

Copyright (c) 2017 Daniel Vorberg

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell
from tensorflow.python.training import coordinator
from tensorflow.python.training import queue_runner

import numpy as np
from matplotlib import pyplot as plt, lines
from collections import namedtuple

import warnings
import os
try:
    import biosppy
except ImportError:
    warnings.warn("Failed to load biosppy required for gradient calculation",
                  ImportWarning)


def cached_property(function):
    """ decorator that store propery after first call.
    """
    attr_name = '_cached_' + function.__name__

    @property
    def _chached_property(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, function(self))
        return getattr(self, attr_name)
    return _chached_property


class NeuralNetworks:
    """ The graph of the neural network (without vabiables)
    
    Attributes (all are tensorflow tensors):
        batch: list of the attributes [time, input.sequence, target, input.length]
        input: namedtuple of (sequence, length)
            sequence (tensor[BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_FEATURES])
                placeholder for the input data
            length (tensor[BATCH_SIZE] of tf.int32): the length of the 
                each example in sequence (which is padded)
        target (tensor[BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_TARGETS]):
            placeholder of the target (default reading from train_batch)
        prediction (tensor[BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_FEATURES]): 
            the prediction of the network
        loss (tensor[]): the average loss of the batch
        loss_per_example (tensor[BATCH_SIZE]): the loss for each example
            in the batch
        error (tensor[]): the average loss of the batch.
            At the moment equal to loss, but this might change in
            the future
        error_per_example (tensor[BATCH_SIZE]): the loss for each example
            in the batch. At the moment equal to error_per_example, but 
            this might change in the future
        initializer (tensorflow op): initialize (when run) all local and 
            global variables        
        global_step (tensor[]): counts the global step
        learning_rate (tensor[]): placeholder for the learning rate            
    """


    MAX_SEQUENCE_LEN = None  # allow dynamically choosen sequence length
    BATCH_SIZE = None  # allow dynamically choosen batch size
    NUM_FEATURES = 1
    NUM_TARGETS = 1
        
    Input = namedtuple('Input', ('sequence', 'length'))

    def __init__(self, name=None, model_save_path=None, 
                 default_batch=None, lstm_sizes=[15, 15]):
        """ 
        Args:
            name (str): to generate the path (model_path/<name>)
            model_path (str): to generate the path (model_path/<name>)
            default_batch (list of tensors: 
                [time, ecg, target respiration, length]):
                if None, new placeholders are builded
        """
        if name is not None and model_save_path is not None:
            self.path = os.path.join(model_save_path, name)
        else:
            self.path = None
        self.default_batch = default_batch
        self._lstm_sizes = lstm_sizes

        self._build_tensors()

    @classmethod
    def from_meta_graph(cls, model_path):
        saver = tf.train.import_meta_graph(model_files + ".meta")
        graph = tf.get_default_graph()
        
        class ImportedNeuralNetworks:
            pass

        net = ImportedNeuralNetworks()
        net.input = self.Input(
            sequence=graph.get_tensor_by_name("input_sequence:0"),
            length=graph.get_tensor_by_name("input_seq_length:0"))
        net.target = graph.get_tensor_by_name("target:0")
        net.prediction = graph.get_tensor_by_name("prediction:0")
        net.loss = graph.get_tensor_by_name("loss:0")
        net.prediction = cls.prediction
        return net

    def _build_tensors(self):
        self.prediction
        self.target
        self.loss
        self.error
        self.train_step
        self.global_step

    @cached_property
    def batch(self):
        """ list of tensors: [time, ecg, target respiration, length]
        
        """
        if self.default_batch is None:
            time = tf.placeholder(
                tf.float32, 
                shape=[self.BATCH_SIZE, self.MAX_SEQUENCE_LEN, self.NUM_FEATURES],
                name="time")
        else:
            time = self.default_batch[0]
        return [time, self.input.sequence, self.target, self.input.length]
            
    @cached_property
    def input(self):
        """ namedtuple (sequence, length): the input (batch)
        
        """
        if self.default_batch is not None:
            input_sequence = tf.placeholder_with_default(
                self.default_batch[1], 
                shape=[self.BATCH_SIZE, self.MAX_SEQUENCE_LEN, self.NUM_FEATURES],
                name="input_sequence")
            input_seq_len = tf.placeholder_with_default(
                self.default_batch[3],
                shape=(None), 
                name="input_seq_length")
        else:
            input_sequence = tf.placeholder(
                tf.float32, 
                shape=[self.BATCH_SIZE, self.MAX_SEQUENCE_LEN, self.NUM_FEATURES],
                name="input_sequence")
            input_seq_len = tf.placeholder(
                tf.int32, 
                shape=(None), 
                name="input_seq_length")
        input_seq_len.set_shape((None,))    
        return self.Input(sequence=input_sequence,
                     length=input_seq_len)

    @cached_property
    def target(self):
        """ target batch
        """
        if self.default_batch is not None:
            target = tf.placeholder_with_default(
                self.default_batch[2], 
                shape=[self.BATCH_SIZE, self.MAX_SEQUENCE_LEN, self.NUM_TARGETS],
                name="target")
        else:
            target = tf.placeholder(
                tf.float32, 
                shape=[self.BATCH_SIZE, self.MAX_SEQUENCE_LEN, self.NUM_TARGETS],
                name="target")
        return target
    
    @cached_property
    def prediction(self):
        """ BLSTM logits build from the input
        """
        lstm_layer_input = self.input.sequence
        for n_layer, num_hidden_neurons in enumerate(self._lstm_sizes):
            lstm_cell_fw = LSTMCell(
                num_hidden_neurons, state_is_tuple=True)
            lstm_cell_bw = LSTMCell(
                num_hidden_neurons, state_is_tuple=True)
            outputs, _ = tf.nn.bidirectional_dynamic_rnn(
                lstm_cell_fw, lstm_cell_bw,
                inputs=lstm_layer_input,
                sequence_length=self.input.length,
                dtype=tf.float32,
                scope='LSTM_' + str(n_layer + 1))
            output_fw = outputs[0]
            output_bw = outputs[1]
            lstm_layer_output = tf.concat([output_fw, output_bw], 2)
            lstm_layer_input = lstm_layer_output
        output = tf.reshape(lstm_layer_output, [-1, 2 * num_hidden_neurons])
        W = tf.Variable(tf.truncated_normal(
            [2 * num_hidden_neurons, self.NUM_TARGETS],
            stddev=0.1))
        b = tf.Variable(tf.constant(0.1, shape=[self.NUM_TARGETS]))
        output = tf.matmul(output, W) + b
        batch_size = tf.shape(self.input.sequence)[0]
        output = tf.reshape(output, [batch_size, -1, self.NUM_TARGETS],
                            name="prediction")
        return output
    
    @cached_property
    def loss(self):
        """ lost/cost function of the batch
        """
        loss = tf.reduce_mean(self.loss_per_example, name="loss")
        tf.summary.scalar('loss', loss)
        return loss
    
    @cached_property
    def loss_per_example(self):
        """ tensor[batch_size]: the loss for each example
        """
        mask = tf.sequence_mask(self.input.length, 
                                tf.shape(self.input.sequence)[1],
                                tf.float32)
        mask = tf.reshape(mask, [tf.shape(self.input.sequence)[0],
                                 tf.shape(self.input.sequence)[1], 1])
        total_loss = tf.reduce_sum(
            mask * (self.prediction - self.target)**2,
            reduction_indices=[1, 2])
        loss = (total_loss / tf.to_float(self.input.length))**0.5
        return loss
    
    @cached_property
    def error(self):
        """ error for the batch
        """
        error = self.loss
        tf.summary.scalar('error', error)
        return error

    @cached_property
    def error_per_example(self):
        """ tensor[batchsize]: error for each example in the batch
        """
        return self.loss_per_example
        
        
    @cached_property
    def initializer(self):
        """ initialize local and global variables
        """
        return [tf.global_variables_initializer(),
                tf.local_variables_initializer()]
    
    @cached_property
    def global_step(self):
        """ total number of optimization steps
        """
        return tf.Variable(0, name='global_step', trainable=False)

    @cached_property
    def learning_rate(self):
        return tf.placeholder(tf.float32, [], name="learning_rate")
                
    @cached_property
    def _saver(self):
        return tf.train.Saver(keep_checkpoint_every_n_hours=0.5)
       
    def save_checkpoint(self, sess, global_step=None):
        name = "model.ckpt"
        try:
            path = os.path.join(self.path, name)
        except TypeError:
            raise ValueError("provide name and savepath in NeuralNetwork(...) to be able to save")
        self._saver.save(
            sess, path, global_step=global_step)
        print("variables have been saved")

    def restore(self, sess, path):
        """ restore the model (but not the graph from meta) in a given session
        
        Args:
            sess: tensorflow-session
            path: path to the ckpt-file
        """
        self._saver.restore(sess, path)   

    @cached_property
    def train_step(self):
        """ single training step
        """
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        train_step = optimizer.minimize(self.loss,
                                        global_step=self.global_step)
        return train_step
                
    @cached_property
    def summary(self):
        self.error, self.loss
        return tf.summary.merge_all()
    
    def get_gradients(self, time_indices):
        return [tf.gradients(self.prediction[:, i], self.input.sequence) 
                for i in time_indices]


class ImportedNeuralNetworks(NeuralNetworks):
    input = None
    target = None
    prediction = None
    loss = None

    def __init__(self, model_path):
        saver = tf.train.import_meta_graph(model_path + ".meta")
        graph = tf.get_default_graph()

        #if name is not None and model_path is not None:
        #    self.path = os.path.join(model_path, name)
        #else:
        self.path = None
        
        self.input = self.Input(
            sequence=graph.get_tensor_by_name("input_sequence:0"),
            length=graph.get_tensor_by_name("input_seq_length:0"))
        self.target = graph.get_tensor_by_name("target:0")
        self.prediction = graph.get_tensor_by_name("prediction:0")
        self.loss = graph.get_tensor_by_name("loss:0")
        self.op
        self.error
        self.train_step
        self.global_step


   
    
def record_to_batch(
    filenames, batch_size, shuffle=False,
     allow_smaller_final_batch=False, num_epochs=None):
    """ read records in batches

    Args:
        filenames (list of str), paths to the tfrecords to
            feed the queue
        batch_size (int or tf.Placeholder): number of examples per batch
        shuffle: whether all examples in the queue are suffled
        allow_smaller_final_batch (bool): the last batch might by smaller
            than batch_size
        num_epochs: number of times each file is read (None -> inf)
    
    Returns:
        list [time, ecg, respiration, length]
            time (tensor[BATCH_SIZE, MAX_SEQUENCE_LEN, 1])
            ecg (tensor[BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_FEATURES])
            respiration (tensor[BATCH_SIZE, MAX_SEQUENCE_LEN, NUM_TARGETS])
            length (tensor[BATCH_SIZE] of tf.int32)
    """
    def read_example(filenames, num_epochs):
        filename_queue = tf.train.string_input_producer(
                filenames, num_epochs=num_epochs)
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)

        _, sequence = tf.parse_single_sequence_example(
            serialized_example,
            sequence_features={
                'time': tf.FixedLenSequenceFeature(
                    [1], tf.float32),
                'ecg': tf.FixedLenSequenceFeature(
                    [1], tf.float32),
                'respiration': tf.FixedLenSequenceFeature(
                    [1], tf.float32)})
        length = tf.shape(sequence['ecg'])[0]
        time = sequence['time']
        ecg = sequence['ecg']
        respiration = sequence['respiration']
        return [time, ecg, respiration, length]
   
    def shuffled(tensors):
        """ return the tensor(s) in shuffled order
        """
        shuffle = tf.RandomShuffleQueue(capacity=10000,
            dtypes=[tensor.dtype for tensor in tensors],
            #shapes=[tensor.shape for tensor in tensors],  # does not work
            min_after_dequeue=5000)

        shuffle_ops = [shuffle.enqueue(tensors)]

        tf.train.add_queue_runner(tf.train.QueueRunner(shuffle, shuffle_ops))

        tensors_shuffled = shuffle.dequeue()
        for tensor_shuffled, tensor in zip(tensors_shuffled, tensors):
            tensor_shuffled.set_shape(tensor.get_shape())
        return tensors_shuffled

    def batched_example(example, batch_size):
        """ return examples in minibatches
        """
        batch = tf.train.batch(
            example,
            batch_size=batch_size, capacity=1000, dynamic_pad=True,
            allow_smaller_final_batch=allow_smaller_final_batch)
        return batch

    example = read_example(filenames, num_epochs)

    if shuffle:
        example = shuffled(example)
    return batched_example(example, [batch_size])



class Plot(object):
    """ Interactive plot to monitor training progress.
    
    Show two panels in the upper panel the input and in the 
    lower the target and prediction. The prediciton can be
    updated to shown progress.
    """
    def __init__(self, x, signal, target, prediction_init):
        """

        Args:
            x (array): the common x-value for all other signals
            signal (array like x): fixed signal to show in upper panel
            target (array like x): fixed target to shown in lower panel
            prediction_init (array like x): initial prediction to
                shown in lower panel. This can be updated via update()

        """
        plt.ion()
        self.fig , (ax_input, ax_output) = plt.subplots(2, 1)
        ax_input.set_xlim(min(x), max(x))
        ax_input.set_ylim(-2, 2)
        ax_output.set_xlim(min(x), max(x))
        ax_output.set_ylim(-1.5, 1.5)

        self.signal = lines.Line2D([], [], color='r', label="input")
        self.target = lines.Line2D([], [], color='b', label="target")
        self.prediction = lines.Line2D([], [], color='g', label="prediction")

        ax_input.add_line(self.signal)
        ax_output.add_line(self.target)
        ax_output.add_line(self.prediction)
        plt.legend()
        plt.show()

        self.signal.set_xdata(x)
        self.target.set_xdata(x)
        self.prediction.set_xdata(x)
        self.signal.set_ydata(signal)
        self.target.set_ydata(target)
        self.prediction.set_ydata(prediction_init)
        self.fig.canvas.draw()

    def update(self, prediction):
        self.prediction.set_ydata(prediction)
        self.fig.canvas.draw()


class Batches:
    """ Load the tf-record in batches.

    Attributes:
        SUBJECTS_TRAIN: list of all subject in training set, only preselected segments
        SUBJECTS_TEST: list of all subject in test set, only preselected segments
        SUBJECTS_UNSEGMENTED_TRAIN:  list of all subject in training set, all data
        SUBJECTS_UNSEGMENTED_TEST:  list of all subject in test set, all data
        SYNTHETIC_TRAIN: list of different training synthetic subjects of varios parameters
        SYNTHETIC_TEST: list of different test synthetic subjects of varios parameters
    """
    SUBJECTS_TRAIN = ['f1o04', 'f1o05', 'f1o06', 'f1o07', 'f1o08', 'f1o09', 'f1o10',  'f1y04', 'f1y05', 'f1y06', 'f1y07', 'f1y08', 'f1y09', 'f1y10',  'f2o04', 'f2o05', 'f2o06', 'f2o07', 'f2o08', 'f2o09', 'f2o10',  'f2y04', 'f2y05', 'f2y06', 'f2y07', 'f2y08', 'f2y09', 'f2y10']
    SUBJECTS_TEST = ['f1o01', 'f1o02', 'f1o03',  'f1y01', 'f1y02', 'f1y03','f2o01', 'f2o02', 'f2o03', 'f2y01', 'f2y02', 'f2y03']
    SUBJECTS_UNSEGMENTED_TRAIN = ['{}_all_train'.format(subject) for subject in SUBJECTS_TRAIN] #train means first hour
    SUBJECTS_UNSEGMENTED_TEST = ['{}_all_train'.format(subject) for subject in SUBJECTS_TEST] #train means first hour
    SUBJECTS_UNSEGMENTED_TRAIN_SIGNLE = 'f1o01_all_train'
    SUBJECTS_UNSEGMENTED_TEST_SIGNLE = 'f1o01_all_test'
    _EVALUATION_BATCH_SIZE = 10000  # 
    DEFAULT_BATCH_SIZE = 64

    
    def __init__(self, data_path):
        self._data_path = data_path
    
    def _filename(self, subject):
        return "{}.tfrecords".format(subject)

    def _path(self, subject):
        filename = self._filename(subject)
        return os.path.join(self._data_path, filename)

    @classmethod
    def synthetic_subject(self, esk, rsa, type_="test"):
        return "synthetic_esk={}_rsa={}_{}".format(esk, rsa, type_)
    
    def symbolic(self, subjects, batch_size=None):
        """ Return symbolic batch (list of tensors) for the given subject(s)

        The Batches are suffled.

        Args:
            subjects (str or list of str): the subjects
            batch_size (int): number of examples per batch  

        Returns:
            list [time, ecg, respiration, length]: were each is a tensor
        """
        if type(subjects) == str:
            subjects = [subjects]
        filepaths = [self._path(subject) for subject in subjects]
        print("open record:", filepaths)
        if batch_size is None:
            batch_size = tf.placeholder_with_default(tf.constant(self.DEFAULT_BATCH_SIZE, tf.int32), [])
        return record_to_batch(filepaths, batch_size, shuffle=True)
    
    def evaluated(self, subjects, batch_size=_EVALUATION_BATCH_SIZE):
        """ Return evaluted batch (list of arrays) for the given subject(s) 

        Args:
            subjects (str or list of str): the subjects
            batch_size (int): number of examples

        Returns:
            list [time, ecg, respiration, length]: were each is a array
        """
        if type(subjects) == str:
            subjects = [subjects]
        filepaths = [self._path(subject) for subject in subjects]
        with tf.Graph().as_default():
            batch = record_to_batch(filepaths,
                                    batch_size=batch_size,
                                    allow_smaller_final_batch=True,
                                    shuffle=True, num_epochs=1)
            coord = coordinator.Coordinator()
            with tf.Session() as sess:
                sess.run([tf.global_variables_initializer(),
                          tf.local_variables_initializer()])  
                threads = queue_runner.start_queue_runners(sess, coord)
                evaled_batch = sess.run(batch)
                coord.request_stop()
                coord.join(threads)
        print("loaded evaluation from {} with {} elements"\
            .format(filepaths, len(evaled_batch[0])))

        return evaled_batch


class Training:
    """ Train a given tensorflow graph.

    The graph must have a build in queue providing the traing data.
    The learning_rate placeholder of the graph is used to control the
    learing shedule.
    """
    Evaluation = namedtuple("Evaluation", ["batch", "writer", "name", "plot"])
    
        
    
    def __init__(self, num_steps=2000, learning_rate_default=0.003, 
                 learning_rate_final=0.0003, num_final_steps=500,
                 validation_interval=5, show_evaluation=False):
        self.LEARNING_RATE_DEFAULT = learning_rate_default
        self.NUM_STEPS = num_steps
        self.LEARNING_RATE_FINAL = learning_rate_final
        self.NUM_FINAL_STEPS = num_final_steps
        self.VALIDATION_INTERVAL = validation_interval
        self._is_interactive_plot = show_evaluation
    def _get_evaluation(self, batch, writer_path=None, name=None, show=False):
        graph = tf.get_default_graph()
        if writer_path is not None:
            writer = tf.summary.FileWriter(writer_path, graph)
        else:
            writer = None
        
        if self._is_interactive_plot and show:
            index = 0
            length = batch[3][index]
            plot = Plot(batch[0][index][:length], 
                        batch[1][index][:length],
                        batch[2][index][:length],
                        np.zeros(length))
        else:
            plot = None
        return self.Evaluation(batch=batch, writer=writer, name=name, plot=plot)

    def _evaluate(self, net, sess, evaluation):
        feed_dict = {key: value for key, value in zip(net.default_batch, evaluation.batch)}
        summary, error, prediction, target, step = sess.run(
            [net.summary, net.error, net.prediction,
             net.target, net.global_step], 
            feed_dict=feed_dict)

        evaluation.writer.add_summary(summary, step)
        print("Evaluation ({}): {:2.0f}%".format(evaluation.name, 100*error))
        if evaluation.plot is not None:
            index = 0
            length = evaluation.batch[3][index]
            evaluation.plot.update(prediction[index][:length].reshape(-1))
        return error
    
    def train(self, net, evaluation_batches, restore=None):
        """ Train a given net

        Args:
            net (NeuralNetworks): the graph to train its variables
            evaluation_batches (list of evaled_batch): a list of evaled batches
            restore: str or None. If not None the model (the variables)
                were loaded from this path before training
        
       Returns: None
            It saves checkpoints of the trained graph (see NeuralNetworks)
        """


    
              
        evaluations = [self._get_evaluation(batch, 
                                            os.path.join(net.path, str(i)), 
                                            "evaluation_{}".format(i), 
                                            (i == 0))
                       for i, batch in enumerate(evaluation_batches)]
    
        
        with tf.Session() as sess:
            sess.run(net.initializer)
            if restore is not None:
                net.restore(sess, restore)
            tf.train.start_queue_runners(sess=sess)
            while True:
                step = sess.run(net.global_step)
                if step % 10 == 0:
                    print(step)
                if step == self.NUM_STEPS:
                    break
                if step < self.NUM_STEPS - self.NUM_FINAL_STEPS:
                    learning_rate = self.LEARNING_RATE_DEFAULT
                else:
                    learning_rate = self.LEARNING_RATE_FINAL
                sess.run(net.train_step, 
                    feed_dict={net.learning_rate: learning_rate})
 
                if (step % self.VALIDATION_INTERVAL) == 0:
                    for evaluation in evaluations:
                        self._evaluate(net, sess, evaluation)

                if step % 200 == 0:
                    net.save_checkpoint(sess, step)


def detect_rpeaks_indices(ecg, sampling_rate, rate_zoom_factor=4):
    """ Return the times of the R-peaks in a ECG signal

    Args:
        ecg (1D array): the ecg signal
    Returns:
         list(float): the times od the R-peaks
    """       
    ecg = np.squeeze(ecg)
    ecg_extended = np.column_stack(rate_zoom_factor * [ecg]).reshape(-1)
    out = biosppy.ecg.ecg(signal=ecg_extended, 
                          sampling_rate=rate_zoom_factor * sampling_rate,
                          show=False)
    return out["rpeaks"]/rate_zoom_factor


HALF_WINDOW = 100
def find_window_around_closest_rpeak(signal, time_indices, ecgs, lengths, times):
    """


    """
    #from .resp import Resp
    #respiration = Resp(respiration, sampling_rate)
    for time_idx, signal_t in zip(time_indices, signal):
        for ecg, gradients_example, length, time in zip(ecgs, signal_t, lengths, times):
            rpeaks = detect_rpeaks_indices(ecg, 25)
            peak_idx = int(round(rpeaks[np.argmin(abs(rpeaks - time_idx))]))
            if HALF_WINDOW < peak_idx < length - HALF_WINDOW:
                window = slice(peak_idx-HALF_WINDOW, peak_idx+HALF_WINDOW)
                yield gradients_example[window], ecg[window], time[window]
                


class Inference:
    """ Infer the learned knowlege of a given net.

    Attributed (cached properties which are only calculated on demand):
        ranked_example (int[BATCHSIZE]): example indices sorted by there error
        default_idx (int): indice of the example with median error
        gradient (float[TIME_STEPS, BATCH_SIZE, max_sequence_length]):
            gradient[TARGET_TIME, EXAMPLE, INPUT_TIME] of the target[TARGET_TIME]
            with respect to input[INPUT_TIME] for EXAMPLE
        mean_gradient: gradients average over all examples in the batc 
    """
    TIME_STEPS = 50
    SAMPLING_RATE = 25.
    def __init__(self, net, sess, batch, name):
        self._sess = sess
        self._net = net
        self._batch = batch
        self._name = name
        self._feed_dict = {key: value for key, value in zip(net.batch, batch)}
        self.error, self.time, self.ecg, self.respiration, self.prediction, self.length \
            = sess.run([net.error_per_example, net.batch[0], net.input.sequence, 
                        net.target, net.prediction, net.input.length],
                       feed_dict=self._feed_dict)
       
    @classmethod
    def with_new_session(cls, net, restore_path, batch, name=None):
        sess = tf.Session()
        net.restore(sess, restore_path)
        if name is None:
            name = restore_path.split("/")[1].replace("_train", "")
        return cls(net, sess, batch, name)
    
    @cached_property
    def ranked_example(self):
        return np.argsort(self.error)
    
    
    def plot_historgram(self, save=False):
        plt.hist(self.error, 60, label="out of sample error")
        plt.title(self._name)
        plt.legend()
        #plt.title()
        if save:
            plt.savefig("Figures/histogram_{}.png".format(
                self._name))
        plt.show()
        
    @property
    def default_idx(self):
        #print(self.error[self.ranked_example])
        return self.ranked_example[len(self.ranked_example) // 2]
    
    def plot_single_example(self, idx=None, save=False):
        if idx is None:
            idx = self.default_idx
        plot(self.time[idx][:self.length[idx]], 
             self.ecg[idx][:self.length[idx]], 
             self.respiration[idx][:self.length[idx]], 
             self.prediction[idx][:self.length[idx]],
                title="error: {}".format(self.error[idx]))
        if save:
            plt.savefig("Figures/single_example_{}.png".format(
              self._name))
        plt.show()

        
    @cached_property
    def _time_indices(self):
        return list(range(0, len(self.time[0]), self.TIME_STEPS))

    @cached_property
    def gradients(self):
        print("generate symbolic gradients")
        symbolic_gradients = self._net.get_gradients(self._time_indices)
        print("evalute gradients")
        return np.squeeze(np.array(
            self._sess.run(symbolic_gradients, feed_dict=self._feed_dict)))
        
    def plot_single_gradient(self, time_idx, idx=None, save=False):
        if idx is None:
            idx = self.default_idx
        plot(self.time[idx][:self.length[idx]], 
             self.ecg[idx][:self.length[idx]], 
             self.respiration[idx][:self.length[idx]], 
             prediction=self.prediction[idx][:self.length[idx]],
             gradient=self.gradients[time_idx][idx][:self.length[idx]])
        if save:
            plt.savefig("Figures/single_gradient_train_{}.png".format(
                self._name))
        plt.show()

    @cached_property
    def mean_gradient(self):
        print("average gradients")
        mean_gradient = np.zeros(2 * HALF_WINDOW)
        mean_ecg = np.zeros(2 * HALF_WINDOW)
        correlation = np.zeros(2 * HALF_WINDOW)
        slope_correlation = np.zeros(2 * HALF_WINDOW)  # correlation between slope and gradient
        i=0
        for single_gradient, single_ecg, _ in find_window_around_closest_rpeak(self.gradients, self._time_indices, self.ecg, self.length, self.time):
            mean_gradient += single_gradient
            correlation += np.squeeze(single_ecg) * single_gradient
            slope_correlation += np.gradient(np.squeeze(single_ecg)) * single_gradient
            mean_ecg += np.squeeze(single_ecg)
            i += 1
        mean_ecg = mean_ecg / i
        mean_gradient = mean_gradient / i
        correlation = correlation / i
        slope_correlation = slope_correlation / i
        mea
        return [mean_ecg, mean_gradient, correlation, slope_correlation]

    def plot_mean_gradient(self, save=False):
        mean_ecg, mean_gradient, correlation, slope_correlation = self.mean_gradient
        time = np.arange(-HALF_WINDOW, HALF_WINDOW)/self.SAMPLING_RATE
        ax = plt.subplot(211)
        plt.title(self._name)
        ax.plot(time, mean_ecg, label="ecg")
        ax.plot(time, mean_gradient, label="gradient")
        plt.subplot(212, sharex=ax)
        plt.plot(time, correlation, label="correlation")
        plt.plot(time, slope_correlation, label="slope_correlation")
        plt.legend()
        if save:
            plt.savefig("Figures/mean_gradient_train_{}.png".format(
                self._name))
        plt.show()
    def phase_error(self):
        pass
        
def plot(time, ecg, respiration, prediction=None, gradient=None, title=None):
    """ plot ecg, and respiration (target and prediction).
    
    all arrays must have same dimension
    Args:
        time (array):
        ecg (array like time):
        respiration (array like time):
        prediciton (array like time) or None:
        gradient (array like time) or None:
        title (str or None):
    
    """
    ax = plt.subplot(211)
    if title is not None:
        plt.title(title)
    ax.plot(time, ecg, label="ecg")
    plt.legend()
    if gradient is not None:
        ax.plot(time, gradient, label="gradient")
    plt.legend()
    plt.subplot(212, sharex=ax)
    plt.plot(time, respiration, 
             label="resp" if prediction is None else "target resp")
    if prediction is not None:
        plt.plot(time, prediction, label="predicted resp")
    plt.legend()


def main(data_path="/scratch/dv/ekg/tfrecords",
         model_path="/data/ekg/tfmodels/",
         model_name="unsegmented_large"):
    
    batches = Batches(data_path)
    #train_subject = model.Batches.SUBJECTS_UNSEGMENTED_TRAIN
    train_subject = [subject + "_all_train" for subject in batches.SUBJECTS_TRAIN]
    train_subject += [subject + "_all_test" for subject in batches.SUBJECTS_TRAIN]
    #train_subject = [batches.synthetic_subject(esk=esk,rsa=rsa,type_="train")
    #                  for esk in [0, 0.03, 0.1]
    #                  for rsa in [0, 0.3, 1.]]
    #test_subject = batches.synthetic_subject(esk=0.03,rsa=0.3,type_="test")
    test_subject = batches.SUBJECTS_TEST[0] + "_all_train"
    test_batch = batches.evaluated(test_subject)
    batch = batches.symbolic(train_subject)
    net = NeuralNetworks(
        default_batch=batch,
        lstm_sizes=[25, 25],
        model_path=model_path,
        name=model_name)

    training = Training(num_steps=4000)
    training.train(net, [test_batch])

if __name__ == "__main__":
    main()
