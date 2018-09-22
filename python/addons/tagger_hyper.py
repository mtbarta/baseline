import math
import json
from baseline.model import Tagger, create_tagger_model, load_tagger_model
import os
import tensorflow as tf
import numpy as np
import pickle
import time
import random
from random import shuffle
import math
import argparse
from datetime import datetime
now = datetime.now()


def str2bool(answer):
    answer = answer.lower()
    if answer in ['y', 'yes']:
        return True

    if answer in ['n', 'no']:
        return False

    print('Invalid answer: ' + answer)
    print('Exiting..')
    exit()


parser = argparse.ArgumentParser()

parser.add_argument("--base_name", type=str, help="", default='')

# Path to the folder where your *_dataset/ folders live.
parser.add_argument("--root_path", type=str, help="Root path", default='/path/to/your/data/folders/')

parser.add_argument("--dataset", type=str, help="SNLI/PRFX10/PRFX30/PRFX50", default='SNLI')

# Word embeddings params
parser.add_argument("--word_dim", type=int, help="Word and hidden state embedding dimensions", default=5)
parser.add_argument("--word_init_avg_norm", type=float, help="Word init max val per dim.", default=0.001)
parser.add_argument("--inputs_geom", type=str, help="Input geometry: eucl/hyp.", default='eucl')


# RNN params
parser.add_argument("--cell_type", type=str, help="rnn/gru/TFrnn/TFgru/TFlstm", default='rnn')
parser.add_argument("--cell_non_lin", type=str, help="id/relu/tanh/sigmoid.", default='id')
parser.add_argument("--sent_geom", type=str, help="Sentence geometry eucl/hyp", default='eucl')
parser.add_argument("--bias_geom", type=str, help="RNN bias geometry: eucl/hyp.", default='eucl')

parser.add_argument("--fix_biases", type=str2bool, help="Biases are not trainable: y/n", default='n')
parser.add_argument("--fix_matrices", type=str2bool, help="y/n: If y, matrix weights are kept fixed as eye matrices", default='n')
parser.add_argument("--matrices_init_eye", type=str2bool, help="Matrix weights are initialized as eye matrices: y/n", default='n')


# FFNN and MLR params
parser.add_argument("--before_mlr_dim", type=int, help="Embedding dimension after FFNN, but before MLR.", default=5)
parser.add_argument("--ffnn_non_lin", type=str, help="id/relu/tanh/sigmoid.", default='id')
parser.add_argument("--ffnn_geom", type=str, help="FFNN geometry: eucl/hyp.", default='eucl')
parser.add_argument("--additional_features", type=str, help="Input of the final FFNN, besides sentence emb. Either empty string or dsq (distance squared, Euclidean or hyperbolic).", default='')
parser.add_argument("--dropout", type=float, help="dropout probability for FFNN layers", default=1.0)
parser.add_argument("--mlr_geom", type=str, help="MLR geometry: eucl/hyp.", default='eucl')

parser.add_argument("--proj_eps", type=float, help="", default=1e-5)


# L2 regularization:
parser.add_argument("--reg_beta", type=float, help="", default=0.0)

# Optimization params
parser.add_argument("--hyp_opt", type=str, help="Optimization technique. Only when inputs_geom is hyp. projsgd/rsgd.", default='rsgd')
parser.add_argument("--lr_ffnn", type=float, help="learning rate for the FFNN and MLR layers", default=0.01)
parser.add_argument("--lr_words", type=float, help="learning rate for words (updated rarely).", default=0.1)

parser.add_argument("--batch_size", type=int, help="", default=64)
parser.add_argument("--burnin", type=str2bool, help="y/n on whether to do burnin", default='n') ##### Seems to hurt

parser.add_argument("--c", type=float, help="c", default=1.0)

parser.add_argument("--restore_model", type=str2bool, help="y/n: restore model", default='n')
parser.add_argument("--restore_from_path", type=str, help="", default="")

args = parser.parse_args()

######################### Parse arguments ####################################
num_classes = 2
num_epochs = 30


root_path = args.root_path

PROJ_EPS = args.proj_eps

import util
util.PROJ_EPS = PROJ_EPS
import rnn_impl

dataset = args.dataset
assert dataset in ['SNLI', 'PRFX10', 'PRFX30', 'PRFX50']

c_val = args.c

cell_non_lin = args.cell_non_lin
ffnn_non_lin = args.ffnn_non_lin

cell_type = args.cell_type
word_dim = args.word_dim
hidden_dim = word_dim
word_init_avg_norm = args.word_init_avg_norm

additional_features = args.additional_features
assert additional_features in ['', 'dsq']

dropout = args.dropout
hyp_opt = args.hyp_opt


sent_geom = args.sent_geom
inputs_geom = args.inputs_geom
bias_geom = args.bias_geom

ffnn_geom = args.ffnn_geom
mlr_geom = args.mlr_geom
before_mlr_dim = args.before_mlr_dim

assert hyp_opt in ['rsgd', 'projsgd']
assert sent_geom in ['eucl', 'hyp']
assert inputs_geom in ['eucl', 'hyp']
assert bias_geom in ['eucl', 'hyp']
assert mlr_geom in ['eucl', 'hyp']
assert ffnn_geom in ['eucl', 'hyp']

if sent_geom == 'eucl':
    assert inputs_geom == 'eucl'
    assert bias_geom == 'eucl'
    assert ffnn_geom == 'eucl'
    assert mlr_geom == 'eucl'

if ffnn_geom == 'hyp':
    assert sent_geom == 'hyp'

if ffnn_geom == 'eucl':
    assert mlr_geom == 'eucl'

if mlr_geom == 'hyp':
    assert ffnn_geom == 'hyp'
    assert sent_geom == 'hyp'

fix_biases = args.fix_biases
fix_biases_str = ''
if fix_biases:
    fix_biases_str = 'FIX'

fix_matrices = args.fix_matrices
matrices_init_eye = args.matrices_init_eye
mat_str = ''
if fix_matrices or matrices_init_eye:
    mat_str = 'W'
    if fix_matrices:
        mat_str = mat_str + 'FIXeye'
    elif matrices_init_eye:
        mat_str = mat_str + 'eye'

burnin = args.burnin
lr_ffnn = args.lr_ffnn
lr_words = args.lr_words
batch_size = args.batch_size

reg_beta = args.reg_beta

restore_model = args.restore_model
assert  restore_model == False
restore_from_path = args.restore_from_path

if inputs_geom == 'hyp' or bias_geom == 'hyp' or ffnn_geom =='hyp' or mlr_geom == 'hyp':
    hyp_opt_str = hyp_opt + '_lrW' + str(lr_words) + '_lrFF' + str(lr_ffnn) + '_'
else:
    hyp_opt_str = ''

if c_val != 1.0:
    c_str = 'C'  + str(c_val) + '_'
else:
    c_str = ''

if dropout != 1.0:
    drp_str = 'drp' + str(dropout) + '_'
else:
    drp_str = ''

burnin_str = ''
if burnin:
    burnin_str = 'burn' + str(burnin).lower()

reg_beta_str = ''
if reg_beta > 0.0:
    reg_beta_str = 'reg' + str(reg_beta) + '_'

additional_features_str = additional_features
if additional_features != '':
    additional_features_str = additional_features + '_'

tensorboard_name = args.base_name + '_' +\
                   dataset + '_' +\
                   'W' + str(word_dim) + 'd,' + str(word_init_avg_norm) + 'init_' + \
                   cell_type + '_' + \
                   'cellNonL' + cell_non_lin + '_' +\
                   'SENT' + sent_geom + '_' + \
                   'INP' + inputs_geom + '_' + \
                   'BIAS' + bias_geom + fix_biases_str + '_' + mat_str +\
                   'FFNN' + ffnn_geom + str(before_mlr_dim) + ffnn_non_lin + '_' +\
                   additional_features_str + \
                   drp_str +\
                   'MLR' + mlr_geom + '_' + \
                   reg_beta_str + \
                   hyp_opt_str + \
                   c_str +\
                   'prje' + str(PROJ_EPS) + '_' + \
                   'bs' + str(batch_size) + '_' +\
                   burnin_str +  '__' + now.strftime("%H:%M:%S,%dM")

name_experiment = tensorboard_name
logger = util.setup_logger(name_experiment, logs_dir= os.path.join(root_path, 'logs/'), also_stdout=True)
logger.info('PARAMS :  ' + name_experiment)
logger.info('')
logger.info(args)

dtype = tf.float64


class HyperbolicRNNModel(Tagger):
    def __init__(self, word_to_id, id_to_word):
        # self.word_to_id = word_to_id
        # self.id_to_word = id_to_word

        # self.construct_placeholders()
        # self.construct_execution_graph()

    def save_values(self, basename):
        self.saver.save(self.sess, basename)

    def save_md(self, basename):

        path = basename.split('/')
        base = path[-1]
        outdir = '/'.join(path[:-1])

        state = {"mxlen": self.mxlen, "maxw": self.maxw, "crf": self.crf, "proj": self.proj, "crf_mask": self.crf_mask, 'span_type': self.span_type}
        with open(basename + '.state', 'w') as f:
            json.dump(state, f)

        tf.train.write_graph(self.sess.graph_def, outdir, base + '.graph', as_text=False)
        with open(basename + '.saver', 'w') as f:
            f.write(str(self.saver.as_saver_def()))

        with open(basename + '.labels', 'w') as f:
            json.dump(self.labels, f)
            
        if len(self.word_vocab) > 0:
            with open(basename + '-word.vocab', 'w') as f:
                json.dump(self.word_vocab, f)

        with open(basename + '-char.vocab', 'w') as f:
            json.dump(self.char_vocab, f)

    def make_input(self, batch_dict, do_dropout=False):
        x = batch_dict['x']
        y = batch_dict.get('y', None)
        xch = batch_dict['xch']
        lengths = batch_dict['lengths']

        pkeep = 1.0-self.pdrop_value if do_dropout else 1.0

        if do_dropout and self.pdropin_value > 0.0:
            UNK = self.word_vocab['<UNK>']
            PAD = self.word_vocab['<PAD>']
            drop_indices = np.where((np.random.random(x.shape) < self.pdropin_value) & (x != PAD))
            x[drop_indices[0], drop_indices[1]] = UNK
        feed_dict = {self.x: x, self.xch: xch, self.lengths: lengths, self.pkeep: pkeep}
        if y is not None:
            feed_dict[self.y] = y
        return feed_dict

    def save(self, basename):
        self.save_md(basename)
        self.save_values(basename)

    def predict_text(self, text):
        summary, loss, argmax_idx = \
          sess.run([self.summary_merged, self.loss, self.argmax_idx], feed_dict={
              self.word_ids_1: batch_word_ids_1,
              self.num_words_1: batch_num_words_1,
              self.word_ids_2: batch_word_ids_2,
              self.num_words_2: batch_num_words_2,
              self.label_placeholder: batch_label,
              self.dropout_placeholder: 1.0
          })
        self.test_summary_writer.add_summary(summary, summary_i)

    @staticmethod
    def load(basename, **kwargs):
        basename = unzip_model(basename)
        model = RNNTaggerModel()
        model.sess = kwargs.get('sess', tf.Session())
        checkpoint_name = kwargs.get('checkpoint_name', basename)
        checkpoint_name = checkpoint_name or basename
        with open(basename + '.state') as f:
            state = json.load(f)
            model.mxlen = state.get('mxlen', 100)
            model.maxw = state.get('maxw', 100)
            model.crf = bool(state.get('crf', False))
            model.crf_mask = bool(state.get('crf_mask', False))
            model.span_type = state.get('span_type')
            model.proj = bool(state.get('proj', False))

        with open(basename + '.saver') as fsv:
            saver_def = tf.train.SaverDef()
            text_format.Merge(fsv.read(), saver_def)

        with gfile.FastGFile(basename + '.graph', 'rb') as f:
            gd = tf.GraphDef()
            gd.ParseFromString(f.read())
            model.sess.graph.as_default()
            tf.import_graph_def(gd, name='')

            model.sess.run(saver_def.restore_op_name, {saver_def.filename_tensor_name: checkpoint_name})
            model.x = tf.get_default_graph().get_tensor_by_name('x:0')
            model.xch = tf.get_default_graph().get_tensor_by_name('xch:0')
            model.y = tf.get_default_graph().get_tensor_by_name('y:0')
            model.lengths = tf.get_default_graph().get_tensor_by_name('lengths:0')
            model.pkeep = tf.get_default_graph().get_tensor_by_name('pkeep:0')
            model.best = tf.get_default_graph().get_tensor_by_name('output/ArgMax:0')
            model.probs = tf.get_default_graph().get_tensor_by_name('output/Reshape_1:0')  # TODO: rename
            try:
                model.A = tf.get_default_graph().get_tensor_by_name('Loss/transitions:0')
                #print('Found transition matrix in graph, setting crf=True')
                if not model.crf:
                    print('Warning: meta-data says no CRF but model contains transition matrix!')
                    model.crf = True
            except:
                if model.crf is True:
                    print('Warning: meta-data says there is a CRF but not transition matrix found!')
                model.A = None
                model.crf = False

        with open(basename + '.labels', 'r') as f:
            model.labels = json.load(f)

        model.word_vocab = {}
        if os.path.exists(basename + '-word.vocab'):
            with open(basename + '-word.vocab', 'r') as f:
                model.word_vocab = json.load(f)

        with open(basename + '-char.vocab', 'r') as f:
            model.char_vocab = json.load(f)

        model.saver = tf.train.Saver(saver_def=saver_def)
        return model

    def save_using(self, saver):
        self.saver = saver

    def _compute_word_level_loss(self, mask):

        nc = len(self.labels)
        # Cross entropy loss
        cross_entropy = tf.one_hot(self.y, nc, axis=-1) * tf.log(tf.nn.softmax(self.probs))
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        all_loss = tf.reduce_mean(cross_entropy, name="loss")
        return all_loss

    def _compute_sentence_level_loss(self):

        if self.crf_mask:
            assert self.span_type is not None, "To mask transitions you need to provide a tagging span_type, choices are `IOB`, `BIO` (or `IOB2`), and `IOBES`"
            A = tf.get_variable(
                "transitions_raw",
                shape=(len(self.labels), len(self.labels)),
                dtype=tf.float32,
                trainable=True
            )

            self.mask = crf_mask(self.labels, self.span_type, self.labels['<GO>'], self.labels['<EOS>'], self.labels.get('<PAD>'))
            self.inv_mask = tf.cast(tf.equal(self.mask, 0), tf.float32) * tf.constant(-1e4)

            self.A = tf.add(tf.multiply(A, self.mask), self.inv_mask, name="transitions")
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths, self.A)
        else:
            ll, self.A = tf.contrib.crf.crf_log_likelihood(self.probs, self.y, self.lengths)
        return tf.reduce_mean(-ll)

    def create_loss(self):

        with tf.variable_scope("Loss"):
            gold = tf.cast(self.y, tf.float32)
            mask = tf.sign(gold)

            if self.crf is True:
                print('crf=True, creating SLL')
                all_loss = self._compute_sentence_level_loss()
            else:
                print('crf=False, creating WLL')
                all_loss = self._compute_word_level_loss(mask)

        return all_loss

    def __init__(self):
        super(RNNTaggerModel, self).__init__()
        pass

    def get_vocab(self, vocab_type='word'):
        return self.word_vocab if vocab_type == 'word' else self.char_vocab

    def get_labels(self):
        return self.labels

    def predict(self, batch_dict):

        feed_dict = self.make_input(batch_dict)
        lengths = batch_dict['lengths']
        # We can probably conditionally add the loss here
        preds = []
        if self.crf is True:

            probv, tranv = self.sess.run([self.probs, self.A], feed_dict=feed_dict)
            batch_sz, _, label_sz = probv.shape
            start = np.full((batch_sz, 1, label_sz), -1e4)
            start[:, 0, self.labels['<GO>']] = 0
            probv = np.concatenate([start, probv], 1)

            for pij, sl in zip(probv, lengths):
                unary = pij[:sl + 1]
                viterbi, _ = tf.contrib.crf.viterbi_decode(unary, tranv)
                viterbi = viterbi[1:]
                preds.append(viterbi)
        else:
            # Get batch (B, T)
            bestv = self.sess.run(self.best, feed_dict=feed_dict)
            # Each sentence, probv
            for pij, sl in zip(bestv, lengths):
                unary = pij[:sl]
                preds.append(unary)

        return preds

    @staticmethod
    def create(labels, embeddings, **kwargs):

        word_vec = embeddings['word']
        char_vec = embeddings['char']
        model = RNNTaggerModel()
        model.sess = kwargs.get('sess', tf.Session())

        model.mxlen = kwargs.get('maxs', 100)
        model.maxw = kwargs.get('maxw', 100)

        hsz = int(kwargs['hsz'])
        pdrop = kwargs.get('dropout', 0.5)
        pdrop_in = kwargs.get('dropin', 0.0)
        rnntype = kwargs.get('rnntype', 'blstm')
        nlayers = kwargs.get('layers', 1)
        model.labels = labels
        model.crf = bool(kwargs.get('crf', False))
        model.crf_mask = bool(kwargs.get('crf_mask', False))
        model.span_type = kwargs.get('span_type')
        model.proj = bool(kwargs.get('proj', False))
        model.feed_input = bool(kwargs.get('feed_input', False))
        model.activation_type = kwargs.get('activation', 'tanh')

        char_dsz = char_vec.dsz
        nc = len(labels)
        model.x = kwargs.get('x', tf.placeholder(tf.int32, [None, model.mxlen], name="x"))
        model.xch = kwargs.get('xch', tf.placeholder(tf.int32, [None, model.mxlen, model.maxw], name="xch"))
        model.y = kwargs.get('y', tf.placeholder(tf.int32, [None, model.mxlen], name="y"))
        model.lengths = kwargs.get('lengths', tf.placeholder(tf.int32, [None], name="lengths"))
        model.pkeep = kwargs.get('pkeep', tf.placeholder(tf.float32, name="pkeep"))
        model.pdrop_value = pdrop
        model.pdropin_value = pdrop_in
        model.word_vocab = {}

        inputs_geom = kwargs.get("inputs_geom", "hyp")
        bias_geom = kwargs.get("bias_geom", "hyp")
        ffnn_geom = kwargs.get("ffnn_geom", "hyp")
        mlr_geom = kwargs.get("mlr_geom", "hyp")
        c_val = kwargs.get("c_val", 1.0)
        cell_non_lin = kwargs.get("cell_non_lin", "id") #"id/relu/tanh/sigmoid."
        ffnn_non_lin = kwargs.get("ffnn_non_lin", "id")

        if word_vec is not None:
            model.word_vocab = word_vec.vocab
        # model.char_vocab = char_vec.vocab
        seed = np.random.randint(10e8)
        if word_vec is not None:
            word_embeddings = embed(model.x, len(word_vec.vocab), word_vec.dsz,
                                    initializer=tf.constant_initializer(word_vec.weights, dtype=tf.float32, verify_shape=True))

        # Wch = tf.Variable(tf.constant(char_vec.weights, dtype=tf.float32), name="Wch")
        # ce0 = tf.scatter_update(Wch, tf.constant(0, dtype=tf.int32, shape=[1]), tf.zeros(shape=[1, char_dsz]))

        # word_char, _ = pool_chars(model.xch, Wch, ce0, char_dsz, **kwargs)
        # joint = word_char if word_vec is None else tf.concat(values=[word_embeddings, word_char], axis=2)
        joint = word_embeddings
        embedseq = tf.nn.dropout(joint, model.pkeep)
        if (mlr_geom == 'hyp'):
                embedseq = util.tf_exp_map_zero(embedseq, c_val)
        

        if cell_type == 'rnn' and sent_geom == 'hyp':
            cell_class = lambda h_dim: rnn_impl.HypRNN(num_units=h_dim,
                                                       inputs_geom=inputs_geom,
                                                       bias_geom=bias_geom,
                                                       c_val=c_val,
                                                       non_lin=cell_non_lin,
                                                       fix_biases=False,
                                                       fix_matrices=False,
                                                       matrices_init_eye=False,
                                                       dtype=tf.float32)
        elif cell_type == 'gru' and sent_geom == 'hyp':
            cell_class = lambda h_dim: rnn_impl.HypGRU(num_units=h_dim,
                                                       inputs_geom=inputs_geom,
                                                       bias_geom=bias_geom,
                                                       c_val=c_val,
                                                       non_lin=cell_non_lin,
                                                       fix_biases=fix_biases,
                                                       fix_matrices=fix_matrices,
                                                       matrices_init_eye=False,
                                                       dtype=tf.float32)
        
        if rnntype == 'rnn':
            rnnout = cell_class(hsz)
            rnnout = tf.contrib.rnn.DropoutWrapper(rnnout)
            rnnout, state = tf.nn.dynamic_rnn(rnnout, embedseq, sequence_length=model.lengths, dtype=tf.float32)
        elif rnntype == 'bi':
            rnnfwd = cell_class(hsz)
            rnnbwd = cell_class(hsz)
            rnnout, state = tf.nn.bidirectional_dynamic_rnn(rnnfwd, rnnbwd, embedseq, sequence_length=model.lengths, dtype=tf.float32)
            rnnout = tf.concat(axis=2, values=rnnout)
        else:
            rnnout = cell_class(hsz)
            rnnout = tf.contrib.rnn.DropoutWrapper(rnnout)
            rnnout, state = tf.nn.dynamic_rnn(rnnout, embedseq, sequence_length=model.lengths, dtype=tf.float32)
        
        tf.summary.scalar('RNN/word_emb1', tf.reduce_mean(tf.norm(word_embeddings, axis=2)))
        sent1_norm = util.tf_norm(state)
        tf.summary.scalar('RNN/sent1', tf.reduce_mean(sent1_norm))

        ################## first feed forward layer ###################

        # Define variables for the first feed-forward layer: W1 * s1 + W2 * s2 + b + bd * d(s1,s2)
        W_ff_s1 = tf.get_variable('W_ff_s1',
                                  dtype=dtype,
                                  shape=[hidden_dim, before_mlr_dim],
                                  initializer= tf.contrib.layers.xavier_initializer())

        # W_ff_s2 = tf.get_variable('W_ff_s2',
        #                           dtype=dtype,
        #                           shape=[hidden_dim, before_mlr_dim],
        #                           initializer= tf.contrib.layers.xavier_initializer())

        b_ff = tf.get_variable('b_ff',
                               dtype=dtype,
                               shape=[1, before_mlr_dim],
                               initializer=tf.constant_initializer(0.0))

        # b_ff_d = tf.get_variable('b_ff_d',
        #                          dtype=dtype,
        #                          shape=[1, before_mlr_dim],
        #                          initializer=tf.constant_initializer(0.0))

        eucl_vars += [W_ff_s1]
        if ffnn_geom == 'eucl' or bias_geom == 'eucl':
            eucl_vars += [b_ff]
            # if additional_features == 'dsq':
            #     eucl_vars += [b_ff_d]
        else:
            hyp_vars += [b_ff]
            # if additional_features == 'dsq':
            #     hyp_vars += [b_ff_d]


        if ffnn_geom == 'eucl' and sent_geom == 'hyp': # Sentence embeddings are Euclidean after log, except the proper distance (Eucl or hyp) is kept!
            model.sent_1 = util.tf_log_map_zero(model.sent_1, c_val)
            # model.sent_2 = util.tf_log_map_zero(model.sent_2, c_val)

        ####### Build output_ffnn #######
        if ffnn_geom == 'eucl':
            output_ffnn = tf.matmul(model.sent_1, W_ff_s1) + b_ff
            if additional_features == 'dsq': # [u, v, d(u,v)^2]
                output_ffnn = output_ffnn + b_ff_d

        else:
            assert sent_geom == 'hyp'
            ffnn_s1 = util.tf_mob_mat_mul(W_ff_s1, model.sent_1, c_val)
            # ffnn_s2 = util.tf_mob_mat_mul(W_ff_s2, model.sent_2, c_val)
            output_ffnn = util.tf_mob_add(ffnn_s1, c_val)

            hyp_b_ff = b_ff
            if bias_geom == 'eucl':
                hyp_b_ff = util.tf_exp_map_zero(b_ff, c_val)
            output_ffnn = util.tf_mob_add(output_ffnn, hyp_b_ff, c_val)

            # if additional_features == 'dsq': # [u, v, d(u,v)^2]
            #     hyp_b_ff_d = b_ff_d
            #     if bias_geom == 'eucl':
            #         hyp_b_ff_d = util.tf_exp_map_zero(b_ff_d, c_val)

            #     output_ffnn = util.tf_mob_add(output_ffnn,
            #                                   util.tf_mob_scalar_mul(d_sq_s1_s2, hyp_b_ff_d, c_val),
            #                                   c_val)

        if ffnn_geom == 'eucl':
            output_ffnn = util.tf_eucl_non_lin(output_ffnn, non_lin=ffnn_non_lin)
        else:
            output_ffnn = util.tf_hyp_non_lin(output_ffnn,
                                              non_lin=ffnn_non_lin,
                                              hyp_output = (mlr_geom == 'hyp' and dropout == 1.0),
                                              c=c_val)
        # Mobius dropout
        if dropout < 1.0:
            # If we are here, then output_ffnn should be Euclidean.
            output_ffnn = tf.nn.dropout(output_ffnn, keep_prob=model.pkeep)
            if (mlr_geom == 'hyp'):
                output_ffnn = util.tf_exp_map_zero(output_ffnn, c_val)


        ################## MLR ###################
        # output_ffnn is batch_size x before_mlr_dim

        A_mlr = []
        P_mlr = []
        logits_list = []
        for cl in range(num_classes):
            A_mlr.append(tf.get_variable('A_mlr' + str(cl),
                                         dtype=dtype,
                                         shape=[1, before_mlr_dim],
                                         initializer=tf.contrib.layers.xavier_initializer()))
            eucl_vars += [A_mlr[cl]]

            P_mlr.append(tf.get_variable('P_mlr' + str(cl),
                                         dtype=dtype,
                                         shape=[1, before_mlr_dim],
                                         initializer=tf.constant_initializer(0.0)))

            if mlr_geom == 'eucl':
                eucl_vars += [P_mlr[cl]]
                logits_list.append(tf.reshape(util.tf_dot(-P_mlr[cl] + output_ffnn, A_mlr[cl]), [-1]))

            elif mlr_geom == 'hyp':
                hyp_vars += [P_mlr[cl]]
                minus_p_plus_x = util.tf_mob_add(-P_mlr[cl], output_ffnn, c_val)
                norm_a = util.tf_norm(A_mlr[cl])
                lambda_px = util.tf_lambda_x(minus_p_plus_x, c_val)
                px_dot_a = util.tf_dot(minus_p_plus_x, tf.nn.l2_normalize(A_mlr[cl]))
                logit = 2. / np.sqrt(c_val) * norm_a * tf.asinh(np.sqrt(c_val) * px_dot_a * lambda_px)
                logits_list.append(tf.reshape(logit, [-1]))

        model.logits = tf.stack(logits_list, axis=1)

        model.argmax_idx = tf.argmax(model.logits, axis=1, output_type=tf.int32)

        return model
        # with tf.variable_scope("output"):
        #     if model.feed_input is True:
        #         rnnout = tf.concat(axis=2, values=[rnnout, embedseq])

        #     # Converts seq to tensor, back to (B,T,W)
        #     hout = rnnout.get_shape()[-1]
        #     # Flatten from [B x T x H] - > [BT x H]
        #     rnnout_bt_x_h = tf.reshape(rnnout, [-1, hout])
        #     init = xavier_initializer(True, seed)

        #     with tf.contrib.slim.arg_scope([fully_connected], weights_initializer=init):
        #         if model.proj is True:
        #             hidden = tf.nn.dropout(fully_connected(rnnout_bt_x_h, hsz,
        #                                                    activation_fn=tf_activation(model.activation_type)), model.pkeep)
        #             preds = fully_connected(hidden, nc, activation_fn=None, weights_initializer=init)
        #         else:
        #             preds = fully_connected(rnnout_bt_x_h, nc, activation_fn=None, weights_initializer=init)
        #     model.probs = tf.reshape(preds, [-1, model.mxlen, nc])
        #     model.best = tf.argmax(model.probs, 2)
        # return model

    ###############################################################################################
    def construct_execution_graph(self):

        # Collect vars separately. Word embeddings are not used here.
        eucl_vars = []
        hyp_vars = []

        ################## word embeddings ###################

        # Initialize word embeddings close to 0, to have average norm equal to word_init_avg_norm.
        maxval = (3. * (word_init_avg_norm ** 2) / (2. * word_dim)) ** (1. / 3)
        initializer = tf.random_uniform_initializer(minval=-maxval, maxval=maxval, dtype=dtype)
        self.embeddings = tf.get_variable('embeddings',
                                          dtype=dtype,
                                          shape=[len(self.word_to_id), word_dim],
                                          initializer=initializer)

        if inputs_geom == 'eucl':
            eucl_vars += [self.embeddings]

        ################## RNNs for sentence embeddings ###################

        if cell_type == 'TFrnn':
            assert sent_geom == 'eucl'
            cell_class = lambda h_dim: tf.contrib.rnn.BasicRNNCell(h_dim)
        elif cell_type == 'TFgru':
            assert sent_geom == 'eucl'
            cell_class = lambda h_dim: tf.contrib.rnn.GRUCell(h_dim)
        elif cell_type == 'TFlstm':
            assert sent_geom == 'eucl'
            cell_class = lambda h_dim: tf.contrib.rnn.BasicLSTMCell(h_dim)
        elif cell_type == 'rnn' and sent_geom == 'eucl':
            cell_class = lambda h_dim: rnn_impl.EuclRNN(h_dim, dtype=dtype)
        elif cell_type == 'gru' and sent_geom == 'eucl':
            cell_class = lambda h_dim: rnn_impl.EuclGRU(h_dim, dtype=dtype)
        elif cell_type == 'rnn' and sent_geom == 'hyp':
            cell_class = lambda h_dim: rnn_impl.HypRNN(num_units=h_dim,
                                                       inputs_geom=inputs_geom,
                                                       bias_geom=bias_geom,
                                                       c_val=c_val,
                                                       non_lin=cell_non_lin,
                                                       fix_biases=fix_biases,
                                                       fix_matrices=fix_matrices,
                                                       matrices_init_eye=matrices_init_eye,
                                                       dtype=dtype)
        elif cell_type == 'gru' and sent_geom == 'hyp':
            cell_class = lambda h_dim: rnn_impl.HypGRU(num_units=h_dim,
                                                       inputs_geom=inputs_geom,
                                                       bias_geom=bias_geom,
                                                       c_val=c_val,
                                                       non_lin=cell_non_lin,
                                                       fix_biases=fix_biases,
                                                       fix_matrices=fix_matrices,
                                                       matrices_init_eye=matrices_init_eye,
                                                       dtype=dtype)
        else:
            logger.error('Not valid cell type: %s and sent_geom %s' % (cell_type, sent_geom))
            exit()

        # RNN 1
        with tf.variable_scope(cell_type + '1'):
            word_embeddings_1 = tf.nn.embedding_lookup(self.embeddings, self.word_ids_1) # bs x num_w_s1 x dim

            cell_1 = cell_class(hidden_dim)
            initial_state_1 = cell_1.zero_state(batch_size, dtype)
            outputs_1, state_1 = tf.nn.dynamic_rnn(cell=cell_1,
                                                   inputs=word_embeddings_1,
                                                   dtype=dtype,
                                                   initial_state=initial_state_1,
                                                   sequence_length=self.num_words_1)
            if cell_type == 'TFlstm':
                self.sent_1 = state_1[1]
            else:
                self.sent_1 = state_1


            sent1_norm = util.tf_norm(self.sent_1)


        # RNN 2
        with tf.variable_scope(cell_type + '2'):
            word_embeddings_2 = tf.nn.embedding_lookup(self.embeddings, self.word_ids_2)
            # tf.summary.scalar('word_emb2', tf.reduce_mean(tf.norm(word_embeddings_2, axis=2)))

            cell_2 = cell_class(hidden_dim)
            initial_state_2 = cell_2.zero_state(batch_size, dtype)
            outputs_2, state_2 = tf.nn.dynamic_rnn(cell=cell_2,
                                                   inputs=word_embeddings_2,
                                                   dtype=dtype,
                                                   initial_state=initial_state_2,
                                                   sequence_length=self.num_words_2)
            if cell_type == 'TFlstm':
                self.sent_2 = state_2[1]
            else:
                self.sent_2 = state_2


            sent2_norm = util.tf_norm(self.sent_2)


        tf.summary.scalar('RNN/word_emb1', tf.reduce_mean(tf.norm(word_embeddings_1, axis=2)))
        tf.summary.scalar('RNN/sent1', tf.reduce_mean(sent1_norm))
        tf.summary.scalar('RNN/sent2', tf.reduce_mean(sent2_norm))


        eucl_vars += cell_1.eucl_vars + cell_2.eucl_vars
        if sent_geom == 'hyp':
            hyp_vars += cell_1.hyp_vars + cell_2.hyp_vars

 
        ## Compute d(s1, s2)
        if sent_geom == 'eucl':
            d_sq_s1_s2 = util.tf_euclid_dist_sq(self.sent_1, self.sent_2)
        else:
            d_sq_s1_s2 = util.tf_poinc_dist_sq(self.sent_1, self.sent_2, c = c_val)


        ##### Some summaries:

        # For summaries and debugging, we need these:
        pos_labels = tf.reshape(tf.cast(self.label_placeholder, tf.float64), [-1, 1])
        neg_labels = 1. - pos_labels
        weights_pos_labels = pos_labels / tf.reduce_sum(pos_labels)
        weights_neg_labels = neg_labels / tf.reduce_sum(neg_labels)

        ################## first feed forward layer ###################

        # Define variables for the first feed-forward layer: W1 * s1 + W2 * s2 + b + bd * d(s1,s2)
        W_ff_s1 = tf.get_variable('W_ff_s1',
                                  dtype=dtype,
                                  shape=[hidden_dim, before_mlr_dim],
                                  initializer= tf.contrib.layers.xavier_initializer())

        W_ff_s2 = tf.get_variable('W_ff_s2',
                                  dtype=dtype,
                                  shape=[hidden_dim, before_mlr_dim],
                                  initializer= tf.contrib.layers.xavier_initializer())

        b_ff = tf.get_variable('b_ff',
                               dtype=dtype,
                               shape=[1, before_mlr_dim],
                               initializer=tf.constant_initializer(0.0))

        b_ff_d = tf.get_variable('b_ff_d',
                                 dtype=dtype,
                                 shape=[1, before_mlr_dim],
                                 initializer=tf.constant_initializer(0.0))

        eucl_vars += [W_ff_s1, W_ff_s2]
        if ffnn_geom == 'eucl' or bias_geom == 'eucl':
            eucl_vars += [b_ff]
            if additional_features == 'dsq':
                eucl_vars += [b_ff_d]
        else:
            hyp_vars += [b_ff]
            if additional_features == 'dsq':
                hyp_vars += [b_ff_d]


        if ffnn_geom == 'eucl' and sent_geom == 'hyp': # Sentence embeddings are Euclidean after log, except the proper distance (Eucl or hyp) is kept!
            self.sent_1 = util.tf_log_map_zero(self.sent_1, c_val)
            self.sent_2 = util.tf_log_map_zero(self.sent_2, c_val)

        ####### Build output_ffnn #######
        if ffnn_geom == 'eucl':
            output_ffnn = tf.matmul(self.sent_1, W_ff_s1) + tf.matmul(self.sent_2, W_ff_s2) + b_ff
            if additional_features == 'dsq': # [u, v, d(u,v)^2]
                output_ffnn = output_ffnn + d_sq_s1_s2 * b_ff_d

        else:
            assert sent_geom == 'hyp'
            ffnn_s1 = util.tf_mob_mat_mul(W_ff_s1, self.sent_1, c_val)
            ffnn_s2 = util.tf_mob_mat_mul(W_ff_s2, self.sent_2, c_val)
            output_ffnn = util.tf_mob_add(ffnn_s1, ffnn_s2, c_val)

            hyp_b_ff = b_ff
            if bias_geom == 'eucl':
                hyp_b_ff = util.tf_exp_map_zero(b_ff, c_val)
            output_ffnn = util.tf_mob_add(output_ffnn, hyp_b_ff, c_val)

            if additional_features == 'dsq': # [u, v, d(u,v)^2]
                hyp_b_ff_d = b_ff_d
                if bias_geom == 'eucl':
                    hyp_b_ff_d = util.tf_exp_map_zero(b_ff_d, c_val)

                output_ffnn = util.tf_mob_add(output_ffnn,
                                              util.tf_mob_scalar_mul(d_sq_s1_s2, hyp_b_ff_d, c_val),
                                              c_val)

        if ffnn_geom == 'eucl':
            output_ffnn = util.tf_eucl_non_lin(output_ffnn, non_lin=ffnn_non_lin)
        else:
            output_ffnn = util.tf_hyp_non_lin(output_ffnn,
                                              non_lin=ffnn_non_lin,
                                              hyp_output = (mlr_geom == 'hyp' and dropout == 1.0),
                                              c=c_val)
        # Mobius dropout
        if dropout < 1.0:
            # If we are here, then output_ffnn should be Euclidean.
            output_ffnn = tf.nn.dropout(output_ffnn, keep_prob=self.dropout_placeholder)
            if (mlr_geom == 'hyp'):
                output_ffnn = util.tf_exp_map_zero(output_ffnn, c_val)


        ################## MLR ###################
        # output_ffnn is batch_size x before_mlr_dim

        A_mlr = []
        P_mlr = []
        logits_list = []
        for cl in range(num_classes):
            A_mlr.append(tf.get_variable('A_mlr' + str(cl),
                                         dtype=dtype,
                                         shape=[1, before_mlr_dim],
                                         initializer=tf.contrib.layers.xavier_initializer()))
            eucl_vars += [A_mlr[cl]]

            P_mlr.append(tf.get_variable('P_mlr' + str(cl),
                                         dtype=dtype,
                                         shape=[1, before_mlr_dim],
                                         initializer=tf.constant_initializer(0.0)))

            if mlr_geom == 'eucl':
                eucl_vars += [P_mlr[cl]]
                logits_list.append(tf.reshape(util.tf_dot(-P_mlr[cl] + output_ffnn, A_mlr[cl]), [-1]))

            elif mlr_geom == 'hyp':
                hyp_vars += [P_mlr[cl]]
                minus_p_plus_x = util.tf_mob_add(-P_mlr[cl], output_ffnn, c_val)
                norm_a = util.tf_norm(A_mlr[cl])
                lambda_px = util.tf_lambda_x(minus_p_plus_x, c_val)
                px_dot_a = util.tf_dot(minus_p_plus_x, tf.nn.l2_normalize(A_mlr[cl]))
                logit = 2. / np.sqrt(c_val) * norm_a * tf.asinh(np.sqrt(c_val) * px_dot_a * lambda_px)
                logits_list.append(tf.reshape(logit, [-1]))

        self.logits = tf.stack(logits_list, axis=1)

        self.argmax_idx = tf.argmax(self.logits, axis=1, output_type=tf.int32)

        self.loss = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.label_placeholder,
                                                           logits=self.logits))
        tf.summary.scalar('classif/unreg_loss', self.loss)

        if reg_beta > 0.0:
            assert num_classes == 2
            distance_regularizer = tf.reduce_mean(
                (tf.cast(self.label_placeholder, dtype=dtype) - 0.5) * d_sq_s1_s2)

            self.loss = self.loss + reg_beta * distance_regularizer

        self.acc = tf.reduce_mean(tf.to_float(tf.equal(self.argmax_idx, self.label_placeholder)))
        tf.summary.scalar('classif/accuracy', self.acc)




        ######################################## OPTIMIZATION ######################################
        all_updates_ops = []

        ###### Update Euclidean parameters using Adam.
        optimizer_euclidean_params = tf.train.AdamOptimizer(learning_rate=1e-3)
        eucl_grads = optimizer_euclidean_params.compute_gradients(self.loss, eucl_vars)
        capped_eucl_gvs = [(tf.clip_by_norm(grad, 1.), var) for grad, var in eucl_grads]  ###### Clip gradients
        all_updates_ops.append(optimizer_euclidean_params.apply_gradients(capped_eucl_gvs))


        ###### Update Hyperbolic parameters, i.e. word embeddings and some biases in our case.
        def rsgd(v, riemannian_g, learning_rate):
            if hyp_opt == 'rsgd':
                return util.tf_exp_map_x(v, -self.burn_in_factor * learning_rate * riemannian_g, c=c_val)
            else:
                # Use approximate RSGD based on a simple retraction.
                updated_v = v - self.burn_in_factor * learning_rate * riemannian_g
                # Projection op after SGD update. Need to make sure embeddings are inside the unit ball.
                return util.tf_project_hyp_vecs(updated_v, c_val)


        if inputs_geom == 'hyp':
            grads_and_indices_hyp_words = tf.gradients(self.loss, self.embeddings)
            grads_hyp_words = grads_and_indices_hyp_words[0].values
            repeating_indices = grads_and_indices_hyp_words[0].indices
            unique_indices, idx_in_repeating_indices = tf.unique(repeating_indices)
            agg_gradients = tf.unsorted_segment_sum(grads_hyp_words,
                                                    idx_in_repeating_indices,
                                                    tf.shape(unique_indices)[0])

            agg_gradients = tf.clip_by_norm(agg_gradients, 1.) ######## Clip gradients
            unique_word_emb = tf.nn.embedding_lookup(self.embeddings, unique_indices)  # no repetitions here

            riemannian_rescaling_factor = util.riemannian_gradient_c(unique_word_emb, c=c_val)
            rescaled_gradient = riemannian_rescaling_factor * agg_gradients

            all_updates_ops.append(tf.scatter_update(self.embeddings,
                                                     unique_indices,
                                                     rsgd(unique_word_emb, rescaled_gradient, lr_words))) # Updated rarely

        if len(hyp_vars) > 0:
            hyp_grads = tf.gradients(self.loss, hyp_vars)
            capped_hyp_grads = [tf.clip_by_norm(grad, 1.) for grad in hyp_grads]  ###### Clip gradients


            for i in range(len(hyp_vars)):
                riemannian_rescaling_factor = util.riemannian_gradient_c(hyp_vars[i], c=c_val)
                rescaled_gradient = riemannian_rescaling_factor * capped_hyp_grads[i]
                all_updates_ops.append(tf.assign(hyp_vars[i], rsgd(hyp_vars[i], rescaled_gradient, lr_ffnn)))  # Updated frequently

        self.all_optimizer_var_updates_op = tf.group(*all_updates_ops)


        self.summary_merged = tf.summary.merge_all()
        self.test_summary_writer = tf.summary.FileWriter(
            os.path.join(root_path, 'tb_28may/' + tensorboard_name + '/'))


def run():
    word_to_id = pickle.load(open(word_to_id_file_path, 'rb'))
    id_to_word = pickle.load(open(id_to_word_file_path, 'rb'))

    model = HyperbolicRNNModel(word_to_id=word_to_id,
                               id_to_word=id_to_word)

    training_data, dev_data, test_data = get_datasets()

    model.train(training_data=training_data,
                dev_data=dev_data,
                test_data=test_data,
                save_model=False,
                save_to_path='models/' + name_experiment,
                restore_model=restore_model,
                restore_from_path=restore_from_path)

if __name__ == '__main__':
    run()
