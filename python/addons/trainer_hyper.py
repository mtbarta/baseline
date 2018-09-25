from baseline.tf.tagger.train import TaggerTrainerTf, TaggerEvaluatorTf
import tensorflow as tf
import numpy as np
from baseline.utils import to_spans, f_score, listify, revlut, get_model_file
from baseline.progress import create_progress_bar
from baseline.reporting import basic_reporting
from baseline.tf.tfy import optimizer
from baseline.train import EpochReportingTrainer, create_trainer
import os
from baseline.utils import zip_model

class HyperbolicTrainer(EpochReportingTrainer):
    def __init__(self, model, **kwargs):
        super(HyperbolicTrainer, self).__init__()
        self.model = model
        # self.loss = self.model.create_loss()
        span_type = kwargs.get('span_type', 'iob')
        verbose = kwargs.get('verbose', False)
        self.evaluator = TaggerEvaluatorTf(model, span_type, verbose)
        # self.loss = model.create_loss()
        # self.global_step = tf.train.create_global_step()
        # self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-lm-%d/lm" % os.getpid())

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-lm-%d" % os.getpid())
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    def _train(self, ts, debug=False):
        total_loss = 0
        steps = len(ts)

        metrics = {}
        pg = create_progress_bar(steps)
        for batch_dict in ts:
            feed_dict = self.model.make_input(batch_dict, do_dropout=True)
            preds, lossv, _ = self.model.sess.run([self.model.probs, self.model.loss, self.model.all_optimizer_var_updates_op], feed_dict=feed_dict)
            total_loss += lossv
            pg.update()
        pg.done()
        metrics['avg_loss'] = float(total_loss)/steps
        return metrics

    def _test(self, ts):
        return self.evaluator.test(ts)


def create_trainer(model, **kwargs):
    return HyperbolicTrainer(model, **kwargs)