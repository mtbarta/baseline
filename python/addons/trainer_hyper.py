from baseline.tf.tagger.train import TaggerTrainerTf

class HyperbolicTrainer(TaggerTrainerTf):
    def __init__(self, model, **kwargs):
        super(HyperbolicTrainer, self).__init__()
        # self.model = model
        # self.loss = model.create_loss()
        # self.global_step, self.train_op = optimizer(self.loss, **kwargs)

    def checkpoint(self):
        self.model.saver.save(self.model.sess, "./tf-lm-%d/lm" % os.getpid(), global_step=self.global_step)

    def recover_last_checkpoint(self):
        latest = tf.train.latest_checkpoint("./tf-lm-%d" % os.getpid())
        print("Reloading " + latest)
        self.model.saver.restore(self.model.sess, latest)

    def _train(self, ts):
        total_loss = 0
        steps = len(ts)
        metrics = {}
        pg = create_progress_bar(steps)
        for batch_dict in ts:
            feed_dict = self.model.make_input(batch_dict, do_dropout=True)
            _, step, lossv = self.model.sess.run([self.train_op, self.global_step, self.loss], feed_dict=feed_dict)
            total_loss += lossv
            pg.update()
        pg.done()
        metrics['avg_loss'] = float(total_loss)/steps
        return metrics

    def _test(self, ts):
        return self.evaluator.test(ts)

def fit(model, ts, vs, es, **kwargs):
    epochs = int(kwargs['epochs']) if 'epochs' in kwargs else 5
    patience = int(kwargs['patience']) if 'patience' in kwargs else epochs
    conll_output = kwargs.get('conll_output', None)
    span_type = kwargs.get('span_type', 'iob')
    txts = kwargs.get('txts', None)
    model_file = get_model_file(kwargs, 'tagger', 'tf')
    after_train_fn = kwargs['after_train_fn'] if 'after_train_fn' in kwargs else None
    trainer = create_trainer(TaggerTrainerTf, model, **kwargs)
    tables = tf.tables_initializer()
    model.sess.run(tables)
    init = tf.global_variables_initializer()
    model.sess.run(init)
    saver = tf.train.Saver()
    model.save_using(saver)
    do_early_stopping = bool(kwargs.get('do_early_stopping', True))
    verbose = bool(kwargs.get('verbose', False))
    if do_early_stopping:
        early_stopping_metric = kwargs.get('early_stopping_metric', 'acc')
        patience = kwargs.get('patience', epochs)
        print('Doing early stopping on [%s] with patience [%d]' % (early_stopping_metric, patience))

    reporting_fns = listify(kwargs.get('reporting', basic_reporting))
    print('reporting', reporting_fns)

    max_metric = 0
    last_improved = 0
    for epoch in range(epochs):

        trainer.train(ts, reporting_fns)
        if after_train_fn is not None:
            after_train_fn(model)

        test_metrics = trainer.test(vs, reporting_fns, phase='Valid')

        if do_early_stopping is False:
            trainer.checkpoint()
            model.save(model_file)

        elif test_metrics[early_stopping_metric] > max_metric:
            last_improved = epoch
            max_metric = test_metrics[early_stopping_metric]
            print('New max %.3f' % max_metric)
            trainer.checkpoint()
            model.save(model_file)

        elif (epoch - last_improved) > patience:
            print('Stopping due to persistent failures to improve')
            break

    if do_early_stopping is True:
        print('Best performance on max_metric %.3f at epoch %d' % (max_metric, last_improved))
    if es is not None:

        trainer.recover_last_checkpoint()
        # What to do about overloading this??
        evaluator = TaggerEvaluatorTf(model, span_type, verbose)
        test_metrics = evaluator.test(es, conll_output=conll_output, txts=txts)
        for reporting in reporting_fns:
            reporting(test_metrics, 0, 'Test')
    if kwargs.get("model_zip", False):
        zip_model(model_file)


