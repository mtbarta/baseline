import os
import pickle
import numpy as np
import baseline
from baseline.utils import (
    export,
    unzip_files,
    find_model_basename,
    find_files_with_prefix,
    import_user_module,
    read_json,
    is_sequence,
    revlut,
    load_vectorizers,
    load_vocabs
)
from baseline.model import (
    load_model,
    load_tagger_model,
    load_seq2seq_model,
    load_lang_model
)


__all__ = []
exporter = export(__all__)

from tensorflow_serving.apis import (predict_pb2 as predictpb, 
                         inference_pb2 as infpb, 
                         prediction_service_pb2_grpc as servicepb
                         )
# from tensorflow.contrib.utils import make_tensor_proto
# import tensorflow as tf
import grpc

class RemoteModel(object):
    def __init__(self, remote, name, signature):
        self.remote = remote
        self.name = name
        self.signature = signature

        self.channel = grpc.insecure_channel(remote)

    def classify(self, examples):
        return self._transform(examples)

    def predict(self, examples):
        return self._transform(examples)

    def predict_next(self, examples):
        return self._transform(examples)

    def _transform(self, examples):
        request = self.create_request(examples)
        stub = servicepb.PredictionServiceStub(self.channel)
        outcomes_list = stub.Predict(request)
        #local models return an nbest list. to duplicate, we nest the result in lists.
        outcomes_list = [self.deserialize_response(outcomes_list)]

        return outcomes_list

    def create_request(self, examples):
        request = predictpb.PredictRequest()
        request.model_spec.name = self.name
        request.model_spec.signature_name = self.signature

        for feature in examples.keys():
            if isinstance(examples[feature], np.ndarray): 
                shape =examples[feature].shape
            else:
                shape = [1]

            import tensorflow
            tensor_proto = tensorflow.contrib.util.make_tensor_proto(examples[feature], shape=shape)
            request.inputs[feature].CopyFrom(
                tensor_proto
            )
        
        return request

    def deserialize_response(self, predict_response):
        """
        this is basically a map.

        here's the relevant piece of the proto:
            map<string, TensorProto> inputs = 2;

        the predict endpoint happens to have the ability to filter output for certain keys, but
        we do not support this currently. There are two keys we want to extract: classes and scores.

        :params predict_response: a PredictResponse protobuf object, 
                    as defined in tensorflow_serving proto files
        """
        classes = predict_response.outputs.get('classes').string_val
        if not classes:
            # s2s returns int values.
            classes = predict_response.outputs.get('classes').int_val
            return [classes]

        scores = predict_response.outputs.get('scores')
        if not scores:
            return [(k,) for k in classes]

        scores = scores.float_val
        return [(k, v) for k,v in zip(classes, scores)]


@exporter
class ClassifierService(object):

class Service(object):
    task_name = None
    task_load = None
    def __init__(self, vocabs=None, vectorizers=None, model=None):
        self.vectorizers = vectorizers
        self.model = model
        self.vocabs = vocabs
            

    def get_vocab(self, vocab_type='word'):
        return self.vocabs.get(vocab_type)

    def get_labels(self):
        return self.model.get_labels()

    @classmethod
    def load(cls, bundle, **kwargs):
        remote = kwargs.get("remote", None)
        name = kwargs.get("name", None)

        # can delegate
        if os.path.isdir(bundle):
            directory = bundle
        else:
            directory = unzip_files(bundle)

        vocabs = load_vocabs(directory)
        vectorizers = load_vectorizers(directory)

        if remote:
            return cls(vocabs, vectorizers, None, remote, name)

        model_basename = find_model_basename(directory)
        be = kwargs.get('backend', 'tf')
        import_user_module('baseline.{}.embeddings'.format(be))
        import_user_module('baseline.{}.{}'.format(be, cls.task_name))
        model = cls.task_load(model_basename, **kwargs)
        return cls(vocabs, vectorizers, model)

@exporter
class ClassifierService(Service):
    task_name = 'classify'
    task_load = load_model

    def predict(self, tokens):
        """Take tokens and apply the internal vocab and vectorizers.  The tokens should be either a batch of text
        single utterance of type ``list``
        """
        mxlen = 0
        mxwlen = 0
        if type(tokens[0]) == str:
            tokens_seq = (tokens,)
        else:
            tokens_seq = tokens

        for tokens in tokens_seq:
            mxlen = max(mxlen, len(tokens))
            for token in tokens:
                mxwlen = max(mxwlen, len(token))

        examples = dict()
        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k].append(vec)
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key].append(length)

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])

        outcomes_list = self.model.classify(examples)
            
        results = []
        for outcomes in outcomes_list:
            results += [sorted(outcomes, key=lambda tup: tup[1], reverse=True)]
        return results


@exporter
class TaggerService(Service):
    task_name = 'tagger'
    task_load = load_tagger_model

    def transform(self, tokens, **kwargs):
        """
        Utility function to convert lists of sentence tokens to integer value one-hots which
        are then passed to the tagger.  The resultant output is then converted back to label and token
        to be printed.

        This method is not aware of any input features other than words and characters (and lengths).  If you
        wish to use other features and have a custom model that is aware of those, use `predict` directly.

        :param tokens: (``list``) A list of tokens

        """
        label_field = kwargs.get('label', 'label')

        mxlen = 0
        mxwlen = 0
        if type(tokens[0]) == str:
            mxlen = len(tokens)
            tokens_seq = []
            for t in tokens:
                mxwlen = max(mxwlen, len(t))
                tokens_seq += [dict({'text': t})]
            tokens_seq = [tokens_seq]
        else:
            # Better be a sequence, but it could be pre-batched, [[],[]]
            # But what kind of object is at the first one then?
            if is_sequence(tokens[0]):
                tokens_seq = []
                # Then what we have is [['The', 'dog',...], ['I', 'cannot']]
                # [[{'text': 'The', 'pos': 'DT'}, ...

                # For each of the utterances, we need to make a dictionary
                if type(tokens[0][0]) == str:

                    for utt in tokens:
                        utt_dict_seq = []
                        mxlen = max(mxlen, len(utt))
                        for t in utt:
                            mxwlen = max(mxwlen, len(t))
                            utt_dict_seq += [dict({'text': t})]
                        tokens_seq += [utt_dict_seq]
                # Its already in dict form so we dont need to do anything
                elif type(tokens[0][0]) == dict:
                    for utt in tokens:
                        mxlen = max(mxlen, len(utt))
                        for t in utt['text']:
                            mxwlen = max(mxwlen, len(t))
            # If its a dict, we just wrap it up
            elif type(tokens[0]) == dict:
                mxlen = max(len(tokens))
                for t in tokens:
                    mxwlen = max(mxwlen, len(t))
                tokens_seq = [tokens]
            else:
                raise Exception('Unknown input format')

        if len(tokens_seq) == 0:
            return []

        # This might be inefficient if the label space is large

        label_vocab = revlut(self.get_labels())

        examples = dict()
        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.vectorizers.items():
                vec, length = vectorizer.run(tokens, self.vocabs[k])
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in self.vectorizers.keys():
            examples[k] = np.stack(examples[k])
            lengths_key = '{}_lengths'.format(k)
            examples[lengths_key] = np.stack(examples[lengths_key])
        outcomes = self.model.predict(examples)
        outputs = []
        for i, outcome in enumerate(outcomes):
            output = []
            for j, token in enumerate(tokens_seq[i]):
                new_token = dict()
                new_token.update(token)
                new_token[label_field] = label_vocab[outcome[j].item()]
                output += [new_token]
            outputs += [output]
        return outputs


@exporter
class LanguageModelService(Service):
    task_name = "lm"
    task_load = load_lang_model

    def __init__(self, *args, **kwargs):
        super(LanguageModelService, self).__init__(*args, **kwargs)
        self.idx_to_token = revlut(self.vocabs[self.model.tgt_key])

    @classmethod
    def load(cls, bundle, **kwargs):
        kwargs['batchsz'] = 1
        return super(LanguageModelService, cls).load(bundle, **kwargs)

    # Do a greedy decode for now, everything else will be super slow
    def run(self, tokens, **kwargs):
        mxlen = kwargs.get('mxlen', 10)
        mxwlen = kwargs.get('mxwlen', 40)

        for k, vectorizer in self.vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen

        token_buffer = tokens
        tokens_seq = tokens
        examples = dict()
        for i in range(mxlen):

            for k, vectorizer in self.vectorizers.items():
                vectorizer.mxlen = len(token_buffer)
                vec, length = vectorizer.run(token_buffer, self.vocabs[k])
                if k in examples:
                    examples[k] = np.append(examples[k], vec)
                else:
                    examples[k] = vec

                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key in examples:
                        examples[lengths_key] += length
                    else:
                        examples[lengths_key] = np.array(length)
            batch_dict = {k: v.reshape((1,) + v.shape) for k, v in examples.items()}
            softmax_tokens = self.model.predict_next(batch_dict)
            next_token = np.argmax(softmax_tokens, axis=-1)[-1]

            token_str = self.idx_to_token.get(next_token, '<PAD>')
            if token_str == '<EOS>':
                break
            if token_str != '<PAD>':
                tokens_seq += [token_str]
            token_buffer = [token_str]
        return tokens_seq


@exporter
class EncoderDecoderService(Service):
    task_name = 'seq2seq'
    task_load = load_seq2seq_model

    def __init__(self, vocabs=None, vectorizers=None, model=None):
        super(EncoderDecoderService, self).__init__(None, None, model)
        self.src_vocabs = {}
        self.tgt_vocab = None
        for k, vocab in vocabs.items():
            if k == 'tgt':
                self.tgt_vocab = vocab
            else:
                self.src_vocabs[k] = vocab

        self.tgt_idx_to_token = revlut(self.tgt_vocab)
        self.src_vectorizers = {}
        self.tgt_vectorizer = None
        for k, vectorizer, in vectorizers.items():
            if k == 'tgt':
                self.tgt_vectorizer = vectorizer
            else:
                self.src_vectorizers[k] = vectorizer

    def get_src_vocab(self, vocab_type):
        return self.src_vocabs[vocab_type]

    def get_tgt_vocab(self):
        return self.tgt_vocab

    @classmethod
    def load(cls, bundle, **kwargs):
        kwargs['predict'] = kwargs.get('predict', True)
        kwargs['beam'] = kwargs.get('beam', 5)
        return super(EncoderDecoderService, cls).load(bundle, **kwargs)

    def transform(self, tokens, **kwargs):

        mxlen = 0
        mxwlen = 0
        if type(tokens[0]) == str:
            tokens_seq = (tokens,)
        else:
            tokens_seq = tokens

        for tokens in tokens_seq:
            mxlen = max(mxlen, len(tokens))
            for token in tokens:
                mxwlen = max(mxwlen, len(token))

        examples = dict()

        for k, vectorizer in self.src_vectorizers.items():
            if hasattr(vectorizer, 'mxlen') and vectorizer.mxlen == -1:
                vectorizer.mxlen = mxlen
            if hasattr(vectorizer, 'mxwlen') and vectorizer.mxwlen == -1:
                vectorizer.mxwlen = mxwlen
            examples[k] = []

        for i, tokens in enumerate(tokens_seq):
            for k, vectorizer in self.src_vectorizers.items():
                vec, length = vectorizer.run(tokens, self.src_vocabs[k])
                examples[k] += [vec]
                if length is not None:
                    lengths_key = '{}_lengths'.format(k)
                    if lengths_key not in examples:
                        examples[lengths_key] = []
                    examples[lengths_key] += [length]

        for k in self.src_vectorizers.keys():
            examples[k] = np.stack(examples[k])
    
        outcomes = self.model.predict(examples)

        results = []
        for i in range(len(outcomes)):
            best = outcomes[i][0]
            out = []
            for j in range(len(best)):
                word = self.tgt_idx_to_token.get(best[j], '<PAD>')
                if word != '<PAD>' and word != '<EOS>':
                    out += [word]
        results += [out]
        return results
