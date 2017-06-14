"""
Daniel Roudnitsky
"""
import logging
import os
import subprocess
import jieba
import sklearn

from sklearn.feature_extraction.text import TfidfVectorizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TokenizeAndVectorize(object):
    """
    Class of tokenizers that read and tokenize a labeled corpus for sentiment analysis (pos or neg labels)
    then vectorizes them using tf-idf(term frequency inverse document frequency) and returns two objects:
    the tf-idf feature matrix and corresponding labels (1 for pos and 0 for neg) randomly shuffled

    Tokenization methods take the parameters that the tokenizer requires
    """

    def __init__(self, corpus_dir):
        self.corpus_dir = corpus_dir
        self.corpus = []
        self.labels = []
        self.read_corpus()
        self.combine_corpus()

    def read_corpus(self):
        """
        Read each document in '/self.corpus_dir/pos' and '/self.corpus_dir/neg' and write to self.corpus and
        create corresponding labels for each document in self.labels
        """

        def read_and_append_contents(file_dir, list_to_append):
            try:
                open(file_dir).read()
            except (UnicodeDecodeError, UnicodeEncodeError):
                pass
            else:
                list_to_append.append(open(file_dir).read().strip())  # remove trailing whitespace in docs

        neg_dir = os.path.join(self.corpus_dir, 'neg')
        pos_dir = os.path.join(self.corpus_dir, 'pos')

        neg_docs, pos_docs = [], []
        for pos_doc, neg_doc in zip(os.listdir(pos_dir), os.listdir(neg_dir)):
            read_and_append_contents(os.path.join(pos_dir, pos_doc), pos_docs)
            read_and_append_contents(os.path.join(neg_dir, neg_doc), neg_docs)

        self.corpus = pos_docs + neg_docs
        self.labels = [1 for _ in range(len(pos_docs))] + [0 for _ in range(len(neg_docs))]

    def combine_corpus(self):
        """
        Creates a file containing all of the documents in the corpus for the sentencepiece `--input`
        parameter. The file is located at `corpus_dir/combined_corp.txt`
        """
        with open(os.path.join(self.corpus_dir, 'combined_corp.txt'), 'w') as f:
            for doc in self.corpus:
                f.write(doc + ' \n')

    def shuffle_and_vectorize(self, corpus, labels):
        """
        Shuffle corpus and labels, return a tf idf matrix for the corpus and its corresponding labels
        :param corpus: A list of documents(strings)
        :param labels: Corresponding labels for corpus/list of documents
        :return: tf_idf_matrix, corresponding_labels(list)
        """
        shuffled = sklearn.utils.shuffle(corpus, labels)
        tokenized_docs, labels = shuffled[0], shuffled[1]
        return TfidfVectorizer().fit_transform(tokenized_docs), labels

    def no_tokenizer(self):
        """
        Use raw documents to create tf-idf feature matrix
        :return: tf_idf_matrix, corresponding_labels(list)
        """
        return self.shuffle_and_vectorize(self.corpus, self.labels)

    def jieba(self, cut_all=False, HMM=True):
        """
        Jieba tokenizer (https://github.com/fxsjy/jieba)
        :return: tf_idf_matrix, corresponding_labels(list)
        """
        tokenized_corp = [' '.join(list(jieba.cut(doc, cut_all, HMM))) for doc in self.corpus]
        return self.shuffle_and_vectorize(tokenized_corp, self.labels)

    def sentence_piece_model(self, model_prefix, vocab_size, model_type='unigram'):
        """
        Google's SentencePiece (https://github.com/google/sentencepiece). Train specified model and then tokenize/encode
        SentencePiece has to be installed already
        :param model_prefix: output model name prefix. <model_name>.model and <model_name>.vocab are generated.
        :param vocab_size: vocabulary size, e.g., 8000, 16000, or 32000
        :param model_type:  model type. Choose from unigram (default), bpe, char, or word. The input sentence must be
                            pretokenized when using word type.
        :return: tf_idf_matrix, corresponding_labels(list)
        """
        if model_type not in ['unigram', 'bpe', 'char', 'word']:
            raise ValueError("Invalid model_type, the only valid parameters are 'unigram', 'bpe', 'char', 'word'")

        combined_corp_path = os.path.join(self.corpus_dir, 'combined_corp.txt')

        input_param = '--input={}'.format(combined_corp_path)
        model_prefix_param = '--model_prefix={}'.format(model_prefix)
        vocab_size_param = '--vocab_size={}'.format(vocab_size)
        model_type_param = '--model_type={}'.format(model_type)
        spm_train_command = 'spm_train {} {} {} {}'.format(input_param, model_prefix_param,
                                                           vocab_size_param, model_type_param)

        logger.info("Training {} model...".format(model_type))
        try:
            subprocess.check_output(spm_train_command, shell=True)
        except subprocess.CalledProcessError:
            raise ValueError("{} model with vocab_size={} raises 'Aborted(core dumped)'".format(model_type, vocab_size))

        logger.info("Encoding all documents in corpus...")
        tokenized_corp = []
        for doc in self.corpus:
            spm_encode_command = 'echo "{}" | spm_encode --model={}'.format(doc, model_prefix + '.model')
            output = subprocess.run(spm_encode_command, shell=True, stdout=subprocess.PIPE).stdout
            tokenized_corp.append(output.decode())

        return self.shuffle_and_vectorize(tokenized_corp, self.labels)
