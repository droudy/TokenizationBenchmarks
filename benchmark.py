"""
Daniel Roudnitsky
"""
from tokenizers import TokenizeAndVectorize
from sklearn.linear_model import LogisticRegression


def train_and_evaluate_classifier(feature_matrix, labels, train_test_split=.9):
    """
    Split feature_matrix and labels into a training set and testing set, train logistic regression on the training set
    and evaluate the model with the testing set, returns the accuracy on the testing set
    :param train_test_split: fraction representing percentage of examples to use for training, save the rest for testing
    :return: Accuracy on the testing set
    """
    split = int(train_test_split * len(labels))  # index at which to split
    training_X, training_y = feature_matrix[:split], labels[:split]
    testing_X, testing_y = feature_matrix[split:], labels[split:]
    lr_model = LogisticRegression(C=1e5).fit(training_X, training_y)
    return lr_model.score(testing_X, testing_y)


def evaluate_tokenizer(tokenizer, *args):
    """
    Get average accuracy(sentiment analysis) of a tokenizer over 5 runs
    :param tokenizer: Tokenization method that returns a tfidf matrix and labels
    :param args: Arguments to tokenization method
    :return: Average accuracy over num_runs using regularized logistic regression
    """
    accuracy = []
    for _ in range(5):
        X, y = tokenizer(*args)
        accuracy.append(train_and_evaluate_classifier(X, y))
    return sum(accuracy) / 5


tokenizers = TokenizeAndVectorize('/home/droudy/Desktop/ChnSuperCorp/ChnSentiCorp_htl_ba_6000')
results = []

# raw documents
score = evaluate_tokenizer(tokenizers.no_tokenizer)
results.append("Raw documents: {}".format(score))

# jieba tokenizer, default params
score = evaluate_tokenizer(tokenizers.jieba)
results.append("Jieba: {}".format(score))

# all spm tokenizers with different vocab_sizes
spm_models = ['unigram', 'bpe', 'char', 'word']
vocab_sizes = [2000, 4000, 8000, 16000]

for model_type in spm_models:
    for vocab_size in vocab_sizes:
        try:
            score = evaluate_tokenizer(tokenizers.sentence_piece_model, 'm', vocab_size, model_type)
            results.append("{} vocab_size={} : {} ".format(model_type, vocab_size, score))
        except ValueError:
            results.append("{} vocab_size={} : ABORTED (CORE DUMPED)".format(model_type, vocab_size))

print(results)
