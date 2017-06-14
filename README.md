# TokenizationBenchmarks
Comparison of various supervised and unsupervised tokenization algorithms using sentiment analysis on a chinese corpus (ChnSentiCorp)

### Results
Regularized logistic regression trained on 5205 examples and tested on 579 examples (90/10 split) 

|Tokenizer   | Accuracy|
|------------|---------|
|no tokenzier|83.07    |
|jieba       |89.32    |

|SPM   |vocab_size=2000|vocab_size=4000|vocab_size=8000|vocab_size=16000|
|------|---------------|---------------|---------------|----------------|
|Unigram|Aborted |87.21|90.43|90.08|
|Byte Pair Encoding|Aborted|86.70|**90.81**|90.81|
|Char|53.36|48.46|48.98|47.35|
|Word|85.18|85.73|84.59|Aborted|

Aborted = `vocab_size` was either too small or too large for that particular model 
