# code
This is the source code and some evaluation scripts for our paper [PEN: Prediction-Explanation Network to Forecast Stock Price Movement with Better Explainability]().
Our code is based on https://github.com/yumoxu/stocknet-code


## Dependencies
- Python 2.7.11
- Tensorflow 1.4.0
- Scipy 1.0.0
- NLTK 3.2.5


## Directories
- src: source files;
- res: resource files including,
    - Vocabulary file `vocab.txt`;
    - Pre-trained embeddings of [GloVe](https://github.com/stanfordnlp/GloVe). We used the GloVe obtained from the Twitter corpora which you could download [here](http://nlp.stanford.edu/data/wordvecs/glove.twitter.27B.zip).
- data:
    - ACL18 consisting of tweets and prices which you could download [here](https://github.com/yumoxu/stocknet-dataset).
    - DJIA consisting of news and prices which you could download [here](https://www.kaggle.com/datasets/aaron7sun/stocknews).


## Configurations
All details about hyper-parameters are listed in `config.yml` and `config_d.yml`. 

See more information in `Experimental Setup` of our paper.

## Running
Use `sh src/run.sh` or `python Main.py` in your terminal to start model training and testing. 

The default code corresponds to ACL18.
For DJIA, simply replace `Executor` to `Executor_d` in `Main.py`.
