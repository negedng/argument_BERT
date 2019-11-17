# Arguing with BERT: Argumentation Mining using Contextualised Embedding and Transformers
## Masters Project at the University of Manchester
 - Supervisor: Dr André Freitas
 
## Abstract
In this thesis, I investigate the possibilities of using contextualised word embedding vectors and Transformers in Argumentation Mining. I propose a model that uses only sentence-level embedding vectors. Therefore, eliminating the token level features, this model uses only sentence-level features, therefore, can be trained as a feed-forward network for argument classification problems. The experiment is tested on the argument component classification and argument relation detection subtasks of Argumentation Mining, achieving better performance on relation detection than the current state-of-the-art. Moreover, I show that Transfer Learning using Transformers can be applied in Argumentation Mining problems with competitive performance. Using the novel models described in the thesis, I built a pipeline architecture tool to perform Argumentation Mining tasks on the general text. The output of this tool is an annotation file in a format that is available for the most used Argumentation Mining corpora.

# Corpora
 - AraucariaDB: http://corpora.aifdb.org/araucaria
 - Student Essay Corpus: https://www.informatik.tu-darmstadt.de/ukp/research_6/data/index.en.jsp

## Data preparation
To generate data from the publicly available corpora, see Tobias Milz's work: https://github.com/Milzi/arguEParser
Additional third party parsers for feature generation: https://github.com/jiyfeng/DPLP

Milz's ArguE classifier: https://github.com/Milzi/ArguE

## Dependencies
The following packages are required in addition to a standard Colab environment:
```
!pip install bert-embedding
!pip install -e git+https://github.com/negedng/bert-embedding#egg=bert_embedding

!pip install vaderSentiment
!pip install xmltodict
!pip install gensim
!pip install -U spacy
#!python -m spacy download en_core_web_lg

import nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('tagsets')
```
An example of a Colab notebook [here](Colab_argument_BERT.ipynb).
