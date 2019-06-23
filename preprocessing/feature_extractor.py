import nltk
import numpy as np
import pandas as pd
import gensim
from keras.preprocessing.sequence import pad_sequences as kp_pad_sequences


WORD2WEC_EMBEDDING_FILE = '/root/input/GoogleNews-vectors-negative300.bin.gz'

def get_propositions(dataset):
    """Parse propositions
    dataset: the original dataframe
    Output:
        propositionSet: list of the propositions of the arg1
        parsedPropositions: parsed prop. in the arg1
    """
    
    propositionSet = list(set(dataset['arg1']))
    parsedPropositions = list()

    for proposition in propositionSet:
        words = nltk.tokenize.word_tokenize(proposition)
        parsedPropositions.append(nltk.pos_tag(words))

    return propositionSet, parsedPropositions
    

def add_word_vector_feature(dataset, propositionSet, parsedPropositions):
    """Add word2vec feature to the dataset
    """

    word2VecModel = gensim.models.KeyedVectors.load_word2vec_format(WORD2WEC_EMBEDDING_FILE, binary=True)

    wordVectorFeature = list()
    feature_vector = np.zeros(300)

    for proposition in parsedPropositions:
        propositionVector = list()
        for word in proposition:
            if word[0] in word2VecModel.vocab:
                feature_vector = word2VecModel[word[0]]

            propositionVector.append(feature_vector)

        wordVectorFeature.append(propositionVector)

    wordVectorFeature = np.array(wordVectorFeature)
    wordVectorFeature = kp_pad_sequences(wordVectorFeature, value=0, padding='post', dtype=float)

    wordVectorFrame = pd.DataFrame({"arg1": propositionSet, "vector1": wordVectorFeature.tolist()})
    dataset = pd.merge(dataset, wordVectorFrame, on='arg1')
    wordVectorFrame = wordVectorFrame.rename(columns={'arg1':'arg2', "vector1":"vector2"})
    dataset = pd.merge(dataset, wordVectorFrame, on='arg2')

    return dataset
    
