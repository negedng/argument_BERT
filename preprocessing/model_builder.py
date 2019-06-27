from argument_BERT.preprocessing import feature_extractor
import timeit
import numpy as np
from keras.layers import LSTM, Input, concatenate, Dense
from keras.models import Model, Sequential


def select_w2v_features(dataset):
    """Select features to match w2v model setup
    """

    arg1vectors = dataset['vector1'].to_numpy()
    arg1vectors = np.stack(arg1vectors.ravel())
    arg2vectors = dataset['vector2'].to_numpy()
    arg2vectors = np.stack(arg2vectors.ravel())
    
    return arg1vectors, arg2vectors


def select_argue_features(dataset,
                          shared_feature_list=['claimIndicatorArg1',
                                               'premiseIndicatorArg1',
                                               'claimIndicatorArg2',
                                               'premiseIndicatorArg2',
                                               'sameSentence',
                                               'sharedNouns',
                                               'numberOfSharedNouns',
                                               'tokensArg1',
                                               'tokensArg2'],
                          sentence_feature_list=['vector', 'pos']):
    """Select features to match ArguE model setup"""
    
    sentence1 = None
    sentence2 = None
    
    for feature in sentence_feature_list:
        
        next_f1 = np.stack(dataset[(feature+'1')].to_numpy().ravel())
        next_f2 = np.stack(dataset[(feature+'2')].to_numpy().ravel())
        
        if(sentence1 is None):
            sentence1 = next_f1
        else:
            sentence1 = np.concatenate((sentence1, next_f1), axis=-1)
        if(sentence2 is None):
            sentence2 = next_f2
        else:
            sentence2 = np.concatenate((sentence2, next_f2), axis=-1)        
    
    sharedFeatures = dataset[shared_feature_list]
    
    return sentence1, sentence2, sharedFeatures
    
    
def build_simple_w2v_LSTM(input_dim, output_dim=2,
                    units_LSTM=16, units_Dense=500,
                    loss='binary_crossentropy', optimizer='adam'):
    """Set up a simple w2v lstm model
    """
    sentence1 = Input(input_dim, name="sentence1")
    sentence2 = Input(input_dim, name="sentence2")

    lstm1 = LSTM(units_LSTM, return_sequences=False)(sentence1)
    lstm2 = LSTM(units_LSTM, return_sequences=False)(sentence2)
    concatenateLayer = concatenate([lstm1, lstm2], axis=-1)
    dense = Dense(units_Dense, activation='sigmoid')(concatenateLayer)
    softmax = Dense(output_dim, activation='softmax')(dense)

    model = Model(inputs=[sentence1, sentence2], outputs=[softmax])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model


def build_argue_RNN(lstm_input_dim, sharedFeatures_input_dim, output_dim=2,
              units_LSTM=16, units_Dense=500,
              loss='binary_crossentropy', optimizer='adam'):
    """ArguE model build"""

    sentence1 = Input(lstm_input_dim, name="sentence1")
    sentence2 = Input(lstm_input_dim, name="sentence2")
    sharedFeatures = Input(sharedFeatures_input_dim, name="sharedFeatures")

    lstm1 = LSTM(units_LSTM, return_sequences=False)(sentence1)
    lstm2 = LSTM(units_LSTM, return_sequences=False)(sentence2)
    concatenateLayer = concatenate([lstm1, lstm2, sharedFeatures], axis=-1)
    dense = Dense(units_Dense, activation='sigmoid')(concatenateLayer)
    softmax = Dense(output_dim, activation='softmax')(dense)

    model = Model(inputs=[sentence1, sentence2,  sharedFeatures], outputs=[softmax])
    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model