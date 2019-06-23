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

    
def simple_w2v_LSTM(input_dim, output_dim=2,
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
