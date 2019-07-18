from argument_BERT.preprocessing import feature_extractor
import timeit
import numpy as np
from keras.layers import LSTM, Input, concatenate, Dense, Dropout
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
    
    sentence1 = np.array([])
    sentence2 = np.array([])
    first_iteration = True
    
    for feature in sentence_feature_list:
        next_f1 = np.stack(dataset[str(feature+'1')].to_numpy().ravel())
        next_f2 = np.stack(dataset[str(feature+'2')].to_numpy().ravel())
        if first_iteration:
            sentence1 = next_f1
        else:
            sentence1 = np.concatenate((sentence1, next_f1), axis=-1)
        if first_iteration:
            sentence2 = next_f2
        else:
            sentence2 = np.concatenate((sentence2, next_f2), axis=-1)        
        first_iteration = False
        
    sharedFeatures = dataset[shared_feature_list]
    
    return sentence1, sentence2, sharedFeatures


def select_FFNN_features(dataset, shared_features=True,
                         shared_feature_list=None,
                         original_bert=False,
                         has_2=True):
    """Only global features"""
    
    if shared_feature_list == None:
        shared_feature_list = dataset.columns
        shared_feature_list = shared_feature_list.drop(['arg1', 'arg2', 'originalArg1', 'originalArg2', 'fullText1', 'argumentationID', 'label', 'originalLabel', 'bertArg1', 'bertArg2', 'bertOriginalArg1', 'bertOriginalArg2', 'vector1', 'vector2', 'pos1', 'pos2', 'bertVector1', 'bertVector2', 'results', 'predicted_label', 'expected_label'], errors='ignore')
    
    sent1 = np.stack(dataset["bertArg1"].to_numpy().ravel())
    if has_2:
        sent2 = np.stack(dataset["bertArg2"].to_numpy().ravel())
    sharedFeatures = dataset[shared_feature_list]
    if not original_bert:
        if not shared_features:
            if not has_2:
                return [sent1]
            return [sent1, sent2]
        if not has_2:
            return [sent1, sharedFeatures]
        return [sent1, sent2, sharedFeatures]
    
    orig1 = np.stack(dataset["bertOriginalArg1"].to_numpy().ravel())
    if has_2:
        orig2 = np.stack(dataset["bertOriginalArg2"].to_numpy().ravel())
    if not shared_features:
        if not has_2:
            return [sent1, orig1]
        return [sent1, sent2, orig1, orig2]
    if not has_2:
        return [sent1, orig1, sharedFeatures]
    return [sent1, sent2, orig1, orig2, sharedFeatures]


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


def build_FFNN(shared_feature_dim, output_dim=2,
               no_separate_layers=2, separate_start_units=300,
               no_concat_layers=2, concat_start_units=300,
               layer_decrease_rate=0.4, dropout_rate=0.2,
               has_shared_features=True, has_original_sentences=False,
               no_sent_arg_layers=2, sent_arg_start_units=300,
               optimizer="adam", activation='sigmoid',
               has_2=True):
    """Model using sentence level embeddings of the 2 arguments 
       and shared features"""
       
    # Input layers
    sentence1 = Input((768,), name="sentence1") # See BERT for input dim
    if has_2:
        sentence2 = Input((768,), name="sentence2")
    if has_shared_features:
        sharedF = Input(shared_feature_dim, name="sharedFeatures")
    if has_original_sentences:
        original1 = Input((768,), name="original1")
        if has_2:
            original2 = Input((768,), name="original2")
    
    # Dense layers of the inputs
    separate_units = separate_start_units
    dense1 = Dense(separate_units, activation=activation)(sentence1)
    if has_2:
        dense2 = Dense(separate_units, activation=activation)(sentence2)
    if has_original_sentences:
        denseOriginal1 = Dense(separate_units, activation=activation)(original1)
        if has_2:
            denseOriginal2 = Dense(separate_units, activation=activation)(original2)
    for i in range(1,no_separate_layers):
        separate_units = int(separate_units*layer_decrease_rate)
        dense1 = Dropout(rate=dropout_rate)(dense1)
        dense1 = Dense(separate_units, activation=activation)(dense1)
        if has_2:
            dense2 = Dropout(rate=dropout_rate)(dense2)
            dense2 = Dense(separate_units, activation=activation)(dense2)
        if has_original_sentences:
            denseOriginal1 = Dropout(rate=dropout_rate)(denseOriginal1)
            denseOriginal1 = Dense(separate_units, activation=activation)(denseOriginal1)
            if has_2:
                denseOriginal2 = Dropout(rate=dropout_rate)(denseOriginal2)
                denseOriginal2 = Dense(separate_units, activation=activation)(denseOriginal2)
    
    # Concat argument and sentence layers
    if has_original_sentences:
        arg_sent_units = sent_arg_start_units
        concat1 = concatenate([dense1, denseOriginal1], axis=-1)
        dense1 = Dense(arg_sent_units, activation=activation)(concat1)
        if has_2:
            concat2 = concatenate([dense2, denseOriginal2], axis=-1)
            dense2 = Dense(arg_sent_units, activation=activation)(concat2)
        for i in range(1, no_sent_arg_layers):
            arg_sent_units = int(arg_sent_units*layer_decrease_rate)
            dense1 = Dropout(rate=dropout_rate)(dense1)
            dense1 = Dense(arg_sent_units, activation=activation)(dense1)
            if has_2:
                dense2 = Dropout(rate=dropout_rate)(dense2)
                dense2 = Dense(arg_sent_units, activation=activation)(dense2)
    
    # Concat with shared features and each other
    if has_shared_features:
        if has_2:
            concatenateLayer = concatenate([dense1, dense2, sharedF], axis=-1)
        else:
            concatenateLayer = concatenate([dense1, sharedF], axis=-1)
    else:
        if has_2:
            concatenateLayer = concatenate([dense1, dense2], axis=-1)
        else:
            concatenateLayer = concatenate([dense1], axis=-1)
        
    concat_units = concat_start_units
    dense = Dense(concat_units, activation=activation)(concatenateLayer)
    for i in range(1,no_concat_layers):
        concat_units = int(concat_units*layer_decrease_rate)
        dense = Dropout(rate=dropout_rate)(dense)
        dense = Dense(concat_units, activation=activation)(dense)
        
    # Softmax output
    softmax = Dense(output_dim, activation='softmax')(dense)
    
    if has_shared_features:
        if has_original_sentences:
            if has_2:
                model = Model(inputs=[sentence1, sentence2, original1, original2, sharedF], outputs=[softmax])
            else:
                model = Model(inputs=[sentence1, original1, sharedF], outputs=[softmax])
        else:
            if has_2:
                model = Model(inputs=[sentence1, sentence2, sharedF], outputs=[softmax])
            else:
                model = Model(inputs=[sentence1, sharedF], outputs=[softmax])
    else:
        if has_original_sentences:
            if has_2:
                model = Model(inputs=[sentence1, sentence2, original1, original2], outputs=[softmax])
            else:
                model = Model(inputs=[sentence1, original1], outputs=[softmax])
        else:
            if has_2:
                model = Model(inputs=[sentence1, sentence2], outputs=[softmax])
            else:
                model = Model(inputs=[sentence1], outputs=[softmax])

    if output_dim == 2:
        loss="binary_crossentropy"
    else:
        loss="categorical_crossentropy"

    model.compile(loss=loss, optimizer=optimizer, metrics=['accuracy'])
    
    return model
