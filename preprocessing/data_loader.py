# Based on https://github.com/Milzi/ArguE/blob/master/DataLoader.py
import os
import xmltodict
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


def load_from_directory(directory):
    """Load all files in the directory, creates relation matrix for them"""
    print("Loading data from directory")
    print("Detected files: "+str(len(os.listdir(directory))))
    data_list = list()
    for e, annotation_file in enumerate(os.listdir(directory)):
        annotation_file_path = os.path.join(directory, annotation_file)
        file_data = load_single_file(e, annotation_file_path)
        data_list = data_list + file_data
    dataFrame = pd.DataFrame.from_dict(data_list, orient='columns')
    print("Loaded data length: " + str(len(dataFrame)))
    return dataFrame


def load_single_file(fileID, file_path):
    """Load a single file, creates relation matrix
    Output:
        arg1, arg2 - the arguments
        argumentID - specific ID for the file
        label - relation between the arguments
        originalArg1, originalArg2 - the original sentence for the argument
    """
    file_data = list()
    relationMatrix = {}
    with open(file_path, "r") as myfile:
        data = myfile.read()

    xmlData = xmltodict.parse(data)

    argumentationID = fileID

    matrixLength = len(xmlData["Annotation"]["Proposition"])
    relationCount = 0
    totalRelation = matrixLength*matrixLength
    relationMatrix = (matrixLength, matrixLength)
    relationMatrix = np.zeros(relationMatrix)

    propositions = xmlData["Annotation"]["Proposition"]

    for prop_id in range(len(propositions)):
        currentProposition = propositions[prop_id]

        if(currentProposition["ADU"]["@type"] != "conclusion" and
           "Relation" in currentProposition.keys()):

            partners = list()
            relationTypeList = list()

            if currentProposition["Relation"].__class__ == list().__class__:
                for relation in range(len(currentProposition["Relation"])):
                    relation_data = currentProposition["Relation"][relation]

                    partners.append(relation_data["@partnerID"])
                    relationTypeList.append(relation_data["@typeBinary"])

            else:
                relation_data = currentProposition["Relation"]
                partners.append(relation_data["@partnerID"])
                relationTypeList.append(relation_data["@typeBinary"])

            for partner_id in range(len(partners)):
                for prop_id2 in range(len(propositions)):
                    if partners[partner_id] == propositions[prop_id2]["@id"]:
                        if relationTypeList[partner_id] == "0":
                            relationMatrix[prop_id][prop_id2] = 1
                            relationMatrix[prop_id2][prop_id] = -1

                        elif relationTypeList[partner_id] == "1":
                            relationMatrix[prop_id][prop_id2] = 2
                            relationMatrix[prop_id2][prop_id] = -2
                        else:
                            relationMatrix[prop_id][prop_id2] = -3

    for i in range(len(relationMatrix)):
        for j in range(len(relationMatrix[i])):
            if i != j and relationMatrix[i][j] > -3:
                proposition1 = propositions[i]["text"]
                proposition2 = propositions[j]["text"]
                if(fit_tokenize_length_threshold(proposition1) or
                   fit_tokenize_length_threshold(proposition2)):
                    continue

                originalSentenceArg1 = propositions[i]["text"]
                originalSentenceArg2 = propositions[j]["text"]

                if "TextPosition" in propositions[i].keys():
                    if(propositions[i]["TextPosition"]["@start"] != "-1" or
                       propositions[j]["TextPosition"]["@start"] != "-1"):
                        original_text = xmlData["Annotation"]["OriginalText"]
                        sent_tokenize_list = sent_tokenize(original_text)
                        
                        if propositions[i]["TextPosition"]["@start"] != "-1":
                            for sentence in sent_tokenize_list:

                                if propositions[i]["text"] in sentence:
                                    originalSentenceArg1 = sentence

                        if propositions[j]["TextPosition"]["@start"] != "-1":

                            for sentence in sent_tokenize_list:
                                if propositions[j]["text"] in sentence:
                                    originalSentenceArg2 = sentence
                file_data.append({'argumentationID': argumentationID,
                                  'arg1': propositions[i]["text"],
                                  'originalArg1': originalSentenceArg1,
                                  'arg2': propositions[j]["text"],
                                  'originalArg2': originalSentenceArg2,
                                  'label': relationMatrix[i][j]})
    return file_data


def fit_tokenize_length_threshold(proposition):
    """Drop out too long tokens"""
    if len(sent_tokenize(proposition)) > 1:
        return True
    elif len(word_tokenize(proposition)) > 30:
        return True
    else:
        return False


def prepare_data_for_training(dataset, model_type=None):
    """Prepare dataset to training. Starts feature extracting.
    Returns x_data, y_data"""
    y_data = dataset['label'].to_numpy()
    numberOfLabels = np.unique(y_data).shape[0]
    y_data = np.identity(numberOfLabels)[y_data.astype(int).flatten()]

    x_data = dataset.drop(['label', 'argumentationID'], axis=1)

    return x_data, y_data

