#!/usr/bin/python
# -*- coding: utf-8 -*-

# Based on https://github.com/Milzi/ArguE/blob/master/DataLoader.py

import os
import xmltodict
import numpy as np
import pandas as pd
from nltk.tokenize import sent_tokenize, word_tokenize


def load_from_directory(directory, rst_files=False, ADU=False):
    """Load all files in the directory, creates relation matrix for them
    Input:
        directory: directory with annotation files
        rst_file: True, if the directory stores RST files as well
        ADU: True for proposition type data loading
    Output:
        dataFrame: pandas DataFrame with samples as rows"""

    print('Loading data from directory')
    print('Detected files: ' + str(len(os.listdir(directory))))
    data_list = list()
    for (e, annotation_file) in enumerate(os.listdir(directory)):
        if annotation_file[-7:] not in ['ann.xml', 'son.xml']:
            continue
        annotation_file_path = os.path.join(directory, annotation_file)
        if not ADU:
            file_data = load_single_file(e, annotation_file_path, rst_files)
        else:
            file_data = load_for_ADU_types(e, annotation_file_path)
        data_list = data_list + file_data
    dataFrame = pd.DataFrame.from_dict(data_list, orient='columns')
    print('Loaded data length: ' + str(len(dataFrame)))
    return dataFrame


def load_single_file(fileID, file_path, rst_files=False):
    """Load a single file, creates relation matrix
    Input:
        fileID: index for the processed files
        file_path: filename
        rst_files: True if RST files are stored and used
    Output:
        file_data: dictionary with the features stored:
                   arg1, arg2, argumentationID, label,
                   originalArg1, originalArg2, fullText1,
                   rstCon, rstConParent - only if RST active,
                   positionDiff, positArg1, positArg2,
                   sentenceDiff, sen1, sen2 - only if full text exists
    """

    file_data = list()
    relationMatrix = {}
    with open(file_path, 'r') as myfile:
        data = myfile.read()

    xmlData = xmltodict.parse(data)

    if rst_files:
        (recovered_string, prop_edu_dict) = load_merge(file_path)
        edges = load_brackets(file_path)

    argumentationID = fileID

    matrixLength = len(xmlData['Annotation']['Proposition'])
    relationCount = 0
    totalRelation = matrixLength * matrixLength
    relationMatrix = (matrixLength, matrixLength)
    relationMatrix = np.zeros(relationMatrix)

    propositions = xmlData['Annotation']['Proposition']
    if 'OriginalText' in xmlData['Annotation']:
        original_text = xmlData['Annotation']['OriginalText']
        original_text2 = original_text.replace('\n', ' ')
        sent_tokenize_list = sent_tokenize(original_text)
        sens = len(sent_tokenize_list)

    for prop_id in range(len(propositions)):
        currentProposition = propositions[prop_id]

        if currentProposition['ADU']['@type'] != 'conclusion' \
           and 'Relation' in currentProposition.keys():

            partners = list()
            relationTypeList = list()

            if currentProposition['Relation'].__class__ \
               == list().__class__:
                for relation in range(len(currentProposition['Relation'])):
                    relation_data = currentProposition['Relation'][relation]

                    partners.append(relation_data['@partnerID'])
                    relationTypeList.append(relation_data['@typeBinary'])
            else:

                relation_data = currentProposition['Relation']
                partners.append(relation_data['@partnerID'])
                relationTypeList.append(relation_data['@typeBinary'])

            for partner_id in range(len(partners)):
                for prop_id2 in range(len(propositions)):
                    if partners[partner_id] \
                       == propositions[prop_id2]['@id']:
                        if relationTypeList[partner_id] == '0':
                            relationMatrix[prop_id][prop_id2] = 1
                            relationMatrix[prop_id2][prop_id] = -1
                        elif relationTypeList[partner_id] == '1':

                            relationMatrix[prop_id][prop_id2] = 2
                            relationMatrix[prop_id2][prop_id] = -2
                        else:
                            relationMatrix[prop_id][prop_id2] = -3

    for i in range(len(relationMatrix)):
        for j in range(len(relationMatrix[i])):
            if i != j and relationMatrix[i][j] > -3:
                proposition1 = propositions[i]['text']
                proposition2 = propositions[j]['text']
                if fit_tokenize_length_threshold(proposition1) \
                   or fit_tokenize_length_threshold(proposition2):
                    continue

                originalSentenceArg1 = propositions[i]['text']
                originalSentenceArg2 = propositions[j]['text']

                if 'TextPosition' in propositions[i].keys():
                    if propositions[i]['TextPosition']['@start'] \
                       != '-1' or propositions[j]['TextPosition'
                                                  ]['@start'] != '-1':

                        if propositions[i]['TextPosition']['@start'] \
                           != '-1':
                            for sentence in sent_tokenize_list:

                                if propositions[i]['text'] in sentence:
                                    originalSentenceArg1 = sentence
                                    sen1 = sent_tokenize_list.index(sentence)

                        if propositions[j]['TextPosition']['@start'] != '-1':

                            for sentence in sent_tokenize_list:
                                if propositions[j]['text'] in sentence:
                                    originalSentenceArg2 = sentence
                                    sen2 = sent_tokenize_list.index(sentence)

                line_data = {
                    'argumentationID': argumentationID,
                    'arg1': propositions[i]['text'],
                    'originalArg1': originalSentenceArg1,
                    'arg2': propositions[j]['text'],
                    'originalArg2': originalSentenceArg2,
                    'label': relationMatrix[i][j],
                    'fullText1': original_text2,
                    }

                if rst_files:
                    arg1_range = get_edus(propositions[i]['text'],
                                          recovered_string, prop_edu_dict)
                    arg2_range = get_edus(propositions[j]['text'],
                                          recovered_string, prop_edu_dict)
                    arg1_rsts = get_rst_stats(arg1_range, edges)
                    arg2_rsts = get_rst_stats(arg2_range, edges)
                    cn1 = arg1_rsts['connected_nodes']
                    cn2 = arg2_rsts['connected_nodes']
                    conn = False
                    conn_parent = any([z in cn1 for z in cn2])
                    for c in cn1:
                        if c in arg2_range:
                            conn = True
                    for c in cn2:
                        if c in arg1_range:
                            conn = True
                    line_data['rstCon'] = (1 if conn else 0)
                    line_data['rstConParent'] = \
                        (1 if conn_parent else 0)

#                    line_data['posEduArg1'] = arg1_range[0]
#                    line_data['posEduArg2'] = arg2_range[0]

                positArg1 = int(propositions[i]['TextPosition']['@start'])
                positArg2 = int(propositions[j]['TextPosition']['@start'])
                if positArg1 != -1 and positArg2 != -1:
                    posit = abs((positArg1 - positArg2) / len(original_text))
                    line_data['positionDiff'] = posit
                    line_data['positArg1'] = positArg1 / len(original_text)
                    line_data['positArg2'] = positArg2 / len(original_text)
                    senit = abs(sen1 - sen2)
                    line_data['sentenceDiff'] = senit / sens
                    line_data['sen1'] = sen1 / sens
                    line_data['sen2'] = sen2 / sens

                file_data.append(line_data)
    return file_data


def load_for_ADU_types(fileID, file_path):
    """Loads ADU type features.
    Input:
        fileID - index for the processed files
        file_path: filename
    Output:
        file_data: dictionary with the features stored:
                   arg1, argumentationID, label,
                   originalArg1, fullText1, positArg1
    """
    file_data = list()
    relationMatrix = {}
    with open(file_path, 'r') as myfile:
        data = myfile.read()

    argumentationID = fileID
    xmlData = xmltodict.parse(data)

    matrixLength = len(xmlData['Annotation']['Proposition'])
    relationCount = 0
    totalRelation = matrixLength * matrixLength
    relationMatrix = (matrixLength, matrixLength)
    relationMatrix = np.zeros(relationMatrix)
    original_text2 = " "

    xmlData = xmltodict.parse(data)

    propositions = xmlData['Annotation']['Proposition']
    if 'OriginalText' in xmlData['Annotation']:
        original_text = xmlData['Annotation']['OriginalText']
        original_text2 = original_text.replace('\n', ' ')
        sent_tokenize_list = sent_tokenize(original_text)
        sens = len(sent_tokenize_list)

    for prop_id in range(len(propositions)):
        currentProposition = propositions[prop_id]

        if currentProposition['ADU']['@type'] == 'conclusion':
            aduType = 2
        elif currentProposition['ADU']['@type'] == 'claim':
            aduType = 1
        elif currentProposition['ADU']['@type'] == 'premise':
            aduType = 0
        else:
            err_ADU = currentProposition['ADU']['@type']
            raise ValueError('Unexpected ADU type: ' + err_ADU)

        arg1 = currentProposition['text']
        originalSentenceArg1 = arg1
        positArg1 = -1

        if currentProposition['TextPosition']['@start'] != '-1':
            for sentence in sent_tokenize_list:

                if arg1 in sentence:
                    originalSentenceArg1 = sentence
                    sen1 = sent_tokenize_list.index(sentence)

            positArg1 = int(currentProposition['TextPosition']['@start'])
        line_data = {
            'argumentationID': argumentationID,
            'arg1': arg1,
            'originalArg1': originalSentenceArg1,
            'label': aduType,
            'fullText1': original_text2,
            'positArg1': positArg1 / len(original_text2),
            }
        file_data.append(line_data)
    return file_data


def fit_tokenize_length_threshold(proposition, min_len=1, max_len=30):
    """Drop out too long tokens"""

    if len(sent_tokenize(proposition)) > min_len:
        return True
    elif len(word_tokenize(proposition)) > max_len:
        return True
    else:
        return False


# See: https://github.com/jiyfeng/DPLP for data parsing

edge_type_list = [
    'span',
    'purpose',
    'textualorganization',
    'attribution',
    'elaboration',
    'list',
    'circumstance',
    'antithesis',
    'same_unit',
    'manner',
    'reason',
    'explanation',
    'condition',
    'means',
    'topic',
    'example',
    'temporal',
    'concession',
    'contrast',
    'result',
    'question',
    'comparison',
    'consequence',
    'sequence',
    'summary',
    'restatement',
    ]


def load_merge(file_path):
    """Load from merge file"""

    merge_file_path = file_path.replace('ann.xml', 'txt.merge')
    props = []
    with open(merge_file_path) as f:
        for line in f:
            if line == '\n':
                continue
            props.append(line.split('\t'))
    recovered_string = ''
    prop_edu_dict = {}
    for proposition in props:
        ws = ''
        if proposition[2] not in [
                                  "'",
                                  '.',
                                  ',',
                                  '?',
                                  '!',
                                  "'s",
                                  ]:
            ws = ' '
        prop_edu_dict[len(recovered_string)] = int(proposition[-1])
        recovered_string += ws + proposition[2]
    recovered_string = recovered_string.lstrip()
    return (recovered_string, prop_edu_dict)


def load_brackets(file_path):
    """Load from brackets file"""

    bracket_file_path = file_path.replace('ann.xml', 'txt.brackets')
    edges = []
    with open(bracket_file_path) as f:
        for line in f:
            line_p = line.strip().replace('(', '').replace(')', '')
            line_p = line_p.replace("'", '').replace(',', '').split(' ')
            edges.append({
                'node_1': int(line_p[0]),
                'node_2': int(line_p[1]),
                'node_type': line_p[2],
                'edge_type': line_p[3],
                })
    return edges


def get_edus(arg, edu_string, edu_dict):
    """Get starting and ending EDU for argument"""

    prefix = edu_string.find(arg)
    length = len(arg)
    for i in range(prefix, prefix + length):
        if i in edu_dict:
            start = edu_dict[i]
            break
    for i in range(prefix + length, prefix, -1):
        if i in edu_dict:
            end = edu_dict[i] + 1
            break
    return range(start, end)


def get_rst_stats(edus, edges):
    """Get info from brackets of arg EDUs"""

    satelite_no = 0
    nucleus_no = 0
    edge_types = [0] * len(edge_type_list)
    connected_nodes = []
    for edge in edges:
        if edge['node_1'] in edus or edge['node_2'] in edus:
            if edge['node_type'] == 'Satellite':
                satelite_no += 1
            if edge['node_type'] == 'Nucleus':
                nucleus_no += 1

            edge_types[edge_type_list.index(edge['edge_type'])] += 1

            if edge['node_1'] not in edus:
                if edge['node_1'] not in connected_nodes:
                    connected_nodes.append(edge['node_1'])
            if edge['node_2'] not in edus:
                if edge['node_2'] not in connected_nodes:
                    connected_nodes.append(edge['node_2'])
    return {
        'nucleus': nucleus_no,
        'satelite': satelite_no,
        'edge_types': edge_types,
        'connected_nodes': connected_nodes,
        }
