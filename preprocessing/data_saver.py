#!/usr/bin/python
# -*- coding: utf-8 -*-
from lxml import etree
import json
json.encoder.FLOAT_REPR = lambda x: format(x, '.5f')

def save2file(data, filename):
    if filename[-4:]=='.xml':
        root = etree.Element('Annotation')
        root.set("corpus", data['corpus'])
        
        for prop in data['propositions']:
            propE = etree.Element('Proposition')
            propE.set('id', prop['id'])
            root.append(propE)
            
            aduE = etree.Element('ADU')
            aduE.set('type', prop['ADU']['type'])
            aduE.set('confidence', str(prop['ADU']['confidence']))
            propE.append(aduE)
            
            textE = etree.Element('text')
            textE.text = prop['text']
            propE.append(textE)
            
            positionE = etree.Element('TextPosition')
            positionE.set('start', str(prop['textPosition']['start']))
            positionE.set('end', str(prop['textPosition']['end']))
            propE.append(positionE)
            
            for relation in prop['relations']:
                relationE = etree.Element('Relation')
                relationE.set('relationID', relation['id'])
                relationE.set('type', relation['type'])
                relationE.set('typeBinary', str(relation['typeBinary']))
                relationE.set('partnerID', relation['partnerID'])
                propE.append(relationE)
        originalTextE = etree.Element('OriginalText')
        originalTextE.text = data['originalText']
        root.append(originalTextE)
        
        xmlData = etree.tostring(root, pretty_print=True)
        
        with open(filename, 'wb') as f:
            f.write(xmlData)
    else:
        if filename[-5:]!='.json':
            print("Warning: No file format specified.")
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4, sort_keys=True)