#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import random
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.load("../config/spm.model")

type_dict = {"chitchat": 30001, "knowledge": 30002, "recommend": 30003}

input_file = './b_0'
output_file = './ori_token.txt'
tokenarray = []
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        ids = line.split(' ')
        tokens = sp.DecodePieces(ids)
        tokenarray.append(tokens)

    fp.close()

with open(output_file, encoding='utf8') as fp:
    for line in tokenarray:
        fp.writelines(line)

    fp.close()