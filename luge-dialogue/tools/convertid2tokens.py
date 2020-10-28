#!/usr/bin/env python
# -*- coding: utf-8 -*-


import json
import random
import sentencepiece as spm


sp = spm.SentencePieceProcessor()
sp.load("../config/spm.model")

type_dict = {30001: "chitchat" , 30002:"knowledge",  30003 :"recommend"}


input_file = './b_0'
output_file = './ori_token.txt'
tokenarray = []
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        arrays = line.split(';')
        tmp = []
        for string in arrays:
            tokens = []
            ids = [int(item) for item in string.split(' ')]
            for char in ids:
                if char in type_dict.keys():
                    tokens.append(type_dict[char])
                else:
                    tokens.append(sp.DecodeIds([char]))
            tmp.append(tokens)
        tokenarray.append(';'.join(tmp))

    fp.close()

with open(output_file, encoding='utf8') as fp:
    for line in tokenarray:
        fp.writelines(line)

    fp.close()