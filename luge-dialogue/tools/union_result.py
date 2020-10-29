#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

input_file = '../data/ori_test.txt'
oridata = []
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        oridata.append(json.loads(line))

UnifiedTransformer = []
input_file = './unif_result.txt'
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        UnifiedTransformer.append(line)

Plato = []
input_file = './plato_result.txt'
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        Plato.append(line)

addNsp_Plato = []
input_file = 'nsp_result.txt'
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        addNsp_Plato.append(line)
print(len(addNsp_Plato))

all_result = []
for index, item in enumerate(oridata):
    item.pop('response')
    item['context'] = item['context'].split('\t')
    item['UniTransformer_response'] = UnifiedTransformer[index]
    item['Plato_response'] = Plato[index]
    item['NSPPlato_response'] = addNsp_Plato[index]

    all_result.append(item)

result = open("./result_eval.txt", 'w')
for item in all_result:
    # print(sample)
    result.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")

