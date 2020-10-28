#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json

input_file = '../data/ori_data.txt'
oridata = []
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        oridata.append(json.loads(line))

UnifiedTransformer = []
input_file = './1st_inference_output.txt'
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        UnifiedTransformer.append(line.strip())

Plato = []
input_file = './2st_inference_output.txt'
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        Plato.append(line.strip())

addNsp_Plato = []
input_file = 'inference_output.txt'
with open(input_file, encoding='utf8') as fp:
    for line in fp:
        addNsp_Plato.append(line.strip())
print(len(addNsp_Plato))

all_result = []
for index, item in enumerate(oridata):
    item.pop('response')
    item['UniTransformer_response'] = UnifiedTransformer[index]
    item['Plato_response'] = Plato[index]
    item['NSPPlato_response'] = addNsp_Plato[index]

    all_result.append(item)

    if index == 31:
        break

result = open("./result_eval.txt", 'w')
for item in all_result:
    # print(sample)
    result.write(json.dumps(item, ensure_ascii=False, indent=4) + "\n")

