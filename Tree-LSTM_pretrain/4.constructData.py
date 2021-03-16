"""
Construct the training set, validation set, and test set that need to be put into the pre-trained tree-LSTM model.
"""

from utils import Node
import pickle
import json
import random
import numpy as np


def build_tree(data_name, data_type):
    path = data_name + '/' + data_type + '/split_' + data_type + '_ast.json'
    # path = "test\\split_test_ast.json"   # train valid test
    roots_list = []
    file = []
    if data_name == 'Java' :
        with open(path, "r", encoding='utf-8') as f:
            for line in f:   # Read a line
                root_list = []   # The current row has several small subtrees
                file = eval(line)
                for part in file:  # Each little subtree in a row
                    json_object = json.dumps(part)
                    json_object = json.loads(json_object)
                    nodes = [Node(num=i, children=[]) for i in range(len(json_object))]
                    for i in range(len(json_object)):
                        nodes[i].label = str(json_object[i].get('type')) + "_" + str(json_object[i].get('value'))
                        # print(i, " ", nodes[i].label)
                        if json_object[i].get('children') != None:
                            children = json_object[i].get('children')
                            for c in children:
                                nodes[i].children.append(nodes[c])
                                nodes[c].parent = nodes[i]
                    root_list.append(nodes[0])
                roots_list.append(root_list)  # [ [] , [] , [], [] , [] ]

    elif data_name == 'Python':
        with open(path, "r", encoding='utf-8') as f:
            for line in f:  # Read a line
                root_list = []  # The current row has several small subtrees
                line = json.loads(line)
                json_object = line
                for item in json_object:
                    nodes = [Node(num=i, children=[]) for i in range(len(item))]
                    i = 0
                    for l in item:
                        # print(l, ':', item[l])
                        nodes[i].label = str(item[l].get('node'))
                        # print(i, " ", nodes[i].label)
                        if item[l].get('children') is not None:
                            children = item[l].get('children')
                            if len(children) == 1 and '1*NODEFIX' not in children[0]:
                                nodes[i].label += '_' + str(item[l].get('children')[0].strip().replace('\n', ''))
                                i += 1
                                continue
                            for c in children:
                                c = int(c.replace('1*NODEFIX', '')) - 1
                                nodes[i].children.append(nodes[c])
                                nodes[c].parent = nodes[i]
                        i += 1
                    root_list.append(nodes[0])
                roots_list.append(root_list)

    return roots_list

def set_seed(SEED=2020):
    random.seed(SEED)
    np.random.seed(SEED)


if __name__ == '__main__':
    set_seed()
    dataName = ['Java', 'Python']
    dataType = ['test', 'valid', 'train']
    for name in dataName:
        for type in dataType:
            roots_list = build_tree(name, type)
            pickle.dump(roots_list, open(name + '/' + type + '/' + type + '_pre_ast_' + name + '.pkl', "wb"))
            print(len(roots_list))
            print(name + '/' + type + ' success!!')