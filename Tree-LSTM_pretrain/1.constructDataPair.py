"""
Construct sentence pairs and vocab of tree-lstm pre-training model
"""


from utils import Node
from collections import Counter
import json
import random
import numpy as np
import pickle
from tqdm import tqdm

def build_tree(data_name, data_type):
    path = data_name + '/' + data_type + '/split_' + data_type + '_ast.json'
    # path = "train/split_train_ast.json"   # train valid test
    roots_list = []
    file = []
    with open(path, "r", encoding='utf-8') as f:
        for line in tqdm(f):
            root_list = []
            if data_name == 'Java':
                file = eval(line)
                for part in file:
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
                roots_list.append(root_list)
            elif data_name == 'Python':
                line = json.loads(line)
                json_object = line
                for item in json_object:
                    # print(item)
                    nodes = [Node(num=i, children=[]) for i in range(len(item))]
                    # print(len(nodes))
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
                    # for i in range(len(nodes)):
                    #     print(nodes[i].label)
                    root_list.append(nodes[0])
                roots_list.append(root_list)
    list1 = []
    list2 = []
    label = []
    for i in range(len(roots_list)):
        cur_line = roots_list[i]
        if len(cur_line) > 3:
            for j in range(len(cur_line)-1):
                list1.append(cur_line[j])
                list2.append(cur_line[j+1])
                label.append(1)

    print("label1: ", len(list1)," ",len(list2), " ",len(label))

    for i in range(len(roots_list)-1):
        first = roots_list[i]
        second = roots_list[i+1]
        index1 = random.randint(0, len(first) - 1)
        index2 = random.randint(0, len(second) - 1)
        list1.append(first[index1])
        list2.append(second[index2])
        label.append(0)
        index1 = random.randint(0, len(first) - 1)
        index2 = random.randint(0, len(second) - 1)
        list1.append(first[index1])
        list2.append(second[index2])
        label.append(0)
    print('label0 + label1: ', len(list1), " ", len(list2), " ", len(label))
    # print(len(roots_list))
    newindex = list(np.random.permutation(len(label)))
    list1 = [list1[i] for i in newindex]
    list2 = [list2[i] for i in newindex]
    label = [label[i] for i in newindex]
    return list1, list2, label


def build_vocab(name):
    path = name + '/train/split_train_ast.json'
    vo = []
    file = []
    with open(path, "r", encoding='utf-8') as f:
        if name == 'Java':
            for line in tqdm(f):
                root_list = []
                file = eval(line)
                for part in file:
                    json_object = json.dumps(part)
                    json_object = json.loads(json_object)
                    nodes = [Node(num=i, children=[]) for i in range(len(json_object))]
                    for i in range(len(json_object)):
                        vo.append(str(json_object[i].get('type')) + "_" + str(json_object[i].get('value')))
        elif name == 'Python':
            for line in tqdm(f):
                root_list = []
                line = json.loads(line)
                json_object = line
                for item in json_object:
                    for l in item:
                        res = str(item[l].get('node'))
                        if item[l].get('children') is not None:
                            children = item[l].get('children')
                            if len(children) == 1 and '1*NODEFIX' not in children[0]:
                                res += '_' + str(item[l].get('children')[0].strip().replace('\n', ''))
                        vo.append(res)
    vocab = Counter(vo)
    ast_i2w = {i: w for i, w in enumerate(
        ["<UNK>"] + sorted([x[0] for x in vocab.most_common(50000)]))}
    ast_w2i = {w: i for i, w in enumerate(
        ["<UNK>"] + sorted([x[0] for x in vocab.most_common(50000)]))}
    print(len(ast_w2i))
    with open(name + "/vocab.ast", "w", encoding='utf-8') as f:
        for x in ast_w2i.keys():
            f.write(x + "\n")
    return ast_i2w, ast_w2i

def set_seed(SEED=2020):
    random.seed(SEED)
    np.random.seed(SEED)

if __name__ == '__main__':
    set_seed()
    dataName = ['Java', 'Python']
    # get vocab
    for name in dataName:
        ast_i2w , ast_w2i = build_vocab(name)
        pickle.dump(ast_i2w, open(name + "/ast_i2w_" + name + ".pkl", "wb"))
        pickle.dump(ast_w2i, open(name + "/ast_w2i_" + name + ".pkl", "wb"))
        print(name + ' save ast vocab sucessful!!!')

    # Construct sentence pairs
    dataType = ['test', 'valid', 'train']
    for name in dataName:
        for type in dataType:
            list1, list2, label = build_tree(name, type)
            pickle.dump(list1, open(name + "/" + type + "/" + type + "_ast1_" + name + ".pkl", "wb"))
            pickle.dump(list2, open(name + "/" + type + "/" + type + "_ast2_" + name + ".pkl", "wb"))
            pickle.dump(label, open(name + "/" + type + "/" + type + "_label_" + name + ".pkl", "wb"))
            print(name + '/' + type + ' success!!')