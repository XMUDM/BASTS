#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import json
import os
from tqdm import tqdm
from copy import deepcopy
from graphviz import Digraph


def read_cfg(file_path):
    file = open(file_path, encoding='utf-8')
    node_dict = {}
    for line in file.readlines():
        line_dict = json.loads(line)
        node_dict[line_dict['id']] = line_dict
    file.close()
    return node_dict


def draw_graph(node_dict, name='GRAPH', display_code=False):
    g = Digraph(name)
    for idx, node in node_dict.items():
        # node represent source code or id
        if display_code:
            g.node(name=node['id'], label=node['source_code'])
        else:
            g.node(name=node['id'])
        for next_node in node['next_nodes']:
            g.edge(node['id'], next_node)
    g.view(directory='img/')


def update_pre(node_dict):
    id_visited = {}
    for idx, node in node_dict.items():
        id_visited[idx] = False
    node_dict['1']['pre_nodes'] = []
    dfs('1', node_dict, id_visited)
    return node_dict


def dfs(current_id, node_dict, id_visited):   # update pre nodes
    next_nodes = node_dict[current_id]['next_nodes']
    if len(next_nodes) == 0:
        return
    if not id_visited[current_id]:
        for next_node_id in next_nodes:
            if next_node_id != '':
                next_node = node_dict[next_node_id]
                if 'pre_nodes' not in next_node.keys():
                    node_dict[next_node_id]['pre_nodes'] = [current_id]
                else:
                    node_dict[next_node_id]['pre_nodes'].append(current_id)
        id_visited[current_id] = True
        for next_node_id in next_nodes:
            if next_node_id != '':
                dfs(next_node_id, node_dict, id_visited)


def find_dominators(cfg):
    in_dict = {}
    out_dict = {}
    for idx, node in cfg.items():
        if idx == '1':
            out_dict[idx] = [idx]
        else:
            out_dict[idx] = list(cfg.keys())
        in_dict[idx] = []
    changed = True
    while changed:
        changed = False
        temp_in = deepcopy(in_dict)
        temp_out = deepcopy(out_dict)
        for idx, node in cfg.items():
            if idx == '1':
                continue
            for pre_node in node['pre_nodes']:
                if len(in_dict[idx]) == 0:
                    in_dict[idx] = sorted(out_dict[pre_node], key=lambda x: int(x))
                else:
                    in_dict[idx] = sorted(list(set(in_dict[idx]).intersection(out_dict[pre_node])), key=lambda x: int(x))
            tmp = deepcopy(in_dict[idx])
            tmp.append(idx)
            out_dict[idx] = sorted(tmp, key=lambda x: int(x))
        if out_dict != temp_out or in_dict != temp_in:
            changed = True
    return in_dict, out_dict


def build_dom_tree(out_dict):
    tree_json = {}
    for idx, dom in out_dict.items():
        if idx == '1':
            continue
        if dom[-2] not in tree_json.keys():
            tree_json[dom[-2]] = {'id': dom[-2], 'next_nodes': [idx]}
        else:
            tree_json[dom[-2]]['next_nodes'].append(idx)
        tree_json[idx] = {'id': idx, 'next_nodes':[]}
    return tree_json


def split_dom_tree(block_list, node_dict, current_node='1', current_block=None):
    if current_block is None:
        current_block = []
    current_node_dict = node_dict[current_node]
    if len(node_dict) == 1:  # end node
        if current_block is not None and len(current_block) > 0:
            block_list.append(current_block)
        return
    # If the current node is a branch node, then the current node is added to the current block,
    # and the current block is added to the block_list
    if len(current_node_dict['next_nodes']) > 1:
        if current_node != '1':
            current_block.append(current_node)
        del node_dict[current_node]
        if current_block is not None and len(current_block) > 0:
            block_list.append(current_block)
        # Recursion, the current node is the subsequent node
        sorted_node = sorted(current_node_dict['next_nodes'], key=lambda x: int(x))
        for next_node in sorted_node:
            current_block = []
            split_dom_tree(block_list, node_dict, next_node, current_block)
    else:  # The current node has the only successor, or there is no
        if current_node != '1':
            current_block.append(current_node)
        del node_dict[current_node]
        if len(current_node_dict['next_nodes']) > 0:
            split_dom_tree(block_list, node_dict, current_node_dict['next_nodes'][0], current_block)
        else:
            block_list.append(current_block)


def split_code(cfg, block_list):
    code_block_list = []
    for block in block_list:
        code_block = ''
        for node_id in block:
            code = cfg[node_id]['source_code']
            # delete catch、finally to generate AST
            if code.find('catch') >= 0 or code.find('finally') >= 0:
                continue
            code_block += code
        # Completing curly braces
        missing_count = code_block.count('{') - code_block.count('}')
        if missing_count > 0:
            code_block += '}'*missing_count
        if code_block != '':
            code_block_list.append(code_block)
    return code_block_list


def get_code_blocks(cfg_path):
    cfg = read_cfg(cfg_path)
    # draw CFG
    draw_graph(cfg, name='CFG')
    # Get predecessor node
    cfg = update_pre(cfg)
    # find all dominator of each node
    in_dict, out_dict = find_dominators(cfg)
    # generate dominator tree，The predecessor of each node is immediate dominator
    dom_tree = build_dom_tree(out_dict)
    # drwa dominator tree
    draw_graph(dom_tree, name='DOM_TREE')
    # Segment the dominator tree
    block_list = []
    split_dom_tree(block_list, dom_tree)
    # Obtain each code snippet from the segmentation result
    code_list = split_code(cfg, block_list)
    return code_list


if __name__ == '__main__':
    cfg_dir = 'E:\\dataset\\code comment\\new_dataset\\train\\final_cfgs\\'
    code_split_dir = 'E:\\dataset\\code comment\\new_dataset\\train\\code_split\\'
    cfg_file_list = os.listdir(cfg_dir)
    os.chdir(cfg_dir)
    for cfg_path in tqdm(cfg_file_list):
        try:
            code_list = get_code_blocks(cfg_path)
        except KeyError:
            print(cfg_path)
            continue
        idx = cfg_path.replace('.json', '')
        code_split_file = open(code_split_dir + idx + '.txt', 'w', encoding='utf-8')
        for code in code_list:
            code_split_file.write('<sep>' + code)
        code_split_file.close()

    # for test
    # code_list = get_code_blocks('E:\\dataset\\code comment\\cfg_40w\\test\\final_cfgs\\1000.json')
    # print(code_list)


