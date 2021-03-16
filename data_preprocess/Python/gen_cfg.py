#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import json
import os
from tqdm import tqdm


def data2java(dataset_file, java_dir):  # json formate
    file = open(dataset_file, encoding='utf-8')
    i = 0
    for line in file.readlines():
        json_txt = json.loads(line)
        code_id = ''.join(map(lambda x: str(x), [json_txt['id']]))
        # batch
        i += 1
        idx = i // 30000
        new_file_path = java_dir + 'java_files' + str(idx) + '/' + code_id + '.java'
        # new_file_path = java_dir + code_id + '.java'
        new_file = open(new_file_path, 'w', encoding='utf-8')
        raw_code = ''.join(map(lambda x: str(x), [json_txt['code']]))
        code = 'public class A' + code_id + ' { \r\n' + raw_code + '\r\n' + '}'
        new_file.write(code)
        new_file.close()
    file.close()


def data2java_2(dataset_file, java_dir):  # txt format dataset
    file = open(dataset_file, encoding='utf-8')
    i = 0
    for line in file.readlines():
        i += 1
        # idx = i // 30000
        code = line.replace('\n', '').strip()
        new_file_path = java_dir + str(i) + '.java'
        code = 'public class A' + str(i) + ' { \r\n' + code + '\r\n' + '}'
        new_file = open(new_file_path, 'w', encoding='utf-8')
        new_file.write(code)
        new_file.close()
    file.close()


def skip_empty_node(nodes):
    for _ in range((len(nodes)) // 4):
        for i in range(len(nodes)):
            next_nodes = nodes[i][6:]
            nodes[i] = nodes[i][:6]
            if len(next_nodes) and next_nodes[0] != '':
                # Handling the colon case
                for node in next_nodes:
                    if node.find(':') > 0:
                        temp = node
                        next_nodes.remove(node)
                        next_nodes += temp.split(':')
                # Skip the next empty node
                for node_id in next_nodes:
                    # The next node is an empty node
                    if node_id == '':
                        continue
                    if nodes[int(node_id) - 1][1] == '' and int(node_id) != len(nodes):
                        next_next_nodes = nodes[int(node_id) - 1][6:]
                        nodes[i] += next_next_nodes
                    else:
                        nodes[i] += [node_id]


def process_cfg(cfg_dir, java_dir, final_cfg_dir):   # read source code by line
    cfg_file_list = os.listdir(cfg_dir)
    os.chdir(cfg_dir)
    for item in tqdm(cfg_file_list):
        if item.find('.txt') > 0:
            cfg_file = open(item, encoding='utf-8')
            cfg_id = item.split('.')[0].replace('A', '')
            try:
                java_file = open(java_dir + cfg_id + '.java', encoding='utf-8')
            except FileNotFoundError:
                print(cfg_id)
                continue
            source_code = java_file.readlines()
            nodes = cfg_file.read().split(';')
            for i in range(len(nodes)):
                nodes[i] = nodes[i].split(',')
            skip_empty_node(nodes)
            new_nodes = ''
            for i in range(len(nodes)):
                if i != 0 and i != len(nodes) - 1 and nodes[i][2] == '':
                    continue
                node_attrs = nodes[i]
                code_list = '' if node_attrs[1] == '' else source_code[int(node_attrs[1]) - 1:int(node_attrs[3])]
                code = ""
                for j in range(len(code_list)):
                    code_list[j] = code_list[j].replace('\t', '').replace('\n', '').strip()
                    code += code_list[j]
                    if j != len(code_list) - 1:
                        code += "\n"
                next_nodes = list(set(node_attrs[6:]))
                # Generate a new CFG graph node
                if i != len(nodes):
                    new_nodes += json.dumps(
                        {"id": str(i + 1), "source_code": code.replace('\n', ''), "next_nodes": next_nodes}) + '\n'
            new_cfg_file = open(final_cfg_dir + cfg_id + '.json', 'w',
                                encoding='utf-8')
            new_cfg_file.write(new_nodes)
            new_cfg_file.close()
            java_file.close()
            cfg_file.close()

def process_cfg_py(cfg_dir, java_dir, final_cfg_dir):   # read source code by line
    cfg_file_list = os.listdir(cfg_dir)
    os.chdir(cfg_dir)
    for item in tqdm(cfg_file_list):
        if item.find('.txt') > 0:
            cfg_file = open(item, encoding='utf-8')
            cfg_id = item.split('.')[0].replace('A', '')
            # print(cfg_file)
            try:
                java_file = open(java_dir + cfg_id + '.py', encoding='utf-8')
                # print(java_dir + cfg_id + '.py')
                source_code = java_file.readlines()
                nodes = cfg_file.read().split(';')
                for i in range(len(nodes)):
                    nodes[i] = nodes[i].split(',')
                skip_empty_node(nodes)
                new_nodes = ''
                for i in range(len(nodes)):
                    if i != 0 and i != len(nodes) - 1 and nodes[i][2] == '':
                        continue
                    node_attrs = nodes[i]
                    code_list = '' if node_attrs[1] == '' else source_code[int(node_attrs[1]) - 1:int(node_attrs[3])]
                    code = ""
                    for j in range(len(code_list)):
                        # code_list[j] = code_list[j].replace('\t', '').replace('\n', '').strip()
                        code += code_list[j]
                        # if j != len(code_list) - 1:
                        #     code += "\n"
                    next_nodes = list(set(node_attrs[6:]))
                    # Generate a new CFG graph node
                    if i != len(nodes):
                        new_nodes += json.dumps(
                            {"id": str(i + 1), "source_code": code, "next_nodes": next_nodes}) + '\n'
                new_cfg_file = open(final_cfg_dir + cfg_id + '.json', 'w',
                                    encoding='utf-8')
                new_cfg_file.write(new_nodes)
                new_cfg_file.close()
                java_file.close()
                cfg_file.close()
            except FileNotFoundError:
                print(cfg_id)
                # continue


def process_cfg_2(cfg_dir, java_dir, final_cfg_dir):  # Get source code by column
    cfg_file_list = os.listdir(cfg_dir)
    os.chdir(cfg_dir)
    for item in tqdm(cfg_file_list):
        if item.find('.txt') > 0:
            cfg_file = open(item, encoding='utf-8')
            cfg_id = item.split('.')[0].replace('A', '')
            try:
                java_file = open(java_dir + cfg_id + '.java', encoding='utf-8')
            except FileNotFoundError:
                print(cfg_id)
                continue
            source_code = java_file.readlines()[2]
            nodes = cfg_file.read().split(';')
            for i in range(len(nodes)):
                nodes[i] = nodes[i].split(',')
            skip_empty_node(nodes)
            new_nodes = ''
            for i in range(len(nodes)):
                if i != 0 and i != len(nodes) - 1 and nodes[i][2] == '':
                    continue
                node_attrs = nodes[i]
                code = ''
                if node_attrs[2] != '':
                    code = source_code[int(node_attrs[2]):int(node_attrs[4]) + 1]
                next_nodes = list(set(node_attrs[6:]))
                # Generate a new CFG graph node
                new_nodes += json.dumps({"id": str(i + 1), "source_code": code, "next_nodes": next_nodes}) + '\n'
                # if i != len(nodes) - 1:
                #     new_nodes += ',\n'
                # else:
                #     new_nodes += '\n'
            new_cfg_file = open(final_cfg_dir + cfg_id + '.json', 'w',
                                encoding='utf-8')
            new_cfg_file.write(new_nodes)
            new_cfg_file.close()
            java_file.close()
            cfg_file.close()


def clean_dataset(dataset_file, new_file, cfg_dir):  # json data set cleaning
    cfg_file_list = os.listdir(cfg_dir)
    dataset = open(dataset_file, encoding='utf-8')
    new = open(new_file, 'w', encoding='utf-8')
    for line in tqdm(dataset.readlines()):
        json_txt = json.loads(line)
        code_id = ''.join(map(lambda x: str(x), [json_txt['id']])) + '.json'
        if code_id in cfg_file_list:
            new.write(line)
    dataset.close()
    new.close()


def clean_dataset_2(origin_dir, origin_clean_dir, code_split_dir, batch_type):
    code_split_list = os.listdir(code_split_dir)
    source_file = open(origin_dir + batch_type + '.source', encoding='utf-8')
    code_file = open(origin_dir + batch_type + '.token.code', encoding='utf-8')
    nl_file = open(origin_dir + batch_type + '.token.nl', encoding='utf-8')
    # sbt_file = open(origin_dir + batch_type + '.token.sbt', encoding='utf-8')
    # ast_file = open(origin_dir + batch_type + '_ast.json', encoding='utf-8')
    source_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.source', 'w', encoding='utf-8')
    code_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.token.code', 'w', encoding='utf-8')
    nl_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.token.nl', 'w', encoding='utf-8')
    # sbt_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.token.sbt', 'w', encoding='utf-8')
    # ast_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '_ast.json', 'w', encoding='utf-8')

    source_lines = source_file.readlines()
    code_lines = code_file.readlines()
    nl_lines = nl_file.readlines()
    # sbt_lines = sbt_file.readlines()
    # ast_lines = ast_file.readlines()
    for path in tqdm(code_split_list):
        idx = path.replace('.txt', '')
        source_line = source_lines[int(idx) - 1]
        source_clean_file.write(source_line)
        code_line = code_lines[int(idx) - 1]
        code_clean_file.write(code_line)
        nl_line = nl_lines[int(idx) - 1]
        nl_clean_file.write(nl_line)
        # sbt_line = sbt_lines[int(idx) - 1]
        # sbt_clean_file.write(sbt_line)
        # ast_line = ast_lines[int(idx) - 1]
        # ast_clean_file.write(ast_line)

    # ast_clean_file.close()
    # ast_file.close()
    # sbt_clean_file.close()
    # sbt_file.close()
    nl_clean_file.close()
    code_clean_file.close()
    source_clean_file.close()
    nl_file.close()
    code_file.close()
    source_file.close()


def add_head(code_split_dir, source_path, new_split_path):  # The method header is also added to the code snippet
    code_split_list = os.listdir(code_split_dir)
    # source_file = open(source_path, 'r', encoding='utf-8')
    # source_lines = source_file.readlines()
    # new_file = open(new_split_path, 'w', encoding='utf-8')
    os.chdir(code_split_dir)
    for f in tqdm(code_split_list):
        file = open(f, encoding='utf-8')
        idx = f.replace('.txt', '')
        source = open(source_path + idx + '.py', 'r', encoding='utf-8')
        code_split_file = open(new_split_path + idx + '.txt', 'w', encoding='utf-8')
        lines = file.readlines()
        if len(lines) == 0:
            line = ''
        else:
            line = "".join(lines)
        # s_line = source_lines[int(idx)-1]
        # s_line = s_line.split(":")[0] + ":"
        s_line = "".join(source.readlines()).split('):')[0] + "):"
        line = s_line + line + '\n'
        code_split_file.write(line)
        code_split_file.close()
        source.close()
    # source_file.close()


if __name__ == '__main__':
    process_cfg_py(cfg_dir='E:\\chao\\codeSum\\code_process\\train\\cfgs2\\',
                java_dir='E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train_py2\\',
                final_cfg_dir='E:\\chao\\codeSum\\code_process\\train\\final_cfgs\\')

    # clean_dataset_2(origin_dir='E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train\\origin\\',
    #                 origin_clean_dir='E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train\\origin_clean\\',
    #                 code_split_dir='E:\\chao\\codeSum\\code_process\\train\\code_split\\',
    #                 batch_type='train')
    add_head(code_split_dir='E:\\chao\\codeSum\\code_process\\train\\code_split\\',
             source_path='E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train_py2\\',
             new_split_path='E:\\chao\\codeSum\\code_process\\train\\code_split_addhead\\')
