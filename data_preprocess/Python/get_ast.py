#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import javalang
import json
from tqdm import tqdm
import collections
from graphviz import Digraph
import sys
import lib
import argparse
import codecs
import ast, asttokens
import optparse
import sys
import os
import copy

NODE_FIX = '1*NODEFIX'#'1*NODEFIX'

def python2tree(line):
    atok = asttokens.ASTTokens(line, parse=True)
    return atok, atok.tree

def traverse_python_tree(atok, root):
    iter_children = asttokens.util.iter_children_func(root)
    node_json = {}
    current_global = {}
    current_idx, global_idx = 1, 1
    for node in asttokens.util.walk(root):
        if not next(iter_children(node), None) is None:
            child_num = 0
            for child in iter_children(node):
                child_num += 1
            global_idx = global_idx + child_num
            current_global[current_idx] = global_idx
        current_idx += 1
    # print current_global
    current_idx = 1
    for node in asttokens.util.walk(root):
        # print current_idx
        # idx_upper = current_idx
        node_json["%s%s" % (NODE_FIX, current_idx)] = {"node": type(node).__name__, "children": [],
                                                                 "parent": None}
        # idx_upper = len(node_json)
        if not next(iter_children(node), None) is None:
            child_idx = 0
            for child in iter_children(node):
                child_idx += 1
                node_json["%s%s" % (NODE_FIX, current_idx)]['children'].insert(0, "%s%s" % (
                NODE_FIX, current_global[current_idx] - child_idx + 1))
        else:  # leaf node
            node_json["%s%s" % (NODE_FIX, current_idx)]['children'].append(atok.get_text(node))

        current_idx += 1

    # update_parent
    for k, node in node_json.items():
        children = [c for c in node['children'] if c.startswith(NODE_FIX)]
        if len(children):
            for c in children:
                node_json[c]['parent'] = k

    return node_json

def process_source(file_name, save_file):  # referred from EMSE-DeepCom
    with open(file_name, 'r', encoding='utf-8') as source:
        lines = source.readlines()
    with open(save_file, 'w+', encoding='utf-8') as save:
        for line in lines:
            code = line.strip()
            tokens = list(javalang.tokenizer.tokenize(code))
            tks = []
            for tk in tokens:
                if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
                    tks.append('STR_')
                elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
                    tks.append('NUM_')
                elif tk.__class__.__name__ == 'Boolean':
                    tks.append('BOOL_')
                else:
                    tks.append(tk.value)
            save.write(" ".join(tks) + '\n')


def get_ast(file_name, w):  # referred from EMSE-DeepCom
    with open(file_name, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    with open(w, 'w+', encoding='utf-8') as wf:
        ign_cnt = 0
        for line in tqdm(lines):
            code = line.strip()
            tokens = javalang.tokenizer.tokenize(code)
            token_list = list(javalang.tokenizer.tokenize(code))
            length = len(token_list)
            parser = javalang.parser.Parser(tokens)
            try:
                tree = parser.parse_member_declaration()
            except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
                print('Error')
                continue
            flatten = []
            for path, node in tree:
                flatten.append({'path': path, 'node': node})

            ign = False
            outputs = []
            stop = False
            for i, Node in enumerate(flatten):
                d = collections.OrderedDict()
                path = Node['path']
                node = Node['node']
                children = []
                for child in node.children:
                    child_path = None
                    if isinstance(child, javalang.ast.Node):
                        child_path = path + tuple((node,))
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                                children.append(j)
                    if isinstance(child, list) and child:
                        child_path = path + (node, child)
                        for j in range(i + 1, len(flatten)):
                            if child_path == flatten[j]['path']:
                                children.append(j)
                d["id"] = i
                d["type"] = str(node.__class__.__name__)
                if children:
                    d["children"] = children
                value = None
                if hasattr(node, 'name'):
                    value = node.name
                elif hasattr(node, 'value'):
                    value = node.value
                elif hasattr(node, 'position') and node.position:
                    for i, token in enumerate(token_list):
                        if node.position == token.position:
                            pos = i + 1
                            value = str(token.value)
                            while (pos < length and token_list[pos].value == '.'):
                                value = value + '.' + token_list[pos + 1].value
                                pos += 2
                            break
                elif type(node) is javalang.tree.This \
                        or type(node) is javalang.tree.ExplicitConstructorInvocation:
                    value = 'this'
                elif type(node) is javalang.tree.BreakStatement:
                    value = 'break'
                elif type(node) is javalang.tree.ContinueStatement:
                    value = 'continue'
                elif type(node) is javalang.tree.TypeArgument:
                    value = str(node.pattern_type)
                elif type(node) is javalang.tree.SuperMethodInvocation \
                        or type(node) is javalang.tree.SuperMemberReference:
                    value = 'super.' + str(node.member)
                elif type(node) is javalang.tree.Statement \
                        or type(node) is javalang.tree.BlockStatement \
                        or type(node) is javalang.tree.ForControl \
                        or type(node) is javalang.tree.ArrayInitializer \
                        or type(node) is javalang.tree.SwitchStatementCase:
                    value = 'None'
                elif type(node) is javalang.tree.VoidClassReference:
                    value = 'void.class'
                elif type(node) is javalang.tree.SuperConstructorInvocation:
                    value = 'super'

                if value is not None and type(value) is type('str'):
                    d['value'] = value
                if not children and not value:
                    # print('Leaf has no value!')
                    print(type(node))
                    print(code)
                    ign = True
                    ign_cnt += 1
                    # break
                outputs.append(d)
            if not ign:
                wf.write(json.dumps(outputs))
                wf.write('\n')
    print(ign_cnt)


def build_ast(line):
    code = line.strip()
    tokens = list(javalang.tokenizer.tokenize(code))
    tks = []
    for tk in tokens:
        if tk.__class__.__name__ == 'String' or tk.__class__.__name__ == 'Character':
            tks.append('STR_')
        elif 'Integer' in tk.__class__.__name__ or 'FloatingPoint' in tk.__class__.__name__:
            tks.append('NUM_')
        elif tk.__class__.__name__ == 'Boolean':
            tks.append('BOOL_')
        else:
            tks.append(tk.value)
    line = " ".join(tks)

    code = line.strip()
    tokens = javalang.tokenizer.tokenize(code)
    token_list = list(javalang.tokenizer.tokenize(code))
    length = len(token_list)
    parser = javalang.parser.Parser(tokens)
    try:
        tree = parser.parse_member_declaration()
    except (javalang.parser.JavaSyntaxError, IndexError, StopIteration, TypeError):
        print('Error: ' + line)
        return None
    flatten = []
    for path, node in tree:
        flatten.append({'path': path, 'node': node})
    outputs = []
    for i, Node in enumerate(flatten):
        d = collections.OrderedDict()
        path = Node['path']
        node = Node['node']
        children = []
        for child in node.children:
            if isinstance(child, javalang.ast.Node):
                child_path = path + tuple((node,))
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path'] and child == flatten[j]['node']:
                        children.append(j)
            if isinstance(child, list) and child:
                child_path = path + (node, child)
                for j in range(i + 1, len(flatten)):
                    if child_path == flatten[j]['path']:
                        children.append(j)
        d["id"] = i
        d["type"] = str(node.__class__.__name__)
        if children:
            d["children"] = children
        value = None
        if hasattr(node, 'name'):
            value = node.name
        elif hasattr(node, 'value'):
            value = node.value
        elif hasattr(node, 'position') and node.position:
            for i, token in enumerate(token_list):
                if node.position == token.position:
                    pos = i + 1
                    value = str(token.value)
                    while (pos < length and token_list[pos].value == '.'):
                        value = value + '.' + token_list[pos + 1].value
                        pos += 2
                    break
        elif type(node) is javalang.tree.This \
                or type(node) is javalang.tree.ExplicitConstructorInvocation:
            value = 'this'
        elif type(node) is javalang.tree.BreakStatement:
            value = 'break'
        elif type(node) is javalang.tree.ContinueStatement:
            value = 'continue'
        elif type(node) is javalang.tree.TypeArgument:
            value = str(node.pattern_type)
        elif type(node) is javalang.tree.SuperMethodInvocation \
                or type(node) is javalang.tree.SuperMemberReference:
            value = 'super.' + str(node.member)
        elif type(node) is javalang.tree.Statement \
                or type(node) is javalang.tree.BlockStatement \
                or type(node) is javalang.tree.ForControl \
                or type(node) is javalang.tree.ArrayInitializer \
                or type(node) is javalang.tree.SwitchStatementCase:
            value = 'None'
        elif type(node) is javalang.tree.VoidClassReference:
            value = 'void.class'
        elif type(node) is javalang.tree.SuperConstructorInvocation:
            value = 'super'

        if value is not None and type(value) is type('str'):
            d['value'] = value
        if not children and not value:
            # print('Leaf has no value!')
            print(type(node))
            print(code)
            # break
        outputs.append(d)
    return outputs


def gen_all_ast_py(code_split_dir, ast_file_path):

    code_split_list = os.listdir(code_split_dir)
    os.chdir(code_split_dir)
    syntax_error_count = 0
    empty_count = 0
    # ast_file = open(ast_file_path, 'w', encoding='utf-8')

    for f in tqdm(code_split_list):
        try:
            ast_list = []
            file = open(f, encoding='utf-8')
            idx = f.replace('.txt', '')
            # print('当前id:', idx)
            lines = file.readlines()
            if len(lines) == 0:
                line = ''
            else:
                line = "".join(lines)
            code_list = line.split('<sep>')
            head = code_list[0]
            if len(code_list) > 1:
                code_list = code_list[1:]
                # Add method headers to each code segment to generate AST
                for code in code_list:
                    if code.find('except') >= 0:
                        continue
                    code = head + "\n" + code
                    # print(code)
                    code = clean_code_py(code)
                    # print('code2:' + code)
                    atok, tree = python2tree(code)
                    tree_json = traverse_python_tree(atok, tree)
                    # print(tree_json)
                    if tree_json is not None:
                        ast_list.append(tree_json)
                    else:
                        syntax_error_count += 1
            if len(ast_list) == 0:
                empty_count += 1
                continue
            ast_file = open(ast_file_path + idx + '.json', 'w', encoding='utf-8')
            ast_file.write(json.dumps(ast_list))
            ast_file.close()

            file.close()
        except Exception:
            print(f)
            pass
        continue
        print('syntax_error_count: ' + str(syntax_error_count))
        print('empty_count: ' + str(empty_count))




def clean_code(code):
    code = code.replace('\n', '')
    code = code.replace('default:', '')
    if code.find('throw') >= 0:
        return ''
    while code.find('case') >= 0:
        # code = code[code.rfind(':') + 1:]
        code = code[code.find(':') + 1:]
        if code.find(':') < 0:
            break
    if code.find('switch (') == 0 and code.find('{'):
        code = code[code.find('{')+1:code.find('}')]
    if code.find('for') >= 0 or code.find('while') >= 0:
        code += '{}'
    return code

def get_real_arr(arr):
    """
    Return arr after removing all null values
    """
    arr_copy = copy.deepcopy(arr)
    arr_copy = list(filter(None, arr_copy))
    while '' in arr_copy:
        arr_copy.remove('')
    return arr_copy


def clean_code_py(code):
    # code = code.replace('\n', '')
    # code = code.replace('default:', '')
    # if code.find('throw') >= 0:
    #     return ''
    # while code.find('case') >= 0:
    #     # code = code[code.rfind(':') + 1:]
    #     code = code[code.find(':') + 1:]
    #     if code.find(':') < 0:
    #         break
    # if code.find('switch (') == 0 and code.find('{'):
    #     code = code[code.find('{')+1:code.find('}')]
    num = 0
    # print('code:' + code)
    code_tmp2 = code.strip()
    tmp = code.split(':')[0] + ':'
    code_tmp = code.replace(tmp, '', 1)
    code_tmp = code_tmp.split('\n')
    code_tmp3 = get_real_arr(code_tmp)
    for c in code_tmp3:
        if c.find('def') >= 0:
            if (c.strip().index('def') == 0) or (c.strip().index('def') == 6):
                code = code.replace(c, '')

    if (code.find('for ') >= 0 and code_tmp2.find(':', code_tmp2.index('for ')) != -1) or (code.find('while ') >= 0 and code_tmp2.find(':', code_tmp2.index('while ')) != -1):

        # Determine if for or while is the last line
        i = 0
        for j in range(len(code_tmp3)):
            i += 1
            if code_tmp3[j].find('for ') >= 0:
                if code_tmp3[j].strip().index('for ') == 0 and j == len(code_tmp3)-1:
                    code_tmp = code_tmp3[j]
                    num = i
                elif code_tmp3[j].strip().index('for ') == 0 and j == len(code_tmp3)-2 and code_tmp3[j+1].find('if') <0 and code_tmp3[j+1].find(':') >= 0 :
                    code_tmp = code_tmp3[j]
                    num = i+1
            if code_tmp3[j].find('while ') >= 0:
                if code_tmp3[j].strip().index('while ') == 0 and j == len(code_tmp3)-1:
                    code_tmp = code_tmp3[j]
                    num = i
                elif code_tmp3[j].strip().index('while ') == 0 and j == len(code_tmp3)-2 and code_tmp3[j+1].find('if') <0 and code_tmp3[j+1].find(':') >= 0:
                    code_tmp = code_tmp3[j]
                    num = i+1

        # print(num)
        # print('after:\n', code)
        # print(len(code_tmp3))
        if num == len(code_tmp3):
            num = 0
            while code_tmp.find('    ') >= 0:
                code_tmp = code_tmp.replace('    ', '', 1)
                num += 1
            code += "    " * (num+1) + 'continue'

    if code_tmp2.find('if') >= 0 and code_tmp2.find(':', code_tmp2.index('if')) != -1:
        # if code_tmp2.index(':', code_tmp2.index('if')) == len(code_tmp2)-1:

        # Determine if is the last line
        i = 0
        for c in code_tmp3:
            i += 1
            if c.find('if') >= 0:
                if c.strip().index('if') == 0:
                    code_tmp = c
                    num = i

        # print('num:', num)
        # print(len(code_tmp3))
        if num == len(code_tmp3):
            num = 0
            while code_tmp.find('    ') >= 0:
                code_tmp = code_tmp.replace('    ', '', 1)
                num += 1
            code += "    " * (num+1) + 'print("")'

    if code_tmp2.find('elif') >= 0 and code_tmp2.find(':', code_tmp2.index('elif')) != -1:
        # if code_tmp2.index(':', code_tmp2.index('elif')) == len(code_tmp2)-1:

        i = 0
        for c in code_tmp3:
            i += 1
            if c.find('elif') >= 0:
                if c.strip().index('elif') == 0:
                    code_tmp = c
                    num = i

        if num == len(code_tmp3):
            num = 0
            while code_tmp.find('    ') >= 0:
                code_tmp = code_tmp.replace('    ', '', 1)
                num += 1

            code += "    " * (num+1) + 'print("")'
        code = code.replace('elif', 'if')
    # print('code3:' + code)

    if code_tmp2.find('else') >= 0 and code_tmp2.find(':', code_tmp2.index('else')) != -1:
        # for c in code_tmp3:
        #     if c.find('else') >= 0:
        #         if c.strip().index('else') == 0:
        #             code_tmp = c
        #
        # if num == len(code_tmp3):
        #     num = 0
        #     while code_tmp.find('    ') >= 0:
        #         code_tmp = code_tmp.replace('    ', '', 1)
        #         num += 1
        #
        #     code += "    " * num + 'print("")'
        code = code.replace('else:', '')
    # print('code3:' + code)

    return code


def clean_head(head):
    first = head[0:head.find('(')-1]
    second = head[head.find('(')-1:]
    words = first.split(' ')
    name = words[-1]
    head = 'void ' + name + second
    return head


def read_ast(ast_file_path):
    ast_file = open(ast_file_path, encoding='utf-8')
    for line in ast_file.readlines():
        ast_list = json.loads(line)
        for ast in ast_list:
            draw_ast(ast)


def draw_ast(ast):
    g = Digraph('AST')
    for node in ast:
        g.node(name=str(node['id']), label=node['type'])
        if 'children' in node.keys():
            for c in node['children']:
                g.edge(str(node['id']), str(c))
    g.view(directory='img/')


def clean_dataset_2(origin_dir, origin_clean_dir, code_split_dir, batch_type):
    code_split_list = os.listdir(code_split_dir)
    print(code_split_list)
    source_file = open(origin_dir + batch_type + '.source', encoding='utf-8')
    code_file = open(origin_dir + batch_type + '.token.code', encoding='utf-8')
    nl_file = open(origin_dir + batch_type + '.token.nl', encoding='utf-8')
    # sbt_file = open(origin_dir + batch_type + '.token.sbt', encoding='utf-8')
    # ast_file = open(origin_dir + batch_type + '_ast.json', encoding='utf-8')
    source_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.source', 'w', encoding='utf-8')
    code_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.token.code', 'w', encoding='utf-8')
    nl_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.token.nl', 'w', encoding='utf-8')
    split_ast_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.split.ast', 'w', encoding='utf-8')
    # sbt_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '.token.sbt', 'w', encoding='utf-8')
    # ast_clean_file = open(origin_clean_dir + 'clean_' + batch_type + '_ast.json', 'w', encoding='utf-8')

    source_lines = source_file.readlines()
    code_lines = code_file.readlines()
    nl_lines = nl_file.readlines()
    # sbt_lines = sbt_file.readlines()
    # ast_lines = ast_file.readlines()
    for path in tqdm(code_split_list):
        print(path)
        idx = path.replace('.json', '')
        source_line = source_lines[int(idx)]
        source_clean_file.write(source_line)
        code_line = code_lines[int(idx)]
        code_clean_file.write(code_line)
        nl_line = nl_lines[int(idx)]
        nl_clean_file.write(nl_line)
        split_ast = open(code_split_dir+path, encoding='utf-8')
        split_ast_json = split_ast.readlines()[0].strip()
        split_ast_clean_file.write(split_ast_json + '\n')
        split_ast.close()
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
    split_ast_clean_file.close()

if __name__ == '__main__':
    # code for test
    # process_source('ast_files/8.java', 'ast_files/source.code')
    # get_ast('ast_files/source.code', 'ast_files/8.json')


    gen_all_ast_py(code_split_dir='E:\\chao\\codeSum\\code_process\\train\\code_split_addhead2\\',
                ast_file_path='E:\\chao\\codeSum\\code_process\\train\\split_ast\\')

    # clean_dataset_2(origin_dir='E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train\\origin\\',
    #                 origin_clean_dir='E:\\chao\\codeSum\\ASE2020\\transformer-ast\\python_datautils\\python_datautils\\data\\train\\final_clean\\',
    #                 code_split_dir='E:\\chao\\codeSum\\code_process\\train\\split_ast\\',
    #                 batch_type='train')