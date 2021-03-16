#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: zzz_jq
# Time: 2020/3/15 20:06

import javalang
import json
from tqdm import tqdm
import collections
from graphviz import Digraph
import sys


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


def gen_all_ast(code_split_path, ast_file_path):
    code_file = open(code_split_path, encoding='utf-8')
    ast_file = open(ast_file_path, 'w', encoding='utf-8')
    syntax_error_count = 0
    empty_count = 0
    for line in tqdm(code_file.readlines()):
        ast_list = []
        code_list = line.split('<sep>')
        head = clean_head(code_list[0])
        head_ast = build_ast(head)
        if head_ast is not None:
            ast_list.append(head_ast)
        else:
            syntax_error_count += 1
        if len(code_list) > 1:
            code_list = code_list[1:]
            # Add method headers to each code segment to generate AST
            for code in code_list:
                if code.replace('\n', '').strip() != '':
                    code = clean_code(code)
                    if len(code) <= 0:continue
                    code = head.replace(';', '') + '{' + code + '}'
                    ast = build_ast(code)
                    if ast is not None:
                        ast_list.append(ast)
                    else:
                        syntax_error_count += 1
        if len(ast_list) == 0:
            empty_count += 1
        ast_file.write(json.dumps(ast_list) + '\n')
    print('syntax_error_count: ' + str(syntax_error_count))
    print('empty_count: ' + str(empty_count))
    ast_file.close()
    code_file.close()


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


if __name__ == '__main__':
    # code for test
    # process_source('ast_files/8.java', 'ast_files/source.code')
    # get_ast('ast_files/source.code', 'ast_files/8.json')

    gen_all_ast(code_split_path='Java/split_train.txt',
                ast_file_path='Java/split_train_ast.json')
    # read_ast('ast_files/1000_.json')

