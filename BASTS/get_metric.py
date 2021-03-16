#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# Author: Zhichao Ouyang
# Time: 2021/3/11 16:19

import os
import re
import nltk
from nltk.translate.bleu_score import *
import subprocess
import xlwt
from tqdm import tqdm
from get_rouge import *


def nltk_sentence_bleu(ref_path, hyp_path):
    print(type(ref_path))
    cc = SmoothingFunction()
    with open(ref_path, 'r') as r:
        lines = r.readlines()
        ref = []
        for line in lines:
            ref.append(line.strip().split(' '))

    with open(hyp_path, 'r') as g:
        lines = g.readlines()
        gen = []
        for line in lines:
            gen.append(line.strip().split(' '))

    all_score = 0.0
    count = 0
    for r, g in zip(ref, gen):
        if len(g) < 4:
            continue
        else:

            score = nltk.translate.bleu([r], g, smoothing_function=cc.method4)
            all_score += score
            count += 1
    # print("average score: %f/%d %f" % (all_score, count, all_score / count))
    return (all_score / count) * 100

def get_c_bleu(ref, output):
    get_bleu_score = "perl multi-bleu.perl {} < {} > {}".format(ref, output, "temp2")
    os.system(get_bleu_score)
    bleu_score_report = open("temp2", "r", encoding='utf-8').read()
    bleu_score_report = re.findall("BLEU = ([^,]+)", bleu_score_report)[0]
    res = bleu_score_report
    os.remove("temp2")
    return float(res)

def get_metor(ref, output):
    get_metor_score = "java -Xmx2G -jar meteor-1.5/meteor-1.5.jar {} {} -l en -norm ".format(output, ref)
    p = subprocess.Popen(get_metor_score, shell=True, stdout=subprocess.PIPE)
    out, err = p.communicate()
    out = str(out, encoding='utf-8')
    lines = []
    for line in out.splitlines():
        lines.append(line)
    res = lines[-1]
    res = float(res.replace("Final score:", "").strip()) * 100
    return res

if __name__ == "__main__":
    dataName = 'Java'  # Default dataset is java
    # dataName = 'Python'
    ref = 'code_sum_dataset/' + dataName + 'test.token.nl'
    workbook = xlwt.Workbook(encoding='utf-8')
    sheet1 = workbook.add_sheet("test", cell_overwrite_ok=True)
    sheet1.write(0, 0, "epoch") 
    sheet1.write(0, 1, "S-BLEU")
    sheet1.write(0, 2, "C-BLEU")  
    sheet1.write(0, 3, "METOR")
    sheet1.write(0, 4, "ROUGE-1") 
    sheet1.write(0, 5, "ROUGE-2")
    sheet1.write(0, 6, "ROUGE-L")
    flag = 1
    for epoch in tqdm(range(0, 101, 5)):   # Output results every 5 epochs
        print(epoch)
        output = './BASTS_model_' + dataName + '/BASTS_output_epoch' + str(epoch) + '.txt'
        c_bleu = get_c_bleu(ref, output)
        s_bleu = nltk_sentence_bleu(ref, output)
        metor = get_metor(ref, output)
        sheet1.write(flag, 0, epoch)
        sheet1.write(flag, 1, s_bleu)
        sheet1.write(flag, 2, c_bleu)
        sheet1.write(flag, 3, metor)
        with open(ref, 'r') as r:
            lines = r.readlines()
            ref_list = []
            for line in lines:
                ref_list.append(line.strip())

        with open(output, 'r') as g:
            lines = g.readlines()
            gen = []
            for line in lines:
                gen.append(line.strip())

        avg_rouge_1, avg_rouge_2, avg_rouge_l = avg_rouge(gen, ref_list)
        sheet1.write(flag, 4, float(avg_rouge_1[2]) * 100)
        sheet1.write(flag, 5, float(avg_rouge_2[2]) * 100)
        sheet1.write(flag, 6, float(avg_rouge_l[2]) * 100)
        flag += 1
    workbook.save('BASTS_epoch1-100.xls')
    print("save!!!")



