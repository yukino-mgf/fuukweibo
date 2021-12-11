import json
import jieba
import simhash
import math
# import numpy as np

from features import same_uid_process
delimiter = "\t"

def weibo_predict(uid_features_str_file_path, weibo_predict_file_path,weibo_result_file_path):
    uid_features_str_file = open(uid_features_str_file_path,encoding= 'utf-8')   
    weibo_predict_file = open(weibo_predict_file_path,encoding= 'utf-8')   
    weibo_result_file = open(weibo_result_file_path,"w",encoding= 'utf-8')   
    table={}
    for line in open(uid_features_str_file_path,encoding= 'utf-8'):
        uid,values=line.split("\t")
        table[""+uid+""]=values
    for line in open(weibo_predict_file_path,encoding= 'utf-8'):
        uid=line.split(delimiter)[0]
        mid=line.split(delimiter)[1]
        if uid in table.keys():
            weibo_result_file.write(uid+"\t"+mid+"\t"+table[""+uid+""])
        else:
            weibo_result_file.write(uid+"\t"+mid+"\t"+str(0)+ ","+str(0)+ ","+str(0)+ "\n")
            
def cross_same_predict(same_uid_file_path, weibo_predict_file_path, weibo_result_file_path):
    with open(same_uid_file_path, 'r', encoding='utf-8') as fin:
        uid_dics = json.load(fin)
        for k,v in uid_dics.items():
            for x in v:
                # x.append(simhash.Simhash(x[3]))
                x.append(set(x[3]))
    with open(weibo_predict_file_path, 'r', encoding='utf-8') as prefin:
        lines = prefin.readlines()
    reslines = []
    for line in lines:
        resstr = ""
        uid, mid, time, content = line.split(delimiter)
        # content_set = set(list(content))
        
        content_cut = jieba.lcut(content)
        # content_hash = simhash.Simhash(content_cut)
        content_set = set(content_cut)
        
        if uid not in uid_dics:
            resstr = uid+"\t"+mid+"\t"+str(0)+ ","+str(0)+ ","+str(0)+ "\n"
        else:
            uid_list = uid_dics[uid]
            
            # sim_sum = 0
            # sim_list = []
            # for tp in uid_list:
                # keyset = set(tp[3])
                # inter = content_set.intersection(keyset)
                # sim = len(inter)/(len(content_set) + len(keyset) - len(inter))
                
            sim_sum = 0
            sim_list = []
            for tp in uid_list:
                # dis = content_hash.distance(tp[4])
                # sim = (64-dis)/64.0
                
                inter = content_set.intersection(tp[4])
                sim = len(inter)/(len(content_set) + len(tp[4]) - len(inter))
                sim = sim**10
                # sim = math.exp(7*sim)-1
                sim_list.append(sim)
                sim_sum += sim
            res = [0.0, 0.0, 0.0]
            for id,sim in enumerate(sim_list):
                res[0] += (int(uid_list[id][0]) * sim)
                res[1] += (int(uid_list[id][1]) * sim)
                res[2] += (int(uid_list[id][2]) * sim)
            final_res = [int(x/sim_sum) for x in res]
            
            # AR: have a try
            ar_tuple = (final_res[0], final_res[1], final_res[2], "", content_set)
            uid_dics[uid].append(ar_tuple)
            
            resstr = uid+"\t"+mid+"\t{},{},{}\n".format(final_res[0], final_res[1], final_res[2])
        reslines.append(resstr)
    with open(weibo_result_file_path, 'w', encoding='utf-8') as fout:
        fout.writelines(reslines)
