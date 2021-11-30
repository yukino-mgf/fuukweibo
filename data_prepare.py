import jieba
import pandas as pd
import numpy as np
import re
import zhon.hanzi
import string
import os
from tqdm import tqdm
# 测试集路劲，训练集路径，训练集数据属性，测试集数据属性
predict_data_path = './Weibo Data/Weibo Data/weibo_predict_data(new)/weibo_predict_data.txt'
train_data_path = './Weibo Data/Weibo Data/weibo_train_data(new)/weibo_train_data.txt'
names_train = ["uid", "mid", "time", "forward_count", "comment_count", "like_count", "content"]
names_predict = ["uid", "mid", "time", "content"]
word_weight_dict={}
# 计算content部分的权重时仅考虑是否包含出现频率在前valid_word_num的单词
valid_word_num = 30
"""
注意第40行~43行，请在其他文件中通过import调用指定位置的列表，用于检索所有uid,本文件中process_content返回的
data_all_train的第一列的数字对应的是train_uid中的uid索引
"""

#获取数据
def get_data(train=True):
    """
    :param train:是否为训练集
    :return: 读取到的数据集，dataframe
    """
    if train==True:
        data_path=train_data_path
        names = names_train
    else:
        data_path=predict_data_path
        names = names_predict
    # 读取txt文件中的数据
    data_pd = pd.read_csv(data_path,sep='\t',header=None,names=names)
    # print(len(data_pd['uid']))
    # print(len(set(data_pd['uid'])))
    return data_pd

# 分别为训练集，预测集，所有数据的uid
train_uid = sorted(list(set(get_data(True)['uid'])))
predict_uid = sorted(list(set(get_data(False)['uid'])))
all_uid = sorted(list(set(train_uid+predict_uid)))


# 对数据中的content部分进行简单的处理，计算每个单词的权重（根据权重占比前20的单词来计算）
def get_word_dict(data_pd):
    """
    :param data_pd:训练集数据
    :return: 20个权重最高的单词及其他们对应的权重，dict
    """
    data_select = data_pd[["forward_count", "comment_count", "like_count", "content"]]
    word_dict={}
    for i in tqdm(range(data_select.shape[0])):
    # for i in range(1000):
        # 去除中文标点符号并切割
        word_list = jieba.lcut(delete_punctuation(str(data_select['content'].iloc[i])))
        for word in word_list:
            if word in word_dict.keys():
                word_dict[word]=word_dict[word]+(data_select['forward_count'].iloc[i]+data_select['comment_count'].iloc[i]+data_select['like_count'].iloc[i])
            else:
                word_dict[word]=data_select['forward_count'].iloc[i]+data_select['comment_count'].iloc[i]+data_select['like_count'].iloc[i]
        if i%20000==0 or i==data_select.shape[0]-1:
            np.save('word_dict'+str(valid_word_num)+'.npy', word_dict)
    if os.path.exists('word_dict'+str(valid_word_num)+'.npy'):
        word_dict = np.load('word_dict'+str(valid_word_num)+'.npy',allow_pickle=True).item()
    my_dict = sorted(word_dict.items(), key=lambda word: word[1], reverse=True)
    # print(my_dict)
    # 只取权重前valid_word_num（默认为20）的词（至少2字）
    i=0
    word_num=0
    while word_num<=valid_word_num:
        key = my_dict[i][0]
        if len(key)>1:
            value = my_dict[i][1]
            word_weight_dict[key] = value/data_select.shape[0]
            word_num = word_num+1
        i=i+1
    # print(word_weight_dict)
    np.save('word_weight_dict'+str(valid_word_num)+'.npy', word_weight_dict)
    return word_weight_dict


# 根据事先计算好的word_dict更新数据内容
def process_content(word_weight_dict,data_pd):
    """
    :param word_weight_dict: 20个权重最高的单词及其他们对应的权重，dict
    :param data_pd: 待处理的数据集
    :return:content经过量化处理后的dataframe
    """
    data_new = data_pd.iloc[:,0:data_pd.shape[1]-1]
    # print(data_new.columns)
    weight_list = []
    if os.path.exists('weight_list.npy'):
        weight_list = list(np.load('weight_list.npy'))
    # print(weight_list)
    # 刷新对应语句中的所有权重
    for i in tqdm(range(len(weight_list),data_new.shape[0])):
    # for i in tqdm(range(100)):
        # 初始化权重计算
        weight=0
        # 去除中英文标点符号并切割
        word_list = jieba.lcut(delete_punctuation(str(data_pd['content'].iloc[i])))
        for word in word_list:
            if word in word_weight_dict.keys():
                weight = weight+word_weight_dict[word]
        weight_list.append(weight)
        # 储存列表，可以多次训练
        if i%2000==0:
            np.save('weight_list.npy',weight_list)
    data_content = pd.DataFrame(weight_list,columns=['content'])
    data_new = pd.concat([data_new,data_content],axis=1)
    os.remove('weight_list.npy')
    return data_new


# 删除对应语句中的中文字符、英文字符和一些自定义字符
def delete_punctuation(line):
    # 删除中文字符
    line = re.sub(r"[%s]+" % zhon.hanzi.punctuation,"",line)
    # 删除英文字符
    line = re.sub(r"[%s]+" % string.punctuation,"",line)
    # 手动添加一些字符
    filters = ['\.',' ','\t','\n','\x97', '\x96']
    line =re.sub("|".join(filters), "", line)
    return line


# 数据处理函数
def process_data(start_month=2, end_month=8):
    """
    :param start_month:训练集起始月份
    :param end_month: 训练集终止月份
    :return: data_all_train:训练集（numpy),每一行对应一个用户，纵列方向分别对应
    ["uid", "month_1", "forward_count_month_1", "comment_count_month_1", "like_count_month_1", "keyword_weight_month_1","month_2", "forward_count_month_2", "comment_count_month_2", "like_count_month_2", "keyword_weight_month_2"……]
    用户id索引 月份_1      当月平均转发数             当月平均评论数             当月平均点赞数         当月内容平均总权重
    data_val_pd:验证集（DataFrame）,对应训练集所处月份后一个月的数据，可以通过前几个的数据以及该月数据进行训练和预测
    data_test_pd：测试集（DataFrame）
    """
    # 如果已经创建完毕则直接读取
    dataset_path = './dataset/dataset_'+str(start_month)+'_'+str(end_month)+'_'+str(valid_word_num)
    if os.path.exists(dataset_path+'/data_val_pd.csv'):
        data_all_train=np.load(dataset_path+'/data_all_train.npy')
        data_val_pd=pd.read_csv(dataset_path+'/data_val_pd.csv')
        data_test_pd=pd.read_csv('data_test'+str(valid_word_num)+'.csv')
        return data_all_train, data_val_pd, data_test_pd
    # 训练集数据
    data_train_pd = get_data(True)
    # 充当测试集的数据
    data_test_pd = get_data(False)
    if os.path.exists('word_weight_dict'+str(valid_word_num)+'.npy'):
        word_weight_dict = np.load('word_weight_dict'+str(valid_word_num)+'.npy',allow_pickle=True).item()
    else:
        word_weight_dict = get_word_dict(data_train_pd)
    print(word_weight_dict)
    # 对所有训练/测试数据的content进行处理
    if os.path.exists('data_train_1_'+str(valid_word_num)+'.csv'):
        data_train_pd_1 = pd.read_csv('data_train_1_'+str(valid_word_num)+'.csv')
        data_train_pd_2 = pd.read_csv('data_train_2_' + str(valid_word_num) + '.csv')
        data_train_pd = pd.concat([data_train_pd_1,data_train_pd_2],axis=0)
    else:
        data_train_pd = process_content(word_weight_dict, data_train_pd)
        data_train_pd_1 = data_train_pd[0:500000]
        data_train_pd_2 = data_train_pd[500000:]
        data_train_pd_1.to_csv('data_train_1_'+str(valid_word_num)+'.csv', index=0)  # 不保存行索引
        data_train_pd_2.to_csv('data_train_2_' + str(valid_word_num) + '.csv', index=0)  # 不保存行索引
    if os.path.exists('data_test'+str(valid_word_num)+'.csv'):
        data_test_pd = pd.read_csv('data_test'+str(valid_word_num)+'.csv')
    else:
        data_test_pd = process_content(word_weight_dict,data_test_pd)
        data_test_pd.to_csv('data_test'+str(valid_word_num)+'.csv',index=0)

    # 根据时间范围对数据进行过滤,取训练集所处时间范围后一个的时间作为验证集
    train_start_time = '2015-0'+str(start_month)+'-01 00:00:00'
    train_end_time = '2015-0'+str(end_month)+'-01 00:00:00'
    val_end_time = '2015-0'+str(end_month+1)+'-01 00:00:00'
    data_train_pd_new = data_train_pd[(data_train_pd['time']>=train_start_time) & (data_train_pd['time']<train_end_time)]
    # print(data_pd.head()['time'])
    # 充当验证集的数据
    data_val_pd = data_train_pd[(data_train_pd['time']>=train_end_time) & (data_train_pd['time']<val_end_time)]

    """
    用于储存数据，每一行对应一个用户，纵列方向分别对应
    ["uid", "month", "forward_count_month", "comment_count_month", "like_count_month", "keyword_weight_month"],
    其中month的值从start_month到end_month-1
    """
    data_all_train = np.zeros((len(train_uid),1+(end_month-start_month)*5))
    if os.path.exists(dataset_path)!=True:
        os.makedirs(dataset_path)
    for i in tqdm(range(len(train_uid))):
        # 每一行对应train_uid索引
        data_all_train[i][0]=i
        uid = train_uid[i]
        # 筛选出对应uid的用户的数据
        data_train_uid = data_train_pd_new[(data_train_pd_new['uid']==uid)]
        # 如果dataframe非空，读取数据
        if data_train_uid.empty==False:
            for j in range(end_month-start_month):
                cur_start_month = start_month+j
                cur_end_month = cur_start_month+1
                # 依次筛选每个月的数据
                cur_start_time = '2015-0' + str(cur_start_month) + '-01 00:00:00'
                cur_end_time = '2015-0' + str(cur_end_month) + '-01 00:00:00'
                # 填写训练集内容，包含月份、当月平均转发数、当月平均评论数、当月平均点赞数、当月平均内容中关键词权重和
                data_all_train[i][1+5*j]=cur_start_month
                data = data_train_uid[(data_train_uid['time']>=cur_start_time) & (data_train_uid['time']<cur_end_time)]
                if data.empty==False:
                    data_np = np.array(data)
                    # 当月平均转发数
                    forward_count_month=0
                    # 当月平均评论数
                    comment_count_month=0
                    # 当月平均点赞数
                    like_count_month = 0
                    # 当月内容中平均关键词权重和
                    keyword_weight_month =0
                    for k in range(data.shape[0]):
                        forward_count_month = int(data['forward_count'].iloc[k])+forward_count_month
                        comment_count_month = int(data['comment_count'].iloc[k])+comment_count_month
                        like_count_month = int(data['like_count'].iloc[k])+like_count_month
                        keyword_weight_month = float(data['content'].iloc[k])+keyword_weight_month
                        # forward_count_month = int(data_np[k,3])+forward_count_month
                        # comment_count_month = int(data_np[k,4])+comment_count_month
                        # like_count_month = int(data_np[k,5])+like_count_month
                        # keyword_weight_month = float(data_np[k,6])+keyword_weight_month
                    forward_count_month = forward_count_month/data.shape[0]
                    comment_count_month = comment_count_month/data.shape[0]
                    like_count_month = like_count_month/data.shape[0]
                    keyword_weight_month =  keyword_weight_month/data.shape[0]
                    # 填写内容
                    data_all_train[i][2+5*j] = forward_count_month
                    data_all_train[i][3+5*j] = comment_count_month
                    data_all_train[i][4+5*j] = like_count_month
                    data_all_train[i][5+5*j] = keyword_weight_month
        if i%1000==0:
            np.save(dataset_path+'/data_all_train.npy',data_all_train)
    data_val_pd.to_csv(dataset_path+'/data_val_pd.csv', index=0)  # 不保存行索引
    return data_all_train,data_val_pd,data_test_pd


if __name__ == '__main__':
    # 一些测试代码，不用管，可直接删除
    # train_uid = sorted(list(set(get_data(True)['uid'])))
    # predict_uid = sorted(list(set(get_data(False)['uid'])))
    # all_uid = sorted(list(set(train_uid+predict_uid)))
    # print(len(train_uid))
    # print(len(predict_uid))
    # print(len(all_uid))
    # process_data()
    # 测试jieba库
    # line = "我是[中，国人。？！ afad "
    # line = delete_punctuation(line)
    # print(jieba.lcut(line))
    # df = pd.DataFrame([[1, 2, 3, 4], [100, 200, 300, 400], [1000, 2000, 3000, 4000]],
    #                   columns=['a', 'b', 'c', 'd'])
    # print(df)
    # e=[5,500,5000]
    # df_new = pd.DataFrame(e,columns=['e'])
    # print(df_new)
    # df_new1 = pd.concat([df,df_new],axis=1)
    # print(df_new1)
    # df = get_data()
    # df.info()
    # 以上内容可直接删除


    train_data,val_data_pd,test_data_pd = process_data()
    print('读取完毕')
    pass