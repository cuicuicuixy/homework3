import math
import jieba
import os  # 用于处理文件路径
import random
import numpy as np
from gensim import corpora, models
import gensim
import matplotlib.pyplot as plt    
def content_deal(content):  # 语料预处理，进行断句，去除一些广告和无意义内容
    ad = '本书来自www.cr173.com免费txt小说下载站\n更多更新免费电子书请关注www.cr173.com'
    content = content.replace(ad, '')   
    return content        
def read_novel(path):
    file_list = os.listdir(path)  
    data_list = []
    data_label = []
    test_list = []
    test_label = []

    for file in file_list:
        novel_path = "E:/DLNL/--main/NLP-3/jinyong" + '/' + file 
        with open (novel_path,'r',encoding='ANSI')as f:
            content = f.read()
            word_list0 = content_deal(content)
            
            #大于500词的段落
            for para in word_list0.split('\n'):
                if len(para)<500:
                    continue
                data_list.append(para)
                data_label.append(file)
                f.close()
    #随机200段落
    random_way = random.sample(range(len(data_list)), 200)
    test_list.extend([data_list[i] for i in random_way])
    test_label.extend([data_label[i] for i in random_way])
    #返回所有段落及其标签，选取的200段落及其标签
    return data_list,data_label,test_list,test_label      




if __name__ == '__main__':
    path = r"E:\DLNL\--main\NLP-3\jinyong"
    stop_word_file   = r"E:\DLNL\--main\NLP-3\cn_stopwords.txt"
    punctuation_file = r"E:\DLNL\--main\NLP-3\cn_punctuation.txt"
    ll = r"E:/DLNL/--main/NLP-3/jinyong"
    #读取停词列表
    stop_word_list = [] 
    with open(stop_word_file,'r',encoding='utf-8') as f:
        for line in f:
            stop_word_list.append(line.strip())
    stop_word_list.extend("\u3000")
    stop_word_list.extend(['～',' ','没','听','一声','道', '见', '中', '便', '说', '一个','说道'])
    #读取段落
    [data_list,data_label,test_list,test_label] = read_novel(ll)   
    #分词
    #词模式
    fenci_word= []
    fenci_word_label=[]
    fenci_char = []
    fenci_char_label = []
    
    for index,text in enumerate(test_list):
        fenci = [word for word in jieba.lcut(sentence=text) if word not in stop_word_list]
        fenci_word.append(fenci)
        fenci_word_label.append(test_label[index])
    #字模式
        t = []
        for word1 in fenci:
            t.extend([char for char in word1])
        fenci_char.append(t)
        fenci_char_label.append(test_label[index])


    #构建词典,形成稀疏向量
    dic_word = corpora.Dictionary(fenci_word)
    cor_word = [dic_word.doc2bow(i)for i in fenci_word]
    dic_char = corpora.Dictionary(fenci_char)
    cor_char = [dic_char.doc2bow(i)for i in fenci_char]
    
    #训练lda    
    num_topic = 6  
    # topic = []
    # lda_word = models.ldamodel.LdaModel(corpus=cor_word, id2word=dic_word, num_topics=num_topic)
    # print(lda_word) 
    # #topic-word分布
    # for topic in lda_word.print_topics(num_words=10):
    #     print(topic)
    # #para-topic分布
    # for e, values in enumerate(lda_word.inference(cor_word)[0]):
    #     print(test_list[e])
    #     for ee, value in enumerate(values):
    #         print('\t主题%d推断值%.2f' % (ee, value))   
 
    # # 对于每个主题，所有词对应的概率，求和=1
    # print('概率总和', sum(i[1] for i in lda_word.show_topic(0, 9999)))




    x = [] # x轴
    perplexity_values_word = [] # 困惑度
    coherence_values_word = []   # 一致性
    perplexity_values_char = [] # 困惑度
    coherence_values_char = []   # 一致性
    model_list = [] # 存储对应主题数量下的lda模型,便于生成可视化网页

    for topic in range(num_topic):
        print("主题数量：", topic+1)
        lda_word=models.ldamodel.LdaModel(corpus=cor_word, num_topics=topic+1, id2word =dic_word, chunksize = 2000, passes=20, iterations = 400)
        lda_char=models.ldamodel.LdaModel(corpus=cor_char, num_topics=topic+1, id2word =dic_char, chunksize = 2000, passes=20, iterations = 400)
        model_list.append(lda_word)
        x.append(topic+1)
        perplexity_values_word.append(-lda_word.log_perplexity(cor_word))
        coherencemodel_word = models.CoherenceModel(model=lda_word, texts=fenci_word, dictionary=dic_word, coherence='c_v')
        coherence_values_word.append(coherencemodel_word.get_coherence())
        
        perplexity_values_char.append(-lda_char.log_perplexity(cor_char))
        coherencemodel_char = models.CoherenceModel(model=lda_char, texts=fenci_char, dictionary=dic_char, coherence='c_v')
        coherence_values_char.append(coherencemodel_char.get_coherence())
        
        print("该主题评价完成\n")