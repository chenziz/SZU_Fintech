# pip install 都可以下载的包
from ahocorapy.keywordtree import KeywordTree
from sklearn import feature_extraction
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import re
import wordcloud
import nltk
import jieba
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
import collections
import numpy as np
import numpy
from PIL import Image
import matplotlib.pyplot as plt


# 读文件里面的数据转化为二维列表
def Read_list(filename):
    file1 = open(filename + ".txt", "r")
    list_row = file1.readlines()
    list_source = []
    for i in range(len(list_row)):
        column_list = list_row[i].strip().split("\t")  # 每一行split后是一个列表
        list_source.append(column_list)  # 在末尾追加到list_source
    file1.close()
    return list_source


def Save_list(list1, filename):
    file2 = open(filename + '.txt', 'w')
    for i in range(len(list1)):
        for j in range(len(list1[i])):
            file2.write(str(list1[i][j]))  # write函数不能写int类型的参数，所以使用str()转化
            file2.write('\t')  # 相当于Tab一下，换一个单元格
        file2.write('\n')  # 写完一行立马换行
    file2.close()


# 创建去除非中文字符的函数
# 数据清洗，去除标点符号，数字，等其它非中文字符
# 匹配[^\u4e00-\u9fa5]
def find_chinese(file):
    pattern = re.compile(r'[^\u4e00-\u9fa5]')
    chinese_txt = re.sub(pattern, '', file)
    return chinese_txt


# 文件读取
def read_txt(filepath):
    file = open(filepath, 'r', encoding='utf-8')
    txt = file.read()
    return txt


# 中文分词
def cut_word(text):
    # 精准模式
    jieba_list = jieba.cut(text, cut_all=False)
    return jieba_list


# 去除停用词
def seg_sentence(list_txt):
    # 读取停用词表
    stopwords = read_txt('stopwordslist.txt')
    seg_txt = [w for w in list_txt if w not in stopwords]
    return seg_txt


# 词频统计
def counter(txt):
    seg_list = txt
    c = Counter()
    for w in seg_list:
        if w != ' ':
            c[w] += 1
    return c


def create__file(file_path, msg):
    f = open(file_path, "a")
    f.write(msg)
    f.close


def searching(list):
    l = []
    l1 = []
    kwtree = KeywordTree(case_insensitive=True)
    kwtree.add('经济')
    kwtree.add('金融')
    kwtree.add('财政')
    kwtree.add('税')
    kwtree.add('人民银行')
    kwtree.add('央行')
    kwtree.add('赤字')
    kwtree.add('利率')
    kwtree.add('外汇')
    kwtree.finalize()
    resulta = kwtree.search(str(list))
    resultsa = kwtree.search_all(str(list))
    for result in resultsa:
        l.append(result[0])
    kwtree = KeywordTree(case_insensitive=True)
    kwtree.add('不确定')
    kwtree.add('不明确')
    kwtree.add('波动')
    kwtree.add('震荡')
    kwtree.add('动荡')
    kwtree.add('未明')
    kwtree.add('不稳')
    kwtree.add('难料')
    kwtree.add('预计')
    kwtree.add('估计')
    kwtree.add('影响')
    kwtree.add('预测')
    kwtree.add('风险')
    kwtree.finalize()
    resultb = kwtree.search(str(list))
    resultsb = kwtree.search_all(str(list))
    for result in resultsb:
        l.append(result[0])
    kwtree = KeywordTree(case_insensitive=True)
    kwtree.add('政策')
    kwtree.add('制度')
    kwtree.add('体制')
    kwtree.add('战略')
    kwtree.add('措施')
    kwtree.add('规章')
    kwtree.add('规例')
    kwtree.add('条例')
    kwtree.add('政治')
    kwtree.add('执政')
    kwtree.add('国家')
    kwtree.add('改革')
    kwtree.add('整治')
    kwtree.add('监管')
    kwtree.add('规管')
    kwtree.add('整改')
    kwtree.add('财政')
    kwtree.add('税')
    kwtree.add('人民银行')
    kwtree.add('央行')
    kwtree.add('赤字')
    kwtree.add('利率')
    kwtree.add('外汇')
    kwtree.finalize()
    resultc = kwtree.search(str(list))
    resultsc = kwtree.search_all(str(list))
    for result in resultsc:
        l.append(result[0])
    if ((resulta != None) and (resultb != None) and (resultc != None)):
        return 1, l
    else:
        return 0, l


# 主函数
if __name__ == "__main__":
    # 读取文本信息
    news = read_txt('2008.txt')
    # print("原文：",news)
    # 清洗数据,去除无关标点
    chinese_news = find_chinese(news)
    # print("原文文本长度：",news)
    # print("纯中文文本：",chinese_news)
    # 结巴分词
    chinese_cut = cut_word(chinese_news)
# print(chinese_cut)
chinese_sentence = seg_sentence(chinese_cut)
test_words = chinese_sentence[0:8400000]
print(len(test_words))
# with open("finaltest_rmrb08.txt","w") as f:
#  f.write(str(test_words))

b = numpy.array(test_words).reshape(12000, 700)  # reshape(列的长度，行的长度)

print(b[0])  # 1行
Save_list(b, 'daily_rmrb08')

K = Read_list('daily_rmrb08')

# 统计每一个月的EPU值
list_words = []
sum = [0] * 12
for i in range(0, 12):
    for j in range(1000 * i, 1000 * i + 999):
        sum[i] += searching(K[j])[0]
        list_words.append((searching(K[j])[1]))

print(sum)
Save_list(list_words, 'list_words_rmrb08')

# 打印处理后的EPU
k = 100 / sum[0]
finalepu = [] * 12
for i in range(0, 12):
    sum[i] = k * sum[i]
print(sum)

# 画折线图
x_data = ['JAN', 'FEB', 'MAR', 'APR', 'MAY', 'JUN', 'JUL', 'AUG', 'SEP', 'OCT', 'NOV', 'DEC']
y_data = sum
plt.title("2008-EPU")
plt.plot(x_data, y_data)
plt.show()

# 计算tf-idf值
corpus = []
print(list_words[0])
print(str(list_words[1][1]))
print(len(list_words))
for i in range(0, 11988):
    corpus.append(str(list_words[i]))
# step 1
vectoerizer = CountVectorizer(min_df=1, max_df=1.0, token_pattern='\\b\\w+\\b')
# step 2
vectoerizer.fit(corpus)
# step 3
bag_of_words = vectoerizer.get_feature_names()
print("Bag of words:")
print(bag_of_words)
print(len(bag_of_words))
# step 4
X = vectoerizer.transform(corpus)
print("Vectorized corpus:")
print(X.toarray())
from sklearn.feature_extraction.text import TfidfTransformer

# step 1
tfidf_transformer = TfidfTransformer()
# step 2
tfidf_transformer.fit(X.toarray())

# step 3
for idx, word in enumerate(vectoerizer.get_feature_names()):
    print(word, tfidf_transformer.idf_[idx])
print(vectoerizer.get_feature_names())
# step 4
tfidf = tfidf_transformer.transform(X)
print(tfidf.toarray())

# 绘制图云


# 读取文件
fn = open('list_words_rmrb08.txt')
string_data = fn.read()
fn.close()

# 文本预处理
pattern = re.compile(u'\t|\n|\.|-|:|;|\)|\(|\?|"')  # 定义正则表达式匹配模式
string_data = re.sub(pattern, '', string_data)  # 将符合模式的字符去除

# 文本分词
seg_list_exact = jieba.cut(string_data, cut_all=False)  # 精确模式分词
object_list = []
remove_words = [u'的', u'，', u'和', u'是', u'随着', u'对于', u'对', u'等', u'能', u'都', u'。',
                u' ', u'、', u'中', u'在', u'了', u'通常', u'如果', u'我们', u'需要']  # 自定义去除词库

for word in seg_list_exact:  # 循环读出每个分词
    if word not in remove_words:  # 如果不在去除词库中
        object_list.append(word)  # 分词追加到列表

# 词频统计
word_counts = collections.Counter(object_list)  # 对分词做词频统计
word_counts_top10 = word_counts.most_common(40)  # 获取前40最高频的词
print (word_counts_top10)  # 输出检查

# 词频展示
mask = np.array(Image.open('中国地图.jpg'))  # 定义词频背景 需要自己找
wc = wordcloud.WordCloud(
    background_color='white',  # 设置背景颜色
    font_path='/System/Library/Fonts/Hiragino Sans GB.ttc',  # 设置字体格式 我这是mac win自己改路径
    mask=mask,  # 设置背景图
    max_words=200,  # 最多显示词数
    max_font_size=100,  # 字体最大值
    scale=32  # 调整图片清晰度，值越大越清楚
)

wc.generate_from_frequencies(word_counts)  # 从字典生成词云
image_colors = wordcloud.ImageColorGenerator(mask)  # 从背景图建立颜色方案
wc.recolor(color_func=image_colors)  # 将词云颜色设置为背景图方案
wc.to_file("cnepu3.jpg")  # 将图片输出为文件
plt.imshow(wc)
plt.axis("off")
plt.savefig("cnepu3.jpg", dpi=200)
plt.show()