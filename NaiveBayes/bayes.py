# encoding: utf-8
from numpy import *
import operator


def load_data_set():
    posting_list = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'],
                    ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'],
                    ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'],
                    ['stop', 'posting', 'stupid', 'worthless', 'garbage'],
                    ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'],
                    ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    class_vec = [0, 1, 0, 1, 0, 1]  # 1 代表侮辱性文字, 0 代表正常言论
    return posting_list, class_vec


def create_vocab_list(data_set):
    vocab_set = set([])  # create empty set
    for document in data_set:
        vocab_set = vocab_set | set(document)   # union of the two sets
    return list(vocab_set)


def set_of_words2vec(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] = 1
        else:
            print "the word: %s is not in my Vocabulary!" % word
    return return_vec


# 朴素贝叶斯分类器训练函数
def train_nb0(train_matrix, train_category):
    num_train_docs = len(train_matrix)
    num_words = len(train_matrix[0])
    p_abusive = sum(train_category)/float(num_train_docs)
    p0_num = ones(num_words)
    p1_num = ones(num_words)      # change to ones()
    p0_denom = 2.0
    p1_denom = 2.0                        # change to 2.0
    for i in range(num_train_docs):
        if train_category[i] == 1:
            p1_num += train_matrix[i]
            p1_denom += sum(train_matrix[i])
        else:
            p0_num += train_matrix[i]
            p0_denom += sum(train_matrix[i])

    p1_vect = log(p1_num/p1_denom)          # change to log()
    p0_vect = log(p0_num/p0_denom)          # change to log()
    return p0_vect, p1_vect, p_abusive


def classify_nb(vec2classify, p0_vec, p1_vec, p_class1):
    p1 = sum(vec2classify * p1_vec) + log(p_class1)    # element-wise mult
    p0 = sum(vec2classify * p0_vec) + log(1.0 - p_class1)
    if p1 > p0:
        return 1
    else:
        return 0


# 网站恶意留言
def testing_nb():
    list_of_posts, list_classes = load_data_set()
    my_vocab_list = create_vocab_list(list_of_posts)
    train_mat = []
    for postinDoc in list_of_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, postinDoc))
    p0_v, p1_v, p_ab = train_nb0(array(train_mat), array(list_classes))
    test_entry = ['love', 'my', 'dalmation']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab)
    test_entry = ['stupid', 'garbage']
    this_doc = array(set_of_words2vec(my_vocab_list, test_entry))
    print test_entry, 'classified as: ', classify_nb(this_doc, p0_v, p1_v, p_ab)


# 词袋
def bag_of_words2vec_mn(vocab_list, input_set):
    return_vec = [0]*len(vocab_list)
    for word in input_set:
        if word in vocab_list:
            return_vec[vocab_list.index(word)] += 1
    return return_vec


# 分词
def text_parse(big_string):    # input is big string, # output is word list
    import re
    list_of_tokens = re.split(r'\W*', big_string)
    return [tok.lower() for tok in list_of_tokens if len(tok) > 2]


# 垃圾邮件过滤
def spam_test():
    doc_list = []
    class_list = []
    full_text = []
    # 文本解析
    for i in range(1, 26):
        word_list = text_parse(open('email/spam/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(1)
        word_list = text_parse(open('email/ham/%d.txt' % i).read())
        doc_list.append(word_list)
        full_text.extend(word_list)
        class_list.append(0)
    vocab_list = create_vocab_list(doc_list)         # create vocabulary
    # 50个样本，10个作为测试集，40个作为训练集
    training_set = range(50)
    test_set = []                                      # create test set
    for i in range(10):
        rand_index = int(random.uniform(0, len(training_set)))
        test_set.append(training_set[rand_index])
        del(training_set[rand_index])
    train_mat = []
    train_classes = []
    for doc_index in training_set:        # train the classifier (get probs) trainNB0
        train_mat.append(bag_of_words2vec_mn(vocab_list, doc_list[doc_index]))
        train_classes.append(class_list[doc_index])
    p0_v, p1_v, p_spam = train_nb0(array(train_mat), array(train_classes))
    # 错误率
    error_count = 0
    for docIndex in test_set:            # classify the remaining items
        word_vector = bag_of_words2vec_mn(vocab_list, doc_list[docIndex])
        if classify_nb(array(word_vector), p0_v, p1_v, p_spam) != class_list[docIndex]:
            error_count += 1
            print "classification error", doc_list[docIndex]
    print 'the error rate is: ', float(error_count)/len(test_set)


if __name__ == '__main__':
    '''
    list_of_posts, list_classes = load_data_set()
    print(list_of_posts)
    print(list_classes)
    my_vocab_list = create_vocab_list(list_of_posts)
    print(my_vocab_list)
    # print(set_of_words2vec(my_vocab_list, list_of_posts[0]))
    train_mat = []
    # 词->向量
    for postin_doc in list_of_posts:
        train_mat.append(set_of_words2vec(my_vocab_list, postin_doc))

    p0_v, p1_v, p_ab = train_nb0(train_mat, list_classes)
    print(p_ab)
    print(p0_v)
    print(p1_v)
    '''

    # testing_nb()
    spam_test()
