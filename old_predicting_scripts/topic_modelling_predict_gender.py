# -*- coding: utf8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
# from en_stopwords import stopwords # stop-words list from MySQL
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import glob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 30)


def display_wordclouds(model, feature_names, no_top_words, n_topics):
    top_words_weight_dicts = list()
    for topic_idx, topic in enumerate(model.components_):
        top_words_weight_dict = dict()
        for i in topic.argsort()[:-no_top_words - 1:-1]:
            top_words_weight_dict[feature_names[i]] = model.components_[topic_idx][i]
        top_words_weight_dicts.append(top_words_weight_dict)
    for t in range(n_topics):
        plt.figure()
        plt.imshow(WordCloud(background_color='white', color_func=grey_color_func).fit_words(top_words_weight_dicts[t]))
        plt.axis("off")
        plt.title("Topic #" + str(t))
        plt.show()

        #top_words_weight_weights = list(top_words_weight_dict.values())
        #top_words_weight_weights = [i/sum(top_words_weight_weights) for i in top_words_weight_weights]
        #print(sum(top_words_weight_weights))


def display_topics(model, feature_names, no_top_words, n_topics):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
        #top_words_weight_weights = list(top_words_weight_dict.values())
        #top_words_weight_weights = [i/sum(top_words_weight_weights) for i in top_words_weight_weights]
        #print(sum(top_words_weight_weights))
        # print(", ".join([feature_names[i]
        #                  for i in topic.argsort()[::-1][:no_top_words]]))


def display_one_topic(model, feature_names, no_top_words, topic_idx_needed):
    for topic_idx, topic in enumerate(model.components_):
        if topic_idx == topic_idx_needed:
            print("Topic {}:".format(topic_idx))
            print('Topic top-words: ' + ", ".join([feature_names[i]
                                                   for i in topic.argsort()[:-no_top_words - 1:-1]]))


stopwords_ru = open('./stopwords_and_others/stop_ru.txt', 'r', encoding='utf-8').read().split('\n')


# FOR DOCUMENT != words chunk
train_documents = list()
train_documents_titles = list()

test_documents = list()
test_documents_titles = list()

# FOR DOCUMENT = PLAY
train_texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
                   'Programming/github desktop/RusDraCor/Ira_Scripts/' \
                   'TopicModelling/speech_corpus_no_prop_char_names_POS_restriction/byplay/byplay/'
test_texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
                  'Programming/github desktop/RusDraCor/Ira_Scripts/' \
                  'TopicModelling/speech_corpus_no_prop_char_names_POS_restriction/bysex/'
# FOR DOCUMENT = One Characters speech
#texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/speech_corpus/bycharacter'
all_train_texts = glob.glob(train_texts_path+'*.txt')
all_test_texts = glob.glob(test_texts_path+'*.txt')

n = 0
k = 0
for doc in all_train_texts:
    train_documents_titles.append(doc.split('/')[-1].split('.txt')[0])
    doc_text = re.sub('[\.,!\?\(\)\-:;—…́«»–]', '', open(doc, 'r', encoding='utf-8').read()).split()
    chunk_size = 1000
    for i in range(0, len(doc_text), chunk_size):
        one_chunk = ' '.join(doc_text[i:i + chunk_size])
        if len(one_chunk.split()) > 200:
            train_documents.append(one_chunk)
        if 200 < len(one_chunk.split()) < 1000:
            k += 1
        else:
            n += 1
print('Chunks with length in words less than 1000 (did not take):', n)
print('Chunks with length in words more than 1000 and less than 200 (took):', k)


for doc in all_test_texts:
    test_documents_titles.append(doc.split('/')[-1].split('.txt')[0])
    doc_text = re.sub('[\.,!\?\(\)\-:;—…́«»–]', '', open(doc, 'r', encoding='utf-8').read())
    test_documents.append(doc_text)


print('Topic Modeling train text collection size: ', len(train_documents), '\n')
print('Median length of train collection\'s documents: ', np.median([len(d.split()) for d in train_documents]))
print('Mean length of train collection\'s documents: ', np.mean([len(d.split()) for d in train_documents]))
print('Minimum length of train collection\'s documents: ', np.min([len(d.split()) for d in train_documents]))
print('Maximum length of train collection\'s documents: ', np.max([len(d.split()) for d in train_documents]))


def run_TM(n_topics, doprint, doreturn):
    n_topics = n_topics
    no_top_words = 40

    # LDA on raw words counts
    tf_vectorizer = CountVectorizer(max_df=0.8,
                                    min_df=0.2,
                                    stop_words=stopwords_ru,
                                    max_features=1000)
    tf = tf_vectorizer.fit_transform(train_documents)
    tf_feature_names = tf_vectorizer.get_feature_names()


    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, random_state=42)
    lda_doc_topic = lda.fit_transform(tf)
    print('LDA doc-topic shape:', lda_doc_topic.shape)

    print('\nTOPICS\nLDA top terms:')
    display_topics(lda, tf_feature_names, no_top_words, n_topics)

    tf1 = tf_vectorizer.transform(test_documents)
    doc_topic_dist_unnormalized = np.matrix(lda.transform(tf1))

    print('\n\n')

    '''
    doc_topic = lda.doc_topic_
    for i in range(0, 10):
        print("{} (top topic: {})".format(documents_titles[i], doc_topic[i].argmax()))
        print(doc_topic[i].argsort()[::-1][:3])
    '''

    # normalize the distribution (only needed if you want to work with the probabilities)
    doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
    topic_topdocs_dict = dict()

    for play in range(len(doc_topic_dist)):
        top_topic = str(doc_topic_dist.argmax(axis=1)[play].tolist()[0][0])
        # print(documents_titles[n])
        # print(print(doc_topic_dist.argsort()[n].tolist()[0]))
        if top_topic not in topic_topdocs_dict:
            topic_topdocs_dict[top_topic] = list()
            topic_topdocs_dict[top_topic].append(test_documents_titles[play])
        else:
            topic_topdocs_dict[top_topic].append(test_documents_titles[play])

    def create_doc_topic_dict_for_plays():
        doc_3toptopic_dict = dict()
        doc_topicsprobs_dict = dict()
        for play in range(len(doc_topic_dist)):
            play_title = test_documents_titles[play]
            play_topic_dist = (doc_topic_dist[play].tolist()[0])
            play_topic_dist = [round(100*float('{:f}'.format(item))) for item in play_topic_dist]  # creating a list with probs per topic (in 100-notation)
            doc_topicsprobs_dict[play_title] = play_topic_dist
            play_top3_topics = reversed(doc_topic_dist.argsort()[play].tolist()[0][-3::])
            doc_3toptopic_dict[play_title] = play_top3_topics
        return doc_3toptopic_dict, doc_topicsprobs_dict

    def create_doc_topic_dict_for_acts():
        doc_topic_dict = dict()
        for play in range(len(doc_topic_dist)):
            play_title = test_documents_titles[play]
            top_topic = doc_topic_dist.argsort()[play].tolist()[0][-1]
            doc_topic_dict[play_title] = [top_topic]
        return doc_topic_dict

    doc_topic_dict = create_doc_topic_dict_for_plays()[0]
    doc_topicsprobs_dict = create_doc_topic_dict_for_plays()[1]

    if doprint:

        display_wordclouds(lda, tf_feature_names, 100, n_topics)

        print('\nDOCUMENTS PER TOPIC')
        for topic in topic_topdocs_dict:
            display_one_topic(lda, tf_feature_names, no_top_words, int(topic))
            print('Topic plays: ' + ', '.join(topic_topdocs_dict[topic]), '\n')

        print('\n\nTOPICS PER DOCUMENT')
        for play in sorted(list(doc_topic_dict)):
            print(play)
            for topic in doc_topic_dict[play]:
                print(doc_topicsprobs_dict[play])
                display_one_topic(lda, tf_feature_names, no_top_words, int(topic))
            print('\n')

    if doreturn:
        return doc_topicsprobs_dict


run_TM(7, 1, 0)


'''
tfidf_vectorizer = TfidfVectorizer(ngram_range=(1, 1),
                                   max_df=0.7,
                                   min_df=0.2,
                                   stop_words=stopwords_ru,
                                   tokenizer=LemmaTokenizer(),
                                   max_features=1000)
tfidf = tfidf_vectorizer.fit_transform(documents)
tfidf_feature_names = tfidf_vectorizer.get_feature_names()


print()

nmf = NMF(n_components=n_topics)
nmf_doc_topic = nmf.fit_transform(tfidf)
print('NMF doc-topic shape:', nmf_doc_topic.shape)
print('\nNMF top terms:')
display_topics(nmf, tfidf_feature_names, no_top_words)
'''