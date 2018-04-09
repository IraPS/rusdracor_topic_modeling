# -*- coding: utf8 -*-
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
import numpy as np
# from en_stopwords import stopwords # stop-words list from MySQL
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
import glob


class LemmaTokenizer(object):
    def __init__(self):
        self.wnl = WordNetLemmatizer()
    def __call__(self, doc):
        return [self.wnl.lemmatize(t) for t in word_tokenize(doc)]


def display_topics(model, feature_names, no_top_words):
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(", ".join([feature_names[i]
                        for i in topic.argsort()[:-no_top_words - 1:-1]]))
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
documents = list()
documents_titles = list()

# FOR DOCUMENT = PLAY
texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/speech_corpus/byplay/byplay/'

# FOR DOCUMENT = One Characters speech
#texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/speech_corpus/bycharacter'
all_texts = glob.glob(texts_path+'*.txt')

for doc in all_texts:
    documents_titles.append(doc.split('/')[-1].split('.txt')[0])
    doc_text = open(doc, 'r', encoding='utf-8').read()
    documents.append(doc_text)

'''
# FOR DOCUMENT = 1000-words CHUNK
all_plays_texts = open('/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/'
                       'RusDraCor/Ira_Scripts/TopicModelling/speech_corpus/all_in_one_doc/all_plays_texts.txt',
                       'r', encoding='utf-8').read().split()

print(len(all_plays_texts))

chunk_size = 1000
for i in range(0, len(all_plays_texts), chunk_size):
    one_chunk = ' '.join(all_plays_texts[i:i + chunk_size])
    documents.append(one_chunk)
'''

print('Text collection size and median length in symbols:')
print(len(documents), np.median([len(d) for d in documents]))


def run_TM(n_topics, doprint, doreturn):
    if doprint:
        n_topics = n_topics
        no_top_words = 30

        # LDA on raw words counts
        tf_vectorizer = CountVectorizer(max_df=0.6,
                                        min_df=0.2,
                                        stop_words=stopwords_ru,
                                        max_features=1000)
        tf = tf_vectorizer.fit_transform(documents)
        tf_feature_names = tf_vectorizer.get_feature_names()


        lda = LatentDirichletAllocation(n_topics=n_topics)
        lda_doc_topic = lda.fit_transform(tf)
        print('LDA doc-topic shape:', lda_doc_topic.shape)

        print('\nTOPICS\nLDA top terms:')
        display_topics(lda, tf_feature_names, no_top_words)

        tf1 = tf_vectorizer.transform(documents)
        doc_topic_dist_unnormalized = np.matrix(lda.transform(tf))

        print('\n\n')

        '''
        doc_topic = lda.doc_topic_
        for i in range(0, 10):
            print("{} (top topic: {})".format(documents_titles[i], doc_topic[i].argmax()))
            print(doc_topic[i].argsort()[::-1][:3])
        '''

        # normalize the distribution (only needed if you want to work with the probabilities)
        doc_topic_dist = doc_topic_dist_unnormalized/doc_topic_dist_unnormalized.sum(axis=1)
        topic_doc_dict = dict()

        for n in range(len(doc_topic_dist.argmax(axis=1))):
            topic = str(doc_topic_dist.argmax(axis=1)[n].tolist()[0][0])
            # print(documents_titles[n])
            # print(print(doc_topic_dist.argsort()[n].tolist()[0]))
            if topic not in topic_doc_dict:
                topic_doc_dict[topic] = list()
                topic_doc_dict[topic].append(documents_titles[n])
            else:
                topic_doc_dict[topic].append(documents_titles[n])

        print('\nDOCUMENTS PER TOPIC')

        for topic in topic_doc_dict:
            display_one_topic(lda, tf_feature_names, no_top_words, int(topic))
            print('Topic plays: ' + ', '.join(topic_doc_dict[topic]), '\n')

        print('\n\nTOPICS PER DOCUMENT')

        def create_doc_topic_dict_for_plays():
            doc_topic_dict = dict()
            for play in range(len(doc_topic_dist.argsort())):
                # print(doc_topic_dist[play])
                play_title = documents_titles[play]
                play_top3_topics = reversed(doc_topic_dist.argsort()[play].tolist()[0][-3::])
                doc_topic_dict[play_title] = play_top3_topics
            return doc_topic_dict


        def create_doc_topic_dict_for_acts():
            doc_topic_dict = dict()
            for play in range(len(doc_topic_dist.argsort())):
                play_title = documents_titles[play]
                top_topic = doc_topic_dist.argsort()[play].tolist()[0][-1]
                doc_topic_dict[play_title] = [top_topic]
            return doc_topic_dict

        doc_topic_dict = create_doc_topic_dict_for_plays()

        for play in sorted(list(doc_topic_dict)):
            print(play)
            for topic in doc_topic_dict[play]:
                display_one_topic(lda, tf_feature_names, no_top_words, int(topic))
            print('\n')

    if doreturn:
        return 0


run_TM(9, 1, 0)


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