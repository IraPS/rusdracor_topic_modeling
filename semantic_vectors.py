# -*- coding: utf8 -*-
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np
import glob
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
import random
import gensim,logging
from pymystem3 import Mystem

def get_pos_for_semvector(mystem_pos):
    mystemtag_pos_dict = {'S,': '_NOUN', 'S=': '_NOUN',
                          'A,': '_ADJ', 'A=': '_ADJ',
                          'ADV,':  '_ADV', 'ADV=': '_ADV',
                          'V,': '_VERB', 'V=': '_VERB'}
    for tag in mystemtag_pos_dict:
        if mystem_pos.startswith(tag):
            pos = mystemtag_pos_dict[tag]
    return pos

m = Mystem()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
ncrl_model = 'ruscorpora_upos_skipgram_300_5_2018.vec'
ncrl_model = gensim.models.KeyedVectors.load_word2vec_format(ncrl_model, binary=False)
ncrl_model.init_sims(replace=True)


def display_topics(model, feature_names, no_top_words, n_topics):
    """Displays all topics' top-words and semdensity per topic"""
    all_topics_topwords_similarity = list()
    no_top_words_for_semantics = 10
    for topic_idx, topic in enumerate(model.components_):
        print("Topic {}:".format(topic_idx))
        print(", ".join([feature_names[i]
                         for i in topic.argsort()[:-no_top_words - 1:-1]]))
        topwords = [feature_names[i] for i in topic.argsort()[:-no_top_words_for_semantics - 1:-1]]
        topwords_similarity = list()
        not_in_model = 0
        for word1 in topwords:
            pos1 = m.analyze(word1)[0]['analysis'][0]['gr']
            pos1 = get_pos_for_semvector(pos1)
            word1 = word1 + pos1
            for word2 in topwords:
                pos2 = m.analyze(word2)[0]['analysis'][0]['gr']
                pos2 = get_pos_for_semvector(pos2)
                word2 = word2 + pos2
                if word1 in ncrl_model and word2 in ncrl_model and word1 != word2:
                    word1_word2_similarity = ncrl_model.similarity(word1, word2)
                else:
                    word1_word2_similarity = 0
                    not_in_model += 1
                topwords_similarity.append(word1_word2_similarity)
        topwords_similarity = sum(topwords_similarity)/((no_top_words_for_semantics-1)**2 - not_in_model)
        print(topwords_similarity)
        all_topics_topwords_similarity.append(topwords_similarity)
    print('\nMean topics semantic similarity for {0} topics is {1}'.
          format(n_topics, np.mean(all_topics_topwords_similarity)))


def grey_color_func(word, font_size, position, orientation, random_state=None,
                    **kwargs):
    """Establishes colour range for word-clouds"""
    return "hsl(0, 0%%, %d%%)" % random.randint(0, 30)


def display_wordclouds(model, feature_names, no_top_words, n_topics):
    """Displays word-clouds for n topics' top-words"""
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

# Opening a stop-words list for Russian
stopwords_ru = open('./stopwords_and_others/stop_ru.txt', 'r', encoding='utf-8').read().split('\n')

# Determining train texts path (txt-files)
train_texts_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
                   'Programming/github desktop/RusDraCor/Ira_Scripts/' \
                   'TopicModelling/rusdracor_topic_modeling/corpora/' \
                   'speech_corpus_no_prop_char_names_ONLY_NOUNS/byplay/byplay/'

train_documents = list()
train_documents_titles = list()
all_train_texts = glob.glob(train_texts_path+'*.txt')

# Splitting train texts into word-chunks
n = 0
k = 0
chunk_size = 500
min_chunk_size = 100
for doc in all_train_texts:
    train_documents_titles.append(doc.split('/')[-1].split('.txt')[0])
    doc_text = re.sub('[\.,!\?\(\)\-:;—…́«»–]', '', open(doc, 'r', encoding='utf-8').read()).split()
    for i in range(0, len(doc_text), chunk_size):
        one_chunk = ' '.join(doc_text[i:i + chunk_size])
        if len(one_chunk.split()) > min_chunk_size:
            train_documents.append(one_chunk)
        if min_chunk_size < len(one_chunk.split()) < chunk_size:
            k += 1
        if len(one_chunk.split()) < min_chunk_size:
            n += 1
print('Taking chunks of length {0} WORDS'.format(chunk_size))
print('Chunks with length less than {0} (did not take):'.format(min_chunk_size), n)
print('Chunks with length more than {0} and less than {1} (took):'.format(min_chunk_size, chunk_size), k)


# Reporting statistics on the model
print('\nTopic modeling train text collection size: ', len(train_documents))
print('Median length of train collection\'s documents: ', np.median([len(d.split()) for d in train_documents]))
print('Mean length of train collection\'s documents: ', np.mean([len(d.split()) for d in train_documents]))
print('Minimum length of train collection\'s documents: ', np.min([len(d.split()) for d in train_documents]))
print('Maximum length of train collection\'s documents: ', np.max([len(d.split()) for d in train_documents]))


def run_TM(n_topics, doprint):
    """Performs Topic Modeling, present topics and return/print/write in a file model's application results"""
    n_topics = n_topics
    no_top_words = 40

    tf_vectorizer = CountVectorizer(max_df=0.7,
                                    min_df=0.2,
                                    stop_words=stopwords_ru,
                                    max_features=500)
    tf = tf_vectorizer.fit_transform(train_documents)
    tf_feature_names = tf_vectorizer.get_feature_names()

    lda = LatentDirichletAllocation(n_topics=n_topics, max_iter=100, random_state=42)
    lda_doc_topic = lda.fit_transform(tf)

    # Printing topics' 40 top-words, printing topics', semdensity oer topic,
    # displaying word-clouds for 100 topics' top-words if needed
    if doprint:
        print('LDA doc-topic shape:', lda_doc_topic.shape)
        print('\nTOPICS\nLDA top terms:')
        display_topics(lda, tf_feature_names, no_top_words, n_topics)
        print('\n\n')
        # display_wordclouds(lda, tf_feature_names, 100, n_topics)


    print('The TM is finished, the model is applied to the  data, '
          'the semdensity per topc is calculated.')

# Running topic modeling task to build a model with 5 topics
run_TM(5, 1)