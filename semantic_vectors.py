import sys
import gensim, logging
from pymystem3 import Mystem


def get_pos_for_semvector(mystem_pos):
    if mystem_pos.startswith('S,'):
        pos = '_NOUN'
    if mystem_pos.startswith('S='):
        pos = '_NOUN'
    if mystem_pos.startswith('A,'):
        pos = '_ADJ'
    if mystem_pos.startswith('A='):
        pos = '_ADJ'
    if mystem_pos.startswith('ADV,'):
        pos = '_ADV'
    if mystem_pos.startswith('ADV='):
        pos = '_ADV'
    if mystem_pos.startswith('V,'):
        pos = '_VERB'
    if mystem_pos.startswith('V='):
        pos = '_VERB'
    return pos

m = Mystem()

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = 'ruscorpora_upos_skipgram_300_5_2018.vec'

model = gensim.models.KeyedVectors.load_word2vec_format(model, binary=False)

model.init_sims(replace=True)

words = ['есть', 'собака', 'каша']

for word1 in words:
    pos1 = m.analyze(word1)[0]['analysis'][0]['gr']
    pos1 = get_pos_for_semvector(pos1)
    word1 = word1 + pos1
    for word2 in words:
        pos2 = m.analyze(word2)[0]['analysis'][0]['gr']
        pos2 = get_pos_for_semvector(pos2)
        word2 = word2 + pos2
        if word1 in model and word2 in model:
            print(word1, word2, model.similarity(word1, word2))



# print(model.similarity('кот_NOUN', 'матушка_NOUN'))
