import re
import lxml.etree as ET
import glob
from pymystem3 import Mystem

tei_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/dracor-rusdracor/tei/'
all_tei = glob.glob(tei_path+'*.xml')
ns = {'tei': 'http://www.tei-c.org/ns/1.0'}

plays_texts = dict()


def get_body(file): # TEI
    """This function parse the file at initial phase
    and gets its xml body or returns None if the file is invalid"""
    try:
        tree = ET.parse(file)
        tei = tree.getroot()
        text = tei[1]
        body = text.find('tei:body', ns)
        return body
    except:
        print('ERROR while parsing', file)
        return None


def get_divs(file): # TEI
    """This function gets all the divs of the play
    with their contents"""
    body = get_body(file)
    if body is not None:
        divs = body.findall('tei:div', ns)
        return divs
    else:
        return None


def get_sps(divs):
    text = str()
    for div in divs:
        speeches = div.findall('tei:sp', ns)
        if len(speeches) > 0:
            for sp in speeches:
                ps = sp.findall('tei:p', ns)
                for p in ps:
                    if p is not None:
                        text += str(p.text)
                        text += ' '
                lgs = sp.findall('tei:lg', ns)
                for lg in lgs:
                    ls = lg.findall('tei:l', ns)
                    for l in ls:
                        if l is not None:
                            text += str(l.text)
                            text += ' '
                ls = sp.findall('tei:l', ns)
                for l in ls:
                    if l is not None:
                        text += str(l.text)
                        text += ' '
        else:
            subdivs = div.findall('tei:div', ns)
            if len(subdivs) > 0:
                text += get_sps(subdivs)
    return text

texts_file = open('plays_texts.txt', 'w', encoding='utf-8')
for tei in all_tei:
    divs = get_divs(tei)
    play_text = re.sub('\n', '', get_sps(divs))
    play_text = re.sub('  ', ' ', play_text)
    plays_texts[tei.split('/')[-1]] = play_text
    texts_file.write('\n\nNEW PLAY ' + tei.split('/')[-1] + '\n')
    texts_file.write(play_text)
texts_file.close()


m = Mystem()
lemmas_file = open('plays_texts_lemmatized.txt', 'w', encoding='utf-8')
for play in plays_texts:
    print(play)
    text = plays_texts[play]
    lemmas = m.lemmatize(text)
    lemmas_file.write(''.join(lemmas))
    lemmas_file.write('\n\n')
lemmas_file.close()