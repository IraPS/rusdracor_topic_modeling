from bs4 import BeautifulSoup
import glob

tei_path = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/Programming/github desktop/dracor-rusdracor/tei/'
all_tei = glob.glob(tei_path+'*.xml')

play_texts = dict()

for tei in all_tei:
    title = tei.split('/')[-1]
    play_texts[title] = str()
    play = open(tei, 'r', encoding='utf-8').read()
    soup = BeautifulSoup(play, 'xml')
    for p in soup.find_all('p'):
        soup_p = BeautifulSoup(str(p), 'xml')
        stage_tag = soup_p.stage
        if stage_tag is not None:
            stage_tag.decompose()
        play_texts[title] += p.get_text() + ' '
    for l in soup.find_all('l'):
        soup_l = BeautifulSoup(str(l), 'xml')
        stage_tag = soup_l.stage
        if stage_tag is not None:
            stage_tag.decompose()
        play_texts[title] += l.get_text() + ' '

print(play_texts)
