## импорты
import re,codecs
import json
from lxml import etree
from os import walk
from pymystem3 import Mystem
m = Mystem()

## определяем input -- output
infolder = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
           'Programming/github desktop/RusDraCor/Ira_Scripts/' \
           'TopicModelling/tei_without_proper_names'
results = '/Users/IrinaPavlova/Desktop/Uni/Бакалавриат/2015-2016/' \
          'Programming/github desktop/RusDraCor/Ira_Scripts/TopicModelling/speech_corpus'
bycharacter = results+'/bycharacter/' #'/bycharacter'
bysex = results+'/bysex/'
#outfilename = 'stats_per_play_with_dirtext_and_more_stats_07_02_2018.csv'
#outfile = codecs.open (results+outfilename, 'w', 'utf-8')
ns = {'tei': 'http://www.tei-c.org/ns/1.0','xml':'http://www.w3.org/XML/1998/namespace'}
#header = ['play name','written','print', 'number of stages','len_stages_words','len_speeches_words','total_number_of_verbs','verbs set','number of unique verbs','stages_to_speeches_ratio','verb_diversity(uniqueverbs/totalverbs)','all_stage_texts']

def getnameandauthor (root):
    name = (re.sub('\r|\n','',root.find('.//tei:titleStmt/tei:title', ns).text))
    author = (re.sub('\r|\n|,','',root.find('.//tei:titleStmt/tei:author', ns).text))
    return name, author


def getwhen(date):
    if 'when' in date.attrib:
        return date.attrib ['when']
    else:
        return 'not specified in tei'

def getdates (root):
    written = getwhen(root.find('.//tei:bibl/tei:bibl/tei:date[@type="written"]', ns))
    printdate = getwhen(root.find('.//tei:bibl/tei:bibl/tei:date[@type="print"]', ns))
    return (written, printdate)
    #return (.text)

def checknotmult(speaker):
    if re.search (' #', speaker) != None:
        return False
    else:
        return True

def getsex (speaker, root):
    #print (speaker)
    thisperson = root.find('.//tei:listPerson/tei:person[@xml:id="'+speaker+'"]',ns) #', ns)
    if 'sex' in thisperson.attrib:
        return (thisperson.attrib['sex'])
    else:
        return 'undefined'
    #return root.find('.//tei:listPerson/tei:person[@id="'+speaker+'"]/@sex', ns)

def getspeakerandsex(somesp, root):
    speaker = 'unknown'
    sex = 'undefined'
    notmult = False
    if 'who' in somesp.attrib:
        speaker = somesp.attrib['who']
        notmult = checknotmult(speaker)
        speaker = re.sub('#','',speaker)
        if notmult:
            sex = getsex (speaker, root)
    return speaker, sex, notmult

def getverbs(text):
    verbs = set()
    verbcounter = 0
    analysis = m.analyze(text) #json.dumps(m.analyze(text)) # , ensure_ascii=False, encoding='utf-8'
    for word in analysis:
        #print (word)
        if 'analysis' in word:
            if len (word['analysis'])>0:
                word_analysis = word['analysis'][0]
            #print (word_analysis)
                lemma = word_analysis['lex']
                gram = word_analysis ['gr'].split (',')
                if gram[0] == 'V':
                    verbcounter+=1
                    #print (text)
                    #print (lemma)
                    verbs.add(lemma)
    return verbs, verbcounter

def settostring (someset):
    string =''
    for item in someset:
        string+= item+','
    return string

def parse_xml(path, filename):
    fullpath = path+'/'+filename
    root = etree.parse(fullpath)
    name, author = getnameandauthor (root)
    print(name, author)
    for sp in root.iterfind('.//tei:sp', ns):
        speaker, speakersex, notmult = getspeakerandsex (sp, root)
        if notmult:
            speechtext = ' '.join(sp.xpath('.//tei:p/text()|.//tei:l/text()',namespaces={'tei': 'http://www.tei-c.org/ns/1.0'}))
            speechtext = ' '.join(m.lemmatize(speechtext))
            speechcode = '_'.join([speakersex,author,name,speaker])
            textbyplay = codecs.open(results+'/byplay/byplay/'+name+'.txt','a','utf-8')
            textbyauthor = codecs.open(results+'/byauthor/'+author+'.txt','a','utf-8')
            textbycharacter = codecs.open(bycharacter+speechcode+'.txt','a','utf-8')
            textbygender = codecs.open(bysex+speakersex+'.txt','a','utf-8')
            textbyplay.write(speechtext)
            textbyauthor.write(speechtext)
            textbycharacter.write(speechtext)
            textbygender.write(speechtext)
            textbyplay.close()
            textbyauthor.close()
            textbycharacter.close()
            textbygender.close()


for path, dirs, filenames in walk (infolder):
    for filename in filenames:
        if '.xml' in filename:
            print (filename)
            play = parse_xml (path,filename)

print ('DONE PARSING')
