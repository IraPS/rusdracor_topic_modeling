## This is a repository for the project "Topic Modeling 200 Years of Russian Drama"

### The goal of the project is to perform a TM on 90 Russian plays written from 1747 to 1943 (RusDraCor, https://dracor.org/rus). The plays are encoded in the TEI standard. The algorithms can be reproduced and reapplied to the updated corpus.

Here you can find the data and scripts used in the work:

* [the processed TEI-xml files with excluded proper names of the characters](https://github.com/IraPS/rusdracor_topic_modeling/tree/master/tei_without_proper_names)

* [the script for text-extraction from the TEI-file](https://github.com/IraPS/rusdracor_topic_modeling/tree/master/scripts_for_text_extraction)

* [the stop-words and proper-names lists and the script revoming them](https://github.com/IraPS/rusdracor_topic_modeling/tree/master/stopwords_and_others)

* [the preprocessed corpus of 90 Russian plays](https://github.com/IraPS/rusdracor_topic_modeling/tree/master/corpora)

   - each folder has subfolders **byauthor**, **bycharacter**, **byplay**, **bysex**. 

* [the final version that was used for the project](https://github.com/IraPS/rusdracor_topic_modeling/tree/master/corpora/speech_corpus_no_prop_char_names_ONLY_NOUNS)
   - it also includes subfolders **bygenre** and **byyear_range**

   - checkout the TM (modeling only nouns-based topics) you will need only this folder.

The workflow was organised by following these steps:

| Action          | Description   |
| ------------- |:-------------:|
| stopwords_and_others/extract_capitalised_words.py     | Extracting all capitalised words not in the beginning of a sentence |
| stopwords_and_others/characters(proper)\_names.txt    | Filtering the list to keep only character's proper names      |
| stopwords_and_others/remove_characters(proper)\_names_from_TEI.py | Removing proper names from the TEI documents     |
| scripts_for_text_extraction/get_plays_texts_clean_POS_restriction.py | Extracting characters' speech-texts from the TEI documents with POS restictions (different options available)|
| classification_using_TM_vectors_gender.py | Trying to choose the best model with a character's gender classificaton task |
| semantic_vectors.py | Choosing the best model by calculating "semdensity" of topics |
| topic_modeling_predict_year.py | Applying the model to spot topics' temporal distribution |
| topic_modeling_predict_genre.py | Applying the model to spot topics' distribution by genre |
| topic_modeling_predict_author.py | Applying the model to spot topics' distribution by author |
| topic_modeling_predict_gender.py | Applying the model to spot topics' distribution by character's gender |

 
