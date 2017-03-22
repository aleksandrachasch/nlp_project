#coding:utf-8

import logging
import gensim
import re
import os
import pandas as pd
from lxml import etree
from nltk.tokenize import TreebankWordTokenizer
from nltk.corpus import stopwords

def w2v_word_extraction():

    def parsing_xml(name):
        with open(name, encoding='utf-8') as f:
            xml = f.read()

        dict = {}
        text = []
        category = []
        sentiment = []
        term = []

        root = etree.fromstring(xml)
        for child in root:
            for aspect in child[3]:
                if aspect.attrib['type'] == 'implicit':
                    text.append(child[2].text)
                    category.append(aspect.attrib['category'])
                    sentiment.append(aspect.attrib['sentiment'])
                    term.append(aspect.attrib['term'])

        dict['text'] = text
        dict['category'] = category
        dict['sentiment'] = sentiment
        dict['term'] = term

        df = pd.DataFrame(dict)
        return df

    def preprocessing(df):
        stop_words = set(stopwords.words('russian'))

        def clear_punct(text):
            possible = 'ёйцукенгшщзхъфывапролджэячсмитьбю- '
            text = [symbol for symbol in text.lower() if symbol in possible]
            return ''.join(text)

        texts = df.drop_duplicates(['text'])['text']
        cleared = texts.apply(clear_punct)
        tokenized = cleared.apply(TreebankWordTokenizer().tokenize)

        w = open('sents.txt', 'w', encoding='utf-8')
        for sent in tokenized:
            w.write(' '.join(sent) + '\n')

        os.system('mystem -cld sents.txt lemmatized.txt')

        a = open('sentences.txt', 'w', encoding='utf-8')
        with open('lemmatized.txt', 'r', encoding='utf-8') as f:
            sentences = []
            for line in f:
                words = re.findall('{(.+?)\??}', line)
                sentence = [word for word in words if word not in stop_words]
                a.write(' '.join(sentence))

    def create_our_model():
        logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

        data = gensim.models.word2vec.LineSentence('sentences.txt')
        model = gensim.models.Word2Vec(data, size=1000, window=3, min_count=1, sg=1)
        model.init_sims(replace=True)
        model.save('our_model')

    def test_model(model_name):
        est_pos = ['хороший_ADJ', 'вкусный_ADJ', 'замечательный_ADJ', 'приятный_ADJ', 'красивый_ADJ', 'отличный_ADJ']
        est_neg = ['плохой_ADJ', 'ужасный_ADJ', 'худший_ADJ', 'неприятный_ADJ']

        web_model = gensim.models.KeyedVectors.load_word2vec_format(model_name, binary=True)
        # our_model = gensim.models.Word2Vec.load('model')
        web_model.init_sims(replace=True)

        def output_similar(estimators, model):
            els = []
            for el in estimators:
                try:
                    if el in model.wv.vocab:
                        commons = model.most_similar(positive=el, topn=20)
                        els += [common[0] for common in commons]
                except:
                    if el in model.vocab:
                        commons = model.most_similar(positive=el, topn=20)
                        els += [common[0] for common in commons]
            return els

        poses = output_similar(est_pos, web_model)
        poses = [[re.sub('_.+', '', pos), 'positive'] for pos in set(poses)]
        print('POSITIVE\n', poses)

        negs = output_similar(est_neg, web_model)
        negs = [[re.sub('_.+', '', neg), 'negative'] for neg in set(negs)]
        print('NEGATIVE\n', negs)

        final_df = pd.DataFrame(negs+poses, columns = ['collocation', 'polarity'])
        return final_df

    df1 = parsing_xml('SentiRuEval_rest_markup_train.xml')
    df2 = parsing_xml('SentiRuEval_rest_markup_test.xml')
    df = pd.concat([df1, df2])
    preprocessing(df)
    # create_our_model()
    test_model('web_0_300_20.bin')

if __name__ == '__main__':
    w2v_word_extraction()
