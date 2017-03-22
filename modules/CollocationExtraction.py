# -*- coding: utf-8 -*-
"""
Created on Wed Mar 22 22:57:47 2017

@author: Aleksandra 
"""
from lxml import etree
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import math
from nltk.collocations import BigramAssocMeasures, TrigramAssocMeasures, BigramCollocationFinder, TrigramCollocationFinder

class CollocationExtraction:

    def __init__(self):
        self.corpus_xml = []
        self.contrast = ''
        self.corpus = ''
        self.corpus_tokens = []
        self.contrast_tokens = []
        self.bigrams = []
        self.trigrams = []
        
       
    #parse corpus of restaurants in XML file
    #return set of reviews
    def parse_xml(self):
        
        def one_xml(xmltext):
            with open(xmltext, encoding='utf-8') as f:
                xml = f.read()
                f.close()
            d = {}
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
    
            d['text'] = text
            d['category'] = category
            d['sentiment'] = sentiment
            d['term'] = term
            
            return set(text)
        
        text1 = one_xml(self.corpus_xml[0])
        text2 = one_xml(self.corpus_xml[1])
         
        all_text = ''
        for text in text1:
            all_text += text + ' '

        for text in text2:
            all_text += text + ' '
    
        return all_text
    
        #tokenize and clean the corpora
    
    def preprocess(self, text):
        latin = re.compile('[a-zA-Z]+?')
        stop = set(stopwords.words('russian'))
        tmp = word_tokenize(text)
        tokens = []
        for i in tmp:
            if i.isalnum() and i not in stop and not i.isdigit():
                m = re.search(latin, i)
                if m == None:
                    tokens.append(i.lower())
        return tokens
    
    def import_corpus(self, filenames):
        self.corpus_xml = filenames
        self.corpus = self.parse_xml()
        self.corpus_tokens = self.preprocess(self.corpus)
           
    def import_contrast_corpus(self, filename):
        f = open(filename, 'r' , encoding='utf-8')
        self.contrast = f.read()
        self.contrast_tokens = self.preprocess(self.contrast)
        f.close()
    
     
    #concatenate two lists of ngrams into a list of two lists of bigrams (corpus of interest + contrast corpus)
    #this is needed for idf 
    def generate_corpus(self, corpus_fd, contrast_corpus_fd):
    
        result = []
        
        resto = []
        for w in corpus_fd:
            if corpus_fd[w] > 1:
                n = 0
                while n < corpus_fd[w]:
                    resto.append(w)
                    n += 1
                else:
                    resto.append(w)
         
        contrast = []
        for w2 in contrast_corpus_fd:
            if contrast_corpus_fd[w2] > 1:
                n2 = 0
                while n2 < contrast_corpus_fd[w2]:
                    contrast.append(w2)
                    n2 += 1
                else:
                    contrast.append(w2)
        
        result.append(resto)
        result.append(contrast)
        return result
    
    
    #compute tf
    #fd  = FreqDist with collocations (from BigramCollocationFinder())
    #textlen = number of all collocations found in corpora
    
    def compute_tf(self, fd, textlen):
        res = {}
        for i in fd:
            res[i] = fd[i]/textlen
        return res
    
    #compute idf
    #ngram = ngram to be analyzed
    # corpus = all collocations of restaurant corpus + contrast corpus
    
    
    def compute_idf(self, ngram, corpus):
        return math.log10(len(corpus)/sum([1.0 for text in corpus if ngram in text]))
    
    
    #compute TF-IDF of the corpus of interest
    def compute_tf_idf(self, fd, textlen, corpus):
        tf = self.compute_tf(fd, textlen)
        
        tf_idf = {}
        for i in tf:
            tf_idf_value = tf[i] * self.compute_idf(i, corpus)
            if tf_idf_value not in tf_idf.keys():
                x = [i]
                tf_idf[tf_idf_value] = x
            else:
                tf_idf[tf_idf_value].append(i)
            
        return tf_idf
    
    def generate_bigrams(self):
    
        finder = BigramCollocationFinder.from_words(self.corpus_tokens)
        resto_len = finder.N
        
        finder_contrast = BigramCollocationFinder.from_words(self.contrast_tokens)
        contrast_len = finder_contrast.N
        
        corpus = self.generate_corpus(finder.ngram_fd, finder_contrast.ngram_fd)
        
        finder.apply_freq_filter(10)
        finder_contrast.apply_freq_filter(10)
        
        bigrams_resto = finder.ngram_fd
        bigrams_contrast = finder_contrast.ngram_fd
        
        scores = self.compute_tf_idf(bigrams_resto , resto_len, corpus)
        
        for i in scores:
            if i != 0.0:
                for bg in scores[i]:
                    tmp = ''
                    for word in bg:
                        tmp += word + ' '
                    self.bigrams.append(tmp)
    
    
    def generate_trigrams(self):
    
        finder = TrigramCollocationFinder.from_words(self.corpus_tokens)
        resto_len = finder.N
        
        finder_contrast = TrigramCollocationFinder.from_words(self.contrast_tokens)
        contrast_len = finder_contrast.N
        
        corpus = self.generate_corpus(finder.ngram_fd, finder_contrast.ngram_fd)
        
        finder.apply_freq_filter(3)
        finder_contrast.apply_freq_filter(3)
        
        trigrams_resto = finder.ngram_fd
        trigrams_contrast = finder_contrast.ngram_fd
        
        scores = self.compute_tf_idf(trigrams_resto, resto_len, corpus)
        
        for i in scores:
            if i != 0.0:
                for tg in scores[i]:
                    tmp = ''
                    for word in tg:
                        tmp += word + ' '
                    self.trigrams.append(tmp)


if __name__ == '__main__':
    corpora = ['SentiRuEval_rest_markup_test.xml', 'SentiRuEval_rest_markup_train.xml']

    new = CollocationExtraction()
    new.import_corpus(corpora)
    new.import_contrast_corpus('contrast_corpus.txt')

    new.generate_bigrams()
    new.generate_trigrams()

    print(new.bigrams)
    print(new.trigrams)