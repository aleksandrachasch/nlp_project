# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
from lxml import etree

import pymorphy2
from nltk.tokenize import TreebankWordTokenizer
from stop_words import get_stop_words

import gensim

import matplotlib.pyplot as plt

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.base import TransformerMixin
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, f1_score, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn import cross_validation

"""
- Обучение:
Вытащить из корпуса все positive / negative слова
Векторизовать каждое слово с помощью word2vec
Написать простой классификатор (вектор - класс)

- Тестирование:
На вход поступает слово
Вектурезуем его с помощью word2vec и делаем predict

Краткий анализ результатов:
лучшие модели:
- dt при ruscorpora и .sample(frac=1)[:150]
- rt при web и .sample(frac=1)[:150]
- etx при web и .sample(frac=1)[:300]
"""

# Обученная модель из RusVectores
m = 'web_0_300_20.bin' #'ruscorpora_1_300_10.bin'  #

# Тексты для разметки
# collocations_array = ['отвратительный', 'быстро', 'очень плохое обслуживание', 'отличное меню']
collocations_array = ['отличный выбор', 'не советуем', 'очень советуем', 'очень дорого', 'выше всяких похвал', 'в общем прекрасно', 'нам все понравилось', 'в целом ничего', 'отвратительный', 'быстро', 'очень плохое обслуживание', 'отличное меню', 'хороший', 'вкусный', 'замечательный', 'приятный', 'красивый', 'отличный']


# Ввести правильные ответы
true = {'отличный выбор': 'positive',
'не советуем': 'negative',
'очень советуем': 'positive',
'очень дорого': 'negative',
'выше всяких похвал': 'positive',
'в общем прекрасно': 'positive',
'нам все понравилось': 'positive',
'в целом ничего': 'positive',
'отвратительный': 'negative',
'быстро': 'positive',
'очень плохое обслуживание': 'negative',
'отличное меню' : 'positive',
'хороший' : 'positive',
'вкусный' : 'positive',
'замечательный' : 'positive',
'приятный' : 'positive',
'красивый' : 'positive',
'отличный' : 'positive'}


morph = pymorphy2.MorphAnalyzer()
tokenizer = TreebankWordTokenizer()

RUS_LETTERS = u'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'

# Для POS в cleanization
transit = {'ADJF': 'ADJ',
		   'ADJS': 'ADJ',
		   'ADVB': 'ADV',
		   'COMP': 'ADV',
		   'CONJ': 'CCONJ',
		   'GRND': 'VERB',
		   'INFN': 'VERB',
		   'INTJ': 'INTJ',
		   'LATN': 'X',
		   'NOUN': 'NOUN',
		   'NPRO': 'PRON',
		   'NUMB': 'NUM',
		   'NUMR': 'NUM',
		   'PNCT': 'PUNCT',
		   'PRCL': 'PART',
		   'PRED': 'ADV',
		   'PREP': 'ADP',
		   'PRTF': 'ADJ',
		   'PRTS': 'VERB',
		   'ROMN': 'X',
		   'SYMB': 'SYM',
		   'UNKN': 'X',
		   'VERB': 'VERB'}

robj = re.compile('|'.join(transit.keys()))


def parse_xml(filename):
	""" Парсинг входного корпуса"""
	with open(filename, encoding='utf-8') as f:
		xml = f.read()

	dict = {}
	text = []
	category = []
	sentiment = []
	term = []

	root = etree.fromstring(xml)
	for child in root:
		for aspect in child[3]:
			if aspect.attrib['type'] == 'implicit' and aspect.attrib['sentiment']!= 'both' and aspect.attrib['sentiment']!= 'neutral':
				text.append(child[2].text)
				category.append(aspect.attrib['category'])
				sentiment.append(aspect.attrib['sentiment'])
				term.append(aspect.attrib['term'])

	dict['text'] = text
	dict['category'] = category
	dict['sentiment'] = sentiment
	dict['term'] = term
	
	return dict


def cleanization(text):
	"""Функция приводит входной текст в формат лемматизированное слово_POS"""
	for line in text:
		# 1. Все буквы в нижний регистр
		text_text = text.lower()

		# 2. Удаление всех небукв
		letters_only = ''
		for _c in text_text:
			if _c in RUS_LETTERS:
				letters_only += _c
			else:
				letters_only += ' '

		# 3. Заменяем множественные пробелы
		while '  ' in letters_only:
			letters_only = letters_only.replace('  ', ' ')

		# 4. Токенизация
		word_list = tokenizer.tokenize(letters_only)

		# 5. Лемматизация
		clean_word_list = [morph.parse(word)[0].normal_form for word in word_list]  # лемматизация

		# 6. * Удаление стоп-слов + добавление тегов - части речи
		# meaningful_words = [word for word in clean_word_list if word not in get_stop_words('ru')] # стоп-слова
		meaningful_words = [
			str(word) + '_' + robj.sub(lambda m: transit[m.group(0)], str(morph.parse(word)[0].tag.POS)) for word in
			clean_word_list]
		return ' '.join(meaningful_words)


def mean(a):
	return sum(a) / len(a)


def word2vec_mean(text):
	"""Усредняет вектор слов."""
	arr = []
	clean_text = cleanization(text)
	# для каждого слова в тексте выводим его вектор
	for word in clean_text.split(' '):
		# есть ли слово в модели? Может быть, и нет
		if word in model:
			arr.append(model[word])
	if len(list(map(mean, zip(*arr)))) != 0:
		return list(map(mean, zip(*arr)))
	else:
		return [0 for i in range(0, 300)]


class FunctionFeaturizer(TransformerMixin):
	""" Для создания своего вектора я использовала усредненную векторизацию с помощью word2vec"""

	def __init__(self):
		pass

	def fit(self, X, y=None):
		return self

	def transform(self, X):
		fvs = []
		# fvs = word2vec_mean(X)  # если подавать по словам, а не датафрейм
		for datum in X:
			fv = word2vec_mean(datum)
			fvs.append(fv)
		return np.array(fvs)


if __name__ == '__main__':

	text_train = parse_xml('SentiRuEval_rest_markup_train.xml')
	text_test = parse_xml('SentiRuEval_rest_markup_test.xml')

	# Создаем датафрейм из тестового и тренировочного корпуса
	df1 = pd.DataFrame(text_train)
	df2 = pd.DataFrame(text_test)
	frames = [df1, df2]
	df = pd.concat(frames)

	# Делаем датасет сбалансированным
	df = pd.concat([df[df['sentiment'] == 'positive'].sample(frac=1)[:150], df[df['sentiment'] == 'negative']]).sample(frac=1)  # ЗАМЕНА

	# Загружаем модель
	model = gensim.models.KeyedVectors.load_word2vec_format(m, binary=True)
	model.init_sims(replace=True)

	# Делим корпус на тестовый и тернировочный
	X_train, X_test, y_train, y_test = train_test_split(df['term'], df['sentiment'], test_size=0.1)


	def do_smth_with_model(data_train, class_train, data_test, class_test, steps):
		"""Функция получает на взох данные и параметры для pipeline и печатает
		результаты работы обучающей модели на тестовой выборке + возвращает pipeline"""
		print('\nModel train')
		pipeline = Pipeline(steps=steps)

		cv_results = cross_val_score(pipeline,
									 data_train,
									 class_train,
									 cv=10,
									 scoring='accuracy',
									 )
		print(cv_results.mean(), cv_results.std())

		pipeline.fit(data_train, class_train)

		class_predicted = pipeline.predict(data_test)
		print(class_predicted)

		print(classification_report(class_test, class_predicted))

		return pipeline, class_predicted


	w2v_featurizer = FunctionFeaturizer()  # создание своего векторизатора

	# Word2Vec + LogisticRegression
	print('\nCustom Transformer + LogisticRegression')
	lr_pipeline, label_predicted = do_smth_with_model(X_train, y_train,
													  X_test, y_test,
													  steps=[('custom', w2v_featurizer),
															 ('classifier', LogisticRegression())])

	# Word2Vec + ExtraTreesClassifier
	print('\nCustom Transformer + ExtraTreesClassifier')
	etx_pipeline, label_predicted = do_smth_with_model(X_train, y_train,
													   X_test, y_test,
													   steps=[('custom', w2v_featurizer),
															  ('classifier', ExtraTreesClassifier())])

	# Word2Vec + RandomForestClassifier
	print('\nCustom Transformer + RandomForestClassifier')
	rf_pipeline, label_predicted = do_smth_with_model(X_train, y_train,
													   X_test, y_test,
													   steps=[('custom', w2v_featurizer),
															  ('classifier', RandomForestClassifier())])

	# Word2Vec + DecisionTreeClassifier
	print('\nCustom Transformer + DecisionTreeClassifier')
	dt_pipeline, label_predicted = do_smth_with_model(X_train, y_train,
													   X_test, y_test,
													   steps=[('custom', w2v_featurizer),
															  ('classifier', DecisionTreeClassifier())])

	# Проверка работы модели на наших тестовых коллокациях
	def predictor(collocations_array, pipeline):
		mistakes = 0
		arr = []
		df1 = pd.DataFrame({'text': collocations_array})
		for i in df1.text:
			arr.append(i)
		с = 0
		for i in pipeline.predict(df1.text):
			print(arr[с], ':', i)
			if true[arr[с]] != i:
				mistakes += 1
			с += 1
		print(mistakes)


	# ВВЕДИТЕ СЛОВА, КОТОРЫЕ ХОТИТЕ ПРОВЕРИТЬ
	predictor(collocations_array, etx_pipeline)
	print('_'*30)
	predictor(collocations_array, lr_pipeline)
	print('_'*30)
	predictor(collocations_array, rf_pipeline)
	print('_'*30)
	predictor(collocations_array, dt_pipeline)


