## Программа анализа тональности оценочного слова/коллокации (sentiment_analysis)

### Тренировочные данные
Размеченный корпус о ресторанах с DialogEvaluation 2015 года
* 'SentiRuEval_rest_markup_train.xml'
* 'SentiRuEval_rest_markup_test.xml'

### Использованные методы
* Модель с RusVectores Word2Vec ('web_0_300_20.bin' и 'ruscorpora_1_300_10.bin')
* Реализация моделей ExtraTreesClassifier, LogisticRegression, RandomForest, DecissionTree в Sklearn 

### Как запускать:
- На вход модулю поступаем массив коллокаций в формате ['отличный выбор', 'не советуем', 'очень советуем', 'очень дорого']
- На выходе датаврейм с двум колонками: collocation и polarity

### Описание работы программы
- Берется модель из  RusVectores и с ее помощью векторизуются terms (векторизуется каждое слово в коллокации и потом берется средний вектор)
- Дальше на этих векторах и их полярностях обучаю разные модели

### Результат
На выходе - результат работы нескольких моделей (ExtraTreesClassifier, LogisticRegression, RandomForest, DecissionTree) + результат классификации в формате выражение:полярность

### Интерпретация результатов:
- 'web_0_300_20.bin' работает лучше, чем 'ruscorpora_1_300_10.bin'
- Лучше делать корпус сбалансированным (позитивных примеров изначально в раз 10 больше, чем негативных)
- При разных параметрах разные модели показывают лучшие результаты
- В целом на тестовой выборке результаты PR и R около 90, но на них нельзя особо опираться, потому что всего корпус 150 + 150 термов, разбит в соотношении 0,1 тестовый к 0,9 обучающий (примеров так мало, что я решила обучать на как можно большем количестве)

### Пример классификации (RandomForest, web model):
 ```
- отличный выбор : positive
- не советуем : negative
- очень советуем : negative
- очень дорого : negative
- выше всяких похвал : negative
- в общем прекрасно : positive
- нам все понравилось : negative
- в целом ничего : positive
- отвратительный : negative
- быстро : positive
- очень плохое обслуживание : negative
- отличное меню : positive
- хороший : positive
- вкусный : positive
- замечательный : positive
- приятный : positive
- красивый : positive
- отличный : positive
 ```


## Программа выделения оценочных слов (w2v_word_extraction)

### Используемые модели

1) Первая модель взята из RusVectores.

- Корпус интернет текстов за декабрь 2014.
- Объем: 900 миллионов слов, 267 540 лемм.
- Минимальная частота: 30
- Используемый алгоритм: Continuous Bag-of-Words
- Размерность вектора: 300

Модель была выбрана по идеалогическому принципу (корпус интернет текстов должен быть ближе к ресторанным отзывам) и сравнена с моделью ruscorpora, которая работает хуже, чем модель веб-корпуса.

2) Вторая модель обучена нами.

- Модель обучена на объединении данных 'SentiRuEval_rest_markup_train.xml' и 'SentiRuEval_rest_markup_test.xml'.
- Размер вектора: 1000
- Ширина окна: 3
- Минимальная частота: 1
- Используемый алгоритм: SkipGram

Модель была выбрана подгоном ключевых параметров с ориентацией на результат.

### Описание программы

Обученная модель берет на вход предложенные нам изначально оценочные слова:

* Positive: хороший, вкусный, замечательный, приятный, красивый, отличный
* Negative: плохой, ужасный, худший, неприятный

И выдает 20 самых близких к ним слов. Таким образом мы расширяем словарь положительных и отрицательных оценочных слов.

### Результаты

Способ оценивания - интроспекция.

Модель, обученная нами, показала гораздо более низкое качество по сравнению с моделью веб-корпуса.
Поэтому было решено остановиться на модели веб-корпуса для выделения оценочных слов.

#### Модель на веб-корпусе
POSITIVE

 ```
['красивый', 'positive'], ['отменный', 'positive'], ['штрудель', 'positive'], ['порадовать', 'positive'], ['обалденный', 'positive'], ['пахлава', 'positive'], ['вкусно', 'positive'], ['плохой', 'positive'], ['веселый', 'positive'], ['достойный', 'positive'], ['интересный', 'positive']
 ```
 
NEGATIVE
 ```
['дерьмовый', 'negative'], ['неэстетичный', 'negative'], ['тягостный', 'negative'], ['паршивый', 'negative'], ['портить', 'negative'], ['омерзительный', 'negative'], ['неподходящий', 'negative'], ['мерзкий', 'negative'], ['приятный', 'negative'], ['вообще', 'negative'], ['малоприятный', 'negative'], ['хреновый', 'negative']
 ```
 
#### Модель, обученная нами
POSITIVE

 ```
['вид', 'positive'], ['советовать', 'positive'], ['красивый', 'positive'], ['кафе', 'positive'], ['спокойный', 'positive'], ['ненавязчивый', 'positive'], ['ролл', 'positive'], ['год', 'positive'], ['банкет', 'positive'], ['целое', 'positive'], ['оказываться', 'positive'], ['закуска', 'positive'], ['отношение', 'positive'], ['обстановка', 'positive'], ['неплохой', 'positive'], ['большой', 'positive'], ['рекомендовать', 'positive']
 ```
 
NEGATIVE

 ```
['взгляд', 'negative'], ['народ', 'negative'], ['удаваться', 'negative'], ['посетитель', 'negative'], ['плохой', 'negative'], ['стоять', 'negative'], ['получаться', 'negative'], ['вообще', 'negative'], ['находиться', 'negative'], ['любить', 'negative'], ['подавать', 'negative'], ['тк', 'negative'], ['сервис', 'negative'], ['сходить', 'negative'], ['уходить', 'negative'], ['понимать', 'negative'], ['особо', 'negative']
 ```
 
## Программа выделения коллокаций
 
### Описание программы
 
Модуль `CollocationExtraction` позволяет выделить из корпуса отзывов о ресторанах наиболее специфичные биграммы и триграммы.

#### Тип входных данных

1) Массив  - корпус отзывов о ресторанах в формате `XML`. 

 `['SentiRuEval_rest_markup_train.xml' , 'SentiRuEval_rest_markup_test.xml']`
 
2) Контрастный корпус в формате 'txt'.
 
   В нашем случае использовался новостной корпус РИА-новости.

#### Инициализация и предобработка

Для того, чтобы выделить коллокации, необходимо инициализировать объект `CollocationExtraction` и импортировать туда корпус:

```
corpora = ['SentiRuEval_rest_markup_test.xml','SentiRuEval_rest_markup_train.xml']
new = CollocationExtraction()
new.import_corpus(corpora)
new.import_contrast_corpus('contrast_corpus.txt')

```

При импортировании корпуса производится предобработка:

- удаление слов, где присутствуют латинские символы,
- удаление стоп-слов (список из модуля `NLTK.corpus`),
- Перевод строчных символов в прописные,
- токенизация,
- удаление знаков пунктуации, цифр.

#### Выделение биграмм и триграмм

Следующие функции выделяют биграммы и триграммы с наиболее высоким `TF-IDF`:

```
new.generate_bigrams()
new.generate_trigrams()

print(new.bigrams)
print(new.trigrams)
```

Выделение биграмм и триграмм происходит в несколько этапов:

1) Находим все возможные биграммы/триграммы при помощи `BigramCollocationFinder()` / `TrigramCollocationFinder()` из `NLTK`;
2) Генерируем "корпус" всех биграмм/триграмм, присутствующих в изначальном корпусе (необходимо для подсчета **TF**);
3) Фильтруем биграммы/триграммы по количеству вхождений, чтоыбы считать TF-IDF только для наиболее частотных биграмм/триграмм (для     биграмм - 10 , для триграмм - 3);
4) Для каждой биграммы/триграммы считаем `TF-IDF`;
5) Выбираем биграммы/триграммы с наиболее высокой мерой `TF-IDF`.


### Результаты

Выделение коллокаций на основе `TF-IDF` меры лучше, чем выделение на каких-то других метриках (`pmi`, `log-likelihood`, `chi-squared`).

Например, выделение триграмм (топ-10 для каждой метрики):

- **pmi:**
```
('оставляет', 'желать', 'лучшего'),
 ('выразить', 'огромную', 'благодарность'),
 ('близко', 'друг', 'другу'),
 ('выше', 'всяких', 'похвал'),
 ('по', 'крайней', 'мере'),
 ('хочу', 'выразить', 'огромную'),
 ('обслуживание', 'оставляет', 'желать'),
 ('милая', 'девушка', 'проводила'),
 ('что', 'касается', 'кухни'),
 ('ресторана', 'такого', 'уровня')
 
 ```

- **likelihood-ratio:**
```
('остались', 'очень', 'довольны'),
 ('в', 'общем', 'целом'),
 ('в', 'общем', 'это'),
 ('в', 'общем', 'очень'),
 ('отмечали', 'день', 'рождения'),
 ('свой', 'день', 'рождения'),
 ('отпраздновать', 'день', 'рождения'),
 ('праздновали', 'день', 'рождения'),
 ('день', 'рождения', 'нашей'),
 ('что', 'могу', 'сказать'),
 ```
 
 - **chi-square:**
 ```
 ('оставляет', 'желать', 'лучшего'),
 ('выше', 'всяких', 'похвал'),
 ('выразить', 'огромную', 'благодарность'),
 ('близко', 'друг', 'другу'),
 ('по', 'крайней', 'мере'),
 ('хочу', 'выразить', 'огромную'),
 ('обслуживание', 'оставляет', 'желать'),
 ('что', 'касается', 'кухни'),
 ('милая', 'девушка', 'проводила')
 ```
 

А вот список триграмм с наиболее высоким `TF-IDF`:

```
('остались', 'очень', 'довольны')
---------------------------
score: 8.27561231894383e-05
('отмечали', 'день', 'рождения')
('выше', 'всяких', 'похвал')
---------------------------
score: 7.356099839061181e-05
('нам', 'очень', 'понравилось')
---------------------------
score: 6.436587359178533e-05
('все', 'очень', 'вкусно')
---------------------------
score: 5.517074879295886e-05
('что', 'касается', 'кухни')
('оставляет', 'желать', 'лучшего')
('обслуживание', 'высшем', 'уровне')
('очень', 'понравился', 'интерьер')
---------------------------
score: 4.5975623994132386e-05
('гости', 'остались', 'очень')
('очень', 'вкусно', 'очень')
('обслуживание', 'оставляет', 'желать')
('всё', 'очень', 'вкусно')
```

Список биграмм по `TF-IDF`:

```
score: 0.0005241221135331092
('в', 'общем')
---------------------------
score: 0.000422975740746018
('очень', 'вкусно')
---------------------------
score: 0.0004137806159471915
('очень', 'понравилось')
---------------------------
score: 0.0003310244927577532
('очень', 'довольны')
---------------------------
score: 0.00030343911836127376
('остались', 'очень')
('могу', 'сказать')
---------------------------
score: 0.00024826836956831486
('это', 'место')
---------------------------
score: 0.0002390732447694884
('молодой', 'человек')
('очень', 'понравился')
---------------------------
score: 0.00022987811997066191
('отдельное', 'спасибо')
---------------------------
```

Список биграмм по другим метрикам (топ-10):

- **pmi:**
```
('всяких', 'похвал'),
 ('молодым', 'человеком'),
 ('выше', 'всяких'),
 ('первом', 'этаже'),
 ('второй', 'этаж'),
 ('что', 'касается'),
 ('высшем', 'уровне'),
 ('живая', 'музыка'),
 ('молодой', 'человек'),
 ('самое', 'главное')
 ```

- **likelihood-ratio:**
```
('в', 'общем'),
 ('день', 'рождения'),
 ('могу', 'сказать'),
 ('молодой', 'человек'),
 ('в', 'целом'),
 ('отдельное', 'спасибо'),
 ('что', 'касается'),
 ('живая', 'музыка'),
 ('всяких', 'похвал'),
 ('молодым', 'человеком')
 ```
 
 - **chi-square:**
 ```
('всяких', 'похвал'),
 ('молодым', 'человеком'),
 ('что', 'касается'),
 ('выше', 'всяких'),
 ('первом', 'этаже'),
 ('второй', 'этаж'),
 ('день', 'рождения'),
 ('молодой', 'человек'),
 ('живая', 'музыка')
 ```
 
Таким образом, `TF-IDF`, кажется, больше подходит для выделения оценочной лексики, чем какие-то другие метрики.
