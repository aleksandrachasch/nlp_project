import pandas as pd
from modules.sentiment_analysis import sentiment_analysis
from modules.collocation_extraction import collocation_extraction
from modules.w2v_word_extraction import w2v_word_extraction
from modules.tf_idf_word_extraction import tf_idf_word_extraction

# first block

bigrams, trigrams = collocation_extraction()

sa_b_df = sentiment_analysis(bigrams)
sa_t_df = sentiment_analysis(trigrams)

# second block

w2v_df = w2v_word_extraction()

# result compilation

final_df = pd.concat([sa_b_df, sa_t_df, w2v_df])
final_df.drop_duplicates()

print (final_df)

final_df.to_csv('sentiment_collocations_and_words.csv')
