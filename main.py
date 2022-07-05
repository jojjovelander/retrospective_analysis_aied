import collections
import lda
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import rcParams
from sklearn.feature_extraction.text import CountVectorizer
from wordcloud import STOPWORDS

stopwords = STOPWORDS
import re
from collections import Counter


file1 = open("CC1C.txt", "r", encoding="ISO-8859-1")
file2 = open("stopwordlist.txt", "r")
data1 = file1.read()
data2 = file2.read()
data3 = data1.lower()


words_all = re.findall("[A-Za-z]+", data3)
stop_words = re.findall("[A-Za-z]+", data2)
words_nostop = [word for word in words_all if word not in stop_words]
print("The top 50 most frequent words is:")
for i in Counter(words_nostop).most_common(50):
  print(i[0] + " : " + str(i[1]))

counted_words = collections.Counter(words_nostop)
words = []
counts = []
for letter, count in counted_words.most_common(50):
    words.append(letter)
    counts.append(count)
colors = cm.rainbow(np.linspace(0, 1, 20))
rcParams['figure.figsize'] = 20, 10
plt.title('Top words and their count')
plt.xlabel('Count')
plt.ylabel('Words')
plt.barh(words, counts, color=colors)
plt.show()
text = words_nostop


vec = CountVectorizer(analyzer='word', ngram_range=(1,1))
X = vec.fit_transform(text)
c = vec.get_feature_names()
model = lda.LDA(n_topics=10, random_state=1)
model.fit(X)
topic_word = model.topic_word_
vocab = vec.get_feature_names()
n_top_words = 2
for i, topic_dist in enumerate(topic_word):
    topic_words = np.array(vocab)[np.argsort(topic_dist)][:-(n_top_words+1):-1]
    print('Topic {}: {}'.format(i, ' '.join(topic_words)))

