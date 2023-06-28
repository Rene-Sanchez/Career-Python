import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from wordcloud import WordCloud, STOPWORDS, ImageColorGenerator

data = pd.read_csv('nps_comments.csv')

c2021 = data[data['YEAR']==2021 ]
c2021 = c2021[ c2021['NPS_GROUP_NAME'] == 'Detractor']
c2021['SURVEY_COMMENTS'] = c2021['SURVEY_COMMENTS'].astype(str)

text = " ".join(review for review in c2021.SURVEY_COMMENTS)

stopwords = set(STOPWORDS)
stopwords.update(["wedding", "tux", "event", "groom", "everything",
                  "got","able","well","told","Black","go","still",
                  "rental","took","will","use","getting","way",
                  "went","one","day","t","company","never","anyone",
                  "send","week","great","didn","know","wasn","wear",
                  "email","even","needed","sent","person","recommend",
                  "back","guy","need","someone","days","received",
                  "people","u","tuxedo","time"])

wordcloud = WordCloud(stopwords=stopwords,max_font_size=70,width=800, height=400,
                      max_words=50, background_color="white").generate(text)


plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.show()

result = pd.DataFrame.from_dict(wordcloud.words_ ,orient='index',columns=['value'])
