import rpaVectorizer
import scipy.sparse

vectorizer = rpaVectorizer.rpaVectorizer()
#texts =
#vectorizer.get_texts_from_folders([r'C:\Users\swite\Desktop\ВКР2018_text\diplomas_text\2018_ИМИКН_ИБ_ОДО',
                                  #r'C:\Users\swite\Desktop\ВКР2018_text\diplomas_text\2018_ИМИКН_ИБАС_ОДО'])
texts = vectorizer.get_texts_from_folders([r'C:\Users\swite\Desktop\ВКР2018_text\diplomas_text\2018_ИМИКН_ИБ_ОДО'])
tfidf = vectorizer.vectorize_text_tfidf(texts)
print(tfidf[1])
limit = 0.07
tfidf_dence = tfidf[0].toarray()
docindex = 0
termindex = 0
for doc in tfidf_dence:
    print(texts[docindex][0] + " :\n")
    docindex += 1
    for term in doc:
        if term > limit:
            print("     " + str(tfidf[1][termindex]) + " - " + str(term) + " ")
        termindex += 1
    termindex = 0;