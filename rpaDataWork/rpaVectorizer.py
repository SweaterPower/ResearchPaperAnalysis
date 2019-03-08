class rpaVectorizer(object):
    """векторизация текста
        пока только tf-idf"""

    import nltk
    import sklearn
    import pathlib
    import os
    import re
    import pandas as pd

    text_files = []
    locale = 'russian'
    encoding='utf-8'
    stopwords = nltk.corpus.stopwords.words(locale)
    #nltk.download()
    #nltk.download('punkt')

    def get_text_from_file(self, filename):
        textdata = []
        file = open(filename, mode='r', encoding=self.encoding)
        text = file.read()
        textdata.append(file.name.split('.')[0])
        textdata.append(text)
        self.text_files.append(textdata)
        return textdata

    def get_texts_from_folder(self, folder):
        from pathlib import Path
        p = Path(folder)
        #texts = []
        for file in p.iterdir():
            #texts.append(self.get_text_from_file(file))
            self.get_text_from_file(file)
        #return texts
        return self.text_files

    def get_texts_from_folders(self, folders):
        #texts = []
        for folder in folders:
            #texts.append(self.get_texts_from_folder(folder))
            self.get_texts_from_folder(folder)
        #return texts
        return self.text_files
    
    def tokenize_and_stem(self, text):
        from nltk.stem.snowball import RussianStemmer
        
        stemmer = RussianStemmer()
        # first tokenize by sentence, then by word to ensure that punctuation
        # is caught as it's own token
        tokens = [word for sent in self.nltk.sent_tokenize(text, self.locale) for word in self.nltk.word_tokenize(sent, self.locale)]
        #tokens = [word for word in self.nltk.word_tokenize(text[1])]#, language=self.locale)
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens,
        # raw punctuation)
        for token in tokens:
            if self.re.search('[a-zA-Zа-яА-Я]', token):
                if token not in self.stopwords:
                    filtered_tokens.append(token)
        stems = [stemmer.stem(t) for t in filtered_tokens]
        return stems


    def tokenize_only(self, text):
        
        # first tokenize by sentence, then by word to ensure that punctuation
        # is caught as it's own token
        tokens = [word for sent in self.nltk.sent_tokenize(text[1]) for word in self.nltk.word_tokenize(sent)]
        #tokens = [word for word in self.nltk.word_tokenize(text[1])]#, language=self.locale)
        filtered_tokens = []
        # filter out any tokens not containing letters (e.g., numeric tokens,
        # raw punctuation)
        for token in tokens:
            if self.re.search('[a-zA-Z]', token):
                filtered_tokens.append(token)
        return filtered_tokens

    def vectorize_text_tfidf(self, text_files):
        '''
        totalvocab_stemmed = []
        totalvocab_tokenized = []
        for i in text_files:
            allwords_stemmed = self.tokenize_and_stem(i) #for each item in 'synopses', tokenize/stem
            totalvocab_stemmed.extend(allwords_stemmed) #extend the 'totalvocab_stemmed' list
            allwords_tokenized = self.tokenize_only(i)
            totalvocab_tokenized.extend(allwords_tokenized)

        #vocab_frame = self.pd.DataFrame({'words': totalvocab_tokenized}, index = totalvocab_stemmed)
        #print('there are ' + str(vocab_frame.shape[0]) + ' items in vocab_frame')
        '''
        from sklearn.feature_extraction.text import TfidfVectorizer
        #define vectorizer parameters
        tfidf_vectorizer = TfidfVectorizer(max_df=0.9, max_features=200000,
                                        min_df=0.1, 
                                        use_idf=True, tokenizer=self.tokenize_and_stem, ngram_range=(1,3))
        only = [row[1] for row in text_files]
        tfidf_matrix = tfidf_vectorizer.fit_transform(only)
        #return tfidf_matrix;
        terms = tfidf_vectorizer.get_feature_names()
        return [tfidf_matrix, terms]

    #def vectorize_text_tfidf(self):
    #    self.vectorize_text_tfidf(self.text_files)