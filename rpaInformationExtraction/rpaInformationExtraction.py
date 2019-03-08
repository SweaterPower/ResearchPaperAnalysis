import nltk, re, pprint

document = ''
sentences = nltk.sent_tokenize(document, 'russian') [1]
sentences = [nltk.word_tokenize(sent, 'russian') for sent in sentences] [2]
sentences = [nltk.pos_tag(sent, lang='rus') for sent in sentences] [3]

sentence = sentence[1]
grammar = "NP: {<DT>?<JJ>*<NN>}"
cp = nltk.RegexpParser(grammar)
result = cp.parse(sentence)
print(result)
  