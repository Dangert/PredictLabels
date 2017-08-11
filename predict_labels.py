"""
Algorithms for Computational Linguistics - Final Task
Tel Aviv Univerity, course no. 0627.2235, Spring 2017.

An implementation of syntax decider.

@author: <Dan Gertzovsky>
"""

from collections import defaultdict
from functools import lru_cache
import string
import re
import sys
from random import choice
import os

class NGram:
    def __init__(self, n):
        """
        @param n: The size of the NGram.
        @type n: int.
        """
        self.n = n
        self.vocabulary = set()
        self.common_words = set()
        self.occur_dict = {}

    def train(self, corpus):
        """
        @param corpus: Text to train the structure with.
        @type n: str.
        The function "trains" the structure according to the corpus.
        Sentences are reprocessed and only common words remain the same.
        """
        corpus = [token for sentence in self.preprocess(corpus) for token in sentence]
        self.corpus_size = len(corpus)
        bound = self.corpus_size*(1/1600)
        #get occurrences of all words in corpus
        for word in corpus:
            self.vocabulary.add(word)
            if word in self.occur_dict:
                self.occur_dict[word] += 1
            else:
                self.occur_dict[word] = 1
        #words with number of occurences above the bound are defined as 'common words'
        for word in self.occur_dict:
            if self.occur_dict[word] >= bound:
                self.common_words.add(word)
        #every non-common word in corpus is replaced with '__'
        for i in range(len(corpus)):
            if corpus[i] not in self.common_words:
                corpus[i] = "__"
        #get occurrences of all n-grams in corpus
        self.ngrams = defaultdict(int) #values of dict are of type int
        for i in range(self.corpus_size-self.n+1): #for every possible n-gram
            nGram = tuple(corpus[i:i+self.n])
            #to save memory - don't keep unnecessary n-grams:
            #save n-grams within the same sentence
            if (("<start>" not in nGram) or ("<end>" not in nGram)):
                self.ngrams[nGram] += 1
            #save n-grams of the form "<start> ... <end>". (though there could still be a smaller sentence in between)
            elif ((nGram[0] == "<start>") and (nGram[self.n-1] == "<end>")):
                self.ngrams[nGram] += 1
    
    @lru_cache(maxsize=None)
    def get_conditional_probability(self,sequence):
        """
        @param sequence: a sequence of words.
        @type sequence: str.
        The function calculates the probability of the sequence to appear in corpus.
        @return: float representing the probability.
        """
        seq_occurrences = self.ngrams[sequence]
        return float(seq_occurrences+1)/(self.corpus_size + len(self.vocabulary))

    def preprocess(self, text):
        """
        @param text: a sequence of words.
        @type text: str.
        The function "cleans" the text, makes it uniform and divides it to sentences. 
        @return: list of sentences post processing.
        """
        #convert to lowercase and remove multiple whitespaces
        text = ' '.join(text.lower().split())
        #replace all possible sentence endings with 'END' (won't be in text because all chars are lowercase)
        text = re.sub(r'(?!)\s*', 'END', text)
        text = re.sub(r'(?:\?|\.)\s*', 'END', text)
        text = re.sub(r'(!)\s*', 'END', text)
        #remove other punctuations
        text = re.sub(r'-+', ' ', text)
        text = text.translate(str.maketrans('', '', string.punctuation))
        #replace numbers with <num>
        text = re.sub(r'[0-9]+', "<num>", text)
        #remove whitespaces from beginning and end of text (if exist)
        text = text.strip()
        #divide text to sentences (according to 'END' strings)
        sentences = text.split('END')
        lst_of_sentences = [("<start> " + s + " <end>").split() for s in sentences]
        return lst_of_sentences

    def check_sentence(self, string):
        """
        @param string: a sequence of words.
        @type string: str.
        The function takes a new sentence and checks its probability in corpus.
        The probability is a multiplication of every consecutive inner n-gram's probability in sentence.
        @return: float representing the probability.
        """
        #preprocess sentence
        sentence = self.preprocess(string)[0]
        #replace non-common words (of corpus) in sentence with '__'
        for i in range(len(sentence)):
            if sentence[i] not in self.common_words:
                sentence[i] = "__"
        #calculate probability of sentence in corpus
        prob = 1
        for i in range(len(sentence)-1): #check all consecutive n-grams in sentence
            curr_ngram = tuple(sentence[i:i+self.n])
            if len(curr_ngram)==self.n: #making sure n-gram is of size n
                boolean = False
                for word in curr_ngram:
                    if word in (self.common_words-{"__", "<end>", "<start>"}):
                        boolean = True
                if boolean: #sentences consist of only '__'/'<end>'/'<start>' won't be put in calculation
                    prob = prob*(self.get_conditional_probability(curr_ngram))
        return prob



"""
Program Activation
"""
#extract paths
script_dir = os.path.dirname(sys.argv[0])
training_good_path = os.path.join(script_dir, 'good.txt')
training_bad_path = os.path.join(script_dir, 'bad.txt')
#read corpora
good_corpus = open(training_good_path,'r',encoding='utf8').read()
bad_corpus = open(training_bad_path,'r',encoding='utf8').read()
#create NGram for each corpus
good_ngram = NGram(6)
bad_ngram = NGram(6)
#train each NGram according to its corpus
good_ngram.train(good_corpus)
bad_ngram.train(bad_corpus)

output_file = open(sys.argv[2],'w')
#read each line of input file and decide
with open(sys.argv[1], 'r') as input_file:
    for line in input_file:
        if (good_ngram.check_sentence(line)>bad_ngram.check_sentence(line)): #probably syntactic
            output_file.write("1\n")
        elif (good_ngram.check_sentence(line)<bad_ngram.check_sentence(line)): #probably not syntactic
            output_file.write("0\n")
        else: #if can't decide, decide randomly
            output_file.write(choice(["0\n","1\n"]))
output_file.close()
