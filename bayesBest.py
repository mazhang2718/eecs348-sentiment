# Name: Matthew Zhang - mlz855 
# Date: 5/21/16
# Description: Best Bayes Classifier - uses bigrams as well as unigrams to classify 
# To perform tenfold testing, do:
# x = Bayes_Classifier
# x.train() (optional: only if pickled files are already present)
# x.test()

import math, os, pickle, re
from random import shuffle

class Bayes_Classifier:

   def __init__(self):
      """This method initializes and trains the Naive Bayes Sentiment Classifier.  If a 
      cache of a trained classifier has been stored, it loads this cache.  Otherwise, 
      the system will proceed through training.  After running this method, the classifier 
      is ready to classify input text."""
      #Initialize dictionaries for positive words and negative words
      self.posDict = {}
      self.negDict = {}
      #Initilaize bigram dictionaries for positive/negative files
      self.posBiDict = {}
      self.negBiDict = {}
      #Iinitialize list of files that are positive and list of files that are negative
      self.negativeList = []
      self.positiveList = []
      #If the pickled files exist, load them
      if (os.path.isfile("posDict") and os.path.isfile("negDict") and os.path.isfile("posBiDict") and os.path.isfile("negBiDict")):
         self.posDict = self.load("posDict")
         self.negDict = self.load("negDict")
         self.posBiDict = self.load("posBiDict")
         self.negBiDict = self.load("negBiDict")
      #If the pickled files don't exist, generate them through train
      else:
         self.train()

   def train(self):   
      """Trains the Naive Bayes Sentiment Classifier."""
      #initialize the list of files
      lFileList = []
      #add each file in the movies_reviews/ directory to lFileList
      for fFileObj in os.walk('movies_reviews/'):
         lFileList = fFileObj[2]
         break
      #iterate through each file in lFileList
      for files in lFileList:
         #gets the star rating from the filename
         num = int(files[7])
         #tokenize the contents of the file
         tokens = self.tokenize(self.loadFile(files))
         #if the file is negative
         if (num==1):
            #append the file to the negative file list
            self.negativeList.append(files)
            #go through each word in the token
            for word in tokens:
               #if the word isn't already present in the dictionary, initialize it
               if (self.negDict.has_key(word) is False):
                  self.negDict[word] = 1
               #otherwise, update the frequency of the word in the dictionary by 1
               else:
                  self.negDict[word] += 1
            #go through each bigram in the token
            for i in range(0,len(tokens)-1):
               #selects a bigram by appending one token to the next token
               phrase = tokens[i] + tokens[i+1]
               #if the bigram isn't already present, initialize it with 1
               if (self.negBiDict.has_key(phrase) is False):
                  self.negBiDict[phrase] = 1
               #otherwise, update the bigram's frequency in the dictionary
               else:
                  self.negBiDict[phrase] += 1
         #if the file is positive
         elif (num==5):
         #append the file to the positive file list
            self.positiveList.append(files)
            #go through each word in the token
            for word in tokens:
               #if the word isn't already present in the positive dictionary, initialize it
               if (self.posDict.has_key(word) is False):
                  self.posDict[word] = 1
               #otherwise, update the frequency in the positive dictionary
               else:
                  self.posDict[word] += 1
            #go through each bigram in the token
            for i in range(0,len(tokens)-1):
               #selects a bigram by appending one token to the next token
               phrase = tokens[i] + tokens[i+1]
               #if the bigram isn't already present, initialize it with 1
               if (self.posBiDict.has_key(phrase) is False):
                  self.posBiDict[phrase] = 1
               #otherwise, update the bigram's frequency in the dictionary
               else:
                  self.posBiDict[phrase] += 1

      #pickle the negative and positive unigram/bigram dictionaries
      self.save(self.negDict, "negDict")
      self.save(self.posDict, "posDict")
      self.save(self.negBiDict, "negBiDict")
      self.save(self.posBiDict, "posBiDict")


   def classify(self, sText, sigma = 0):
      """Given a target string sText, this function returns the most likely document
      class to which the target string belongs (i.e., positive, negative or neutral).
      """
      #initialize the number of positive/negative words/phrases in each dictionary
      posWords = 0
      negWords = 0
      posBiWords = 0
      negBiWords = 0

      #find the number of words in the positive dictionary
      for val in self.posDict.values():
         posWords+=val
      #find the number of words in the negative dictionary
      for val in self.negDict.values():
         negWords+=val

      #finds number of phrases in the bigram dictionaries
      for val in self.posBiDict.values():
         posBiWords+=val
      for val in self.negBiDict.values():
         negBiWords+=val

      #initialize the probability of positve/negative given features
      posProb = 0
      negProb = 0

      #tokenize the input string
      tokens = self.tokenize(sText)

      #iterate through each word in the token
      for words in tokens:
         #if the word isn't already present in the positive dictionary, the numerator will be 1 (w/ add one smoothing)
         if (self.posDict.has_key(words) is False):
            #update the probability w/ log of probability of feature given that the document is positive (addresses underflow)
            posProb += math.log(float(1) / (posWords) )
         #otherwise, calculate probability normally
         else:
            #update the probability w/ log of probability of feature given that the document is positive
            posProb += math.log(float(1+self.posDict[words]) / (posWords))
         #do the same as above, only with the negative dictionary
         if (self.negDict.has_key(words) is False):
            negProb += math.log(float(1) / (negWords) )
         else:
            negProb += math.log(float(1+self.negDict[words]) / (negWords) )

      #does the same thing as the above for loop, iterating through each bigram conditional and updating the corresponding probability
      for i in range(0,len(tokens)-1):
         phrase = tokens[i] + tokens[i+1]
         if (self.posBiDict.has_key(phrase) is False):
            posProb += math.log(float(1) / (posBiWords) )
         else:
            posProb += math.log(float(1+self.posBiDict[phrase]) / (posBiWords))
         if (self.negBiDict.has_key(phrase) is False):
            negProb += math.log(float(1) / (negBiWords) )
         else:
            negProb += math.log(float(1+self.negBiDict[phrase]) / (negBiWords) )

      #if the positive probability is higher than negative with sigma value, return positive
      if posProb - negProb > sigma:
         return 'positive'
      #if the negative probability is higher than positive with sigma value, return positive
      elif negProb - posProb > sigma:
         return 'negative'
      #if neither positive/negative probablities are greater than each other by sigma, assign document to be neutral
      else:
         return 'neutral'


   def testTrain(self, posTrainingList, negTrainingList):   
      """Trains the Naive Bayes Sentiment Classifier given lists 
      of positive and negative files for tenfold testing purposes."""

      #re-initalizes the positive and negative unigram/bigram dictionaries
      self.posDict = {}
      self.negDict = {}
      self.negBiDict = {}
      self.posBiDict = {}

      #iterates through the given list of positive files and updates the dictionary with each word
      for files in posTrainingList:
         tokens = self.tokenize(self.loadFile(files))
         #iterates through each word and updates the positve dictionary with it
         for word in tokens:
            if (self.posDict.has_key(word) is False):
               self.posDict[word] = 1
            else:
               self.posDict[word] += 1
         #iterates through each bigram and updates its frequency in the bigram dictionary
         for i in range(0,len(tokens)-1):
            phrase = tokens[i] + tokens[i+1]
            if (self.posBiDict.has_key(phrase) is False):
               self.posBiDict[phrase] = 1
            else:
               self.posBiDict[phrase] += 1

      #iterates through given list of negative files
      for files in negTrainingList:
         tokens = self.tokenize(self.loadFile(files))
         #iterates through each word and updates the negative dictionary
         for word in tokens:
            if (self.negDict.has_key(word) is False):
               self.negDict[word] = 1
            else:
               self.negDict[word] += 1
         #iterates through each bigram and updates its frequency in the bigram dictionary
         for i in range(0,len(tokens)-1):
            phrase = tokens[i] + tokens[i+1]
            if (self.negBiDict.has_key(phrase) is False):
               self.negBiDict[phrase] = 1
            else:
               self.negBiDict[phrase] += 1


   def test(self):
      """Applies tenfold testing to the Sentiment Classifier."""

      #initialize the number of true/false positives and true/false negatives
      correctPos = 0
      falsePos = 0
      correctNeg = 0
      falseNeg = 0

      #randomizes the order of positive and negative files
      shuffle(self.negativeList)
      shuffle(self.positiveList)

      #calculate length of list of negative files
      negLen = len(self.negativeList)
      #shorten list of positive files to match length of negative files
      newPositiveList = self.positiveList[0:negLen]
      #length of newPositiveList
      posLen = len(newPositiveList)

      #calculate length of each partition for tenfold testing
      posPartLen = posLen / 10
      negPartLen = negLen / 10

      #iterate ten times for tenfold testing
      for i in range(0,10):

         #slice positive/negative lists to generate 9/10 of files for training
         posTrainingList = newPositiveList[0:posPartLen*i] + newPositiveList[posPartLen*(i+1)-1:posLen]
         negTrainingList = self.negativeList[0:negPartLen*i] + self.negativeList[negPartLen*(i+1)-1:negLen]
         #train classifier on those 9/10 of files
         self.testTrain(posTrainingList, negTrainingList)

         #slice positive/negative lists to generate 1/10 of files for testing
         posTestingList = newPositiveList[posPartLen*i:posPartLen*(i+1)]
         negTestingList = self.negativeList[negPartLen*i:negPartLen*(i+1)]
         
         #do testing for each positive file
         for files in posTestingList:
            words = self.loadFile(files)
            #classify document
            res = self.classify(words)
            #if it thinks positive file is positive, it's correctPos so update it
            if (res == 'positive'):
               correctPos += 1
            #if it thinks positive file is negative, update falseNegative
            else:
               falseNeg += 1

         #do testing for each negative file
         for files in negTestingList:
            words = self.loadFile(files)
            #classify document
            res = self.classify(words)
            #if it classifies negative file as negative, update correctNegative
            if (res == 'negative'):
               correctNeg += 1
            #otherwise, it's a false positive
            else:
               falsePos += 1

      #calculate recall by dividing correctly-identified files by total # of files that were actually pos/negative
      posRecall = float(correctPos)/(correctPos+falseNeg)
      negRecall = float(correctNeg)/(correctNeg+falsePos)

      #calculate precision by dividing correctly-identified positive by total # of files that were classified as pos/negative
      posPrec = float(correctPos)/(correctPos+falsePos)
      negPrec = float(correctNeg)/(correctNeg+falseNeg)

      #calculate f-measure using formula
      posF = (2*posRecall*posPrec)/(posRecall+posPrec)
      negF = (2*negRecall*negPrec)/(negRecall+negPrec)

      #macro-average positive and negative classes
      averageRecall = (posRecall+negRecall)/2
      averagePrec = (posPrec + negPrec)/2
      averageF = (posF + negF)/2

      #return macroaveraged recall, precision, f-measure
      return "Recall: %f Precision: %f F-measure: %f" % (averageRecall, averagePrec, averageF)


   def loadFile(self, sFilename):
      """Given a file name, return the contents of the file as a string."""
      sFilename = 'movies_reviews/' + sFilename
      f = open(sFilename, "r")
      sTxt = f.read()
      f.close()
      return sTxt
   
   def save(self, dObj, sFilename):
      """Given an object and a file name, write the object to the file using pickle."""

      f = open(sFilename, "w")
      p = pickle.Pickler(f)
      p.dump(dObj)
      f.close()
   
   def load(self, sFilename):
      """Given a file name, load and return the object stored in the file."""

      f = open(sFilename, "r")
      u = pickle.Unpickler(f)
      dObj = u.load()
      f.close()
      return dObj

   def tokenize(self, sText): 
      """Given a string of text sText, returns a list of the individual tokens that 
      occur in that string (in order)."""

      lTokens = []
      sToken = ""
      for c in sText:
         if re.match("[a-zA-Z0-9]", str(c)) != None or c == "\"" or c == "_" or c == "-":
            sToken += c
         else:
            if sToken != "":
               lTokens.append(sToken)
               sToken = ""
            if c.strip() != "":
               lTokens.append(str(c.strip()))
               
      if sToken != "":
         lTokens.append(sToken)

      return lTokens
