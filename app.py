import streamlit as st
import pandas as pd
import numpy as np
import nltk
import re
import string
import pickle
import joblib

stopwords = nltk.corpus.stopwords.words('english')
ps = nltk.PorterStemmer()
wn = nltk.WordNetLemmatizer()

st.title('Toxic Comment Analysis.')
st.markdown('''
	Used Natural language Processing to clean and vectorize input data and
	 Machine learning algorithm(Linear SVC) to predict if a comment is toxic or not. 
	 The model had a F1 accuracy of 71%. ''')
# function to remove punctuation, tokenize, remove stopwords and stem
@st.cache
def clean_text(text):
    text = re.sub(r"what's", "what is ", text)
    text = re.sub(r"\'s", " ", text)
    text = re.sub(r"\'ve", " have ", text)
    text = re.sub(r"can't", "cannot ", text)
    text = re.sub(r"n't", " not ", text)
    text = re.sub(r"I'm", "i am ", text)
    text = re.sub(r"i'm", "i am ", text)
    text = re.sub(r"\'re", " are ", text)
    text = re.sub(r"\'d", " would ", text)
    text = re.sub(r"\'ll", " will ", text)
    text = re.sub(r"\'scuse", " excuse ", text)
   
    
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    
    tokens = re.split('\W+', text)
    #words = [wn.lemmatize(word, 'v') for word in tokens]
    text = [ps.stem(word) for word in tokens if word not in stopwords]
    text = [wn.lemmatize(word) for word in text] 
    
    text = " ".join(text)
    text = " ".join([word for word in text.split() if len(word)>2])
    return text
@st.cache
def vectorizing(text):
	new_question = text
	tfidf_vectorizer = pickle.load(open("tfidf.pickle", "rb"))
	vectorized_question = tfidf_vectorizer.transform([new_question])
	return vectorized_question
@st.cache
def create_features(cleaned_text, vectorized_text):
	text = cleaned_text
	vectorized_text = vectorized_text
	label = ['toxic', 'severe_toxic', 'obscene', 'threat','insult', 'identity_hate']
	toxic = ['fuck', 'shit', 'suck', 'stupid', 'bitch', 'idiot', 'asshol', 'gay', 'dick']
	severe_toxic = ['fuck', 'bitch', 'suck', 'shit', 'asshol', 'dick', 'cunt', 'faggot', 'cock']
	obscene =['fuck', 'shit', 'suck', 'bitch', 'asshol', 'dick', 'cunt', 'faggot', 'stupid']
	threat =['kill', 'die', 'fuck', 'shit', 'rape', 'hope', 'bitch', 'death', 'hell']
	insult = ['fuck', 'bitch', 'suck', 'shit', 'idiot', 'asshol', 'stupid', 'faggot', 'cunt']
	identity_hate = ['fuck', 'gay', 'nigger', 'faggot', 'shit', 'jew', 'bitch', 'homosexu', 'suck']
	contains_toxic = []
	contains_severe_toxic = []
	contains_obscene = []
	contains_threat = []
	contains_insult = []
	contains_identity_hate =[]
	for col in range(len(label)):
		toxic_list = vars()[label[col]]
		#st.write(toxic_list)
		value = "contains_"+label[col]
		
		check = any(substring in text for substring in toxic_list) 
		if check is True:
			vars()[value].append(1)
			#st.write("True")
		else:
			vars()[value].append(0)
			#st.write("False")
	inp = list([contains_toxic[0],contains_severe_toxic[0],contains_obscene[0], contains_threat[0], contains_insult[0], contains_identity_hate[0]])
	df = pd.DataFrame([inp], columns=['contains_toxic_word', 'contains_severe_toxic_word', 'contains_obscene_word', 'contains_threat_word', 'contains_insult_word', 'contains_identity_hate_word'])
	X = pd.concat([df, pd.DataFrame(vectorized_text.toarray())], axis=1)
	return X
def predict(features):

	svc_from_joblib = joblib.load('toxicmodel.pkl')  
	y = svc_from_joblib.predict(features)
	return y

def main():
	message = st.text_area('write a comment here:')
	if st.button('Predict'):
		#st.write(message)
		cleaned_text = clean_text(message)
		#st.write(cleaned_text)
		vectorized_text = vectorizing(cleaned_text)
		#st.write(vectorized_text)
		features = create_features(cleaned_text, vectorized_text)
		#st.write(features)
		prediction = predict(features)
		df = pd.DataFrame({
			"contains_toxic": prediction[:, 0],
			"contains_severe_toxic": prediction[:, 1],
			"contains_obscene": prediction[:, 2],
			"contains_threat": prediction[:, 3],
			"contains_insult":prediction[:, 4],
			"contains_identity_hate": prediction[:, 5]
			}, index=['Comment'])
		st.write(df.T)

if __name__ == '__main__':
    main()
