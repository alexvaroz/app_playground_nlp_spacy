import streamlit as st
import spacy_streamlit
import spacy
from wordcloud import WordCloud
import matplotlib.pyplot as plt

st.set_option('deprecation.showPyplotGlobalUse', False)


nlp = spacy.load('pt_core_news_sm')


def main():
	"""A Simple NLP app with Spacy-Streamlit"""
	st.title("Playground NLP SpaCy App")
	menu = ["Tokenization", "NER", "WordCloud"]
	st.sidebar.subheader("Selecione a técnica:")
	choice = st.sidebar.radio('Opções', menu)
	if choice == "Tokenization":
		st.subheader("Tokenization")
		raw_text = st.text_area("Digite o texto de entrada")
		doc = nlp(raw_text)
		if st.button("Tokenize"):
			spacy_streamlit.visualize_tokens(doc, attrs=['text', 'pos_', 'dep_', 'ent_type_'])
	elif choice == "NER":
		st.subheader("Named Entity Recognition")
		raw_text = st.text_area("Digite o texto de entrada")
		doc = nlp(raw_text)
		spacy_streamlit.visualize_ner(doc, labels=nlp.get_pipe('ner').labels)
	elif choice == "WordCloud":
		st.subheader("Word Cloud")
		raw_text = st.text_area("Digite o texto de entrada")
		doc = nlp(raw_text)
		text = [token.lemma_ for token in doc if (token.is_alpha & ~token.is_stop)]
		text = ' '.join(text)
		wordcloud = WordCloud(background_color="#f5f5f5", colormap='Dark2').generate(text)
		plt.imshow(wordcloud, interpolation='bilinear')
		plt.axis("off")
		plt.show()
		st.pyplot()


if __name__ == '__main__':
	main()
