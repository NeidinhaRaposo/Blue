import nltk
import random
import string
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Baixar pacotes necessários do NLTK
nltk.download("punkt")  # Tokenizador
nltk.download("wordnet")  # Lematizador

# Dados para treinar o chatbot
corpus = """
Olá! Eu sou um chatbot. Estou aqui para ajudar você.
Você pode me fazer perguntas simples e eu responderei da melhor forma possível.
Por exemplo, pergunte sobre a minha funcionalidade ou algo relacionado a sistemas automatizados.
Se eu não souber, pedirei desculpas educadamente.
"""

# Pré-processamento do texto
lemmer = nltk.stem.WordNetLemmatizer()
def LemNormalize(text):
    return [lemmer.lemmatize(token.lower()) for token in nltk.word_tokenize(text) if token not in string.punctuation]

# Divida o corpus em sentenças
sentences = nltk.sent_tokenize(corpus)

def chatbot_response(user_input):
    sentences.append(user_input)
    # TF-IDF para calcular a similaridade
    tfidf = TfidfVectorizer(tokenizer=LemNormalize, stop_words="english")
    tfidf_matrix = tfidf.fit_transform(sentences)
    cosine_vals = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    index = cosine_vals.argsort()[0][-1]
    flat = cosine_vals.flatten()
    flat.sort()
    score = flat[-1]
    sentences.pop()  # Remove a entrada do usuário do corpus
    if score == 0:
        return "Desculpe, não entendi sua pergunta."
    else:
        return sentences[index]

# Interface do Chatbot
print("Olá! Eu sou o ChatBot Automático. Digite 'sair' para encerrar.")
while True:
    user_input = input("Você: ")
    if user_input.lower() in ["sair", "tchau", "fim"]:
        print("ChatBot: Até logo!")
        break
    resposta = chatbot_response(user_input)
    print(f"ChatBot: {resposta}")

