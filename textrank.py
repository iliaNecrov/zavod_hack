import nltk
nltk.download('punkt')
from itertools import combinations
from nltk.tokenize import sent_tokenize, RegexpTokenizer
from nltk.stem.snowball import RussianStemmer
import networkx as nx
from transformers import AutoTokenizer

import warnings
warnings.filterwarnings("ignore")

MAX_SEQUENCE_LENGTH = 70

def similarity(s1, s2):
    if not len(s1) or not len(s2):
        return 0.0

    # В качестве меры похожести возьмем отношение количества одинаковых слов в предложениях
    # к суммарной длине предложений. Коэффициент Сёренсена.
    return len(s1.intersection(s2))/(1.0 * (len(s1) + len(s2)))


def textrank(text):
    # Разбиваем текст на предложения с помощью токенизатора из nltk.
    sentences = sent_tokenize(text)

    sentences_splitted = []
    for sentence in sentences:
        sentences_splitted.extend(sentence.split("\n"))

    sentences = sentences_splitted

    # Поделим эти предложения на слова регуляркой, выбрасывая пробелы и знаки препинания.
    tokenizer = RegexpTokenizer(r'\w+')

    # Все слова обработаем стеммером, который обрезает суффиксы и окончания, чтобы 
    # одно и то же слово в разных формах выглядело одинаково.
    lmtzr = RussianStemmer()

    # Выкинем повторяющиеся слова из предложений. После этого у нас есть список предложений, 
    # которые представлены в виде множества слов.
    words = [set(lmtzr.stem(word) for word in tokenizer.tokenize(sentence.lower()))
             for sentence in sentences]
    
    # Для каждой пары предложений вычислим их похожесть
    pairs = combinations(range(len(sentences)), 2)
    scores = [(i, j, similarity(words[i], words[j])) for i, j in pairs]

    # Уберем пары, у которых нет ничего общего (похожесть равна нулю).
    scores = filter(lambda x: x[2], scores)
	
    # Создадим граф. Вершинами в нем являются предложения (если точнее, то номера предложений в тексте), 
    # а ребра между ними имеют вес, равный похожести вершин, которым оно инцидентно.
    g = nx.Graph()
    g.add_weighted_edges_from(scores)

    # Посчитаем пейджранки в этом графе.
    pr = nx.pagerank(g)
	
    # Результатом будет список предложений, отсортированный по их пейджранку.
    return sorted(((i, pr[i], s) for i, s in enumerate(sentences) if i in pr), key=lambda x: pr[x[0]], reverse=True)


# Вывод топ-N предложений
def extract_main_point(text: str, n=5):
    tr = textrank(text)
    top_n = sorted(tr[:n])
    return ' '.join(x[2] for x in top_n)


def preprocess_comment(text: str): 
    """
    Get main point from sentences
    by getting 2 most important sentences
    """
    token_count = len(text.split(" "))

    if token_count > MAX_SEQUENCE_LENGTH:
        text = extract_main_point(text, n=1)
        return f"1) {text}"
    
    return text
