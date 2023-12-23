import torch
from sklearn.preprocessing import MinMaxScaler
from sentence_transformers import SentenceTransformer, util
from utils import get_start_index

import warnings
warnings.filterwarnings("ignore")

print(f"CUDA Device: {torch.cuda.is_available()}")

# Порог для классификации
RELATION_THRESHOLD = 0.5
# Размер батча
BATCH_SIZE = 2


# Модель SentenceTransformer
# sent_trans_model = SentenceTransformer('cointegrated/rubert-tiny2').to('cuda:0')
model = SentenceTransformer('cointegrated/rubert-tiny2').to("cuda:0")


def relation_classifier(post: tuple, comments: list, split_type: str) -> list:
    """
    Функция классификации прямого/косвенного отношения комментариев к посту
    :param post: Хэш и текст поста
    :param comments: Хэш и текст каждого комментария
    :param split_type: Класс, принадлежащие к которому комментарии надо вернуть

    :return: Список Хэш и текст комментариев нужного класса
    """
    if len(post) == 0:
        return comments

    post_ = [post[1]]
    comments_ = [el[1][get_start_index(el[1]):] for el in comments]
    post_and_comments = post_ + comments_

    # Compute embeddings
    embeddings = model.encode(post_and_comments, convert_to_tensor=True, batch_size=BATCH_SIZE)
    post_embedding = embeddings[0]
    comments_embedding = embeddings[1:]
    torch.cuda.empty_cache()

    # Compute cosine-similarities for each sentence with each other sentence
    cosine_scores = util.cos_sim(comments_embedding, post_embedding).tolist()

    # MinMaxScaler
    minmax_scaler = MinMaxScaler()
    cosine_scores_minmax = minmax_scaler.fit_transform(cosine_scores)

    # Если коммент пустой - relevance установить в 0
    for i in range(len(comments_)):
        if len(comments_) == 0:
            cosine_scores_minmax[i][0] = 0

    # Classifying relation
    split_type_comments = []
    for i in range(len(cosine_scores_minmax)):
        if cosine_scores_minmax[i][0] >= RELATION_THRESHOLD:
            label = 'direct'
        else:
            label = 'indirect'
        if label == split_type:
            split_type_comments.append(comments[i])

    if len(split_type_comments) == 0:
        return comments

    return split_type_comments
