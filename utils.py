import json
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

from preprocess import DataPreprocess

import warnings
warnings.filterwarnings("ignore")

def split_data(jsonl_file_path: str) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """
    Splits data to posts and comments
    """
    comments, posts = [], []
    with open(jsonl_file_path, 'r') as json_file:
        print("Reading data...")
        for line in tqdm(json_file):  # Adding line numbers for better error tracking

            jsonl = json.loads(line)
            jsonl["text"] = DataPreprocess.preprocess(jsonl["text"])
            
            if "root_id" in jsonl:
                comments.append(jsonl)
            else:
                posts.append(jsonl)

    return posts, comments


def add_subcomments(comments_data: List[Dict[str, Any]]) ->List[Dict[str, Any]]:
    """
    Creates new dict with comments and their subcomments
    Dict keys: ["comment", "subcomment"]
    """
    comments_structure = {}

    # Create keys for all main comments
    for comment in comments_data:
        if comment['parent_id'] == comment['root_id']:  # It's a main comment
            comments_structure[comment['id']] = {
                'comment': comment,
                'subcomments': []
                }

    # Assign subcomments to their corresponding main comment
    for comment in comments_data:
        if comment['parent_id'] != comment['root_id']:
            if comment['parent_id'] in comments_structure:
                comments_structure[comment['parent_id']]['subcomments'].append(comment)
            else:
                comments_structure[comment['id']] = {
                'comment': comment,
                'subcomments': []
                }

    final_comments_data = list(comments_structure.values())

    for comment in final_comments_data:
        comment['subcomments'] = sorted(comment["subcomments"],
                                                key = lambda x: x['date'])

    # Бывают случаи дубликатов комментариев, assert отложить до лучших дней    
    #amount_comments = 0
    #for comment in final_comments_data:
    #    amount_comments += 1 + len(comment["subcomments"])
    
    #print(len(comments_data), amount_comments)
    #assert len(comments_data) == amount_comments

    return final_comments_data


def create_comments_list(structered_comments: List[Dict[str, Any]]) -> List[Tuple[str, str]]:
    """
    Creates unstructed comments from structed ones
    (Transforms dict to list with tuples: (comment_hash, comment_text))
    """
    unstructered_comments = []
    for index, structered_comment in enumerate(structered_comments, 1):
        main_comment = structered_comment["comment"]

        comment_text =  f"{index}) {main_comment['text']}"
        unstructered_comments.append((main_comment['hash'], comment_text))

        sub_comments = structered_comment["subcomments"]
        for sub_index, sub_comment in enumerate(sub_comments, 1):
            
            sub_comment_text = f"{index}.{sub_index}) {sub_comment['text']}"
            unstructered_comments.append((sub_comment['hash'], sub_comment_text))
            
    return unstructered_comments


def get_start_index(text: str) -> int:
    """
    Deleting points from sentence
    """
    start = text.find(")")
    if start != -1 and start<6:
        return start
    else:
        return 0