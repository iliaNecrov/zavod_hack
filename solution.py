from utils import split_data, add_subcomments, create_comments_list
from tqdm import tqdm
from relation_classifier import relation_classifier
from model_summ import T5summarizer
from textrank import preprocess_comment
import json

import warnings
warnings.filterwarnings("ignore")

import argparse

# GLOBAL
BATCH_SIZE = 6 # model batch size
CHUNK_SIZE = 20 # amount of sentences which we send for prediction

# MODEL
model = T5summarizer("gromoboy/rut5_base_summ_brand")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--split-type", "-st", help="Split type: all/post/topic_comments", required=True)
    parser.add_argument("--input-file", "-i", help="Input file", required=True)
    parser.add_argument("--output-file", "-o", help="Output file", required=True)

    args = parser.parse_args()

    split_type = "all_comments"
    if args.split_type:
        print("Split type: %s" % args.split_type)
        split_type = args.split_type

    input_file = "./dataset.jsonl"
    if args.input_file:
        print("Input file: %s" % args.input_file)
        input_file = args.input_file

    output_file = "./result.jsonl"
    if args.output_file:
        print("Output file: %s" % args.output_file)
        output_file = args.output_file

    model = T5summarizer("gromoboy/rut5_base_summ_brand")

    # ---------------------------------------------------------------------------

    jsonl_file_path = f'data/{input_file}'
    posts, comments = split_data(jsonl_file_path)

    comments_by_root = {}
    for comment in tqdm(comments, desc="Root ID sort comments"):
        root_id = comment["root_id"]

        if root_id in comments_by_root:
            comments_by_root[root_id].append(comment)
        else:
            comments_by_root[root_id] = [comment]

    data = []
    for index, post in enumerate(tqdm(posts, desc="Gathering preprocessed posts/comments together")):
        try:
            comments_structured = add_subcomments(comments_by_root[post["id"]])
            comments_unstructured = create_comments_list(comments_structured)

            data.append([(post['hash'], post['text']),
                        comments_unstructured])
        except:
            print("Error, no post or comment with this ID")

    # ---------------------------------------------------------------------------

    result = []
    for post_data in tqdm(data, desc="Creating summary"):
        post, comments = post_data

        post_hash = post[0]

        if split_type == "post_comments":
            comments = relation_classifier(post, comments, "direct")
        
        if split_type == "topic_comments":
            comments = relation_classifier(post, comments, "indirect")

        comments_texts, comments_hash = [], []
        for comment in comments:
            hash_, text = comment
            comments_texts.append(preprocess_comment(text))
            comments_hash.append(hash_)
        
        chunks = []
        for i in range(0, len(comments_texts), CHUNK_SIZE):
            chunks.append("\n".join(comments_texts[i : i+CHUNK_SIZE]))
        
        if len(chunks) != 0:
            summary = model.batch_summarize(chunks, BATCH_SIZE)
        else:
            summary = "Отсутствует содержание"
        
        result.append({"summary": summary,
                        "post_hash": post_hash,
                        "comments_hash": comments_hash})

    # ---------------------------------------------------------------------------
    filename = f"data/{output_file}"

    with open(filename, 'w', encoding='utf-8') as file:
        for entry in result:
            json_record = json.dumps(entry, ensure_ascii=False)
            file.write(json_record + '\n')