import warnings
warnings.filterwarnings("ignore")

from transformers import logging as hf_logging

# Set the logging level for transformers
hf_logging.set_verbosity_error()

from transformers import pipeline
from tqdm import tqdm
import torch

class T5summarizer:
    def __init__(self, model_path:str) -> None:
        self.model_path = model_path
        self.init_t5()
    
    def init_t5(self) -> None:
        self.pipe = pipeline('summarization', 
                         model=self.model_path, 
                         device="cuda:0", 
                         token="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")
        
    
    
    def summarize(self, text: str) -> list:
        summ = self.pipe(text)
        return [i["summary_text"] for i in summ]


    def batch_summarize(self, text_list: list, batch_size: int) -> str:
        text_list = text_list[:2]
        batch_predict = []
        for i in range(0, len(text_list), batch_size):
            batch = text_list[i:i+batch_size]
            batch_summ = self.pipe(batch)
            batch_predict = batch_predict + [i["summary_text"] for i in batch_summ]
            torch.cuda.empty_cache()
        return "; ".join(batch_predict)
