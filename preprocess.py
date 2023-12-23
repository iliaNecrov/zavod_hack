import re
from bs4 import BeautifulSoup

import warnings
warnings.filterwarnings("ignore")

class DataPreprocess:
    """
    Class to preprocess data, returns clean data in "preprocess" method
    """
    @staticmethod
    def __remove_html_tags(text: str) -> str:
        # Parse the text with BeautifulSoup
        soup = BeautifulSoup(text, "html.parser")
        # Get the text without tags
        clean_text = soup.get_text(separator=' ')
        return clean_text
    
    @staticmethod
    def __remove_emojis(text: str) -> str:
        # Define the pattern to capture all non-ASCII characters and emoji ranges
        emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F700-\U0001F77F"  # alchemical symbols
                               u"\U0001F780-\U0001F7FF"  # Geometric Shapes Extended
                               u"\U0001F800-\U0001F8FF"  # Supplemental Arrows-C
                               u"\U0001F900-\U0001F9FF"  # Supplemental Symbols and Pictographs
                               u"\U0001FA00-\U0001FA6F"  # Chess Symbols
                               u"\U0001FA70-\U0001FAFF"  # Symbols and Pictographs Extended-A
                               u"\U00002702-\U000027B0"  # Dingbats
                               u"\U000024C2-\U0001F251" 
                               "]+", flags=re.UNICODE)
        return emoji_pattern.sub(r'', text)
    
    @staticmethod
    def __replace_links(text: str, replacement: str ="ССЫЛКА") -> str:
        # Regular expression for matching URLs
        #url_pattern = re.compile(r"^https?:\/\/(?:www\.)?[-a-zA-Z0-9@:%._\+~#=]{1,256}\.[a-zA-Z0-9()]{1,6}\b(?:[-a-zA-Z0-9()@:%_\+.~#?&\/=]*)$")
        text = re.sub(r"(https?://\S+|www\.\S+)", replacement, text)
        # Replace URLs with the specified replacement text
        #replaced_text = url_pattern.sub(replacement, text)
        return text #replaced_text
    
    @staticmethod
    def __remove_social(text: str) -> str:
        return re.sub(r'@\w+', 'ПРОФИЛЬ', text)
    
    @staticmethod
    def __remove_phone(text: str) -> str:
        phone_pattern = re.compile(
                    r"""
                    (\+7[\s-]?)?          # Country code (optional)
                    (\(?\d{3}\)?)         # Area code
                    [\s-]*                # Optional separator
                    (\d{3})               # First 3 digits
                    [\s-]*                # Optional separator
                    (\d{2})               # Next 2 digits
                    [\s-]*                # Optional separator
                    (\d{2})               # Last 2 digits
                    """, re.VERBOSE)
        return phone_pattern.sub('ТЕЛЕФОН', text)
    
    def __remove_vk_accounts(text: str) -> str:
        pattern = r'\[.*?\]'
        return re.sub(pattern, '', text).lstrip(", ")

    @classmethod
    def preprocess(cls, text: str) -> str:
        no_html_text = cls.__remove_html_tags(text)
        
        no_emojis_text = cls.__remove_emojis(no_html_text)
        
        no_links_text = cls.__replace_links(no_emojis_text)
        
        no_social = cls.__remove_social(no_links_text)
        
        no_phone = cls.__remove_phone(no_social)

        no_vk_accounts = cls.__remove_vk_accounts(no_phone)

        return no_vk_accounts