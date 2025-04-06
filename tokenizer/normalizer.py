import string
import typing

class Normalizer:
    def __init__(self):
        self.punctuation = set(string.punctuation)
        self.whitespace = set(string.whitespace)
        self.digits = set(string.digits)

        self.url_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-._~:/?#[]@!$&'()*+,;=")
        self.email_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.@_-")
    
    def is_url(self, text: str) -> bool:
        if text.startswith(("http://", "https://", "www.")):
            return all(c in self.url_chars for c in text)
        
        return False
    
    def is_email(self, text: str) -> bool:
        if "@" not in text or "." not in text:
            return False
        
        return all(c in self.email_chars for c in text)
    
    def is_number(self, text: str) -> bool:
        if not text:
            return False
        
        has_digit = False
        for i, c in enumerate(text):
            if c in self.digits:
                has_digit = True
            elif c in {".", ","}:
                if i == 0 or i == len(text) - 1:
                    return False
                
                if not (text[i - 1] in self.digits and text[i + 1] in self.digits):
                    return False
            else:
                return False
        
        return has_digit
    
    def split_on_punctuation(self, text: str) -> typing.List[str]:
        result = []
        current_token = []

        for char in text:
            if char in self.punctuation:
                if current_token:
                    result.append("".join(current_token))
                    current_token = []
                
                result.append(char)
            else:
                current_token.append(char)
        
        if current_token:
            result.append("".join(current_token))
        
        return result
    
    def normalize_whitespace(self, text: str) -> str:
        result = []
        previous_is_space = False

        for char in text:
            if char in self.whitespace:
                if not previous_is_space:
                    result.append(" ")
                    previous_is_space = True
            else:
                result.append(char)
                previous_is_space = False
        
        return "".join(result).strip()