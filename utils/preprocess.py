import re
import emoji as emoji_lib
import regex as re2

URL_RE = re.compile(r'https?://\S+|www\.\S+')
USER_RE = re.compile(r'@\w+')
HASH_RE = re.compile(r'#(\w+)')
NUM_RE  = re.compile(r'\b\d+\b')
SPACE_RE = re.compile(r'\s+')

def demojize(text: str) -> str:
    # convert emojis to :smiling_face: text
    return emoji_lib.demojize(text, language='en')

def normalize_hashtags(text: str) -> str:
    # keep hashtag token word (e.g., #BestDayEver -> best day ever)
    def split_camel(word):
        return re2.sub(r'(\p{Ll})(\p{Lu})', r'\1 \2', word)
    def repl(m):
        w = m.group(1)
        w = split_camel(w)
        return f' {w.lower()} '
    return HASH_RE.sub(repl, text)

def clean_tweet(text: str) -> str:
    text = text.strip()
    text = URL_RE.sub(' <url> ', text)
    text = USER_RE.sub(' <user> ', text)
    text = normalize_hashtags(text)
    text = NUM_RE.sub(' <num> ', text)
    text = demojize(text)
    text = text.lower()
    text = SPACE_RE.sub(' ', text).strip()
    return text
