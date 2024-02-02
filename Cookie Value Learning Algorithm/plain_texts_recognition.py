from urllib.parse import urlparse
import pandas as pd
import re
import nltk
from nltk.corpus import words, webtext, names, wordnet

WORD_LIST = set(words.words())
WORD_LIST = WORD_LIST.union(set(names.words()))
WORD_LIST = WORD_LIST.union(set(list(wordnet.words())[:10000]))
WORD_LIST = WORD_LIST.union(set(list(webtext.words())[:10000]))

def is_potentially_human_readable(separated_value, threshold=0.5):
    # Split the string based on non-alphabetic characters
    potential_words = re.findall(r'\b[a-zA-Z]{2,15}\b', separated_value)
    
    # Count how many segments are potential words
    word_count = len(potential_words)
    
    # Determine if the string is human-readable based on the threshold
    return (word_count / max(1, len(separated_value.split()))) > threshold

def find_plain_texts_in_cookie_value(input_separated_value):
    i = 0
    found_plain_texts = []
    while i < len(input_separated_value):
        max_len_word = ''
        for j in range(i + 1, len(input_separated_value) + 1):
            substring = input_separated_value[i:j]
            if substring.lower() in WORD_LIST and len(substring) > len(max_len_word):
                max_len_word = substring
        if max_len_word:
            found_plain_texts.append(max_len_word)
            i += len(max_len_word)
        else:
            i += 1
    return found_plain_texts

def extract_domains(links):
    domains = []
    for link in links:
        parsed = urlparse(link)
        domain = '.'.join(parsed.netloc.split('.')[-2:])
        domains.append(domain)
    return domains

def clean_extracted_domain_name(all_domain_name):
    cleaned_domain = set()
    for domain_name in all_domain_name:
        if not domain_name:
            continue
        domain = domain_name.split('.')
        cleaned_domain.update(domain)
    return cleaned_domain


final_separated_data_filename = "/Cookie Value Learning Algorithm/3_final_separated_data.csv"
final_separated_data = pd.read_csv(final_separated_data_filename, header=0)

all_plain_texts = set()
all_non_plain_texts = set()
final_separated_data = final_separated_data["all_separated_value"].to_list()
final_separated_data = [separated_data for separated_data in final_separated_data if not pd.isna(separated_data)]

for separated_value in final_separated_data:
    if len(str(separated_value)) < 2:
        continue
    readable = is_potentially_human_readable(str(separated_value))
    if readable:
        print(f"separated_value: {', '.join(separated_value)}")
        current_plain_text = find_plain_texts_in_cookie_value(str(separated_value))
        current_plain_text = [item for item in current_plain_text if len(item) > 1]
        print(f"current_plain_text: {', '.join(current_plain_text)}")
        if len(current_plain_text) > 0:
            all_plain_texts.update(current_plain_text)
            
            # Remove each plain text in the cookie value
            remove_plain_texts = "|".join([re.escape(temp_plain_text) for temp_plain_text in current_plain_text])
            removed_separated_value = [temp_separated_value for temp_separated_value in re.split(remove_plain_texts, separated_value) if temp_separated_value]
            
            if len(removed_separated_value) > 0:
                all_non_plain_texts.update(removed_separated_value)
        
        else:
            all_non_plain_texts.add(separated_value)
    else:
        all_non_plain_texts.add(separated_value)
        
all_non_plain_texts = {text for text in all_non_plain_texts if len(text) > 1}

filename_url_domain_pattern = "/Cookie Value Learning Algorithm/all_segments_url_or_domain_pattern.txt"
with open(filename_url_domain_pattern, "r") as file:
    all_segments_url_or_domain_pattern_string = file.read().strip()
all_segments_url_or_domain_pattern = all_segments_url_or_domain_pattern_string.split(',')

# Extract the domain name from the all_segments_url_or_domain_pattern
all_segments_domain_name = extract_domains(all_segments_url_or_domain_pattern)
all_domain_segment = clean_extracted_domain_name(all_segments_domain_name)

all_plain_texts.update(all_domain_segment)

all_plain_texts_data = pd.DataFrame(all_plain_texts, columns=["plain_texts"])
all_plain_texts_file = "/Cookie Value Learning Algorithm/3_all_plain_texts.csv"
all_plain_texts_data.to_csv(all_plain_texts_file, header=True, index=False)
all_non_plain_texts_data = pd.DataFrame(all_non_plain_texts, columns=["non_plain_texts"])
all_non_plain_texts_file = "/Cookie Value Learning Algorithm/3_all_non_plain_texts.csv"
all_non_plain_texts_data.to_csv(all_non_plain_texts_file, header=True, index=False)