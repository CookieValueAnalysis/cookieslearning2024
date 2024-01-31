import urllib.parse
from urllib.parse import urlparse
import base64
import json
import string
import math
import subprocess
import pandas as pd
import re
import numpy as np
import nltk
from nltk.corpus import words, webtext, names, wordnet
from collections import defaultdict
import jellyfish
from datetime import datetime

VAILD_TIMEZONES = ["EST", "PST", "CST", "MST", "UTC", "EDT", "PDT", "CDT", "MDT"]
CONTINENTS_REGIONS = ["Africa", "America", "Antarctica", "Asia", "Atlantic", "Australia", "Europe", "Indian", "Pacific"]

DATE_PATTERNS = [
    # Valid for 2000-2030
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-0[123456789]-0[123456789])', 
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-0[123456789]-[12]\d{1})', 
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-0[123456789]-3[01])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-1[012]-[0][123456789])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-1[012]-[12]\d{1})',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-1[012]-3[01])',
    # HH:MM:SS like 15:38:48
    # r'^\d{2}:\d{2}:\d{2}',
    r'([01]\d|2[0-3]):([0-5]\d):([0-5]\d)',
    # Day+MM+DD+YYYY like Sat+Feb+13+2021
    r'(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\+\d{1,2}\+\d{4}',
    # like 13 Feb 2021
    r'\d{1,2}\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}',
    # YYYYMMDD like 20210213
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)0[123456789]0[123456789])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)0[123456789][12]\d{1})',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)0[123456789]3[01])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)1[012][0][123456789])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)1[012][12]\d{1})',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)1[012]3[01])',
    # MMDDYYYY
    r'(0[123456789]0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(0[123456789][12]\d{1}20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(0[123456789]3[01]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(1[012][0][123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(1[012][12]\d{1}20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(1[012]3[01]20(0[0-9]|1[0-9]|2[0-9]|30))',
    # DDMMYYYY
    r'(0[123456789]0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'([12]\d{1}0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(3[01]0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'([0][123456789]1[012]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'([12]\d{1}1[012]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(3[01]1[012]20(0[0-9]|1[0-9]|2[0-9]|30))',
    # like "2/13/2021"
    r'\d{1,2}/\d{1,2}/\d{4}', 
    # like "9:40:25+AM"
    r'\d{1,2}:\d{2}:\d{2}\+[APM]{2}',
]

PRE_DEFINED_DATE_FORMAT = [
    # Pattern for GMT format
    r"GMT[+-]\d{2,4}", 
    # More specific pattern for timezone abbreviation, using a predefined list
    r"\b(?:" + "|".join(VAILD_TIMEZONES) + r")\b", 
    r"\b(?:" + "|".join(CONTINENTS_REGIONS) + r")\/[A-Za-z_]+\b", 
    # Pattern for ISO 8601 time offset
    r"[+-]\d{2}:\d{2}", 
]

IPV4_PATTERN = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'

# UUID pattern
UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

# Hexadecimal session ID pattern (32 characters as an example)
SESSIONID_PATTERN = r"^[0-9a-fA-F]{32}$"

# URL or Domain Link, like https://example.com/path or http://test.com or mch-farmjournal.com
FULL_URL_PATTERN = r'https?://(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,6}(?:[^\s&;]*)'
DOMAIN_PATTERN = r'(?:(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+(?:com|org|net|edu|gov|co|info|biz|io|app))'

# Regular expression to find 10-digit numbers
TEN_DIGIT_NUMBERS = re.compile(r'\b\d{10}\b')

DELIMITERS = ['.', ',', '|', '&', '=', ':', '[', ']', '{', '}', '(', ')', '+', '\'', '"', ';', '/']

WORD_LIST = set(words.words())
WORD_LIST = WORD_LIST.union(set(names.words()))
WORD_LIST = WORD_LIST.union(set(list(wordnet.words())[:10000]))
WORD_LIST = WORD_LIST.union(set(list(webtext.words())[:10000]))

LENGTH_THRESHOLD = 25
SIMILARITY_THRESHOLD = 1

def make_print_to_file(output_filename, path='./'):
    '''
    path, it is a path for save your log about fuction print
    example:
    use  make_print_to_file()   and the   all the information of funtion print , will be write in to a log file
    :return:
    '''
    import sys
    import os
    import sys
    import datetime
 
    class Logger(object):
        def __init__(self, filename="Default.log", path="./"):
            self.terminal = sys.stdout
            self.path= os.path.join(path, filename)
            self.log = open(self.path, "a", encoding='utf8',)
            print("save:", os.path.join(self.path, filename))
 
        def write(self, message):
            self.terminal.write(message)
            self.log.write(message)
 
        def flush(self):
            pass
 
 
 
 
    fileName = output_filename + '_' + datetime.datetime.now().strftime('%Y_%m_%d_%H_%M_%S')
    sys.stdout = Logger(fileName + '.log', path=path)
    
    print(fileName.center(60,'*'))
    
def is_regular_text(text):
    printable = set(string.printable)
    printable_ratio = sum(c in printable for c in text) / len(text)
    return printable_ratio > 0.85 

# decode the cookie value using four common methods
def decode_cookie_value(cookie_value):
    # Try ASCII equivalent for binary representation cookie value
    try:
        # Check if the cookie value is a binary representation
        if all(char in '01' for char in cookie_value) and len(cookie_value) % 8 == 0:
            # Split the cookie value into 8-bit segments and decode
            decoded_chars = [chr(int(cookie_value[i:i+8], 2)) for i in range(0, len(cookie_value), 8)]
            if is_regular_text(''.join(decoded_chars)):
                return ''.join(decoded_chars)
    except:
        pass
    
    # Try JWT format decoding
    try:
        rem = len(cookie_value) % 4
        if rem > 0:
            prepare_cookie_value = cookie_value + '=' * (4 - rem)
        base64url_decode = base64.urlsafe_b64decode(prepare_cookie_value.encode('utf-8')).decode('utf-8')
        base64url_decoded_json = json.loads(base64url_decode)
        if is_regular_text(base64url_decoded_json):
            return base64url_decoded_json
    except:
        pass
    
    # Try URL decoding
    try:
        url_decoded = urllib.parse.unquote(cookie_value)
        if url_decoded != cookie_value:  # If decoding makes a difference, it was URL encoded
            return url_decoded
    except:
        pass
    
    try:
        base64_decoded_bytes = base64.b64decode(cookie_value)
        base64_decoded_str = base64_decoded_bytes.decode('utf-8')
        
        if is_regular_text(base64_decoded_str):
            return base64_decoded_str
    except:
        pass

    # Try JSON decoding
    try:
        json_decoded = json.loads(cookie_value)
        return json_decoded
    except:
        pass

    # Try Hexadecimal decoding
    try:
        hex_decoded = bytes.fromhex(cookie_value).decode()
        if is_regular_text(hex_decoded):
            return hex_decoded
    except:
        pass
    
    return cookie_value

def auto_crack_cookie_hash_corrected(hash_value):
    # Corrected mapping of hash lengths to specific hash types (Hashcat mode numbers)
    corrected_methods = {
        32: [0, 50],  # MD5 and potentially HmacMD5
        40: [100, 6000, 150],  # SHA1, RIPEMD-160, and potentially HmacSHA1
        56: [10900],  # Whirlpool
        64: [1400, 5000],  # SHA256 and SHA3-256
        128: [1700, 17600],  # SHA512 and SHA3-512
        16: [1000],  # NTLM
        60: [3200],  # bcrypt
        13: [1500],  # Common mode for DES
    }

    hash_length = len(hash_value)
    methods = corrected_methods.get(hash_length, [])

    if not methods:
        return hash_value

    # Simplified masks for brute-force based on typical cookie values
    alphanumeric_masks = [
        "?a?a?a?a?a?a?a?a",
        "?a?a?a?a?a?a?a?a?a?a",
        "?a?a?a?a?a?a?a?a?a?a?a?a",
        "?a?a?a?a?a?a?a?a?a?a?a?a?a?a",
        "?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a?a"
    ]

    for method in methods:
        # Attempt brute-force with common alphanumeric masks
        for mask in alphanumeric_masks:
            command = ["hashcat", "-m", str(method), "-a", "3", hash_value, mask]
            try:
                result = subprocess.run(command, capture_output=True, text=True,
                                        timeout=30)  # Limiting time for demonstration
                if "Cracked" in result.stdout:
                    cracked_value = result.stdout.split(":")[1].strip()
                    return cracked_value
            except subprocess.TimeoutExpired:
                continue

    return hash_value

# decode the cookie value that is using jwt format
def decode_jwt(jwt_token):
    # Split the token into header, payload, and signature
    header_encoded, payload_encoded, signature = jwt_token.split('.')
    
    # Base64Url decode function
    def base64url_decode(input_str):
        rem = len(input_str) % 4
        if rem > 0:
            input_str += '=' * (4 - rem)
        return base64.urlsafe_b64decode(input_str.encode('utf-8')).decode('utf-8')
    
    # Decode header and payload
    header = json.loads(base64url_decode(header_encoded))
    payload = json.loads(base64url_decode(payload_encoded))
    
    return header, payload

def get_delimiter_pattern(all_delimiter):
    # Create a regex pattern to split by the delimiters
    # pattern = '|'.join(map(re.escape, delimiters))
    separated_delimiters = []
    for delimiter in all_delimiter:
        separated_delimiters.append(re.escape(delimiter))
    delimiter_pattern = '|'.join(separated_delimiters)
    
    return delimiter_pattern

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

def is_random_cookie(cookie_value):
    """
    Returns True if the given cookie value is random, False otherwise.
    """
    # Remove any non-alphanumeric characters
    cookie_value = ''.join(c for c in cookie_value if c.isalnum())

    # Calculate the frequency of each character in the cookie value
    freq = {}
    for c in cookie_value:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1

    # Calculate the entropy of the cookie value
    entropy = 0
    for f in freq.values():
        p = float(f) / len(cookie_value)
        entropy -= p * math.log(p, 2)

    # Compare the entropy to a threshold value
    if len(cookie_value) <= 1:
        return False
    threshold = len(cookie_value) / math.log(len(cookie_value), 2)
    return entropy >= threshold

def get_non_plain_texts_with_correct_length(non_plain_texts, correct_length):
    non_plain_texts_with_correct_length = []
    for non_plain_text in non_plain_texts:
        if len(non_plain_text) >= correct_length:
            non_plain_texts_with_correct_length.append(non_plain_text)
    return non_plain_texts_with_correct_length

def generate_ngrams(word, n):
    # Add special tokens for start (^) and end ($) of the word
    word = '^' + word + '$'
    return [word[i:i+n] for i in range(len(word) - n + 1)]

def extract_prefixes_suffixes(texts, n):
    ngram_freq = defaultdict(int)
    
    for text in texts:
        for ngram in generate_ngrams(text, n):
            ngram_freq[ngram] += 1

    # Filter for prefixes and suffixes
    prefixes = {key: value for key, value in ngram_freq.items() if key[0] == '^'}
    suffixes = {key: value for key, value in ngram_freq.items() if key[-1] == '$'}

    # Sort by frequency and return
    sorted_prefixes = dict(sorted(prefixes.items(), key=lambda item: item[1], reverse=True))
    sorted_suffixes = dict(sorted(suffixes.items(), key=lambda item: item[1], reverse=True))
    
    # Get all text from sorted_prefixes and sorted_suffixes without '^' and '$' character
    prefix_text_with_frequency = [(key[1:], value) for key, value in sorted_prefixes.items()]
    suffix_text_with_frequency = [(key[:-1], value) for key, value in sorted_suffixes.items()]
    
    return prefix_text_with_frequency, suffix_text_with_frequency

def check_similarity_for_last_and_current_learned_segments(last_learned_segments_with_frequency, 
                                                           current_learned_segments_with_frequency):
    need_to_be_removed = set()
    might_need_removed_later = set()
    
    for last_segment_with_frequency in last_learned_segments_with_frequency:
        last_segment = last_segment_with_frequency[0]
        last_frequency = last_segment_with_frequency[1]
        
        for current_segment_with_frequency in current_learned_segments_with_frequency:
            current_segment = current_segment_with_frequency[0]
            current_frequency = current_segment_with_frequency[1]
            
            distance = jellyfish.damerau_levenshtein_distance(last_segment, current_segment)
            if (last_segment in current_segment) and (distance == SIMILARITY_THRESHOLD) and (last_frequency <= current_frequency):
                need_to_be_removed.add(last_segment)
            else:
                might_need_removed_later.add(current_segment)
                
    return need_to_be_removed, might_need_removed_later

def cookie_value_pre_processing(filename="/Dataset/all_collected_cookies.xlsx"):
    example_crawl_data = pd.read_excel(filename, header=0)
    example_crawl_data = example_crawl_data.drop_duplicates(subset='value')
    # Decode all the cookie value into the decoded csv file
    first_decoded_cracked_data = pd.DataFrame(columns=["name", "value", 
                                                    "decoded_value", "cracked_value"])
    total = len(example_crawl_data)
    for index, row in example_crawl_data.iterrows():
        print(f"{index} in {total} cookies")
        cookie_value = row["value"]
        decoded_value = decode_cookie_value(cookie_value)
        cracked_value = auto_crack_cookie_hash_corrected(str(decoded_value))
        
        temp_add = {"name": row["name"], 
                    "value": cookie_value, 
                    "decoded_value": decoded_value, 
                    "cracked_value": cracked_value}
        first_decoded_cracked_data = pd.concat([first_decoded_cracked_data, pd.DataFrame.from_records([temp_add])], ignore_index=True)

    return first_decoded_cracked_data

def rules_based_patterns_extraction(preprocessed_data):
    print(f"Start to extract the common patterns in cookie values: ")
    # extract all date patterns and ip patterns from the decoded csv file
    separated_data = preprocessed_data

    date_pattern = set()
    ip_address_pattern = set()
    uuid_pattern = set()
    url_domain_pattern = set()

    for index, row in separated_data.iterrows():
        cracked_value = row["cracked_value"]
        print("cracked_value :", cracked_value)
        if pd.isna(cracked_value):
            print(f"    -------- It is empty one --------")
            separated_data = separated_data.drop(index=index)
            continue
        
        print(f"    -------- For {cracked_value} --------")
        # find time stamp using pre-defined common format
        print(f"    Check Time Stamp: ")
        for pattern in PRE_DEFINED_DATE_FORMAT: 
            match = re.search(pattern, cracked_value)  # search for the pattern in the string
            if match:  # if a match is found
                time_str = match.group()  # extract the matched time string
                date_pattern.add(time_str)
                print(f"        {time_str}")
                cracked_value = re.sub(pattern, '', cracked_value)  # remove the pattern from the string
                separated_data.at[index, "cracked_value"] = cracked_value
                
        # find timestamp using 10-digit numbers format (Unix timestamp format)
        time_stamp = TEN_DIGIT_NUMBERS.findall(cracked_value)
        for time in time_stamp:
            try:
                # Attempt to convert to a date
                date = datetime.utcfromtimestamp(int(time)).strftime('%Y-%m-%d %H:%M:%S UTC')
                date_pattern.add(time)
                print(f"        {time} can be converted to {date}")
                cracked_value = cracked_value.replace(time, '')
                separated_data.at[index, "cracked_value"] = cracked_value
            except ValueError:
                # Handle invalid timestamps
                continue
        
        # find general time stamp
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, cracked_value)  # search for the pattern in the string
            if match:  # if a match is found
                time_str = match.group()  # extract the matched time string
                date_pattern.add(time_str)
                print(f"        {time_str}")
                cracked_value = re.sub(pattern, '', cracked_value)  # remove the pattern from the string
                separated_data.at[index, "cracked_value"] = cracked_value
                
        
        # find ip address
        print(f"    Check IP Address: ")
        ip_address = re.findall(IPV4_PATTERN, cracked_value)
        if len(ip_address) > 0:
            ip_address_pattern.update(ip_address)
            print(f"        {ip_address}")
            cracked_value = re.sub(IPV4_PATTERN, '', cracked_value)
            separated_data.at[index, "cracked_value"] = cracked_value
            
        # find uuid 
        print(f"    Check UUID: ")
        match = re.search(UUID_PATTERN, cracked_value)
        if match:  # if a match is found
            uuid_str = match.group()
            uuid_pattern.add(uuid_str)
            print(f"        {uuid_str}")
            cracked_value = re.sub(UUID_PATTERN, '', cracked_value)
            separated_data.at[index, "cracked_value"] = cracked_value
        
        # find url link
        print(f"    Check URL Link: ")
        url_link = re.findall(FULL_URL_PATTERN, cracked_value)
        if len(url_link) > 0:
            url_domain_pattern.update(url_link)
            print(f"        {url_link}")
            cracked_value = re.sub(FULL_URL_PATTERN, '', cracked_value)
            separated_data.at[index, "cracked_value"] = cracked_value
            
        # find domain link
        print(f"    Check Domain Link: ")
        domain_link = re.findall(DOMAIN_PATTERN, cracked_value)
        if len(domain_link) > 0:
            url_domain_pattern.update(domain_link)
            print(f"        {domain_link}")
            cracked_value = re.sub(DOMAIN_PATTERN, '', cracked_value)
            separated_data.at[index, "cracked_value"] = cracked_value
            
    filtered_date_file = "2_filtered_date_ip_data.csv"
    separated_data.to_csv(filtered_date_file, header=True, index=False)

    filename_date_pattern = "all_segments_date_pattern.txt"
    with open(filename_date_pattern, "w") as file:
        file.write(",".join(date_pattern))
        
    filename_ip_pattern = "all_segments_ipv4_pattern.txt"
    with open(filename_ip_pattern, "w") as file:
        file.write(",".join(ip_address_pattern))
        
    filename_uuid_pattern = "all_segments_uuid_pattern.txt"
    with open(filename_uuid_pattern, "w") as file:
        file.write(",".join(uuid_pattern))

    filename_url_domain_pattern = "all_segments_url_or_domain_pattern.txt"
    with open(filename_url_domain_pattern, "w") as file:
        file.write(",".join(url_domain_pattern))
        
    return separated_data, date_pattern, ip_address_pattern, uuid_pattern, url_domain_pattern

def pre_processing_with_delimiter(filtered_data, 
                                  all_segments_date_pattern, 
                                  all_segments_ipv4_pattern, 
                                  all_segments_uuid_pattern, 
                                  all_segments_url_or_domain_pattern):
    column_list = filtered_data.columns.to_list()
    column_list.insert(-1, "separated_value")
    separated_data = pd.DataFrame(columns=column_list)

    delimiter_pattern = get_delimiter_pattern(DELIMITERS)

    total = len(filtered_data)
    for index, row in filtered_data.iterrows():
        print(f"{index} in {total}")
        name = row["name"]
        value = row["value"]
        decoded_value = row["decoded_value"]
        cracked_value = row["cracked_value"]
        if pd.isna(cracked_value):
            continue
        temp_separate = re.split(delimiter_pattern, cracked_value)
        for item in temp_separate:
            if not item:
                continue
            temp_add = {
                "name": name, 
                "value": value, 
                "decoded_value": decoded_value, 
                "cracked_value": cracked_value, 
                "separated_value": item
            }
            separated_data = pd.concat([separated_data, 
                                        pd.DataFrame.from_records([temp_add])], ignore_index=True)
    
    first_separated_data = separated_data.copy(deep=True)
    column_list = first_separated_data.columns.to_list()
    column_list.insert(-1, "second_decoded_value")
    column_list.insert(-1, "second_cracked_value")
    second_decoded_cracked_data = pd.DataFrame(columns=column_list)

    total = len(first_separated_data)
    for index, row in first_separated_data.iterrows():
        print(f"{index} in {total}")
        separated_value = row["separated_value"]
        second_decoded_value = decode_cookie_value(separated_value)
        second_cracked_value = auto_crack_cookie_hash_corrected(str(second_decoded_value))
        
        temp_add = {
            "name": row["name"], 
            "value": row["value"], 
            "decoded_value": row["decoded_value"], 
            "cracked_value": row["cracked_value"], 
            "separated_value": separated_value, 
            "second_decoded_value": second_decoded_value, 
            "second_cracked_value": second_cracked_value}
        second_decoded_cracked_data = pd.concat([second_decoded_cracked_data, 
                                                pd.DataFrame.from_records([temp_add])], ignore_index=True)
        
    print(f"Start to extract the common patterns in cookie values: ")
    # extract all date patterns and ip patterns from the decoded csv file
    separated_data = second_decoded_cracked_data

    date_pattern = set(all_segments_date_pattern)
    ip_address_pattern = set(all_segments_ipv4_pattern)
    uuid_pattern = set(all_segments_uuid_pattern)
    url_domain_pattern = set(all_segments_url_or_domain_pattern)

    for index, row in separated_data.iterrows():
        cracked_value = row["second_cracked_value"]
        if pd.isna(cracked_value):
            print(f"    -------- It is empty one --------")
            separated_data = separated_data.drop(index=index)
            continue
        
        print(f"    -------- For {cracked_value} --------")
        # find time stamp using pre-defined common format
        print(f"    Check Time Stamp: ")
        for pattern in PRE_DEFINED_DATE_FORMAT: 
            match = re.search(pattern, cracked_value)  # search for the pattern in the string
            if match:  # if a match is found
                time_str = match.group()  # extract the matched time string
                date_pattern.add(time_str)
                print(f"        {time_str}")
                cracked_value = re.sub(pattern, '', cracked_value)  # remove the pattern from the string
                separated_data.at[index, "second_cracked_value"] = cracked_value
                
        # find timestamp using 10-digit numbers format (Unix timestamp format)
        time_stamp = TEN_DIGIT_NUMBERS.findall(cracked_value)
        for time in time_stamp:
            try:
                # Attempt to convert to a date
                date = datetime.utcfromtimestamp(int(time)).strftime('%Y-%m-%d %H:%M:%S UTC')
                date_pattern.add(time)
                print(f"        {time} can be converted to {date}")
                cracked_value = cracked_value.replace(time, '')
                separated_data.at[index, "second_cracked_value"] = cracked_value
            except ValueError:
                # Handle invalid timestamps
                continue
        
        # find general time stamp
        for pattern in DATE_PATTERNS:
            match = re.search(pattern, cracked_value)  # search for the pattern in the string
            if match:  # if a match is found
                time_str = match.group()  # extract the matched time string
                date_pattern.add(time_str)
                print(f"        {time_str}")
                cracked_value = re.sub(pattern, '', cracked_value)  # remove the pattern from the string
                separated_data.at[index, "second_cracked_value"] = cracked_value
                
        
        # find ip address
        print(f"    Check IP Address: ")
        ip_address = re.findall(IPV4_PATTERN, cracked_value)
        if len(ip_address) > 0:
            ip_address_pattern.update(ip_address)
            print(f"        {ip_address}")
            cracked_value = re.sub(IPV4_PATTERN, '', cracked_value)
            separated_data.at[index, "second_cracked_value"] = cracked_value
            
        # find uuid 
        print(f"    Check UUID: ")
        match = re.search(UUID_PATTERN, cracked_value)
        if match:  # if a match is found
            uuid_str = match.group()
            uuid_pattern.add(uuid_str)
            print(f"        {uuid_str}")
            cracked_value = re.sub(UUID_PATTERN, '', cracked_value)
            separated_data.at[index, "second_cracked_value"] = cracked_value
        
        # find url link
        print(f"    Check URL Link: ")
        url_link = re.findall(FULL_URL_PATTERN, cracked_value)
        if len(url_link) > 0:
            url_domain_pattern.update(url_link)
            print(f"        {url_link}")
            cracked_value = re.sub(FULL_URL_PATTERN, '', cracked_value)
            separated_data.at[index, "second_cracked_value"] = cracked_value
            
        # find domain link
        print(f"    Check Domain Link: ")
        domain_link = re.findall(DOMAIN_PATTERN, cracked_value)
        if len(domain_link) > 0:
            url_domain_pattern.update(domain_link)
            print(f"        {domain_link}")
            cracked_value = re.sub(DOMAIN_PATTERN, '', cracked_value)
            separated_data.at[index, "second_cracked_value"] = cracked_value
            

    final_separated_data = list()
    filtered_data = separated_data.copy(deep=True)
    column_list = filtered_data.columns.to_list()
    column_list.insert(-1, "second_separated_value")
    separated_data = pd.DataFrame(columns=column_list)

    second_delimiters = ['-', '_', '#', '$']
    second_delimiters.extend(DELIMITERS)
    delimiter_pattern = get_delimiter_pattern(second_delimiters)

    total = len(filtered_data)
    for index, row in filtered_data.iterrows():
        print(f"{index} of {total}")
        cracked_value = row["second_cracked_value"]
        if pd.isna(cracked_value):
            continue
        temp_separate = re.split(delimiter_pattern, cracked_value)
        temp_separate = [item for item in temp_separate if item]
        current_temp_separate = list()
        for item in temp_separate:
            temp_decoded_value = decode_cookie_value(item)
            temp_cracked_value = auto_crack_cookie_hash_corrected(str(temp_decoded_value))
            if item != temp_cracked_value:
                temp_temp_separate = re.split(delimiter_pattern, temp_cracked_value)
                temp_temp_separate = [temp_item for temp_item in temp_temp_separate if temp_item]
                current_temp_separate.extend(temp_temp_separate)
            else:
                current_temp_separate.append(item)
            
        temp_add = row.copy()
        temp_add["second_separated_value"] = current_temp_separate
        separated_data = pd.concat([separated_data, pd.DataFrame.from_records([temp_add])], ignore_index=True)
        final_separated_data.extend(current_temp_separate)

    final_separated_data_file = pd.DataFrame(final_separated_data, columns=["all_separated_value"])
    final_separated_data_file.to_csv("3-3_single_second_separated_data.csv", header=True, index=False)
    
    return final_separated_data_file, all_segments_date_pattern, all_segments_ipv4_pattern, all_segments_uuid_pattern, all_segments_url_or_domain_pattern
    
def plain_texts_recognition(final_separated_data, all_segments_url_or_domain_pattern):
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
    
    # Extract the domain name from the all_segments_url_or_domain_pattern
    all_segments_domain_name = extract_domains(all_segments_url_or_domain_pattern)
    all_domain_segment = clean_extracted_domain_name(all_segments_domain_name)

    all_plain_texts.update(all_domain_segment)
    
    all_plain_texts_file = pd.DataFrame(all_plain_texts, columns=["plain_texts"])
    all_non_plain_texts_file = pd.DataFrame(all_non_plain_texts, columns=["non_plain_texts"])
    
    return all_plain_texts_file, all_non_plain_texts_file

def randomness_filter(all_non_plain_texts):
    all_non_plain_values = all_non_plain_texts['non_plain_texts'].to_list()

    all_non_plain_values_with_randomness_test = []
    removed_non_plain_values = []

    for non_plain_value in all_non_plain_values:
        if is_random_cookie(non_plain_value):
            removed_non_plain_values.append(non_plain_value)
        else:
            all_non_plain_values_with_randomness_test.append(non_plain_value)
        
    print(f"We should have {len(all_non_plain_values)} non-plain cookie.")
    print(f"After checking the randomness level, we removed {len(removed_non_plain_values)}.")
    print(f"So, we now should have {len(all_non_plain_values_with_randomness_test)} non-plain values for the next step.")

    all_non_plain_values_after_filter = pd.DataFrame(all_non_plain_values_with_randomness_test, columns=["non_plain_texts"])

    # Function to strip leading and trailing spaces
    def clean_name(cookie_values):
        return cookie_values.strip()
    # Apply the function to the 'non_plain_texts' column
    all_non_plain_values_after_filter['non_plain_texts'] = all_non_plain_values_after_filter['non_plain_texts'].apply(clean_name)
    
    return all_non_plain_values_after_filter

def learning_algorithm_for_prefixs_and_surffixs(all_non_plain_values_after_filter):
    print("------------------------------------------------------------------------")
    all_non_plain_values = all_non_plain_values_after_filter['non_plain_texts'].to_list()

    all_learned_prefixes = []
    all_learned_suffixes = []

    last_learned_prefixes = []
    last_learned_suffixes = []

    might_need_removed_later = set()

    print("Start to learn the prefix and suffix based on the non-plain values.")

    for length in range(3, LENGTH_THRESHOLD + 2):
        print(f"    When length based on the {length - 1}: ")
        
        # get all proper non-plain text, like removing the non-plain texts that are too short for the current length.
        # Convert all elements to strings if the function expects strings
        all_non_plain_values_checked = [str(item) for item in all_non_plain_values]
        non_plain_texts = get_non_plain_texts_with_correct_length(all_non_plain_values_checked, length - 1)
        print(f"    We totally have {len(non_plain_texts)} non-plain texts for this length: ")
        
        # Learn prefixes and suffixes based on the given length
        # Note that: the learned prefixes and suffixes are now with the frequency
        temp_prefixes_with_frequency, temp_suffixes_with_frequency = extract_prefixes_suffixes(non_plain_texts, length)
        print(f"        We learned {len(temp_prefixes_with_frequency)} prefixes.")
        print(f"        We learned {len(temp_suffixes_with_frequency)} suffixes.")
        
        prefix_texts = [item[0] for item in temp_prefixes_with_frequency]
        suffix_texts = [item[0] for item in temp_suffixes_with_frequency]
        
        if last_learned_prefixes:
            # check the distance to reduce the noise
            current_prefixes = temp_prefixes_with_frequency.copy()
            current_suffixes = temp_suffixes_with_frequency.copy()
            need_to_be_removed_prefixes, might_need_removed_prefixes_later = check_similarity_for_last_and_current_learned_segments(last_learned_prefixes, current_prefixes)
            need_to_be_removed_suffixes, might_need_removed_suffixes_later = check_similarity_for_last_and_current_learned_segments(last_learned_suffixes, current_suffixes)
            
            might_need_removed_later.update(might_need_removed_prefixes_later)
            might_need_removed_later.update(might_need_removed_suffixes_later)
            
            print(f"        We need to remove {len(need_to_be_removed_prefixes)} prefixes on {length - 2} length.")
            print(f"        We need to remove {len(need_to_be_removed_suffixes)} suffixes on {length - 2} length.")
            
            # remove the duplicates from the last round.
            # Note that: we need to remove the frequency at this stage.
            last_learned_prefixes_only_segment = [item[0] for item in last_learned_prefixes]
            last_learned_suffixes_only_segment = [item[0] for item in last_learned_suffixes]
            
            for removed_prefix in need_to_be_removed_prefixes:
                last_learned_prefixes_only_segment.remove(removed_prefix)
            
            for removed_suffix in need_to_be_removed_suffixes:
                last_learned_suffixes_only_segment.remove(removed_suffix)
            
            # Add to the total dataset
            all_learned_prefixes.extend(last_learned_prefixes_only_segment)
            all_learned_suffixes.extend(last_learned_suffixes_only_segment)
            
            if (length - 1) == LENGTH_THRESHOLD:
                all_learned_prefixes.extend(prefix_texts)
                all_learned_suffixes.extend(suffix_texts)
            
        last_learned_prefixes = temp_prefixes_with_frequency.copy()
        last_learned_suffixes = temp_suffixes_with_frequency.copy()
        
    all_learned_segments = []
    all_learned_segments.extend(all_learned_prefixes)
    all_learned_segments.extend(all_learned_suffixes)

    print(f"We totally learned {len(all_learned_prefixes)} prefixes.")
    print(f"We totally learned {len(all_learned_suffixes)} suffixes.")
    print(f"We totally learned {len(all_learned_segments)} segments.")
    print(f"There should be {((len(all_learned_prefixes) + len(all_learned_suffixes)) - len(all_learned_segments))} segments that are duplicates on learned prefixes and suffixes.")
        
    all_learned_segments_data = pd.DataFrame(all_learned_segments, columns=["learned_segments"])
    return all_learned_segments_data
    
def combine_all_learned_segments(all_learned_segments_data, 
                                 all_plain_texts, 
                                 all_segments_date_pattern, 
                                 all_segments_ipv4_pattern, 
                                 all_segments_uuid_pattern, 
                                 all_segments_url_or_domain_pattern):
    
    all_learned_segments = all_learned_segments_data["learned_segments"].to_list()
    all_plain_texts = all_plain_texts["plain_texts"].to_list()

    all_learned_segments_finally = pd.DataFrame(columns=["segment", "from_plain_text"])
    for segment in all_learned_segments:
        temp_add = {
            "segment": segment, 
            "from_plain_text": False
        }
        all_learned_segments_finally = pd.concat([all_learned_segments_finally, pd.DataFrame.from_records([temp_add])], ignore_index=True)

    for segment in all_plain_texts:
        temp_add = {
            "segment": segment, 
            "from_plain_text": True
        }
        all_learned_segments_finally = pd.concat([all_learned_segments_finally, pd.DataFrame.from_records([temp_add])], ignore_index=True)

    all_learned_segments_finally = all_learned_segments_finally["segment"].to_list()
    all_learned_segments_finally.extend(all_segments_date_pattern)
    all_learned_segments_finally.extend(all_segments_ipv4_pattern)
    all_learned_segments_finally.extend(all_segments_uuid_pattern)
    all_learned_segments_finally.extend(all_segments_url_or_domain_pattern)

    save_csv = pd.DataFrame(all_learned_segments_finally, columns=["segments"])
    save_csv.to_csv("/Cookie Value Learning Algorithm/all_identified_segments.csv", index=False)

filename = "/Dataset/all_collected_cookies.xlsx"
preprocessed_data = cookie_value_pre_processing(filename)
separated_data, all_segments_date_pattern, all_segments_ipv4_pattern, all_segments_uuid_pattern, all_segments_url_or_domain_pattern = rules_based_patterns_extraction(preprocessed_data)
final_separated_data, all_segments_date_pattern, all_segments_ipv4_pattern, all_segments_uuid_pattern, all_segments_url_or_domain_pattern = pre_processing_with_delimiter(separated_data.copy(deep=True), all_segments_date_pattern, all_segments_ipv4_pattern, all_segments_uuid_pattern, all_segments_url_or_domain_pattern)
all_plain_texts, all_non_plain_texts = plain_texts_recognition(final_separated_data, all_segments_url_or_domain_pattern)
all_non_plain_values_after_filter = randomness_filter(all_non_plain_texts)
all_learned_segments_data = learning_algorithm_for_prefixs_and_surffixs(all_non_plain_values_after_filter)
combine_all_learned_segments(all_learned_segments_data, all_plain_texts, all_segments_date_pattern, all_segments_ipv4_pattern, all_segments_uuid_pattern, all_segments_url_or_domain_pattern)
