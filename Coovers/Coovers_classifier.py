import pandas as pd
import json
import urllib.parse
import base64
import json
import string
import math
import subprocess
import re
from datetime import datetime
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report
import openai
from packaging import version

required_version = version.parse("1.1.1")
current_version = version.parse(openai.__version__)

if current_version < required_version:
    raise ValueError(f"Error: OpenAI version {openai.__version__}"
                     " is less than the required version 1.1.1")
else:
    print("OpenAI version is compatible.")

# -- Now we can get to it
from openai import OpenAI


VAILD_TIMEZONES = ["EST", "PST", "CST", "MST", "UTC", "EDT", "PDT", "CDT", "MDT"]
CONTINENTS_REGIONS = ["Africa", "America", "Antarctica", "Asia", "Atlantic", "Australia", "Europe", "Indian", "Pacific"]

DATE_PATTERNS = [
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-0[123456789]-0[123456789])', 
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-0[123456789]-[12]\d{1})', 
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-0[123456789]-3[01])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-1[012]-[0][123456789])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-1[012]-[12]\d{1})',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)-1[012]-3[01])',
    r'([01]\d|2[0-3]):([0-5]\d):([0-5]\d)',
    r'(Sun|Mon|Tue|Wed|Thu|Fri|Sat)\+(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\+\d{1,2}\+\d{4}',
    r'\d{1,2}\s(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\s\d{4}',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)0[123456789]0[123456789])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)0[123456789][12]\d{1})',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)0[123456789]3[01])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)1[012][0][123456789])',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)1[012][12]\d{1})',
    r'(20(0[0-9]|1[0-9]|2[0-9]|30)1[012]3[01])',
    r'(0[123456789]0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(0[123456789][12]\d{1}20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(0[123456789]3[01]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(1[012][0][123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(1[012][12]\d{1}20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(1[012]3[01]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(0[123456789]0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'([12]\d{1}0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(3[01]0[123456789]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'([0][123456789]1[012]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'([12]\d{1}1[012]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'(3[01]1[012]20(0[0-9]|1[0-9]|2[0-9]|30))',
    r'\d{1,2}/\d{1,2}/\d{4}', 
    r'\d{1,2}:\d{2}:\d{2}\+[APM]{2}', 
]

PRE_DEFINED_DATE_FORMAT = [
    # Pattern for GMT format
    r"GMT[+-]\d{2,4}", 
    # More specific pattern for timezone abbreviation, using a predefined list
    r"\b(?:" + "|".join(VAILD_TIMEZONES) + r")\b", 
    # More specific pattern for IANA timezone database name
    # Only match known continent/region patterns
    r"\b(?:" + "|".join(CONTINENTS_REGIONS) + r")\/[A-Za-z_]+\b", 
    # Pattern for ISO 8601 time offset
    r"[+-]\d{2}:\d{2}", 
]

IPV4_PATTERN = r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b'

# UUID pattern
# UUID_PATTERN = r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
UUID_PATTERN = r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}"

# Hexadecimal session ID pattern (32 characters as an example)
SESSIONID_PATTERN = r"^[0-9a-fA-F]{32}$"

# URL or Domain Link, like https://example.com/path or http://test.com or mch-farmjournal.com
FULL_URL_PATTERN = r'https?://(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+[A-Za-z]{2,6}(?:[^\s&;]*)'
DOMAIN_PATTERN = r'(?:(?:[A-Za-z0-9](?:[A-Za-z0-9\-]{0,61}[A-Za-z0-9])?\.)+(?:com|org|net|edu|gov|co|info|biz|io|app))'

# Regular expression to find 10-digit numbers
TEN_DIGIT_NUMBERS = re.compile(r'\b\d{10}\b')

DELIMITERS = ['.', ',', '|', '&', '=', ':', '[', ']', '{', '}', '(', ')', '+', '\'', '"', ';', '/']

SECOND_DELIMITERS = ['-', '_', '#', '$']
SECOND_DELIMITERS.extend(DELIMITERS)

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

def is_only_special_characters(input_string):
    return all(char in SECOND_DELIMITERS for char in input_string)

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

def get_delimiter_pattern(all_delimiter):
    # Create a regex pattern to split by the delimiters
    # pattern = '|'.join(map(re.escape, delimiters))
    separated_delimiters = []
    for delimiter in all_delimiter:
        separated_delimiters.append(re.escape(delimiter))
    delimiter_pattern = '|'.join(separated_delimiters)
    
    return delimiter_pattern

def is_regular_text(text):
    printable = set(string.printable)
    printable_ratio = sum(c in printable for c in text) / len(text)
    return printable_ratio > 0.85 

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

    # Try Base64 decoding
    # try:
    #     base64_decoded = base64.b64decode(cookie_value).decode()
    #     return base64_decoded
    # except:
    #     pass
    
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

def check_whether_containing_date_pattern(cookie_value):
    all_date_pattern = []
    
    for pattern in PRE_DEFINED_DATE_FORMAT: 
        match = re.search(pattern, cookie_value)  # search for the pattern in the string
        if match:  # if a match is found
            time_str = match.group()  # extract the matched time string
            all_date_pattern.append(time_str)
            cookie_value = re.sub(pattern, '', cookie_value)
    
    # find timestamp using 10-digit numbers format (Unix timestamp format)
    time_stamp = TEN_DIGIT_NUMBERS.findall(cookie_value)
    for time in time_stamp:
        try:
            # Attempt to convert to a date
            date = datetime.utcfromtimestamp(int(time)).strftime('%Y-%m-%d %H:%M:%S UTC')
            all_date_pattern.append(time)
            cookie_value = cookie_value.replace(time, '')
        except ValueError:
            # Handle invalid timestamps
            continue
    
    # find time stamp
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, cookie_value)  # search for the pattern in the string
        if match:  # if a match is found
            time_str = match.group()  # extract the matched time string
            all_date_pattern.append(time_str)
            cookie_value = re.sub(pattern, '', cookie_value)  # remove the pattern from the string
    
    return all_date_pattern, cookie_value

def check_common_patterns(cookie_value):
    temp_segments = []
    
    # check date pattern
    all_date_pattern, cookie_value = check_whether_containing_date_pattern(cookie_value)
    if all_date_pattern:
        for date in all_date_pattern:
            text = "time stamp: " + date
            temp_segments.append(text)
    
    # check ip address
    ip_address = re.findall(IPV4_PATTERN, cookie_value)
    if len(ip_address) > 0:
        cookie_value = re.sub(IPV4_PATTERN, '', cookie_value)
        for address in ip_address:
            text = "ip address: " + address
            temp_segments.append(text)
        
    # check uuid
    check_uuid = []
    match = re.search(UUID_PATTERN, cookie_value)
    if match:  # if a match is found
        uuid_str = match.group()
        check_uuid.append(uuid_str)
        cookie_value = re.sub(UUID_PATTERN, '', cookie_value)
    if check_uuid:
        for uuid in check_uuid:
            text = "UUID: " + uuid
            temp_segments.append(text)
        
    # find url link
    url_link = re.findall(FULL_URL_PATTERN, cookie_value)
    if len(url_link) > 0:
        cookie_value = re.sub(FULL_URL_PATTERN, '', cookie_value)
        for url in url_link:
            text = "url link: " + url
            temp_segments.append(text)
        
    # find domain link
    domain_link = re.findall(DOMAIN_PATTERN, cookie_value)
    if len(domain_link) > 0:
        cookie_value = re.sub(DOMAIN_PATTERN, '', cookie_value)
        for domain in domain_link:
            text = "domain link: " + domain
            temp_segments.append(text)
        
    return cookie_value, temp_segments

def try_decode_and_crack(input_value):
    # try decode method and crack method first
    decoded_value = decode_cookie_value(input_value)
    cracked_value = auto_crack_cookie_hash_corrected(str(decoded_value))
    
    return cracked_value != input_value, cracked_value

def find_contained_segments(split_cookie_value_list, all_learned_segments):
    found_segments = []
    for split_value in split_cookie_value_list:
        for segment in all_learned_segments:
            if segment in split_value:
                found_segments.append(segment)
                split_value = split_value.replace(segment, " ")
                
        split_value_rest = split_value.split(" ")
        split_value_rest = [str for str in split_value_rest if str and not is_only_special_characters(str)]
        found_segments.extend(split_value_rest)
                
    return found_segments

def pre_processing_for_cookie_values(example_crawl_data, segments_file, output_filename):
    make_print_to_file("log_pre-processing_for_detecting_segments_in_cookie_values", path='./')

    all_learned_segments = pd.read_csv(segments_file, header=0)
    all_learned_segments = all_learned_segments["segments"].to_list()

    all_learned_segments = [str(segment) for segment in all_learned_segments]
    all_learned_segments.sort(key=len, reverse=True)

    cookie_value_with_segments = pd.DataFrame(
        columns=["Domain", "name", "segments", "ground_truth_label", "value"]
    )

    print(f"Start to detect the segments in {len(example_crawl_data)} cookie values.")

    for index, row in example_crawl_data.iterrows():
        
        current_segments = []
        cookie_value = str(row["value"])
        label = row["ground_truth_label"]
        
        print(f"    For {index} cookie value -- {cookie_value}:")
        
        if pd.isna(cookie_value):
            continue
        
        # if is_random_cookie(cookie_value):
        #     text = "random: " + cookie_value
        #     current_segments.append(text)
            
        all_value = set()
        need_to_be_check_again = set()
        need_to_be_check_again.add(cookie_value)
        
        delimiter_pattern = get_delimiter_pattern(DELIMITERS)
        
        while True:
            if need_to_be_check_again:
                temp_checked_again = set()
                for cookie_value in need_to_be_check_again:
                    # 1. first decode and crack the cookie value
                    results, cracked_value = try_decode_and_crack(cookie_value)
                    
                    # 2. check common patterns
                    cracked_value, temp_segments = check_common_patterns(cracked_value)
                    current_segments.extend(temp_segments)
                    print(f"{cookie_value}: {current_segments}")
                    
                    if not results:
                        # print(f"        It might be split {cracked_value}. ")
                        all_value.add(cracked_value)
                        continue
                    
                    # 3. split the cookie value using delimiters
                    temp_separate = re.split(delimiter_pattern, cracked_value)
                    cracked_value = [item for item in temp_separate]
                    temp_checked_again.update(cracked_value)
                
                need_to_be_check_again = temp_checked_again.copy()
            else:
                break
            
        found_segments = find_contained_segments(all_value, all_learned_segments)
        current_segments.extend(found_segments)
            
        
        if not current_segments:
            current_segments = [str(row["value"])]
        
        temp_add = {
            "Domain": row["Domain"],
            "name": row["name"], 
            "segments": ",".join(current_segments), 
            "ground_truth_label": label, 
            "value": row["value"], 
        }
        
        cookie_value_with_segments = pd.concat([cookie_value_with_segments, pd.DataFrame.from_records([temp_add])], ignore_index=True)

    cookie_value_with_segments.to_csv(output_filename, header=True, index=False)
    return cookie_value_with_segments


# Fine-tune the Chat-GPT model
api_key = ""
client = OpenAI(api_key=api_key)

# Upload training set data
output_filename_for_train = "/Coovers/training_set_data.jsonl"
training_file = client.files.create(
  file=open(output_filename_for_train, "rb"),
  purpose='fine-tune'
)
training_file_id = training_file.id

# Start fine-tuning job
fine_tuning_job = client.fine_tuning.jobs.create(
  training_file=training_file_id, 
  model="gpt-3.5-turbo"
)

fine_tuning_job_id = fine_tuning_job.id

# Monitor the progress of the fine-tuning job
while fine_tuning_job.status != "completed":
   fine_tuning_job = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
   print(fine_tuning_job.status)
   
# Retrieve the state of a fine-tune
fine_tuning_job = client.fine_tuning.jobs.retrieve(fine_tuning_job_id)
model_id = fine_tuning_job.fine_tuned_model

# Preparing for the testing data set
example_crawl_data = pd.read_excel("/Coovers/testing_set_data.xlsx", header=0)
segments_corpus = "/Coovers/all_learned_segments_example.csv"
output_filename = "/Coovers/testing_set_data_after_pre-processing.csv"
cookie_value_with_segments = pre_processing_for_cookie_values(example_crawl_data=example_crawl_data, 
                                                              segments_file=segments_corpus, 
                                                              output_filename=output_filename)

cookie_value_with_segments.insert(5, "predicted_label", "")

for index, row in cookie_value_with_segments.iterrows():
    prompt = "Could you tell me this cookie's purpose? The cookie's name is " + str(row['name'])
    prompt += " from "
    prompt += str(row['Domain'])
    prompt += " domain. The cookie's value contains these segments: "
    prompt += str(row['segments'])
    prompt += ". The cookie purpose should be one of the four general cookie purposes (necessary cookie, functional cookie, statistics cookie, advertising cookie). You must tell me only one purpose for this cookie."
    
    # Generate model's response
    response = client.chat.completions.create(
        model=model_id,
        messages=[
            {
                "role": "system", 
                "content": "GPT is a great tool to analyze the website cookies' purpose using their values or values' segments. The cookies' purposes have four categories: necessary cookie, functional cookie, statistics cookie, advertising cookie. Each website cookie only has one purpose."
                }, 
            {"role": "user", "content": prompt},
            ]
    )

    generated_text = response.choices[0].message.content
    
    cookie_value_with_segments.at[index, 'predicted_label'] = generated_text
    
cookie_value_with_segments.to_excel("/Coovers/testing_set_data_results.xlsx", header=True, index=False)


# Report the metrics for Coovers's performance
label_name = 'predicted_label'
print(f"Results for {label_name}.")
# Calculating accuracy
accuracy = accuracy_score(cookie_value_with_segments['ground_truth_label'], cookie_value_with_segments[label_name])
print(f"Accuracy: {accuracy}")

# Calculating precision, recall, and F1-score
precision, recall, f1, _ = precision_recall_fscore_support(cookie_value_with_segments['ground_truth_label'], cookie_value_with_segments[label_name], average='weighted')
print(f"Precision: {precision}")
print(f"Recall: {recall}")
print(f"F1 Score: {f1}")

# Alternatively, you can use classification_report to get a summary of these metrics
print(classification_report(cookie_value_with_segments['ground_truth_label'], cookie_value_with_segments[label_name]))