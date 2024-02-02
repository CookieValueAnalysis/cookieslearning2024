import urllib.parse
import base64
import json
import string
import subprocess
import pandas as pd
import re
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

def get_delimiter_pattern(all_delimiter):
    # Create a regex pattern to split by the delimiters
    # pattern = '|'.join(map(re.escape, delimiters))
    separated_delimiters = []
    for delimiter in all_delimiter:
        separated_delimiters.append(re.escape(delimiter))
    delimiter_pattern = '|'.join(separated_delimiters)
    
    return delimiter_pattern



filtered_date_file = "/Cookie Value Learning Algorithm/2_filtered_rules_based_patterns.csv"
filtered_data = pd.read_csv(filtered_date_file, header=0)
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

filename = "/Cookie Value Learning Algorithm/all_segments_date_pattern.txt"
with open(filename, "r") as file:
    all_segments_date_pattern = file.read().strip()
all_segments_date_pattern = all_segments_date_pattern.split(',')

filename = "/Cookie Value Learning Algorithm/all_segments_ipv4_pattern.txt"
with open(filename, "r") as file:
    all_segments_ipv4_pattern = file.read().strip()
all_segments_ipv4_pattern= all_segments_ipv4_pattern.split(',')

filename = "/Cookie Value Learning Algorithm/all_segments_uuid_pattern.txt"
with open(filename, "r") as file:
    all_segments_uuid_pattern = file.read().strip()
all_segments_uuid_pattern= all_segments_uuid_pattern.split(',')
    
filename = "/Cookie Value Learning Algorithm/all_segments_url_or_domain_pattern.txt"
with open(filename, "r") as file:
    all_segments_url_or_domain_pattern = file.read().strip()
all_segments_url_or_domain_pattern= all_segments_url_or_domain_pattern.split(',')


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
final_separated_data_filename = "/Cookie Value Learning Algorithm/3_final_separated_data.csv"
final_separated_data_file.to_csv(final_separated_data_filename, header=True, index=False)


filename_date_pattern = "/Cookie Value Learning Algorithm/all_segments_date_pattern.txt"
with open(filename_date_pattern, "w") as file:
    file.write(",".join(date_pattern))
    
filename_ip_pattern = "/Cookie Value Learning Algorithm/all_segments_ipv4_pattern.txt"
with open(filename_ip_pattern, "w") as file:
    file.write(",".join(ip_address_pattern))
    
filename_uuid_pattern = "/Cookie Value Learning Algorithm/all_segments_uuid_pattern.txt"
with open(filename_uuid_pattern, "w") as file:
    file.write(",".join(uuid_pattern))

filename_url_domain_pattern = "/Cookie Value Learning Algorithm/all_segments_url_or_domain_pattern.txt"
with open(filename_url_domain_pattern, "w") as file:
    file.write(",".join(url_domain_pattern))


