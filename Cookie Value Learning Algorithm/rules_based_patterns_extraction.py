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



print(f"Start to extract the common patterns in cookie values: ")
# extract all date patterns and ip patterns from the decoded csv file
first_decoded_cracked_data_filename = "/Cookie Value Learning Algorithm/1_decoded_cracked_example_crawl.csv"
separated_data = pd.read_csv(first_decoded_cracked_data_filename, header=0)

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
        
filtered_date_file = "/Cookie Value Learning Algorithm/2_filtered_rules_based_patterns.csv"
separated_data.to_csv(filtered_date_file, header=True, index=False)

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