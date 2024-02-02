import urllib.parse
import base64
import json
import string
import subprocess
import pandas as pd


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


filename = "/Dataset/all_collected_cookies.xlsx"
example_crawl_data = pd.read_excel(filename, header=0)
example_crawl_data = example_crawl_data.drop_duplicates(subset='value')
# Decode all the cookie value into the decoded csv file
first_decoded_cracked_data = pd.DataFrame(columns=["name", "value", "decoded_value", "cracked_value"])
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

first_decoded_cracked_data_filename = "/Cookie Value Learning Algorithm/1_decoded_cracked_example_crawl.csv"
first_decoded_cracked_data.to_csv(first_decoded_cracked_data_filename, header=True, index=False)