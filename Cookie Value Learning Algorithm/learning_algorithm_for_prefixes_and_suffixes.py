import pandas as pd
from collections import defaultdict
import jellyfish

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


make_print_to_file("log-5-learning_prefixes_and_suffixes", path='./')
print("------------------------------------------------------------------------")
dtype_spec = {'segments': str}
all_non_plain_values_after_filter_file = "/Cookie Value Learning Algorithm/4_all_non_plain_values_after_randomness_filter.csv"
all_non_plain_values = pd.read_csv(all_non_plain_values_after_filter_file, header=0, dtype=dtype_spec)
all_non_plain_values = all_non_plain_values['non_plain_texts'].to_list()

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
    
all_learned_prefixes_and_suffixes = pd.DataFrame(all_learned_segments, columns=["learned_segments"])
all_learned_prefixes_and_suffixes_file = "/Cookie Value Learning Algorithm/5_all_learned_prefixes_and_suffixes.csv"
all_learned_prefixes_and_suffixes.to_csv(all_learned_prefixes_and_suffixes_file, index=False)


all_plain_texts_file = "/Cookie Value Learning Algorithm/3_all_plain_texts.csv"
all_plain_texts = pd.read_csv(all_plain_texts_file, header=0)
all_plain_texts = all_plain_texts["plain_texts"].to_list()

all_learned_segments.extend(all_plain_texts)

filename = "/Cookie Value Learning Algorithm/all_segments_date_pattern.txt"
with open(filename, "r") as file:
    all_segments_date_pattern_strings = file.read().strip()
all_segments_date_pattern = all_segments_date_pattern_strings.split(',')
all_learned_segments.extend(all_segments_date_pattern)

filename = "/Cookie Value Learning Algorithm/all_segments_ipv4_pattern.txt"
with open(filename, "r") as file:
    all_segments_ipv4_pattern_strings = file.read().strip()
all_segments_ipv4_pattern = all_segments_ipv4_pattern_strings.split(',')
all_learned_segments.extend(all_segments_ipv4_pattern)

filename = "/Cookie Value Learning Algorithm/all_segments_uuid_pattern.txt"
with open(filename, "r") as file:
    all_segments_uuid_pattern_strings = file.read().strip()
all_segments_uuid_pattern = all_segments_uuid_pattern_strings.split(',')
all_learned_segments.extend(all_segments_uuid_pattern)

filename = "/Cookie Value Learning Algorithm/all_segments_url_or_domain_pattern.txt"
with open(filename, "r") as file:
    all_segments_url_or_domain_pattern_strings = file.read().strip()
all_segments_url_or_domain_pattern = all_segments_url_or_domain_pattern_strings.split(',')
all_learned_segments.extend(all_segments_url_or_domain_pattern)


save_csv = pd.DataFrame(all_learned_segments, columns=["segments"])
all_learned_segments_file = "/Cookie Value Learning Algorithm/all_identified_segments.csv"
save_csv.to_csv(all_learned_segments_file, header=True, index=False)

print(f"Finally, we obtain {len(all_learned_segments)} segments.")