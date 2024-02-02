import math
import pandas as pd

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


all_non_plain_texts_file = "/Cookie Value Learning Algorithm/3_all_non_plain_texts.csv"
all_non_plain_values = pd.read_csv(all_non_plain_texts_file, header=0)
all_non_plain_values = all_non_plain_values['non_plain_texts'].to_list()

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

all_non_plain_values_after_filter_file = "/Cookie Value Learning Algorithm/4_all_non_plain_values_after_randomness_filter.csv"
all_non_plain_values_after_filter.to_csv(all_non_plain_values_after_filter_file, header=True, index=False)