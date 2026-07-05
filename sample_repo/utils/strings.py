def slugify(text):
    return text.strip().lower().replace(" ", "-")


def truncate(text, max_len=80):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def reverse(text):
    return text[::-1]

def count_vowels(text):
    vowels = 'aeiouAEIOU'
    count = 0
    for char in text:
        if char in vowels:
            count += 1
    return count
