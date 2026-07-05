def slugify(text):
    return text.strip().lower().replace(" ", "-")

def count_vowels(text):
    vowels = 'aeiou'
    count = 0
    for char in text.lower():
        if char in vowels:
            count += 1
    return count

def truncate(text, max_len=80):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def reverse(text):
    return text[::-1]
