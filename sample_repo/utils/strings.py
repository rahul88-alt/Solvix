def slugify(text):
    return text.strip().lower().replace(" ", "-")


def truncate(text, max_len=80):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def count_vowels(text):
    vowels = set("aeiouAEIOU")
    return sum(1 for char in text if char in vowels)

def reverse(text):
    return text[::-1]
