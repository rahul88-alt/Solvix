def slugify(text):
    return text.strip().lower().replace(" ", "-")


def capitalize_words(text):
    words = text.split()
    capitalized_words = [word.capitalize() for word in words]
    return ' '.join(capitalized_words)


def truncate(text, max_len=80):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def reverse(text):
    return text[::-1]
