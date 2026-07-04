def slugify(text):
    return text.strip().lower().replace(" ", "-")


def is_blank(text):
    return not text.strip()

def truncate(text, max_len=80):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def reverse(text):
    return text[::-1]
