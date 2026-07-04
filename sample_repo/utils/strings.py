def slugify(text):
    return text.strip().lower().replace(" ", "-")


def truncate(text, max_len=80):
    if len(text) <= max_len:
        return text
    return text[:max_len].rstrip() + "..."

def reverse(text):
    return text[::-1]

def capitalize_words(text):
    """
    Capitalizes the first letter of each word in the input text.
    
    Args:
        text (str): The input string to be capitalized.
        
    Returns:
        str: A new string with the first letter of each word capitalized.
    """
    return ' '.join(word.capitalize() for word in text.split())
