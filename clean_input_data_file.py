filename = "./input/good_python.txt"


def process(char):
    # Handle all the accents
    if char in ['à', 'á', 'â', 'ã', 'ä', 'å']:
        return 'a'
    elif char in ['ç']:
        return 'c'
    elif char in ['è', 'é', 'ë']:
        return 'e'
    elif char in ['ì', 'í', 'ï']:
        return 'i'
    elif char in ['ò', 'ó', 'õ', 'ö']:
        return 'o'
    elif char in ['ñ']:
        return 'n'
    elif char in ['ù', 'ú', 'ü', 'µ']:
        return 'u'

    # Remove all non-ascii characters
    elif ord(char) > 128:
        return ''

    # Everything else is good
    else:
        return char


if __name__ == "__main__":
    # Read, process, and write the data
    with open(filename, 'r') as f:
        data = f.read()

    data = "".join([process(c) for c in data])

    with open(filename, 'w') as f:
        f.write(data)
