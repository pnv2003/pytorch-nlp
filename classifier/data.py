import glob
import os
import string
import unicodedata
import torch

# find files in a directory
def find_files(path): 
    return glob.glob(path)

# all ascii letters, and some punctuation
all_letters = string.ascii_letters + " .,;'"
n_letters = len(all_letters)

# convert unicode to ascii
# for example, "Ślusàrski" -> "Slusarski"
def unicode_to_ascii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# store category names and lines
category_lines = {}
all_categories = []

# read lines from a file, and convert to ascii
def read_lines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicode_to_ascii(line) for line in lines]

# read all files in the data/names directory
for filename in find_files('data/names/*.txt'):

    # category is the filename without the extension
    category = os.path.splitext(os.path.basename(filename))[0]

    # add category to all_categories
    all_categories.append(category)

    # read lines from the file, and store in category_lines
    lines = read_lines(filename)
    category_lines[category] = lines

# number of categories
n_categories = len(all_categories)


# convert letter to index
def letter_to_index(letter):
    return all_letters.find(letter)

# convert letter to tensor, via one-hot encoding
def letter_to_tensor(letter):
    tensor = torch.zeros(1, n_letters)
    tensor[0][letter_to_index(letter)] = 1
    return tensor

# convert line (name) to tensor
def line_to_tensor(line):

    # 3D tensor: store each letter in a 2D tensor, and stack them
    tensor = torch.zeros(len(line), 1, n_letters)
    for li, letter in enumerate(line):
        tensor[li][0][letter_to_index(letter)] = 1
    return tensor
