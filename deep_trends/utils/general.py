import sys
import sys


def disable_print(func):
    sys.stdout = open(os.devnull, 'w')
    def wrapper(*args, **kwargs):
        result = func(*args, **kwargs)
        sys.stdout = sys.__stdout__
        return result
    return wrapper
