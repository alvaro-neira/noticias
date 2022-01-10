def remove_from_end(big_str, subs):
    big_str_len = len(big_str)
    try:
        big_str_len = big_str.index(subs)
    except ValueError as ve:
        return big_str
    return big_str[0:big_str_len]


print(remove_from_end('asdasd/view?usp=sharing', '/viewusp=sharing'))
