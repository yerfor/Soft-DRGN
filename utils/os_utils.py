import os


def remove_files(filename_lst):
    for fn in filename_lst:
        os.remove(fn)
