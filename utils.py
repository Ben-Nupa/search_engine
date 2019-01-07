import os
import sys
import numpy as np

def read_data(filename: str) -> dict:
    with open(filename) as file:
        documents = {}
        content = file.readlines()
        title = None
        summary = None
        key_words = None

        for i, line in enumerate(content):
            line = line.strip()
            if line[:2] in [".I", ".T", ".W", ".K", ".B", ".A", ".N", ".X"]:
                if line[:2] in [".I", ".T", ".W", ".K"]:

                    field = line[:2]
                    if field == ".I":
                        if i > 0:
                            # Add previous doc
                            break
                            documents[doc_id] = (title, summary, key_words)
                            title = None
                            summary = None
                            key_words = None
                        # Get doc id
                        _, doc_id = line.split()

                else:
                    field = None
            print(l)
            # Get proper information
            if field == ".T":
                title = line
            elif field == ".W":
                summary = line
            elif field == ".K":
                key_words = line

    return documents

def read_data2(filename: str) -> dict:
    raw = np.loadtxt(filename, dtype=str, delimiter="someneverhappeningstr")
    print(raw[0])

    keep = True

    term_dic = {}

    for row in raw:
        if row.startswith("."):
            if row.startswith(".T") or row.startswith(".W") or row.startswith(".K"):
                keep = True
            else:
                keep = False
        elif keep:
            row.replace("; ", " ")
            row.replace(", ", " ")
            row.replace("(", "")
            row.replace(")", "")

            terms = row.split(" ")  # TODO must split along other word sep too
            for term in terms:
                if term not in term_dic:
                    term_dic[term] = 0
                term_dic[term] += 1
    return term_dic

if __name__ == '__main__':
    # read_data("data/CACM/cacm.all")
    print(read_data2("data/CACM/cacm.all"))