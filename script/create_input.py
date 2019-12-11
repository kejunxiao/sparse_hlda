#coding=utf8
# ========================================================
#   Copyright (C) 2019 All rights reserved.
#   
#   filename : create_input.py
#   author   : ***
#   date     : 2019-12-10
#   desc     : 
# ======================================================== 
import os
import sys

from collections import defaultdict

if __name__ == "__main__":
    with open(sys.argv[1], "r") as fin:
        doc = defaultdict(int)
        last_url = None
        for line in fin:
            items = line.strip().split("\t")
            if len(items) == 5:
                url, _, word, _, _ = items
            elif len(items) == 3:
                url, _, word = items
            else:
                continue
            if last_url and last_url != url:
                print(last_url + "\t" + " ".join("%s:%d:-1" % (word, freq) for word, freq in doc.items()))
                doc.clear()
            doc[word] += 1
            last_url = url
                    
