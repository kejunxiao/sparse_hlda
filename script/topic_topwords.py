import sys

with open(sys.argv[1], "r") as fin:
    for line in fin:
        words = []
        for w_s in line.strip().split(" "):
            items = w_s.split(":")
            if len(items) != 2:
                continue
            w, s = items
            words.append((w, int(s)))
        words = sorted(words, key=lambda x:x[-1], reverse=True)
        words_str = " ".join("%s:%d" % (w, s) for w, s in words)
        print(words_str)

