import sys

visited = set([])

with open(sys.argv[1]) as f:
    for line in f.readlines():
        if line.startswith("# ") and line[2:20] not in visited:
            print(line)
            visited.add(line[2:20])
