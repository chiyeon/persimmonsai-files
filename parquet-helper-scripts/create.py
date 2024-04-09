import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import sys

if len(sys.argv) < 2:
    print("USAGE: python create.py /path/to/file out.parquet")
    exit(0)

topic = sys.argv[3] if len(sys.argv) >= 4 else None

data = {
    "query": [],
    "terms": [],
    "response": [],
    "topic": []
}

with open(sys.argv[1]) as f:
    for conv in f.read().split("\n\n"):
        lines = conv.split("\n")
        if len(lines) < 3:
            continue
        data["query"].append(lines[0][2:])
        data["terms"].append([ lines[1][2:] ])
        data["response"].append(lines[2][2:])
        data["topic"].append(topic)


df = pd.DataFrame(data, index=list(range(0, len(data["query"]))))

table = pa.Table.from_pandas(df)

pq.write_table(table, sys.argv[2])
