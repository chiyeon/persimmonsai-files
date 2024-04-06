import sys
import pyarrow.parquet as pq

if len(sys.argv) < 4:
    print("USAGE: python combine.py output.parquet file1.parquet file2.parquet ...")
    exit(1)

files = sys.argv[2:]

schema = pq.ParquetFile(files[0]).schema_arrow
with pq.ParquetWriter(sys.argv[1], schema=schema) as out:
    for f in files:
        out.write_table(pq.read_table(f, schema=schema))
