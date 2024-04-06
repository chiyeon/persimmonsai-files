import pyarrow.parquet as pq
import pandas as pd
import sys

pd.set_option('display.expand_frame_repr', False)
table = pq.read_table(sys.argv[1])
print(table.to_pandas())
