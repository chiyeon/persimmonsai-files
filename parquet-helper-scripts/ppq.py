"""
PPQ
PERSIMMONS PARQUET (helper)

a command line utility to help you read, create, and combine parquet files for conversation data.

based on the following format:

QUERY              | TERMS       | RESPONSE                            | TOPIC
"This is a query." | [ "query" ] | "This is a response to your query." | null

"""

import pyarrow.parquet as pq
import pyarrow as pa
import numpy as np
import pandas as pd
import sys
import os

USAGE = "USAGE: ppq <read, write, append, combine> ...\n" \
         "\tread: previews parquet file\n" \
         "\tcreate: creates new parquet file from a text dataset\n" \
         "\tappend: adds a text dataset onto an existing parquet file\n" \
         "\tcombine: combines a number of parquet files into one"
USAGE_READ = "USAGE: ppq read data1.parquet data2.parquet ..."
USAGE_CREATE = "USAGE: ppq create in.txt out.parquet <optional topic>"
USAGE_APPEND = "USAGE: ppq append base.parquet new.txt <optional topic>"
USAGE_COMBINE = "USAGE: ppq combine in1.parquet in2.parquet ... out.parquet"
COMMANDS = [ "read", "create", "append", "combine" ]

pd.set_option('display.expand_frame_repr', False)

# if condition is false, print error and stop program
# if exit is False, instead of stopping return True on error and False on safe
def check(condition, error, exit_on_false = True):
   if not condition:
      print(error)
      if exit_on_false: exit(0)
   return not condition

# COMMAND FUNCTIONS
def read_table(path):
   check(os.path.exists(path), f"File path \"{path}\" doesn't exists.")
   check(".parquet" in path, f"\"{path}\" is not a parquet file.")

   return pq.read_table(path).to_pandas()

# outputs preview of parquet file(s)
def read(args):
   check(len(args) > 0, USAGE_READ)

   # preview each file (if exists)
   for arg in args:
      table = read_table(arg)
      print(arg)
      print(table) 

# creates pandas table from a text dataset file
def create_table(path, topic):
   data = { "query": [], "terms": [], "response": [], "topic": [] }

   check(os.path.exists(path), f"File path \"{path}\" doesn't exist.")

   with open(path) as f:
      for conv in f.read().split("\n\n"):
         lines = conv.split("\n")
         if len(lines) < 3: continue

         data["query"].append(lines[0][2:])
         data["terms"].append([ lines[1][2:] ])
         data["response"].append(lines[2][2:])
         data["topic"].append(topic)

   #return pd.DataFrame(data, index=list(range(0, len(data["query"]))))
   return pd.DataFrame(data)

def create(args):
   check(len(args) >= 2, USAGE_CREATE)

   data_path = args[0]
   output_path = args[1]
   topic = args[2] if len(args) >= 3 else None

   table = create_table(data_path, topic)
   pq.write_table(pa.Table.from_pandas(table), output_path)

   print("Generated successfully!")
   read([ output_path ])

def append(args):
   check(len(args) >= 2, USAGE_APPEND)
   
   data_path = args[1]
   base_path = args[0]
   topic = args[2] if len(args) >= 3 else None

   new_data = create_table(data_path, topic)
   base_data = read_table(base_path)
   extended_data = pd.concat([ base_data, new_data ])
   
   pq.write_table(pa.Table.from_pandas(extended_data), base_path)

   print("Generated successfully!")
   print(f"From {base_data.shape} to {extended_data.shape}")
   print(extended_data)

def main():
   # some basic checks
   check(len(sys.argv) >= 2, USAGE)
   check(sys.argv[1] in COMMANDS, USAGE)

   cmd = sys.argv[1]
   args = sys.argv[2:]

   if cmd == "read": read(args)
   elif cmd == "create": create(args)
   elif cmd == "append": append(args)

if __name__ == "__main__":
   main()
