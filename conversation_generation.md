# Conversation Generation
The end goal is to generation training data in the form of conversations for a language model. For this, we will use `llama.cpp`.

## Setup
We'll first need to clone the repository, build the project, and download the model weights. We'll use a "small" model, Phi 2B. We can always use larger models (e.g. Lamma2 7B) if needed.

```console
git clone https://github.com/ggerganov/llama.cpp.git
cd llama.cpp
wget https://huggingface.co/TheBloke/phi-2-GGUF/resolve/main/phi-2.Q4_K_M.gguf
```

Run `make` to build the project, or see below to add support for gpu acceleration.

### GPU Acceleration
Llama.cpp supports GPU Acceleration with CUBLAS. First ensure you have applicable hardware such as an NVIDIA graphics card. Then, install [`nvidia-cuda-toolkit`](https://developer.nvidia.com/cuda-downloads) on your system, ensuring the command `nvcc` works.

Then, we can instead bulid with `make LLAMA_CUBLAS=1`. When starting llama.cpp, we can also use the runtime argument `-ngl` to specify any number of gpu layers.

```
make LLAMA_CUBLAS=1
./main -m phi-2.Q4_K_M.gguf -i -ngl 1000
```

## Testing
If everything ran properly, you should be able to chat and ask questions by running the following:

```console
MODEL=phi-2.Q4_K_M.gguf ./examples/chat-13B.sh
```

# Creating Conversations with the Chat Builder
We can use the `chat-builder` tool to generate conversations on keywords, directing the model with a configuration file. This `config.json` contains the words list, the input prompt, and other kinds of information on what the input response should and should not contain.

## Basic Setup
First, lets clone the repository and copy the script over.

```console
git clone https://github.com/persimmonsai/chat-builder.git
cp chat-builder/get-word-conversations.py /dir/to/llama.cpp/
```

Next, we need a base configuration file. We can use [this example](https://drive.google.com/file/d/1hb9TBalreZ9jzNa925MH7yMjhCcWyCBV/view). Either download it or copy it below into an `example.json` inside your `llama.cpp` directory.

```json
{
	"name": "example",
	"model": "phi-2.Q4_K_M.gguf",
	"prompt_lines": [
		"We end this document with a conversation between a student and a teacher, where the teacher conveys the meaning of \"{word}\"",
		"STUDENT:" 
	],
	"words" : [
		"bridge",
		"leaf",
		"pocket",
		"bear",
		"promises",
		"room",
		"village",
		"camera"
	],
	"exclude": [
		"relig",
		"sex",
		"gender",
		"politic",
		"kill"
	]
}
```

Now we can start generating conversations.

```console
python get-word-conversations.py example.json >> /tmp/example-conv.txt
```

## Conversations on a Specific Topic
Lets build a configuration file for generating conversations on a specific topic, in this case physics. We can ask a model for a list or create one directly.

### Model Generated Lists
We can run something like the following to generate a list of physics related terms.

```console
./main -m phi-2.Q4_K_M.gguf -p 'here is a list of 10000 terms about physics:\n"gravity", "force", "energy", "absolute zero", ' --escape
```

Generally we want the format to be in the form of terms surrounded by quotation marks with commas and newlines in between, but this is something we can easily change later.

### Online Resource Generated Lists
We may also generate lists from in online resource like a glossary, in this case Wikipedia.

```console
wget -O - https://en.wikipedia.org/wiki/Glossary_of_physics  | grep '<dt class="glossary" .*<a .*>.*</a></dfn></dt>' | sed 's:.*><a .*>\(.*\)</a>.*:"\1",:g' > /tmp/physic-terms.txt
```

### Building the Configuration File
With our words list generated, we can transfer them into the `words` array in our new `physics.json` config file, derived from the example above. We can also make sure to adjust our prompt to say "...between a student and a **physics** teacher..."

From there, we can generate conversations like before.

```console
python get-word-conversations.py physics.json >> /tmp/physics-conv.txt
```

Experiment with your configuration file based on output, adjutsing the prompt, words, and excluded words as needed to ensure conversations remain on topic.

### Full Sending Generation
If all loks okay, we can use a simple bash script to generate conversations indefinitely. Note that conversations are written to the output file in "chunks", and you won't immediately see the written output until one iteration of the entire python script is completed.

Put the following into a `run.sh` script and run it to generate conversations.

```console
while True; do
	python get-word-conversations.py $1 >> $2
done
```

This takes arguments for the input configuration file and the output file, letting us generate conversations like this:

```console
./run.sh physics.json /tmp/physics-conv.txt
```

# Training a Model
If we have generated "enough" conversations, we can prep our data & train a model.

## Conversation Formatting
First, we can remove every line that doesn't begin with `STUDENT`, `TEACHER` and `TERM`.

```console
cat /tmp/conv.txt | grep -v '^STUDENT:' | grep -v '^TEACHER:' | grep -v '^TERM:' | grep -v '^$'
```

Next, lets remove trailing `STUDENT` questions/comments, ensuring the `TEACHER` always speaks last.

```console
cat /tmp/conv.txt | tr '\n' @ | sed 's:@@:\n:g' | sed 's/@STUDENT: [^@][^@]*$//g' | sed 's:$:\n:g' | tr '@' '\n'
```

Finally, lets convert `STUDENT`, `TEACHER`, and `TERM` into symbols.

```console
cat /tmp/conv.txt | sed 's/STUDENT:/=/g' | sed 's/TEACHER:/#/g' | sed 's/TERM:/~/g'
```

## Training
For training, we will utilize `tinytext.py` and `train.py` from [`personimmonsai's llama2.c`](https://github.com/persimmonsai/llama2.c).

After cloning the repo, we can copy the training parameters used by `smol-llama-101M` in the `train.py` file.

```console
git clone https://github.com/persimmonsai/llama2.c
```


```bash
diff old_train.py new_train.py
 # -----------------------------------------------------------------------------
 # I/O
 out_dir = "out"
-eval_interval = 2000
+eval_interval = 100
 log_interval = 1
 eval_iters = 100
 eval_only = False  # if True, script exits right after the first eval
@@ -49,15 +49,15 @@ wandb_project = "llamac"
 wandb_run_name = "run" + datetime.now().strftime("%Y_%m_%d_%H_%M_%S")
 # data
 batch_size = 128  # if gradient_accumulation_steps > 1, this is the micro-batch size
-max_seq_len = 256
+max_seq_len = 512
 vocab_source = "llama2" # llama2|custom; use Lllama 2 vocab from Meta, or custom trained
 vocab_size = 32000 # the Llama 2 tokenizer has 32K tokens
-dataset = "tinystories"  # tinystories|tinyshakespeare|tinytext
+dataset = "tinytext"  # tinystories|tinyshakespeare|tinytext
 # model
-dim = 288
+dim = 768
 n_layers = 6
-n_heads = 6
-n_kv_heads = 6
+n_heads = 24
+n_kv_heads = 8
 multiple_of = 32
 dropout = 0.0
 # adamw optimizer
```

We can run the following to finish our preperations.
```console
python tinytext.py pretokenize
```

Letting us finally train our model
```console
python -m train.py --compile=False --eval_iters=10 --batch_size=2
```

# Inference
When our model is complete, we can convert it to `gguf` format with `llama.cpp` tools.

```console
file='/tmp/ckpt.pt'
params='/tmp/params.json'
out='/tmp/out.gguf'
python -c "import torch; import json; c = torch.load('$file')['model_args'] ;  c.setdefault('norm_eps', 1e-05) ; print(json.dumps(c))" > "$params"
python convert.py --outfile "$out" --vocab-dir . "$file" --ctx 512 --pad-vocab
```

With our `gguf` file in hand, we can have a conversation with it using the following.

```console
./main --log-disable -m /tmp/out.gguf --in-prefix '' --escsape --reverse-prompt '\n= ' -i
```
