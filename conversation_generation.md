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

### GPU Acceleration (optional)
Llama.cpp supports GPU Acceleration with CUBLAS. First ensure you have applicable hardware such as an NVIDIA graphics card. Then, install [`nvidia-cuda-toolkit`](https://developer.nvidia.com/cuda-downloads) on your system, ensuring `nvcc` runs.

Then, we can instead build with `make LLAMA_CUBLAS=1`. When starting the binary, we can also use the runtime argument `-ngl` to specify a number of gpu layers.

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

Next, we need a base configuration file. Copy the following into `example.json` in your `llama.cpp` directory.

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

If you built `llama.cpp` with `CUBLAS`, you can also pass in the `ngl` GPU layer argument directly into the `get-word-conversations.py` script.

```console
python get-word-conversations.py example.json -ngl 1000 >> /tmp/example-conv.txt
```

## Conversations on a Specific Topic
Lets build a configuration file for a specific topic, in this case physics. We can create one from an online resource or generate one.

### Online Resource Generated Lists (recommended)
For this example, we will use Wikipedia's glossary of physics terms.

```console
wget -O - https://en.wikipedia.org/wiki/Glossary_of_physics  | grep '<dt class="glossary" .*<a .*>.*</a></dfn></dt>' | sed 's:.*><a .*>\(.*\)</a>.*:"\1",:g' > /tmp/physic-terms.txt
```

### Model Generated Lists
We can run something like the following to generate a list of physics related terms, but (in the writers experience at least) it generally has mixed results.

```console
./main -m phi-2.Q4_K_M.gguf -p 'The following is a list of 1000 physics terms. There will be NOTHING except physics terms next:\n"gravity", "force", "energy", "absolute zero", ' --escape
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
#!/bin/bash

if [ $# -ne 2 ]; then
    echo usage: run.sh config.json output.txt
    exit
fi

while true; do
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
To prepare the conversations for training, we need to remove lines without proper terms (that don't start with STUDENT, TEACHER, TERM), remove trailing lines (conversations should always end with the teacher), and change the key terms to symbols. We can use the `format.py` script in the `chat-builder` repo to do so.

```console
python format.py /tmp/physics-conv.txt >> prepped-physics-conv.txt
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

For pretokenizing, `tinytext.py` will look for our conversation in `data/tinytext.txt` so lets move it there.

```console
mkdir ./llama2.c/data
mv prepped-physics-conv ./llama2.c/data/tinytext.txt
```

We can then run the following to finish our preperations.
```console
python tinytext.py pretokenize
```

After `data/tinytext.bin` is generated, we can use `train.py` to train our model. Adjust the hyperparameters as needed.
```console
python -m train.py --compile=False --eval_iters=10 --batch_size=2
```

# Inference
When our model is complete, we can convert it to `gguf` format with `llama.cpp` tools.

## Conversion
We will have to set some environmental variables first. The one that matters is `file`, which is the `.pt` file `train.py` should have generated in an `out` folder.

```console
file='dir/to/llama2.c/out/ckpt.pt'
params='/tmp/params.json'
out='/tmp/out.gguf'
```

We can then convert our checkpoint file into `/tmp/out.gguf`
```console
python -c "import torch; import json; c = torch.load('$file')['model_args'];c.setdefault('norm_eps', 1e-05) ; print(json.dumps(c))" > "$params"
python convert.py --outfile "$out" --vocab-dir . "$file" --ctx 512 --pad-vocab
```

With our `gguf` file in hand, we can have a conversation with it using the following.

## Testing
Once our `.gguf` file is obtained, we can test it with the following.

```console
./main --log-disable -m /tmp/out.gguf --in-prefix '' --escape --reverse-prompt '\n= ' -i
```
