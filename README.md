# GENERATIVE-TEXT-MODEL

*COMPANY*: CODETECH IT SOLUTIONS

*NAME*: NIKHIL KUMAR

*INTERN ID*: CODF69

*DOMAIN*: AIML

*DURATION*: 4 WEEKS

*MENTOR*: NEELA SANTHOSH KUMAR

> This repository contains a Python script to fine-tune a GPT-2 model on a subset of the WikiText-2 dataset and then use the fine-tuned model for interactive text generation.

> This project demonstrates how to adapt a pre-trained GPT-2 model to a specific corpus, enabling it to generate text that aligns more closely with the style and content of the fine-tuning data.

##   Description

This Python script demonstrates the process of fine-tuning a pre-trained GPT-2 language model on a subset of the WikiText-2 dataset and then using the fine-tuned model for text generation. It leverages the popular Hugging Face Transformers library, which provides powerful tools for working with state-of-the-art natural language processing models.

Key Components and Workflow:

Environment Setup and Device Selection:

The script begins by importing necessary libraries from transformers, datasets, and torch.
It intelligently checks for the availability of a CUDA-enabled GPU and sets the torch.device accordingly, ensuring that computations are performed on the GPU for faster training if available, otherwise defaulting to the CPU. This is crucial for efficient deep learning.
Dataset Loading and Preparation:

The wikitext-2-raw-v1 dataset is loaded using load_dataset. To reduce training time and resource consumption, only the first 50% of the training split is used.
The script then loads the gpt2 tokenizer from Hugging Face. A critical step here is setting tokenizer.pad_token = tokenizer.eos_token. This addresses a common issue where GPT-2's tokenizer doesn't have a dedicated padding token, which is required by many training setups. By assigning the end-of-sequence token as the padding token, it ensures smooth batch processing.
Model Loading and Tokenization:

The GPT2LMHeadModel is loaded, which is a standard GPT-2 model with a language modeling head on top, suitable for text generation tasks. The model is then moved to the selected device (GPU or CPU).
A tokenize_function is defined to process the raw text data. This function applies the loaded tokenizer, ensuring that each text example is padded to a max_length of 64 tokens and truncated if longer. This standardization is vital for creating uniform input batches for the model.
The dataset.map method efficiently applies this tokenize_function across the entire dataset, processing data in batches and removing the original "text" column as it's no longer needed.
Data Collator and Training Arguments:

DataCollatorForLanguageModeling is used to create batches of data. mlm=False indicates that Masked Language Modeling (MLM) is not being used, which is appropriate for a Causal Language Model like GPT-2 (which predicts the next token, not masked tokens).
TrainingArguments defines the various hyperparameters and settings for the training process. This includes the output directory for saving checkpoints, batch size (per_device_train_batch_size=8), number of training epochs (num_train_epochs=1), saving frequency (save_steps=500), logging frequency (logging_steps=100), enabling mixed-precision training (fp16=True for speedup on compatible GPUs), and ensuring the output directory is overwritten.
Model Training:

The Trainer class from transformers simplifies the training loop. It's initialized with the model, training arguments, tokenized dataset, and data collator.
trainer.train() initiates the fine-tuning process. During this phase, the GPT-2 model's weights are adjusted based on the WikiText-2 data, allowing it to learn domain-specific patterns and language nuances present in the dataset.
Model Saving and Text Generation Pipeline:

After training, the fine-tuned model and tokenizer are saved to a local directory (./gpt2-finetuned-wikitext2). This allows for easy reloading and deployment without needing to retrain.
A pipeline for "text-generation" is created, providing a high-level API for inference. It uses the newly saved fine-tuned model and tokenizer. The device is set to 0 for the first GPU or -1 for CPU.
Interactive Text Generation:

The script concludes with an interactive loop where the user can enter prompts. The fine-tuned GPT-2 model then generates text continuations based on the provided input, demonstrating its learned language capabilities. The generated text is printed, allowing immediate interaction with the fine-tuned model.
Points to Note:

Efficiency: Using fp16=True and training on a GPU significantly speeds up the training process for large models like GPT-2.
Fine-tuning Benefits: Fine-tuning a pre-trained model on a specific dataset allows it to adapt to the style, vocabulary, and topics of that dataset, leading to more relevant and coherent generations in that domain.
Scalability: The Hugging Face Trainer and pipeline abstractions make it easy to scale this process to larger datasets and more complex models.




























