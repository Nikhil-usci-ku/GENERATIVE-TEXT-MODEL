from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling, Trainer, TrainingArguments, pipeline
from datasets import load_dataset
import torch

# Check for GPU and set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading WikiText-2...")
# We're using half of the training split of WikiText-2
dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train[:50%]")

# Initialize GPT-2 tokenizer and model
model_name = "gpt2"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token # Essential for consistent padding

model = GPT2LMHeadModel.from_pretrained(model_name).to(device)

# Function to tokenize the dataset for the model
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=64)

print("🔄 Tokenizing dataset...")
tokenized_dataset = dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# Data collator prepares batches for training
data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Configure training parameters
training_args = TrainingArguments(
    output_dir="./gpt2-finetuned-wikitext2",
    per_device_train_batch_size=8,
    num_train_epochs=1, # Training for one epoch
    save_steps=500,
    logging_steps=100,
    fp16=True, # Use mixed precision for faster training if GPU is available
    overwrite_output_dir=True,
    report_to="none"
)

# Set up the Hugging Face Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
    data_collator=data_collator
)

print("Training GPT-2 (1 epoch, 50% WikiText-2, GPU)...")
trainer.train() # Start the training!

print("Saving model...")
# Save the fine-tuned model and tokenizer for later use
model.save_pretrained("./gpt2-finetuned-wikitext2")
tokenizer.save_pretrained("./gpt2-finetuned-wikitext2")

# Prepare for text generation
device = 0 if torch.cuda.is_available() else -1
generator = pipeline("text-generation", model="./gpt2-finetuned-wikitext2", tokenizer="./gpt2-finetuned-wikitext2", device=device)

# Interactive loop for generating text
while True:
    prompt = input("Enter your prompt (or type 'exit'): ")
    if prompt.strip().lower() == "exit":
        break
    output = generator(prompt, max_length=100, num_return_sequences=1)
    print("\n GPT-2 Output:\n")
    print(output[0]['generated_text'])
    print("=" * 80)