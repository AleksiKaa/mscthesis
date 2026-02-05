from transformers import pipeline
import torch

# Initialize the pipeline
pipe = pipeline(
  "text-generation", # Task type
  model="mistralai/Mistral-7B-Instruct-v0.3", # Model name
  device_map="auto", # Let the pipeline automatically select best available device
  max_new_tokens=1000
)

# Prepare prompts
messages = [
  {"role": "system", "content": "You're an helpful assistant. Answer to the questions with the best of your abilities."},
  {"role": "user", "content": "Continue the following sequence: 1, 2, 3, 5, 8"},
]

# Generate text and print the response
response = pipe(messages, return_full_text=False)[0]["generated_text"]
print(response)