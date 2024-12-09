import os
import sys
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig, BitsAndBytesConfig
import torch
import bitsandbytes as bnb
import pandas as pd
import random
import transformers

from huggingface_hub import login   #importing the login function
from bertviz import model_view  #importing the model_view module

with open('hg.token','r') as ft:   #putting in the huggingfcae token and hidind it  putting it into a file
        token = ft.read().strip()

login(token=token) #Log into huggingface

# Use a pipeline as a high-level helper
from transformers import pipeline

pipe = pipeline("text-generation", model="ibm/merlinite-7b")

# Load model directly
from transformers import AutoTokenizer, AutoModelForCausalLM

tokenizer = AutoTokenizer.from_pretrained("ibm/merlinite-7b")
model = AutoModelForCausalLM.from_pretrained("ibm/merlinite-7b") #changed the bnb config and cuda map

#sys_prompt = "You are an AI language model developed by IBM Research. You are a cautious assistant. You carefully follow instructions. You are helpful and harmless and you follow ethical guidelines and promote positive behavior."
sys_prompt = """You are an AI language model assigned to generate variation in the target question using various psycholinguistic parameters like concreteness and noise, where Concreteness is defined as a word's total amount of tangible words. High concreteness means a sentence has sufficient real objects and well-defined words that make a meaningful sentence with a high number of specifics. Low concreteness means a sentence has a large amount of abstract words and undefined objects that still make a meaningful sentence but lack a clear story. E.g. table, chair, coffee, guitar, quadratic equation, etc. (high concreteness) E.g., object, drink, musical instrument, math equation, etc. (low concreteness).And assess the base level noise in the question and then calibrate what is low, medium, or high noise based on the presence of distracting details and irrelevant details in the question. Ensure the noise still has narrative cohesion. Don't add logically impossible scenarios. Make another question using the same competency measured by the reference question. """

input = "generate variations for the target question: 1 dollar is 85 rupees. Who has more money? Person A with 12 dollars. Or a person B with 1000 rupees? "

prompt = f'<|system|>\n{sys_prompt}\n<|user|>\n{input}\n<|assistant|>\n' #changed to input
stop_token = '<|endoftext|>'

inputs = tokenizer.encode(prompt, return_tensors="pt")                     #converting the input into tensors the model can understand
attention_mask = torch.ones_like(inputs) #changed .to('cuda')                                  #giving the input 1s indicating the should e attended  te attention layer
attention_mask = attention_mask #Line added 25/10/24            #running the attention mask on GPU
inputs = inputs #Line added 25/10/24 #changed cuda                           #running the inputs also on GPU
output = model.generate(inputs=inputs, attention_mask=attention_mask, max_new_tokens = 250, do_sample = True) #asking model to generate text based on the input
generated_text = tokenizer.decode(output[0], skip_special_tokens=True)    #converting the tokens ack into readale form

print(generated_text)
