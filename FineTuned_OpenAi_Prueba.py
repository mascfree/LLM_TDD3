import pandas as pd
import pickle
import json
import time
from tqdm import tqdm
import os.path
from tenacity import ( retry, stop_after_attempt, wait_random_exponential ) #for exponential backoff
import argparse

#parser.add_argument('--train', '-t', type=str, help='The path to the train TSV file.')
#parser.add_argument('--test', '-s', type=str, help='The path to the test TSV file.')
#args = parser.parse_arsgs()

train_file = r"C:\Users\miguela.silva\Downloads\LLM_TDD\Dataset Creation\tmp\output\3052688\salida.tsv" #args.train
test_file =  r"C:\Users\miguela.silva\Downloads\LLM_TDD\Dataset Creation\tmp\output\3052688\salida.tsv" #args.test

input_column = 'description'
output_column = 'test_case'

train_dataset_path = train_file
test_dataset_path = test_file

# Reading the train and test files
df_train = pd.read_csv(train_dataset_path,delimiter='\t')
df_test = pd.read_csv(test_dataset_path,delimiter='\t')

#Setting up the fine-tuning parameters
fine_tuning_data = []

for i in tqdm(range(len(df_train))):
    description_content = df_train[input_column][i]
    test_case = df_train[output_column][i]
    fine_tuning_data.append({
        "prompt": description_content,
        "completion": test_case
    })


from openai import OpenAI

client = OpenAI(
   
)

response = client.files.create(
    file=open(r"C:\Users\miguela.silva\Downloads\LLM_TDD\Dataset Creation\tmp\output\3052688\salida.jsonl", "rb"),
    purpose="fine-tune"
)

print(response)
print(type(response))
id =  response.id

client.fine_tuning.jobs.create(
  training_file=id, 
  model="gpt-3.5-turbo"
)

fine_tuning_job  = client.fine_tuning.jobs.retrieve(id)

print(fine_tuning_job )

