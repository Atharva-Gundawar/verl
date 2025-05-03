import json
import pandas as pd
import os
from typing import List, Dict, Union
import logging
from torch.utils.data import Dataset
from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
from transformers import AutoTokenizer
import argparse

logging.basicConfig(level=logging.INFO)

class MazeDataset(Dataset):
    def __init__(self, tokenizer: PreTrainedTokenizerFast, data_path: str, max_length: int = 4096):
        self.tokenizer = tokenizer
        self.examples = []
        self.max_length = max_length
        
        # Load and process the data
        with open(data_path, 'r') as f:
            for line in f:
                example = json.loads(line)
                self.examples.append(example)
        
        logging.info(f"Loaded {len(self.examples)} examples from {data_path}")
    
    def __len__(self):
        return len(self.examples)
    
    def __getitem__(self, i) -> Dict:
        example = self.examples[i]
        
        # Construct the full sequence
        full_text = f"<|bos|> {example['text']} <|eos|>"
        
        # Find the position of "reasoning" to split into prompt and completion
        tokens = full_text.split()
        try:
            reasoning_idx = tokens.index("reasoning")
            
            # Initial split into prompt and completion strings
            prompt_str = " ".join(tokens[:reasoning_idx])  # includes "<|bos|> query ..."
            completion_str = " ".join(tokens[reasoning_idx:])  # includes "reasoning ... solution ... end <|eos|>"

            # Tokenize prompt and completion
            prompt_ids = self.tokenizer.encode(prompt_str)
            completion_ids = self.tokenizer.encode(completion_str)

            # Check combined length and truncate if necessary
            combined_len = len(prompt_ids) + len(completion_ids)
            if combined_len > self.max_length:
                overflow = combined_len - self.max_length
                if overflow >= len(completion_ids):
                    # Prompt alone is too long or leaves no room for completion
                    logging.warning(f"Prompt length ({len(prompt_ids)}) exceeds or matches max_length ({self.max_length}). Truncating prompt and setting completion empty for example {i}.")
                    prompt_ids = prompt_ids[:self.max_length]
                    completion_ids = [] # Effectively empty completion
                else:
                    # Truncate completion
                    completion_ids = completion_ids[:-overflow]

                # Decode back to strings
                prompt = self.tokenizer.decode(prompt_ids)
                completion = self.tokenizer.decode(completion_ids)
                logging.debug(f"Truncated example {i}. New completion length: {len(completion_ids)}")
            else:
                # Use original strings if no truncation needed
                prompt = prompt_str
                completion = completion_str

            # Extract plan from potentially truncated completion (after PLAN_START)
            if "<PLAN_START>" in completion:
                reasoning_str, plan_str = completion.split("<PLAN_START>", 1) # Split only once
                reasoning_str = reasoning_str.strip()
                plan_str = plan_str.strip()
            else:
                reasoning_str = completion
                plan_str = ""

            # Create the processed example in parquet format
            processed_example = {
                "data_source": "maze_dataset",
                "prompt": [{
                    "role": "user",
                    "content": prompt,
                }],
                "ability": "maze_planning",
                "reward_model": {
                    "style": "rule",
                    "ground_truth": plan_str
                },
                "extra_info": {
                    'split': 'train',  # This will be set during conversion
                    'index': i,
                    'answer': completion,
                    "question": prompt,
                    "_id": str(example.get('_id', 'N/A'))
                }
            }
            
            return processed_example
            
        except ValueError:
            logging.warning(f"Could not find 'reasoning' in example: {example}")
            return self.__getitem__((i + 1) % len(self))  # Skip this example and get next

def convert_to_parquet(input_jsonl: str, output_parquet: str, split: str, tokenizer_path: str, max_length: int):
    """
    Convert a JSONL file to Parquet using MazeDataset.
    
    Args:
        input_jsonl (str): Path to input JSONL file
        output_parquet (str): Path to output Parquet file
        split (str): Either 'train' or 'test'
        tokenizer_path (str): Path to the tokenizer
        max_length (int): Maximum sequence length for truncation.
    """
    logging.info(f"Processing {input_jsonl}...")
    
    # Initialize tokenizer and dataset
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    dataset = MazeDataset(tokenizer, input_jsonl, max_length=max_length)
    
    # Process all examples
    processed_data = []
    for i in range(len(dataset)):
        example = dataset[i]
        # Update split in extra_info
        example['extra_info']['split'] = split
        processed_data.append(example)
    
    # Convert to DataFrame
    df = pd.DataFrame(processed_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_parquet), exist_ok=True)
    
    # Save to Parquet
    logging.info(f"Saving {len(df)} examples to {output_parquet}...")
    df.to_parquet(output_parquet, index=False)
    logging.info("Conversion complete!")

def main():
    parser = argparse.ArgumentParser(description='Convert Maze JSONL files to Parquet format')
    parser.add_argument('--train_file', type=str, required=True,
                      help='Path to the training JSONL file')
    parser.add_argument('--test_file', type=str, required=True,
                      help='Path to the test JSONL file')
    parser.add_argument('--tokenizer_path', type=str, required=True,
                      help='Path to the tokenizer')
    parser.add_argument('--output_dir', type=str, required=True,
                      help='Directory to save the parquet files')
    parser.add_argument('--max_length', type=int, default=4096,
                      help='Maximum sequence length for truncation (default: 4096)')
    
    args = parser.parse_args()
    
    # Define input and output paths
    input_files = {
        'train': args.train_file,
        'test': args.test_file
    }
    
    # Process each file
    for split, input_file in input_files.items():
        output_file = os.path.join(args.output_dir, f'{split}.parquet')
        convert_to_parquet(input_file, output_file, split, args.tokenizer_path, args.max_length)

if __name__ == "__main__":
    main() 