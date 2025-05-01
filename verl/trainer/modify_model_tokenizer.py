# Create and save the custom tokenizer
from rich import print
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from typing import List
import os
import argparse

def return_vocabulary(width: int, height: int):
    """Generates the custom vocabulary list."""
    vocabulary: List[str] = []
    # Cell identifiers (c0 to c(width*height - 1))
    vocabulary += [f"c{i}" for i in range(width * height)]
    # Coordinates (0 to max(width, height)-1)
    vocabulary += [str(i) for i in range(max(width, height))]
    # Action/marker tokens
    vocabulary += ["start", "goal", "wall", "create", "close", "plan", "query", "reasoning", "solution", "end"]
    return vocabulary

def create_custom_tokenizer(
    width: int,
    height: int,
    output_save_dir: str,
    model_path: str,
    context_window: int,
    rope_scaling_type: str,
    rope_scaling_factor: float,
    original_max_position_embeddings: int,
    special_tokens: List[str] = ["<|pad|>", "<|unk|>", "<|bos|>", "<|eos|>"]
):
    """Creates and saves a custom tokenizer and model with the specified parameters."""
    # 1. Generate Vocabulary
    custom_vocab = return_vocabulary(width, height)
    full_vocab_list = custom_vocab + special_tokens
    vocab_map = {token: i for i, token in enumerate(full_vocab_list)}
    print(f"Total vocabulary size: {len(vocab_map)}")
    print(f"First 10 vocab items: {list(vocab_map.items())[:10]}")
    print(f"Last 10 vocab items: {list(vocab_map.items())[-10:]}")

    # 2. Initialize and Train Tokenizer (using WordLevel requires a map)
    tokenizer = Tokenizer(WordLevel(vocab=vocab_map, unk_token="<|unk|>"))
    tokenizer.pre_tokenizer = Whitespace() # type: ignore
    num_added_toks = tokenizer.add_special_tokens(special_tokens)
    print(f"Added {num_added_toks} special tokens.")
    if "<|pad|>" in special_tokens:
        pad_token_id = vocab_map["<|pad|>"]
        tokenizer.enable_padding(pad_id=pad_token_id, pad_token="<|pad|>")

    # 3. Save the Tokenizer
    os.makedirs(output_save_dir, exist_ok=True)
    tokenizer_save_path = os.path.join(output_save_dir, "tokenizer.json")
    tokenizer.save(tokenizer_save_path)
    print(f"Custom tokenizer saved to: {tokenizer_save_path}")

    # 4. Load and modify the model
    from transformers.models.gpt_neox import GPTNeoXConfig, GPTNeoXForCausalLM
    vocab_size = len(vocab_map)
    print(f"Initializing model ({model_path}) with vocab_size = {vocab_size}")

    config = GPTNeoXConfig.from_pretrained(model_path)
    print(f"Config before override: {config}")

    # Override config with custom settings
    config.vocab_size = vocab_size
    config.max_position_embeddings = context_window
    if context_window > 2048:
        config.rope_scaling = {"type":rope_scaling_type,"factor":rope_scaling_factor}
    if original_max_position_embeddings is not None:
        config.rope_scaling = {"type":rope_scaling_type,"factor":rope_scaling_factor, "original_max_position_embeddings":original_max_position_embeddings}
    config.pad_token_id = vocab_map["<|pad|>"]
    config.bos_token_id = vocab_map["<|bos|>"]
    config.eos_token_id = vocab_map["<|eos|>"]
    config.unk_token_id = vocab_map["<|unk|>"]

    print(f"Config after override: {config}")

    # Instantiate model with custom config
    model = GPTNeoXForCausalLM(config)
    orig_model = GPTNeoXForCausalLM.from_pretrained(model_path)
    print("Model initialized with custom vocab:", model)

    def count_params(m):
        return sum(p.numel() for p in m.parameters())

    print(f"Total params: {count_params(model)/1e6:.2f} M")
    print(f"Total params (original): {count_params(orig_model)/1e6:.2f} M")

    # 5. Save the model
    os.makedirs(output_save_dir, exist_ok=True)
    model.save_pretrained(output_save_dir)
    print(f"\nModel saved to: {output_save_dir}")

    # Save tokenizer in HuggingFace format
    from transformers.tokenization_utils_fast import PreTrainedTokenizerFast
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=tokenizer_save_path,
        unk_token="<|unk|>",
        pad_token="<|pad|>",
        bos_token="<|bos|>",
        eos_token="<|eos|>",
    )
    hf_tokenizer.save_pretrained(output_save_dir)
    print(f"HuggingFace tokenizer saved to: {output_save_dir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create a custom tokenizer and model for maze tasks")
    parser.add_argument("--maze_width", type=int, default=30, help="Width of the maze")
    parser.add_argument("--maze_height", type=int, default=30, help="Height of the maze")
    parser.add_argument("--output_save_dir", type=str, default="./custom_pythia_output", 
                        help="Directory to save the tokenizer and model")
    parser.add_argument("--model_path", type=str, default="EleutherAI/pythia-70m",
                        help="Base model to use")
    parser.add_argument("--context_window", type=int, default=2048)
    parser.add_argument("--rope_scaling_type", type=str, default="dynamic")
    parser.add_argument("--rope_scaling_factor", type=float, default=32.0)
    parser.add_argument("--original_max_position_embeddings", type=int, default=None)
    args = parser.parse_args()

    create_custom_tokenizer(
        width=args.maze_width,
        height=args.maze_height,
        output_save_dir=args.output_save_dir,
        model_path=args.model_path,
        context_window=args.context_window,
        rope_scaling_type=args.rope_scaling_type,
        rope_scaling_factor=args.rope_scaling_factor,
        original_max_position_embeddings=args.original_max_position_embeddings
    )