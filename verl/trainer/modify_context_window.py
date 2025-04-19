from transformers import GPTNeoXForCausalLM, GPTNeoXConfig, AutoTokenizer
import os

def modify_context_window(model_path, output_path, new_context_window=64000):
    # Load configuration and tokenizer
    config = GPTNeoXConfig.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    # Extend context window
    config.max_position_embeddings = new_context_window
    # Set rope-scaling 
    config.rope_scaling = {"type":"dynamic","factor":32.0}
    # Load model with updated config
    model = GPTNeoXForCausalLM.from_pretrained(model_path, config=config)

    # Save the modified model
    os.makedirs(output_path, exist_ok=True)
    model.save_pretrained(output_path)
    tokenizer.save_pretrained(output_path)
    print(f"Model with {new_context_window} context window saved to {output_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--context_window", type=int, default=64000)
    args = parser.parse_args()
    
    modify_context_window(args.model_path, args.output_path, args.context_window) 