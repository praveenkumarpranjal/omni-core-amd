# Models Directory

Place your GGUF model files here.

## Recommended Models

- SmolLM2-1.7B (fast, good for testing)
- LLaMA-3.2-3B (balanced)
- Gemma-2-2B (high quality)

## Download Models

You can download models from:
- Hugging Face: https://huggingface.co/models?library=gguf
- TheBloke's collection: https://huggingface.co/TheBloke

## Example

```bash
# Download a model (example)
wget https://huggingface.co/...model.gguf -O models/model.gguf

# Run inference
./target/release/omni-core -m models/model.gguf -p "Hello" -n 100
```

