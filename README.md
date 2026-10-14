# harness

A small Go CLI and provider layer for talking to model backends through one
message model.

Two wire adapters cover:

- **Anthropic** (`/v1/messages`) — api.anthropic.com
- **OpenAI-compatible** (`/v1/chat/completions`) — api.openai.com, Ollama, vLLM

Both support chat, SSE streaming and tool calls.

## Usage

```
go build -o harness ./cmd/harness

export ANTHROPIC_API_KEY=...
./harness "Say hello"                                  # anthropic, claude-sonnet-5-5
./harness -provider openai -model <model> "Say hello"  # needs OPENAI_API_KEY
./harness -provider ollama -model llama3 "Say hello"   # localhost:11434
echo "Say hello" | ./harness
```

Flags: `-provider`, `-model`, `-base-url`, `-system`, `-max-tokens`.
Token usage is printed to stderr.

## Layout

- `cmd/harness` — the CLI
- `internal/provider` — `Provider` interface, adapters, SSE scanner, heuristic tokenizer

## Tests

```
go test ./...
```

The tests run against a local fake server; no API keys needed.
