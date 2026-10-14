// Command harness sends one prompt to a model backend and prints the reply.
package main

import (
	"context"
	"flag"
	"fmt"
	"io"
	"os"
	"strings"

	"github.com/abinashstack/harness/internal/provider"
)

func main() {
	if err := run(); err != nil {
		fmt.Fprintln(os.Stderr, "harness:", err)
		os.Exit(1)
	}
}

func run() error {
	name := flag.String("provider", "anthropic", "anthropic | openai | ollama")
	model := flag.String("model", "", "model id (default claude-sonnet-5-5 for anthropic; required otherwise)")
	baseURL := flag.String("base-url", "", "override the backend base URL")
	system := flag.String("system", "", "system prompt")
	maxTokens := flag.Int("max-tokens", 0, "max output tokens (0 = adapter default)")
	flag.Parse()

	cfg, err := presetConfig(*name)
	if err != nil {
		return err
	}
	if *model != "" {
		cfg.Model = *model
	}
	if cfg.Model == "" {
		return fmt.Errorf("-model is required for provider %q", *name)
	}
	if *baseURL != "" {
		cfg.BaseURL = *baseURL
	}

	prompt := strings.Join(flag.Args(), " ")
	if prompt == "" {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		prompt = string(b)
	}
	if strings.TrimSpace(prompt) == "" {
		return fmt.Errorf("no prompt: pass it as arguments or on stdin")
	}

	p, err := provider.New(cfg)
	if err != nil {
		return err
	}
	resp, err := p.Chat(context.Background(), provider.Request{
		System:    *system,
		Messages:  []provider.Message{provider.UserText(prompt)},
		MaxTokens: *maxTokens,
	})
	if err != nil {
		return err
	}
	fmt.Println(resp.Text())
	fmt.Fprintf(os.Stderr, "[%s, %s, %d in / %d out tokens]\n", p.Name(), resp.StopReason, resp.Usage.InputTokens, resp.Usage.OutputTokens)
	return nil
}

func presetConfig(name string) (provider.Config, error) {
	switch name {
	case "anthropic":
		return provider.Config{Name: name, Wire: provider.WireAnthropic, APIKeyEnv: "ANTHROPIC_API_KEY", Model: "claude-sonnet-5-5"}, nil
	case "openai":
		return provider.Config{Name: name, Wire: provider.WireOpenAI, APIKeyEnv: "OPENAI_API_KEY"}, nil
	case "ollama":
		return provider.Config{Name: name, Wire: provider.WireOpenAI, BaseURL: "http://localhost:11434"}, nil
	}
	return provider.Config{}, fmt.Errorf("unknown provider %q", name)
}
