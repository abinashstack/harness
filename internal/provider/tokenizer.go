package provider

import (
	"context"
	"sync"
)

// HeuristicTokenizer estimates token counts from byte counts without embedding
// a BPE vocabulary. It is used only as a guard for context-window budgeting,
// with a wide safety margin, so approximate is fine.
//
// The estimate is bytes/ratio plus small per-part and per-message overheads.
// ratio is model- and wire-specific and self-corrects: after every real model
// call the agent feeds the exact prompt-token count back via Calibrate, and the
// ratio converges on the true tokenizer within a few turns.
type HeuristicTokenizer struct {
	mu     sync.Mutex
	ratios map[string]float64 // keyed by model
	seed   float64            // starting ratio for an unseen model
}

// Default starting ratios: characters per token, measured roughly across
// English-plus-code prompts.
const (
	ratioAnthropic = 3.6
	ratioOpenAI    = 4.0
)

// NewHeuristicTokenizer returns a tokenizer seeded for the given wire.
func NewHeuristicTokenizer(wire Wire) *HeuristicTokenizer {
	seed := ratioAnthropic
	if wire == WireOpenAI {
		seed = ratioOpenAI
	}
	return &HeuristicTokenizer{ratios: map[string]float64{}, seed: seed}
}

func (h *HeuristicTokenizer) ratioFor(model string) float64 {
	h.mu.Lock()
	defer h.mu.Unlock()
	if r, ok := h.ratios[model]; ok {
		return r
	}
	return h.seed
}

// Calibrate nudges the ratio for model toward the observed truth: estBytes is
// what CountMessages measured in bytes for a prompt, actualTokens is the exact
// input-token count the provider reported for that same prompt. An
// exponential moving average keeps one noisy sample from swinging the estimate.
func (h *HeuristicTokenizer) Calibrate(model string, estBytes, actualTokens int) {
	if estBytes <= 0 || actualTokens <= 0 {
		return
	}
	observed := float64(estBytes) / float64(actualTokens)
	h.mu.Lock()
	defer h.mu.Unlock()
	cur, ok := h.ratios[model]
	if !ok {
		cur = h.seed
	}
	h.ratios[model] = 0.7*cur + 0.3*observed
}

// CountMessages estimates the token size of a full prompt.
func (h *HeuristicTokenizer) CountMessages(_ context.Context, model, system string, msgs []Message, tools []ToolSchema) (int, error) {
	return h.estimate(model, system, msgs, tools), nil
}

// EstimateBytes reports the raw byte total CountMessages works from. The agent
// records this alongside a call so it can Calibrate afterward.
func (h *HeuristicTokenizer) EstimateBytes(system string, msgs []Message, tools []ToolSchema) int {
	b := len(system)
	for _, m := range msgs {
		b += 8 // per-message framing
		for _, p := range m.Content {
			b += 4 // per-part framing
			switch p.Type {
			case PartText:
				b += len(p.Text)
			case PartToolUse:
				if p.ToolCall != nil {
					b += len(p.ToolCall.Name) + len(p.ToolCall.Input)
				}
			case PartToolResult:
				if p.ToolResult != nil {
					b += len(p.ToolResult.Content)
				}
			}
		}
	}
	for _, t := range tools {
		b += len(t.Name) + len(t.Description) + len(t.InputSchema)
	}
	return b
}

func (h *HeuristicTokenizer) estimate(model, system string, msgs []Message, tools []ToolSchema) int {
	bytes := h.EstimateBytes(system, msgs, tools)
	ratio := h.ratioFor(model)
	if ratio <= 0 {
		ratio = h.seed
	}
	return int(float64(bytes)/ratio) + 1
}
