// Package provider abstracts model backends behind one message model. An
// Anthropic-wire and an OpenAI-wire adapter cover api.anthropic.com,
// api.openai.com, and any OpenAI-compatible base_url (Ollama, vLLM).
//
// The unified model is deliberately small: a Message is a Role plus a list of
// Parts, and a Part is text, a tool call, or a tool result. Each wire adapter
// translates that to and from its provider's JSON shape and nothing else in the
// tree needs to know which backend is in use.
package provider

import (
	"context"
	"encoding/json"
	"time"
)

// Role identifies who produced a Message.
type Role string

const (
	RoleUser      Role = "user"
	RoleAssistant Role = "assistant"
)

// PartType tags the populated field of a Part.
type PartType string

const (
	PartText       PartType = "text"
	PartToolUse    PartType = "tool_use"
	PartToolResult PartType = "tool_result"
)

// Part is one piece of a Message: prose, a model-issued tool call, or the
// result of running one. Exactly one of Text/ToolCall/ToolResult is meaningful,
// selected by Type.
type Part struct {
	Type       PartType
	Text       string
	ToolCall   *ToolCall
	ToolResult *ToolResult
}

// Text returns a text Part.
func Text(s string) Part { return Part{Type: PartText, Text: s} }

// ToolCall is a model's request to invoke a tool. Input is the raw JSON
// arguments object, passed through to the tool unmodified.
type ToolCall struct {
	ID    string
	Name  string
	Input json.RawMessage
}

// ToolResult is the outcome of running a ToolCall, addressed back to it by
// CallID. IsError lets the model see and react to a failed call.
type ToolResult struct {
	CallID  string
	Content string
	IsError bool
}

// Message is one conversational turn.
type Message struct {
	Role    Role
	Content []Part
}

// UserText is a convenience constructor for a plain user message.
func UserText(s string) Message {
	return Message{Role: RoleUser, Content: []Part{Text(s)}}
}

// ToolSchema advertises one tool to the model. InputSchema is a JSON Schema
// (draft 2020-12 subset) describing the arguments object.
type ToolSchema struct {
	Name        string
	Description string
	InputSchema json.RawMessage
}

// Request is a single model call. Model is a logical name; the adapter resolves
// it through Config.ModelMap before hitting the wire. Temperature nil means the
// provider default.
type Request struct {
	Model       string
	System      string
	Messages    []Message
	Tools       []ToolSchema
	MaxTokens   int
	Temperature *float64
	Stream      bool
}

// StopReason is the normalized reason a model turn ended.
type StopReason string

const (
	StopEndTurn   StopReason = "end_turn"
	StopToolUse   StopReason = "tool_use"
	StopMaxTokens StopReason = "max_tokens"
	StopRefusal   StopReason = "refusal"
	StopOther     StopReason = "other"
)

// Usage is token accounting for one or more calls. CostUSD is filled in by the
// adapter from Config.Pricing.
type Usage struct {
	InputTokens      int
	OutputTokens     int
	CacheReadTokens  int
	CacheWriteTokens int
	CostUSD          float64
}

// Add accumulates o into u.
func (u *Usage) Add(o Usage) {
	u.InputTokens += o.InputTokens
	u.OutputTokens += o.OutputTokens
	u.CacheReadTokens += o.CacheReadTokens
	u.CacheWriteTokens += o.CacheWriteTokens
	u.CostUSD += o.CostUSD
}

// TotalTokens is input plus output (cache tokens are a subset of input on the
// wire, so they are not added again).
func (u Usage) TotalTokens() int { return u.InputTokens + u.OutputTokens }

// Response is one assistant turn plus its metadata. Raw is the provider-native
// body, persisted verbatim to the run transcript for later inspection.
type Response struct {
	Assistant  Message
	StopReason StopReason
	Usage      Usage
	Raw        json.RawMessage
}

// Text returns the concatenated text Parts of the assistant message.
func (r Response) Text() string {
	var s string
	for _, p := range r.Assistant.Content {
		if p.Type == PartText {
			s += p.Text
		}
	}
	return s
}

// ToolCalls returns the tool_use Parts of the assistant message.
func (r Response) ToolCalls() []ToolCall {
	var out []ToolCall
	for _, p := range r.Assistant.Content {
		if p.Type == PartToolUse && p.ToolCall != nil {
			out = append(out, *p.ToolCall)
		}
	}
	return out
}

// Provider is a model backend. Chat is the core call; Stream is optional
// richness — DefaultStream adapts any Chat into a single-chunk stream.
type Provider interface {
	Name() string
	Chat(ctx context.Context, req Request) (Response, error)
	Stream(ctx context.Context, req Request) (Stream, error)
}

// Stream yields incremental output. Recv returns io.EOF when the turn is done;
// the final non-EOF Chunk carries Done with the assembled Response.
type Stream interface {
	Recv() (Chunk, error)
	Close() error
}

// Chunk is one streaming delta. At most one of TextDelta/ToolCallDelta is set
// on an interim chunk; Done is set on the terminal chunk.
type Chunk struct {
	TextDelta     string
	ToolCallDelta *ToolCall
	Done          *Response
}

// Tokenizer estimates prompt size for the mini-agent's context-window
// budgeting. Implementations range from a zero-dependency heuristic to a live
// count_tokens call.
type Tokenizer interface {
	CountMessages(ctx context.Context, model, system string, msgs []Message, tools []ToolSchema) (int, error)
}

// Wire is the JSON dialect an endpoint speaks.
type Wire string

const (
	WireAnthropic Wire = "anthropic" // POST {base}/v1/messages
	WireOpenAI    Wire = "openai"    // POST {base}/v1/chat/completions
)

// Pricing is dollars per million tokens, by token class.
type Pricing struct {
	InPerMTok         float64
	OutPerMTok        float64
	CacheReadPerMTok  float64
	CacheWritePerMTok float64
}

// Cost returns the dollar cost of u under this pricing.
func (p Pricing) Cost(u Usage) float64 {
	m := func(tok int, rate float64) float64 { return float64(tok) / 1e6 * rate }
	return m(u.InputTokens, p.InPerMTok) +
		m(u.OutputTokens, p.OutPerMTok) +
		m(u.CacheReadTokens, p.CacheReadPerMTok) +
		m(u.CacheWriteTokens, p.CacheWritePerMTok)
}

// Config describes one backend. The canonical hosts and a local Ollama differ
// only in these fields — never in code path.
type Config struct {
	Name      string            // "anthropic" | "openai" | "ollama"
	Wire      Wire              // anthropic | openai
	BaseURL   string            // "" => canonical host for the wire
	APIKeyEnv string            // env var holding the key; "" allowed for localhost
	Model     string            // default provider-side model id
	ModelMap  map[string]string // logical name => provider-side name
	Pricing   Pricing           // for Usage.CostUSD
	Headers   map[string]string // extra static headers (e.g. anthropic-version)
	Timeout   time.Duration     // per-request; 0 => defaultTimeout
}

const defaultTimeout = 5 * time.Minute

// resolveModel maps a logical model name to its provider-side id, falling back
// to the name itself and then to Config.Model.
func (c Config) resolveModel(logical string) string {
	if logical == "" {
		return c.Model
	}
	if mapped, ok := c.ModelMap[logical]; ok {
		return mapped
	}
	return logical
}
