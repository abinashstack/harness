package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
)

const (
	anthropicDefaultBase      = "https://api.anthropic.com"
	anthropicVersion          = "2023-06-01"
	anthropicDefaultMaxTokens = 4096
)

// anthropic speaks POST {base}/v1/messages.
type anthropic struct{ httpClient }

type aBlock struct {
	Type      string          `json:"type"`
	Text      string          `json:"text,omitempty"`
	ID        string          `json:"id,omitempty"`
	Name      string          `json:"name,omitempty"`
	Input     json.RawMessage `json:"input,omitempty"`
	ToolUseID string          `json:"tool_use_id,omitempty"`
	Content   string          `json:"content,omitempty"`
	IsError   bool            `json:"is_error,omitempty"`
}

type aMessage struct {
	Role    string   `json:"role"`
	Content []aBlock `json:"content"`
}

type aTool struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	InputSchema json.RawMessage `json:"input_schema"`
}

type aRequest struct {
	Model       string     `json:"model"`
	MaxTokens   int        `json:"max_tokens"`
	System      string     `json:"system,omitempty"`
	Messages    []aMessage `json:"messages"`
	Tools       []aTool    `json:"tools,omitempty"`
	Temperature *float64   `json:"temperature,omitempty"`
	Stream      bool       `json:"stream,omitempty"`
}

type aUsage struct {
	InputTokens      int `json:"input_tokens"`
	OutputTokens     int `json:"output_tokens"`
	CacheReadTokens  int `json:"cache_read_input_tokens"`
	CacheWriteTokens int `json:"cache_creation_input_tokens"`
}

type aResponse struct {
	Content    []aBlock `json:"content"`
	StopReason string   `json:"stop_reason"`
	Usage      aUsage   `json:"usage"`
}

// aEvent is the union of the streaming event payloads, keyed by Type.
type aEvent struct {
	Type         string                 `json:"type"`
	Index        int                    `json:"index"`
	Message      struct{ Usage aUsage } `json:"message"`
	ContentBlock aBlock                 `json:"content_block"`
	Delta        struct {
		Type        string `json:"type"`
		Text        string `json:"text"`
		PartialJSON string `json:"partial_json"`
		StopReason  string `json:"stop_reason"`
	} `json:"delta"`
	Usage aUsage `json:"usage"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

func anthropicStop(s string) StopReason {
	switch s {
	case "end_turn", "stop_sequence":
		return StopEndTurn
	case "tool_use":
		return StopToolUse
	case "max_tokens":
		return StopMaxTokens
	case "refusal":
		return StopRefusal
	}
	return StopOther
}

// prepare translates req into the wire body and request headers.
func (a *anthropic) prepare(req Request) (aRequest, map[string]string, error) {
	key, err := a.apiKey()
	if err != nil {
		return aRequest{}, nil, err
	}
	body := aRequest{
		Model:       a.cfg.resolveModel(req.Model),
		MaxTokens:   req.MaxTokens,
		System:      req.System,
		Temperature: req.Temperature,
	}
	if body.MaxTokens == 0 {
		body.MaxTokens = anthropicDefaultMaxTokens
	}
	for _, m := range req.Messages {
		am := aMessage{Role: string(m.Role)}
		for _, p := range m.Content {
			switch {
			case p.Type == PartText:
				am.Content = append(am.Content, aBlock{Type: "text", Text: p.Text})
			case p.Type == PartToolUse && p.ToolCall != nil:
				am.Content = append(am.Content, aBlock{Type: "tool_use", ID: p.ToolCall.ID, Name: p.ToolCall.Name, Input: objectOrEmpty(p.ToolCall.Input)})
			case p.Type == PartToolResult && p.ToolResult != nil:
				am.Content = append(am.Content, aBlock{Type: "tool_result", ToolUseID: p.ToolResult.CallID, Content: p.ToolResult.Content, IsError: p.ToolResult.IsError})
			}
		}
		body.Messages = append(body.Messages, am)
	}
	for _, t := range req.Tools {
		body.Tools = append(body.Tools, aTool{Name: t.Name, Description: t.Description, InputSchema: objectOrEmpty(t.InputSchema)})
	}
	headers := map[string]string{"anthropic-version": anthropicVersion}
	if key != "" {
		headers["x-api-key"] = key
	}
	return body, headers, nil
}

func (a *anthropic) usage(u aUsage) Usage {
	out := Usage{
		InputTokens:      u.InputTokens,
		OutputTokens:     u.OutputTokens,
		CacheReadTokens:  u.CacheReadTokens,
		CacheWriteTokens: u.CacheWriteTokens,
	}
	out.CostUSD = a.cfg.Pricing.Cost(out)
	return out
}

func (a *anthropic) Chat(ctx context.Context, req Request) (Response, error) {
	body, headers, err := a.prepare(req)
	if err != nil {
		return Response{}, err
	}
	raw, err := a.post(ctx, a.baseURL(anthropicDefaultBase)+"/v1/messages", headers, body)
	if err != nil {
		return Response{}, err
	}
	var ar aResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return Response{}, err
	}
	out := Response{Assistant: Message{Role: RoleAssistant}, Raw: raw, StopReason: anthropicStop(ar.StopReason), Usage: a.usage(ar.Usage)}
	for _, b := range ar.Content {
		switch b.Type {
		case "text":
			out.Assistant.Content = append(out.Assistant.Content, Text(b.Text))
		case "tool_use":
			out.Assistant.Content = append(out.Assistant.Content, Part{Type: PartToolUse, ToolCall: &ToolCall{ID: b.ID, Name: b.Name, Input: b.Input}})
		}
	}
	return out, nil
}

// aAcc accumulates one streamed content block.
type aAcc struct {
	block aBlock
	text  strings.Builder
	json  strings.Builder
}

func (a *anthropic) Stream(ctx context.Context, req Request) (Stream, error) {
	body, headers, err := a.prepare(req)
	if err != nil {
		return nil, err
	}
	body.Stream = true
	resp, err := a.open(ctx, a.baseURL(anthropicDefaultBase)+"/v1/messages", headers, body)
	if err != nil {
		return nil, err
	}

	var (
		blocks = map[int]*aAcc{}
		order  []int
		usage  aUsage
		stop   string
	)
	handle := func(f sseFrame) ([]Chunk, error) {
		var ev aEvent
		if err := json.Unmarshal(f.data, &ev); err != nil {
			return nil, fmt.Errorf("provider %q: bad stream event: %w", a.cfg.Name, err)
		}
		switch ev.Type {
		case "message_start":
			usage = ev.Message.Usage
		case "content_block_start":
			blocks[ev.Index] = &aAcc{block: ev.ContentBlock}
			order = append(order, ev.Index)
		case "content_block_delta":
			acc := blocks[ev.Index]
			if acc == nil {
				return nil, nil
			}
			switch ev.Delta.Type {
			case "text_delta":
				acc.text.WriteString(ev.Delta.Text)
				return []Chunk{{TextDelta: ev.Delta.Text}}, nil
			case "input_json_delta":
				acc.json.WriteString(ev.Delta.PartialJSON)
			}
		case "content_block_stop":
			if acc := blocks[ev.Index]; acc != nil && acc.block.Type == "tool_use" {
				acc.block.Input = objectOrEmpty(json.RawMessage(acc.json.String()))
				return []Chunk{{ToolCallDelta: &ToolCall{ID: acc.block.ID, Name: acc.block.Name, Input: acc.block.Input}}}, nil
			}
		case "message_delta":
			if ev.Delta.StopReason != "" {
				stop = ev.Delta.StopReason
			}
			if ev.Usage.OutputTokens > 0 {
				usage.OutputTokens = ev.Usage.OutputTokens
			}
		case "message_stop":
			return nil, io.EOF
		case "error":
			return nil, fmt.Errorf("provider %q: stream error: %s: %s", a.cfg.Name, ev.Error.Type, ev.Error.Message)
		}
		return nil, nil
	}
	finish := func() Response {
		out := Response{Assistant: Message{Role: RoleAssistant}, StopReason: anthropicStop(stop), Usage: a.usage(usage)}
		for _, i := range order {
			acc := blocks[i]
			switch acc.block.Type {
			case "text":
				out.Assistant.Content = append(out.Assistant.Content, Text(acc.text.String()))
			case "tool_use":
				out.Assistant.Content = append(out.Assistant.Content, Part{Type: PartToolUse, ToolCall: &ToolCall{ID: acc.block.ID, Name: acc.block.Name, Input: acc.block.Input}})
			}
		}
		return out
	}
	return newSSEStream(resp.Body, handle, finish), nil
}
