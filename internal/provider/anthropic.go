package provider

import (
	"context"
	"encoding/json"
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
}

type aResponse struct {
	Content    []aBlock `json:"content"`
	StopReason string   `json:"stop_reason"`
	Usage      struct {
		InputTokens      int `json:"input_tokens"`
		OutputTokens     int `json:"output_tokens"`
		CacheReadTokens  int `json:"cache_read_input_tokens"`
		CacheWriteTokens int `json:"cache_creation_input_tokens"`
	} `json:"usage"`
}

func (a *anthropic) Stream(ctx context.Context, req Request) (Stream, error) {
	return streamViaChat(ctx, a, req)
}

func (a *anthropic) Chat(ctx context.Context, req Request) (Response, error) {
	key, err := a.apiKey()
	if err != nil {
		return Response{}, err
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
	raw, err := a.post(ctx, a.baseURL(anthropicDefaultBase)+"/v1/messages", headers, body)
	if err != nil {
		return Response{}, err
	}
	var ar aResponse
	if err := json.Unmarshal(raw, &ar); err != nil {
		return Response{}, err
	}

	out := Response{Assistant: Message{Role: RoleAssistant}, Raw: raw}
	for _, b := range ar.Content {
		switch b.Type {
		case "text":
			out.Assistant.Content = append(out.Assistant.Content, Text(b.Text))
		case "tool_use":
			out.Assistant.Content = append(out.Assistant.Content, Part{Type: PartToolUse, ToolCall: &ToolCall{ID: b.ID, Name: b.Name, Input: b.Input}})
		}
	}
	switch ar.StopReason {
	case "end_turn", "stop_sequence":
		out.StopReason = StopEndTurn
	case "tool_use":
		out.StopReason = StopToolUse
	case "max_tokens":
		out.StopReason = StopMaxTokens
	case "refusal":
		out.StopReason = StopRefusal
	default:
		out.StopReason = StopOther
	}
	out.Usage = Usage{
		InputTokens:      ar.Usage.InputTokens,
		OutputTokens:     ar.Usage.OutputTokens,
		CacheReadTokens:  ar.Usage.CacheReadTokens,
		CacheWriteTokens: ar.Usage.CacheWriteTokens,
	}
	out.Usage.CostUSD = a.cfg.Pricing.Cost(out.Usage)
	return out, nil
}
