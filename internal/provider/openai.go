package provider

import (
	"context"
	"encoding/json"
	"fmt"
)

const openaiDefaultBase = "https://api.openai.com"

// openai speaks POST {base}/v1/chat/completions, which Ollama and vLLM also serve.
type openai struct{ httpClient }

type oToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type oMessage struct {
	Role       string      `json:"role"`
	Content    *string     `json:"content"`
	ToolCalls  []oToolCall `json:"tool_calls,omitempty"`
	ToolCallID string      `json:"tool_call_id,omitempty"`
}

type oTool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters"`
	} `json:"function"`
}

type oRequest struct {
	Model       string     `json:"model"`
	Messages    []oMessage `json:"messages"`
	Tools       []oTool    `json:"tools,omitempty"`
	MaxTokens   int        `json:"max_tokens,omitempty"`
	Temperature *float64   `json:"temperature,omitempty"`
}

type oResponse struct {
	Choices []struct {
		Message      oMessage `json:"message"`
		FinishReason string   `json:"finish_reason"`
	} `json:"choices"`
	Usage struct {
		PromptTokens     int `json:"prompt_tokens"`
		CompletionTokens int `json:"completion_tokens"`
		Details          struct {
			CachedTokens int `json:"cached_tokens"`
		} `json:"prompt_tokens_details"`
	} `json:"usage"`
}

func strp(s string) *string { return &s }

func (o *openai) Stream(ctx context.Context, req Request) (Stream, error) {
	return streamViaChat(ctx, o, req)
}

func (o *openai) Chat(ctx context.Context, req Request) (Response, error) {
	key, err := o.apiKey()
	if err != nil {
		return Response{}, err
	}
	body := oRequest{
		Model:       o.cfg.resolveModel(req.Model),
		MaxTokens:   req.MaxTokens,
		Temperature: req.Temperature,
	}
	if req.System != "" {
		body.Messages = append(body.Messages, oMessage{Role: "system", Content: strp(req.System)})
	}
	for _, m := range req.Messages {
		// Tool results are their own role:"tool" messages, emitted before any
		// accompanying user text.
		var text string
		var calls []oToolCall
		for _, p := range m.Content {
			switch {
			case p.Type == PartText:
				text += p.Text
			case p.Type == PartToolUse && p.ToolCall != nil:
				tc := oToolCall{ID: p.ToolCall.ID, Type: "function"}
				tc.Function.Name = p.ToolCall.Name
				tc.Function.Arguments = string(objectOrEmpty(p.ToolCall.Input))
				calls = append(calls, tc)
			case p.Type == PartToolResult && p.ToolResult != nil:
				body.Messages = append(body.Messages, oMessage{Role: "tool", ToolCallID: p.ToolResult.CallID, Content: strp(p.ToolResult.Content)})
			}
		}
		if text == "" && len(calls) == 0 {
			continue
		}
		om := oMessage{Role: string(m.Role), ToolCalls: calls}
		if text != "" || len(calls) == 0 {
			om.Content = strp(text)
		}
		body.Messages = append(body.Messages, om)
	}
	for _, t := range req.Tools {
		ot := oTool{Type: "function"}
		ot.Function.Name = t.Name
		ot.Function.Description = t.Description
		ot.Function.Parameters = objectOrEmpty(t.InputSchema)
		body.Tools = append(body.Tools, ot)
	}

	headers := map[string]string{}
	if key != "" {
		headers["Authorization"] = "Bearer " + key
	}
	raw, err := o.post(ctx, o.baseURL(openaiDefaultBase)+"/v1/chat/completions", headers, body)
	if err != nil {
		return Response{}, err
	}
	var or oResponse
	if err := json.Unmarshal(raw, &or); err != nil {
		return Response{}, err
	}
	if len(or.Choices) == 0 {
		return Response{}, fmt.Errorf("provider %q: response has no choices", o.cfg.Name)
	}

	ch := or.Choices[0]
	out := Response{Assistant: Message{Role: RoleAssistant}, Raw: raw}
	if ch.Message.Content != nil && *ch.Message.Content != "" {
		out.Assistant.Content = append(out.Assistant.Content, Text(*ch.Message.Content))
	}
	for _, tc := range ch.Message.ToolCalls {
		out.Assistant.Content = append(out.Assistant.Content, Part{Type: PartToolUse, ToolCall: &ToolCall{
			ID: tc.ID, Name: tc.Function.Name, Input: json.RawMessage(tc.Function.Arguments),
		}})
	}
	switch ch.FinishReason {
	case "stop":
		out.StopReason = StopEndTurn
	case "tool_calls":
		out.StopReason = StopToolUse
	case "length":
		out.StopReason = StopMaxTokens
	case "content_filter":
		out.StopReason = StopRefusal
	default:
		out.StopReason = StopOther
	}
	out.Usage = Usage{
		InputTokens:     or.Usage.PromptTokens,
		OutputTokens:    or.Usage.CompletionTokens,
		CacheReadTokens: or.Usage.Details.CachedTokens,
	}
	out.Usage.CostUSD = o.cfg.Pricing.Cost(out.Usage)
	return out, nil
}
