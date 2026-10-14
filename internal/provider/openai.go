package provider

import (
	"context"
	"encoding/json"
	"fmt"
	"io"
	"strings"
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

	Stream        bool            `json:"stream,omitempty"`
	StreamOptions *oStreamOptions `json:"stream_options,omitempty"`
}

type oStreamOptions struct {
	IncludeUsage bool `json:"include_usage"`
}

type oUsage struct {
	PromptTokens     int `json:"prompt_tokens"`
	CompletionTokens int `json:"completion_tokens"`
	Details          struct {
		CachedTokens int `json:"cached_tokens"`
	} `json:"prompt_tokens_details"`
}

type oResponse struct {
	Choices []struct {
		Message      oMessage `json:"message"`
		FinishReason string   `json:"finish_reason"`
	} `json:"choices"`
	Usage oUsage `json:"usage"`
}

// oChunk is one streamed chat.completion.chunk. The final chunk carries Usage
// and no choices.
type oChunk struct {
	Choices []struct {
		Delta struct {
			Content   string `json:"content"`
			ToolCalls []struct {
				Index    int    `json:"index"`
				ID       string `json:"id"`
				Function struct {
					Name      string `json:"name"`
					Arguments string `json:"arguments"`
				} `json:"function"`
			} `json:"tool_calls"`
		} `json:"delta"`
		FinishReason string `json:"finish_reason"`
	} `json:"choices"`
	Usage *oUsage `json:"usage"`
	Error *struct {
		Message string `json:"message"`
	} `json:"error"`
}

func strp(s string) *string { return &s }

// prepare translates req into the wire body and request headers.
func (o *openai) prepare(req Request) (oRequest, map[string]string, error) {
	key, err := o.apiKey()
	if err != nil {
		return oRequest{}, nil, err
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
	return body, headers, nil
}

func openaiStop(s string) StopReason {
	switch s {
	case "stop":
		return StopEndTurn
	case "tool_calls":
		return StopToolUse
	case "length":
		return StopMaxTokens
	case "content_filter":
		return StopRefusal
	}
	return StopOther
}

func (o *openai) usage(u oUsage) Usage {
	out := Usage{InputTokens: u.PromptTokens, OutputTokens: u.CompletionTokens, CacheReadTokens: u.Details.CachedTokens}
	out.CostUSD = o.cfg.Pricing.Cost(out)
	return out
}

func (o *openai) Chat(ctx context.Context, req Request) (Response, error) {
	body, headers, err := o.prepare(req)
	if err != nil {
		return Response{}, err
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
	out := Response{Assistant: Message{Role: RoleAssistant}, Raw: raw, StopReason: openaiStop(ch.FinishReason), Usage: o.usage(or.Usage)}
	if ch.Message.Content != nil && *ch.Message.Content != "" {
		out.Assistant.Content = append(out.Assistant.Content, Text(*ch.Message.Content))
	}
	for _, tc := range ch.Message.ToolCalls {
		out.Assistant.Content = append(out.Assistant.Content, Part{Type: PartToolUse, ToolCall: &ToolCall{
			ID: tc.ID, Name: tc.Function.Name, Input: json.RawMessage(tc.Function.Arguments),
		}})
	}
	return out, nil
}

// oAcc accumulates one streamed tool call, keyed by its delta index.
type oAcc struct {
	id, name string
	args     strings.Builder
}

func (o *openai) Stream(ctx context.Context, req Request) (Stream, error) {
	body, headers, err := o.prepare(req)
	if err != nil {
		return nil, err
	}
	body.Stream = true
	body.StreamOptions = &oStreamOptions{IncludeUsage: true}
	resp, err := o.open(ctx, o.baseURL(openaiDefaultBase)+"/v1/chat/completions", headers, body)
	if err != nil {
		return nil, err
	}

	var (
		text   strings.Builder
		calls  = map[int]*oAcc{}
		order  []int
		finish string
		usage  oUsage
	)
	toolCalls := func() []ToolCall {
		var out []ToolCall
		for _, i := range order {
			c := calls[i]
			out = append(out, ToolCall{ID: c.id, Name: c.name, Input: objectOrEmpty(json.RawMessage(c.args.String()))})
		}
		return out
	}
	handle := func(f sseFrame) ([]Chunk, error) {
		if string(f.data) == "[DONE]" {
			// Tool calls are only complete now, so announce them before Done.
			var cs []Chunk
			for _, tc := range toolCalls() {
				tc := tc
				cs = append(cs, Chunk{ToolCallDelta: &tc})
			}
			return cs, io.EOF
		}
		var ck oChunk
		if err := json.Unmarshal(f.data, &ck); err != nil {
			return nil, fmt.Errorf("provider %q: bad stream event: %w", o.cfg.Name, err)
		}
		if ck.Error != nil {
			return nil, fmt.Errorf("provider %q: stream error: %s", o.cfg.Name, ck.Error.Message)
		}
		if ck.Usage != nil {
			usage = *ck.Usage
		}
		if len(ck.Choices) == 0 {
			return nil, nil
		}
		ch := ck.Choices[0]
		if ch.FinishReason != "" {
			finish = ch.FinishReason
		}
		for _, d := range ch.Delta.ToolCalls {
			acc := calls[d.Index]
			if acc == nil {
				acc = &oAcc{}
				calls[d.Index] = acc
				order = append(order, d.Index)
			}
			if d.ID != "" {
				acc.id = d.ID
			}
			if d.Function.Name != "" {
				acc.name = d.Function.Name
			}
			acc.args.WriteString(d.Function.Arguments)
		}
		if ch.Delta.Content != "" {
			text.WriteString(ch.Delta.Content)
			return []Chunk{{TextDelta: ch.Delta.Content}}, nil
		}
		return nil, nil
	}
	done := func() Response {
		out := Response{Assistant: Message{Role: RoleAssistant}, StopReason: openaiStop(finish), Usage: o.usage(usage)}
		if text.Len() > 0 {
			out.Assistant.Content = append(out.Assistant.Content, Text(text.String()))
		}
		for _, tc := range toolCalls() {
			tc := tc
			out.Assistant.Content = append(out.Assistant.Content, Part{Type: PartToolUse, ToolCall: &tc})
		}
		return out
	}
	return newSSEStream(resp.Body, handle, done), nil
}
