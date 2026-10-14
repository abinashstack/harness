package provider

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"testing"
)

var weatherTool = ToolSchema{
	Name:        "weather",
	Description: "get weather",
	InputSchema: json.RawMessage(`{"type":"object","properties":{"city":{"type":"string"}}}`),
}

// history is a user turn, a tool call, and its result.
func history() []Message {
	return []Message{
		UserText("weather in Paris?"),
		{Role: RoleAssistant, Content: []Part{{Type: PartToolUse, ToolCall: &ToolCall{ID: "c1", Name: "weather", Input: json.RawMessage(`{"city":"Paris"}`)}}}},
		{Role: RoleUser, Content: []Part{{Type: PartToolResult, ToolResult: &ToolResult{CallID: "c1", Content: "sunny"}}}},
	}
}

func serve(t *testing.T, reply string, got *map[string]any, hdr *http.Header, path *string) *httptest.Server {
	t.Helper()
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		b, _ := io.ReadAll(r.Body)
		_ = json.Unmarshal(b, got)
		*hdr = r.Header
		*path = r.URL.Path
		w.Write([]byte(reply))
	}))
}

func TestAnthropicChat(t *testing.T) {
	t.Setenv("TEST_KEY", "k1")
	var got map[string]any
	var hdr http.Header
	var path string
	srv := serve(t, `{"content":[{"type":"text","text":"hi"},{"type":"tool_use","id":"c2","name":"weather","input":{"city":"Rome"}}],
		"stop_reason":"tool_use","usage":{"input_tokens":10,"output_tokens":5}}`, &got, &hdr, &path)
	defer srv.Close()

	p, err := New(Config{Name: "anthropic", Wire: WireAnthropic, BaseURL: srv.URL, APIKeyEnv: "TEST_KEY", Model: "m",
		Pricing: Pricing{InPerMTok: 1e6, OutPerMTok: 1e6}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Chat(context.Background(), Request{System: "sys", Messages: history(), Tools: []ToolSchema{weatherTool}})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/messages" || hdr.Get("x-api-key") != "k1" || hdr.Get("anthropic-version") == "" {
		t.Errorf("bad request: path=%s hdr=%v", path, hdr)
	}
	if got["model"] != "m" || got["system"] != "sys" || got["max_tokens"].(float64) != anthropicDefaultMaxTokens {
		t.Errorf("bad body: %v", got)
	}
	last := got["messages"].([]any)[2].(map[string]any)["content"].([]any)[0].(map[string]any)
	if last["type"] != "tool_result" || last["tool_use_id"] != "c1" || last["content"] != "sunny" {
		t.Errorf("bad tool_result: %v", last)
	}
	if resp.Text() != "hi" || resp.StopReason != StopToolUse {
		t.Errorf("resp = %q %s", resp.Text(), resp.StopReason)
	}
	if tc := resp.ToolCalls(); len(tc) != 1 || tc[0].ID != "c2" || string(tc[0].Input) != `{"city":"Rome"}` {
		t.Errorf("tool calls = %+v", tc)
	}
	if resp.Usage.TotalTokens() != 15 || resp.Usage.CostUSD != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestOpenAIChat(t *testing.T) {
	var got map[string]any
	var hdr http.Header
	var path string
	srv := serve(t, `{"choices":[{"message":{"role":"assistant","content":null,"tool_calls":[{"id":"c2","type":"function","function":{"name":"weather","arguments":"{\"city\":\"Rome\"}"}}]},
		"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":10,"completion_tokens":5}}`, &got, &hdr, &path)
	defer srv.Close()

	// No APIKeyEnv, like a local Ollama.
	p, err := New(Config{Name: "ollama", Wire: WireOpenAI, BaseURL: srv.URL + "/", Model: "llama",
		ModelMap: map[string]string{"fast": "llama-small"}})
	if err != nil {
		t.Fatal(err)
	}
	resp, err := p.Chat(context.Background(), Request{Model: "fast", System: "sys", Messages: history(), Tools: []ToolSchema{weatherTool}})
	if err != nil {
		t.Fatal(err)
	}
	if path != "/v1/chat/completions" || hdr.Get("Authorization") != "" {
		t.Errorf("bad request: path=%s hdr=%v", path, hdr)
	}
	msgs := got["messages"].([]any)
	roles := []string{}
	for _, m := range msgs {
		roles = append(roles, m.(map[string]any)["role"].(string))
	}
	if got["model"] != "llama-small" || len(roles) != 4 || roles[0] != "system" || roles[2] != "assistant" || roles[3] != "tool" {
		t.Errorf("model=%v roles=%v", got["model"], roles)
	}
	if tm := msgs[3].(map[string]any); tm["tool_call_id"] != "c1" || tm["content"] != "sunny" {
		t.Errorf("bad tool message: %v", tm)
	}
	if resp.StopReason != StopToolUse {
		t.Errorf("stop = %s", resp.StopReason)
	}
	if tc := resp.ToolCalls(); len(tc) != 1 || tc[0].Name != "weather" || string(tc[0].Input) != `{"city":"Rome"}` {
		t.Errorf("tool calls = %+v", tc)
	}
	if resp.Usage.TotalTokens() != 15 {
		t.Errorf("usage = %+v", resp.Usage)
	}
}

func TestHTTPErrorAndMissingKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, "boom", http.StatusTooManyRequests)
	}))
	defer srv.Close()
	p, _ := New(Config{Name: "x", Wire: WireOpenAI, BaseURL: srv.URL})
	if _, err := p.Chat(context.Background(), Request{Messages: []Message{UserText("hi")}}); err == nil {
		t.Error("want HTTP error")
	}
	p, _ = New(Config{Name: "x", Wire: WireAnthropic, APIKeyEnv: "DEFINITELY_UNSET_KEY"})
	if _, err := p.Chat(context.Background(), Request{}); err == nil {
		t.Error("want missing-key error")
	}
}

// drain reads a stream to EOF, returning the text deltas, tool-call deltas and
// the terminal Response.
func drain(t *testing.T, s Stream) (text string, calls []ToolCall, done *Response) {
	t.Helper()
	defer s.Close()
	for {
		c, err := s.Recv()
		if err == io.EOF {
			return
		}
		if err != nil {
			t.Fatal(err)
		}
		text += c.TextDelta
		if c.ToolCallDelta != nil {
			calls = append(calls, *c.ToolCallDelta)
		}
		if c.Done != nil {
			done = c.Done
		}
	}
}

func TestAnthropicStream(t *testing.T) {
	var got map[string]any
	var hdr http.Header
	var path string
	// Frames deliberately omit "event:" lines, as some proxies do.
	srv := serve(t, "data: {\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":10}}}\n\n"+
		"data: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\"}}\n\n"+
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hel\"}}\n\n"+
		"data: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"lo\"}}\n\n"+
		"data: {\"type\":\"content_block_stop\",\"index\":0}\n\n"+
		"data: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"tool_use\",\"id\":\"c9\",\"name\":\"weather\",\"input\":{}}}\n\n"+
		"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{\\\"city\\\":\"}}\n\n"+
		"data: {\"type\":\"content_block_delta\",\"index\":1,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"\\\"Rome\\\"}\"}}\n\n"+
		"data: {\"type\":\"content_block_stop\",\"index\":1}\n\n"+
		"data: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"tool_use\"},\"usage\":{\"output_tokens\":7}}\n\n"+
		"data: {\"type\":\"message_stop\"}\n\n", &got, &hdr, &path)
	defer srv.Close()

	p, _ := New(Config{Name: "anthropic", Wire: WireAnthropic, BaseURL: srv.URL, Model: "m"})
	s, err := p.Stream(context.Background(), Request{Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	text, calls, done := drain(t, s)
	if got["stream"] != true {
		t.Errorf("stream flag not sent: %v", got)
	}
	if text != "Hello" || len(calls) != 1 || string(calls[0].Input) != `{"city":"Rome"}` {
		t.Errorf("text=%q calls=%+v", text, calls)
	}
	if done == nil || done.Text() != "Hello" || done.StopReason != StopToolUse || len(done.ToolCalls()) != 1 {
		t.Fatalf("done = %+v", done)
	}
	if done.Usage.InputTokens != 10 || done.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v", done.Usage)
	}
}

func TestOpenAIStream(t *testing.T) {
	var got map[string]any
	var hdr http.Header
	var path string
	srv := serve(t, "data: {\"choices\":[{\"delta\":{\"content\":\"Hel\"}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{\"content\":\"lo\"}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"c9\",\"function\":{\"name\":\"weather\",\"arguments\":\"{\\\"city\\\":\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{\"tool_calls\":[{\"index\":0,\"function\":{\"arguments\":\"\\\"Rome\\\"}\"}}]}}]}\n\n"+
		"data: {\"choices\":[{\"delta\":{},\"finish_reason\":\"tool_calls\"}]}\n\n"+
		"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":10,\"completion_tokens\":7}}\n\n"+
		"data: [DONE]\n\n", &got, &hdr, &path)
	defer srv.Close()

	p, _ := New(Config{Name: "ollama", Wire: WireOpenAI, BaseURL: srv.URL, Model: "m"})
	s, err := p.Stream(context.Background(), Request{Messages: []Message{UserText("hi")}})
	if err != nil {
		t.Fatal(err)
	}
	text, calls, done := drain(t, s)
	if got["stream"] != true || got["stream_options"] == nil {
		t.Errorf("stream flags not sent: %v", got)
	}
	if text != "Hello" || len(calls) != 1 || calls[0].ID != "c9" || string(calls[0].Input) != `{"city":"Rome"}` {
		t.Errorf("text=%q calls=%+v", text, calls)
	}
	if done == nil || done.Text() != "Hello" || done.StopReason != StopToolUse || len(done.ToolCalls()) != 1 {
		t.Fatalf("done = %+v", done)
	}
	if done.Usage.InputTokens != 10 || done.Usage.OutputTokens != 7 {
		t.Errorf("usage = %+v", done.Usage)
	}
}
