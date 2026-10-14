package provider

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"os"
	"strings"
)

// New builds the Provider for cfg's wire dialect.
func New(cfg Config) (Provider, error) {
	hc := newHTTPClient(cfg)
	switch cfg.Wire {
	case WireAnthropic:
		return &anthropic{hc}, nil
	case WireOpenAI:
		return &openai{hc}, nil
	}
	return nil, fmt.Errorf("provider %q: unknown wire %q", cfg.Name, cfg.Wire)
}

// httpClient is the transport shared by both wire adapters.
type httpClient struct {
	cfg Config
	hc  *http.Client
}

func newHTTPClient(cfg Config) httpClient {
	t := cfg.Timeout
	if t == 0 {
		t = defaultTimeout
	}
	return httpClient{cfg: cfg, hc: &http.Client{Timeout: t}}
}

func (c httpClient) Name() string { return c.cfg.Name }

// baseURL returns Config.BaseURL, or def when unset.
func (c httpClient) baseURL(def string) string {
	if c.cfg.BaseURL != "" {
		return strings.TrimRight(c.cfg.BaseURL, "/")
	}
	return def
}

// apiKey reads the key from Config.APIKeyEnv; "" means no key is needed.
func (c httpClient) apiKey() (string, error) {
	if c.cfg.APIKeyEnv == "" {
		return "", nil
	}
	k := os.Getenv(c.cfg.APIKeyEnv)
	if k == "" {
		return "", fmt.Errorf("provider %q: env var %s is not set", c.cfg.Name, c.cfg.APIKeyEnv)
	}
	return k, nil
}

// post sends body as JSON and returns the response body for any 2xx status.
func (c httpClient) post(ctx context.Context, url string, headers map[string]string, body any) ([]byte, error) {
	b, err := json.Marshal(body)
	if err != nil {
		return nil, err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, url, bytes.NewReader(b))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/json")
	for k, v := range c.cfg.Headers {
		req.Header.Set(k, v)
	}
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := c.hc.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	data, err := io.ReadAll(resp.Body)
	if err != nil {
		return nil, err
	}
	if resp.StatusCode/100 != 2 {
		msg := string(data)
		if len(msg) > 500 {
			msg = msg[:500] + "..."
		}
		return nil, fmt.Errorf("provider %q: HTTP %d: %s", c.cfg.Name, resp.StatusCode, msg)
	}
	return data, nil
}

// streamViaChat satisfies Provider.Stream for adapters that only implement Chat.
func streamViaChat(ctx context.Context, p Provider, req Request) (Stream, error) {
	resp, err := p.Chat(ctx, req)
	if err != nil {
		return nil, err
	}
	return DefaultStream(resp), nil
}

// objectOrEmpty returns raw, or "{}" when raw is empty, for tool-call arguments.
func objectOrEmpty(raw json.RawMessage) json.RawMessage {
	if len(bytes.TrimSpace(raw)) == 0 {
		return json.RawMessage("{}")
	}
	return raw
}
