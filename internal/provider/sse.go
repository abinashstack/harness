package provider

import (
	"bufio"
	"bytes"
	"io"
)

// sseFrame is one server-sent event: an optional event name and its data
// payload (concatenated data: lines, newline-joined, per the SSE spec).
type sseFrame struct {
	event string
	data  []byte
}

// sseScanner reads text/event-stream frames from r. It tolerates streams that
// omit the "event:" line (some Bedrock-shaped Anthropic streams do this) —
// event is simply "" then.
type sseScanner struct {
	sc      *bufio.Scanner
	event   string
	data    bytes.Buffer
	err     error
	current sseFrame
}

func newSSEScanner(r io.Reader) *sseScanner {
	sc := bufio.NewScanner(r)
	sc.Buffer(make([]byte, 0, 64*1024), 8<<20)
	return &sseScanner{sc: sc}
}

// Scan advances to the next complete frame, returning false at end of stream or
// on error (check Err).
func (s *sseScanner) Scan() bool {
	s.event = ""
	s.data.Reset()
	for s.sc.Scan() {
		line := s.sc.Bytes()
		if len(line) == 0 { // blank line terminates a frame
			if s.data.Len() == 0 && s.event == "" {
				continue // stray blank line between frames
			}
			s.current = sseFrame{event: s.event, data: append([]byte(nil), bytes.TrimSuffix(s.data.Bytes(), []byte("\n"))...)}
			return true
		}
		if bytes.HasPrefix(line, []byte(":")) {
			continue // comment / heartbeat
		}
		field, value, found := bytes.Cut(line, []byte(":"))
		if !found {
			field, value = line, nil
		}
		value = bytes.TrimPrefix(value, []byte(" "))
		switch string(field) {
		case "event":
			s.event = string(value)
		case "data":
			s.data.Write(value)
			s.data.WriteByte('\n')
		}
	}
	if err := s.sc.Err(); err != nil {
		s.err = err
	}
	// Flush a trailing frame with no terminating blank line.
	if s.data.Len() > 0 || s.event != "" {
		s.current = sseFrame{event: s.event, data: append([]byte(nil), bytes.TrimSuffix(s.data.Bytes(), []byte("\n"))...)}
		s.event = ""
		s.data.Reset()
		return true
	}
	return false
}

func (s *sseScanner) Frame() sseFrame { return s.current }

func (s *sseScanner) Err() error {
	if s.err == io.EOF {
		return nil
	}
	return s.err
}

// oneShotStream adapts a completed Response into the Stream interface: one
// terminal chunk, then EOF. DefaultStream uses it so a Provider that only
// implements Chat still satisfies callers that ask for a Stream.
type oneShotStream struct {
	resp Response
	done bool
}

func (o *oneShotStream) Recv() (Chunk, error) {
	if o.done {
		return Chunk{}, io.EOF
	}
	o.done = true
	r := o.resp
	return Chunk{Done: &r}, nil
}

func (o *oneShotStream) Close() error { return nil }

// DefaultStream turns a Chat result into a single-chunk Stream.
func DefaultStream(resp Response) Stream { return &oneShotStream{resp: resp} }
