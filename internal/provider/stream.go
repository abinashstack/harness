package provider

import (
	"io"
)

// sseStream adapts an SSE response body to Stream. handle turns each frame into
// zero or more interim Chunks; finish assembles the final Response once the
// body ends, which Recv emits as the terminal Done chunk.
type sseStream struct {
	body    io.Closer
	sc      *sseScanner
	handle  func(sseFrame) ([]Chunk, error)
	finish  func() Response
	pending []Chunk
	done    bool
}

func newSSEStream(body io.ReadCloser, handle func(sseFrame) ([]Chunk, error), finish func() Response) *sseStream {
	return &sseStream{body: body, sc: newSSEScanner(body), handle: handle, finish: finish}
}

func (s *sseStream) Recv() (Chunk, error) {
	for {
		if len(s.pending) > 0 {
			c := s.pending[0]
			s.pending = s.pending[1:]
			return c, nil
		}
		if s.done {
			return Chunk{}, io.EOF
		}
		if !s.sc.Scan() {
			if err := s.sc.Err(); err != nil {
				return Chunk{}, err
			}
			s.done = true
			r := s.finish()
			return Chunk{Done: &r}, nil
		}
		cs, err := s.handle(s.sc.Frame())
		if err == io.EOF { // handler saw the end-of-stream marker
			s.done = true
			r := s.finish()
			cs = append(cs, Chunk{Done: &r})
		} else if err != nil {
			return Chunk{}, err
		}
		s.pending = cs
	}
}

func (s *sseStream) Close() error { return s.body.Close() }
