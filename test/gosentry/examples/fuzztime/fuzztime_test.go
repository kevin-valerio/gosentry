package fuzztime

import (
	"bytes"
	"testing"
)

func FuzzSomeFunc(f *testing.F) {
	f.Add([]byte("seed"))

	f.Fuzz(func(t *testing.T, data []byte) {
		_ = bytes.Contains(data, []byte("gosentry"))
	})
}
