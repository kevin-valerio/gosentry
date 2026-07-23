package sigsegvrestart

import (
	"bytes"
	"testing"
)

var sigsegvInput = []byte("SIGSEGV")

func FuzzSIGSEGV(f *testing.F) {
	f.Add(sigsegvInput)
	f.Fuzz(func(t *testing.T, data []byte) {
		if bytes.Equal(data, sigsegvInput) {
			triggerSIGSEGV()
		}
	})
}
