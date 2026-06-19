// Copyright 2026 The Go Authors. All rights reserved.
// Use of this source code is governed by a BSD-style
// license that can be found in the LICENSE file.

package test

import (
	"testing"
	"time"
)

func TestParseLibAFLFuzzTime(t *testing.T) {
	tests := []struct {
		name    string
		in      string
		want    time.Duration
		wantErr bool
	}{
		{name: "empty", in: "", want: 0},
		{name: "duration", in: "1m", want: time.Minute},
		{name: "zero duration", in: "0", wantErr: true},
		{name: "bad duration", in: "bad", wantErr: true},
		{name: "count", in: "10x", wantErr: true},
		{name: "bad count", in: "0x", wantErr: true},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseLibAFLFuzzTime(tt.in)
			if (err != nil) != tt.wantErr {
				t.Fatalf("parseLibAFLFuzzTime(%q) error = %v, wantErr %v", tt.in, err, tt.wantErr)
			}
			if got != tt.want {
				t.Fatalf("parseLibAFLFuzzTime(%q) = %v, want %v", tt.in, got, tt.want)
			}
		})
	}
}
