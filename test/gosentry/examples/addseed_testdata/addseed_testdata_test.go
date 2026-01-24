package addseedtestdata

import "testing"

const crashInput = "GOSENTRY-ADD-SEED-CRASH"

func FuzzAddSeedTestdata(f *testing.F) {
	f.Add(crashInput)
	f.Fuzz(func(t *testing.T, s string) {
		if s == crashInput {
			t.Fatalf("boom")
		}
	})
}
