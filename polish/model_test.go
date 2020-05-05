package polish

import (
	"testing"

	"github.com/unixpickle/polish/polish/nn"
)

func BenchmarkShallow(b *testing.B) {
	layer := ModelTypeShallow.Layer()
	input := nn.NewTensor(512, 512, 3)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		layer.Apply(input)
	}
}
