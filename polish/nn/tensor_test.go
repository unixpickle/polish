package nn

import (
	"fmt"
	"math/rand"
	"reflect"
	"testing"
)

func TestPadUnpad(t *testing.T) {
	tensor := NewTensor(5, 10, 3)
	for i := range tensor.Data {
		tensor.Data[i] = float32(rand.NormFloat64())
	}

	runShape := func(top, r, b, l int) {
		t.Run(fmt.Sprintf("%d,%d,%d,%d", top, r, b, l), func(t *testing.T) {
			padded := tensor.Pad(top, r, b, l)
			unpadded := padded.Unpad(top, r, b, l)
			if !reflect.DeepEqual(tensor, unpadded) {
				t.Error("unexpected pad->unpad")
			}
		})
	}

	runShape(0, 0, 0, 0)
	runShape(1, 0, 0, 0)
	runShape(0, 1, 0, 0)
	runShape(0, 0, 1, 0)
	runShape(0, 0, 0, 1)
	runShape(1, 1, 1, 1)
	runShape(1, 2, 3, 4)
}
