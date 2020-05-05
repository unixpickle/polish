package nn

import (
	"math"
	"testing"
)

func TestPatches(t *testing.T) {
	input := NewTensor(4, 3, 2)
	input.Data = []float32{
		1, 2, 3, 4, 5, 6,
		7, 8, 9, 10, 11, 12,
		13, 14, 15, 16, 17, 18,
		19, 20, 21, 22, 23, 24,
	}

	inputs := [][2]int{
		{2, 1},
		{2, 2},
		{3, 1},
		{3, 2},
	}
	sums := [][]float32{
		{
			44, 60,
			92, 108,
			140, 156,
		},
		{
			44,
			140,
		},
		{
			171,
			279,
		},
		{
			171,
		},
	}

	for i, inputArgs := range inputs {
		expected := sums[i]
		var actual []float32
		Patches(input, inputArgs[0], inputArgs[1], func(t *Tensor) {
			var sum float32
			for _, c := range t.Data {
				sum += c
			}
			actual = append(actual, sum)
		})
		if len(actual) != len(expected) {
			t.Errorf("case %d: unexpected length (got %d expected %d)", i,
				len(actual), len(expected))
			continue
		}
		for j, x := range expected {
			a := actual[j]
			if math.Abs(float64(x-a)) > 1e-3 {
				t.Errorf("case %d: element %d: expected %f but got %f", i, j, x, a)
				break
			}
		}
	}
}
