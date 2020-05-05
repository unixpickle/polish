package nn

import "math"

// GroupNorm implements the normalization step of group
// normalization.
type GroupNorm struct {
	NumGroups int
}

// Apply applies the normalization step.
func (g *GroupNorm) Apply(t *Tensor) *Tensor {
	if t.Depth%g.NumGroups != 0 {
		panic("number of groups must divide number of input channels")
	}
	sums := make([]float32, g.NumGroups)
	sqSums := make([]float32, g.NumGroups)
	Groups(t, g.NumGroups, func(group, idx int) {
		v := t.Data[idx]
		sums[group] += v
		sqSums[group] += v * v
	})
	normalize := 1.0 / float32(t.Width*t.Height*t.Depth/g.NumGroups)

	biases := sums
	scales := sqSums
	for i, x := range biases {
		biases[i] = -x * normalize
	}
	for i, sqSum := range scales {
		b := biases[i]
		x := sqSum*normalize - b*b
		if x < 0 {
			x = 0
		}
		scales[i] = float32(1 / math.Sqrt(float64(x)+1e-5))
	}

	res := NewTensor(t.Height, t.Width, t.Depth)
	Groups(t, g.NumGroups, func(group, idx int) {
		res.Data[idx] = (t.Data[idx] + biases[group]) * scales[group]
	})
	return res
}

// Groups iterates over the entries of t in order, but
// adds a groupIdx parameter indicating which group each
// component belongs to for group normalization.
func Groups(t *Tensor, numGroups int, f func(groupIdx, dataIdx int)) {
	if t.Depth%numGroups != 0 {
		panic("number of groups must divide number of input channels")
	}
	groupSize := t.Depth / numGroups
	var idx int
	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			for g := 0; g < numGroups; g++ {
				for z := 0; z < groupSize; z++ {
					f(g, idx)
					idx++
				}
			}
		}
	}
}
