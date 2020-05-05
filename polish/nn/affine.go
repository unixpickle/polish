package nn

// A Bias layer adds a per-channel constant to a Tensor.
type Bias struct {
	Data []float32
}

// Apply adds the bias to the Tensor.
func (b *Bias) Apply(t *Tensor) *Tensor {
	if len(b.Data) != t.Depth {
		panic("depth must match number of bias channels")
	}
	res := NewTensor(t.Height, t.Width, t.Depth)
	var idx int
	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			for z := 0; z < t.Depth; z++ {
				res.Data[idx] = t.Data[idx] + b.Data[z]
				idx++
			}
		}
	}
	return res
}

// A Mul layer multiplies a per-channel mask to a Tensor.
type Mul struct {
	Data []float32
}

// Apply multiplies the mask to the Tensor.
func (m *Mul) Apply(t *Tensor) *Tensor {
	if len(m.Data) != t.Depth {
		panic("depth must match number of bias channels")
	}
	res := NewTensor(t.Height, t.Width, t.Depth)
	var idx int
	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			for z := 0; z < t.Depth; z++ {
				res.Data[idx] = t.Data[idx] * m.Data[z]
				idx++
			}
		}
	}
	return res
}
