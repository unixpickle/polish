package nn

// A ReLU layer applies the rectified linear unit.
type ReLU struct{}

// Apply applies the rectified linear unit.
func (r ReLU) Apply(t *Tensor) *Tensor {
	res := NewTensor(t.Height, t.Width, t.Depth)
	for i, x := range t.Data {
		if x > 0 {
			res.Data[i] = x
		}
	}
	return res
}

// A Pad layer pads input Tensors.
type Pad struct {
	Top    int
	Right  int
	Bottom int
	Left   int
}

// NewPad creates a Pad with the given values.
func NewPad(t, r, b, l int) *Pad {
	return &Pad{t, r, b, l}
}

// Apply pads the Tensor.
func (p *Pad) Apply(t *Tensor) *Tensor {
	return t.Pad(p.Top, p.Right, p.Bottom, p.Left)
}

// An Unpad layer unpads (crops) input Tensors.
type Unpad struct {
	Top    int
	Right  int
	Bottom int
	Left   int
}

// NewUnpad creates an Unpad with the given values.
func NewUnpad(t, r, b, l int) *Unpad {
	return &Unpad{t, r, b, l}
}

// Apply unpads (crops) the Tensor.
func (u *Unpad) Apply(t *Tensor) *Tensor {
	return t.Unpad(u.Top, u.Right, u.Bottom, u.Left)
}
