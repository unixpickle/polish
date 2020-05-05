package nn

// Tensor is a 3D array of numbers.
//
// It is arranged as [height x width x depth], with the
// outer dimension being height.
type Tensor struct {
	Height int
	Width  int
	Depth  int

	Data []float32
}

// NewTensor creates a zero tensor.
func NewTensor(height, width, depth int) *Tensor {
	return &Tensor{
		Height: height,
		Width:  width,
		Depth:  depth,
		Data:   make([]float32, width*height*depth),
	}
}

// At gets a pointer to the given coordinate.
func (t *Tensor) At(x, y, z int) *float32 {
	return &t.Data[z+t.Depth*(y+x*t.Width)]
}

// Pad creates a zero-padded version of the Tensor.
func (t *Tensor) Pad(top, right, bottom, left int) *Tensor {
	res := NewTensor(t.Height+top+bottom, t.Width+left+right, t.Depth)
	rowSize := t.Depth * t.Width
	for i := 0; i < t.Height; i++ {
		start := t.Depth * t.Width * i
		copy(res.Data[res.Depth*(left+(i+top)*res.Width):], t.Data[start:start+rowSize])
	}
	return res
}

// Unpad cuts out the edges of the Tensor, effectively
// inverting the operation done by Pad.
func (t *Tensor) Unpad(top, right, bottom, left int) *Tensor {
	res := NewTensor(t.Height-(top+bottom), t.Width-(left+right), t.Depth)
	rowSize := t.Depth * (t.Width - (left + right))
	for i := top; i < t.Height-bottom; i++ {
		start := t.Depth * (left + t.Width*i)
		copy(res.Data[res.Depth*res.Width*(i-top):], t.Data[start:start+rowSize])
	}
	return res
}
