// Package nn implements a small collection of neural
// network layers for denoising auto-encoders.
//
// It is designed for inference, not for training.
package nn

// A Layer is a Tensor operation.
type Layer interface {
	Apply(t *Tensor) *Tensor
}

// An NN is a special Layer that composes multiple other
// Layers.
type NN []Layer

// Apply applies all the layers in order.
func (n NN) Apply(t *Tensor) *Tensor {
	res := t
	for _, l := range n {
		res = l.Apply(res)
	}
	return res
}

// Residual is a special Layer that composes multiple
// other Layers and adds the output to the input.
type Residual []Layer

// Apply applies the layers in order and adds the output
// to the original input.
func (r Residual) Apply(t *Tensor) *Tensor {
	t1 := NN(r).Apply(t)
	if t1.Width != t.Width || t1.Height != t.Height || t1.Depth != t.Depth {
		panic("dimensions must match for residual connection")
	}
	res := NewTensor(t.Height, t.Width, t.Depth)
	for i, x := range t.Data {
		res.Data[i] = x + t1.Data[i]
	}
	return res
}
