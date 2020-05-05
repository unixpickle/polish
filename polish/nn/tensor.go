package nn

import (
	"image"
	"image/color"
)

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

// NewTensorRGB creates an RGB Tensor from an image.
func NewTensorRGB(img image.Image) *Tensor {
	b := img.Bounds()
	res := NewTensor(b.Dy(), b.Dx(), 3)
	var idx int
	for y := 0; y < res.Height; y++ {
		for x := 0; x < res.Width; x++ {
			red, green, blue, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
			for _, c := range []uint32{red, green, blue} {
				res.Data[idx] = float32(c) / 0xffff
				idx++
			}
		}
	}
	return res
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

// RGB creates an RGB image out of the Tensor.
//
// If the tensor does not have three channels, this will
// panic().
func (t *Tensor) RGB() image.Image {
	if t.Depth != 3 {
		panic("expected exactly 3 output channels")
	}
	res := image.NewRGBA(image.Rect(0, 0, t.Width, t.Height))
	var idx int
	for y := 0; y < t.Height; y++ {
		for x := 0; x < t.Width; x++ {
			var colors [3]uint8
			for i := 0; i < 3; i++ {
				x := t.Data[idx+i]
				if x < 0 {
					x = 0
				} else if x > 1 {
					x = 1
				}
				colors[i] = uint8(x * 255.999)
			}
			idx += 3
			res.SetRGBA(x, y, color.RGBA{
				R: colors[0],
				G: colors[1],
				B: colors[2],
				A: 0xff,
			})
		}
	}
	return res
}
