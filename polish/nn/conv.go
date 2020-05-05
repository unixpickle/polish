package nn

// Conv is a 2D convolution operator.
//
// It contains weights of the shape:
//
//     [out_depth x in_depth x kernel_size x kernel_size]
//
type Conv struct {
	OutDepth   int
	InDepth    int
	KernelSize int
	Stride     int
	Weights    []float32
}

// Apply applies the convolution to a Tensor.
//
// The resulting Tensor's size is determined by
// ConvOutputSize().
func (c *Conv) Apply(t *Tensor) *Tensor {
	if t.Depth != c.InDepth {
		panic("input Tensor does not have the correct number of channels")
	}
	outH, outW := ConvOutputSize(t.Height, t.Width, c.KernelSize, c.Stride)
	out := NewTensor(outH, outW, c.OutDepth)

	var outIdx int
	features := c.transposedFeatures()
	Patches(t, c.KernelSize, c.Stride, func(patch *Tensor) {
		for _, feature := range features {
			var dot float32
			for i, x := range patch.Data {
				dot += x * feature.Data[i]
			}
			out.Data[outIdx] = dot
			outIdx++
		}
	})

	return out
}

func (c *Conv) transposedFeatures() []*Tensor {
	featureStride := c.KernelSize * c.KernelSize * c.InDepth
	var featureIdx int
	var result []*Tensor
	for i := 0; i < c.OutDepth; i++ {
		feature := c.Weights[featureIdx : featureIdx+featureStride]
		featureIdx += featureStride

		tensor := NewTensor(c.KernelSize, c.KernelSize, c.InDepth)
		var idx int
		for y := 0; y < tensor.Height; y++ {
			for x := 0; x < tensor.Width; x++ {
				for z := 0; z < tensor.Depth; z++ {
					tensor.Data[idx] = feature[(y+z*c.KernelSize)*c.KernelSize+x]
					idx++
				}
			}
		}
		result = append(result, tensor)
	}
	return result
}

// Patches extracts image patches for a convolution of the
// given kernel size and stride, and calls f with each
// patch.
//
// The patches are enumerated in left to right, top to
// bottom order, so that f is called in order of the
// pixels in an output image.
func Patches(t *Tensor, kernelSize, stride int, f func(*Tensor)) {
	patch := NewTensor(kernelSize, kernelSize, t.Depth)
	for y := 0; y+kernelSize <= t.Height; y += stride {
		for x := 0; x+kernelSize <= t.Width; x += stride {
			copyPatch(patch, t, x, y)
			f(patch)
		}
	}
}

func copyPatch(dst, src *Tensor, x, y int) {
	dstOffset := 0
	dstStride := dst.Depth * dst.Width
	srcOffset := (x + y*src.Width) * src.Depth
	srcStride := src.Width * src.Depth
	for subY := 0; subY < dst.Height; subY++ {
		copy(dst.Data[dstOffset:dstOffset+dstStride], src.Data[srcOffset:])
		dstOffset += dstStride
		srcOffset += srcStride
	}
}

// ConvOutputSize gets the output dimensions from a
// convolution operation.
func ConvOutputSize(height, width, kernelSize, stride int) (heightOut, widthOut int) {
	for i := 0; i+kernelSize <= width; i += stride {
		widthOut++
	}
	for i := 0; i+kernelSize <= height; i += stride {
		heightOut++
	}
	return
}
