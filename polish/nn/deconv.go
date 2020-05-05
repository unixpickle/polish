package nn

// Deconv is a 2D transposed convolution operator.
//
// It contains weights of the shape:
//
//     [in_depth x out_depth x kernel_size x kernel_size]
//
type Deconv struct {
	OutDepth   int
	InDepth    int
	KernelSize int
	Stride     int
	Weights    []float32
}

// Apply applies the transposed convolution to a Tensor.
//
// The resulting Tensor's size is determined by
// DeconvOutputSize().
func (d *Deconv) Apply(t *Tensor) *Tensor {
	if t.Depth != d.InDepth {
		panic("input Tensor does not have the correct number of channels")
	}
	outH, outW := DeconvOutputSize(t.Height, t.Width, d.KernelSize, d.Stride)
	out := NewTensor(outH, outW, d.OutDepth)

	features := d.transposedFeatures()
	var inIdx int
	for y := 0; y < t.Height; y++ {
		outY := y * d.Stride
		for x := 0; x < t.Width; x++ {
			outX := x * d.Stride
			for _, feature := range features {
				scale := t.Data[inIdx]
				inIdx++
				addPatch(out, feature, outX, outY, scale)
			}
		}
	}

	return out
}

func (d *Deconv) transposedFeatures() []*Tensor {
	featureStride := d.KernelSize * d.KernelSize * d.OutDepth
	var featureIdx int
	var result []*Tensor
	for i := 0; i < d.InDepth; i++ {
		feature := d.Weights[featureIdx : featureIdx+featureStride]
		featureIdx += featureStride

		tensor := NewTensor(d.KernelSize, d.KernelSize, d.OutDepth)
		var idx int
		for y := 0; y < tensor.Height; y++ {
			for x := 0; x < tensor.Width; x++ {
				for z := 0; z < tensor.Depth; z++ {
					tensor.Data[idx] = feature[(y+z*d.KernelSize)*d.KernelSize+x]
					idx++
				}
			}
		}
		result = append(result, tensor)
	}
	return result
}

func addPatch(dst, src *Tensor, outX, outY int, scale float32) {
	var srcIdx int
	dstIdx := (outX + outY*dst.Width) * dst.Depth
	dstStride := dst.Width*dst.Depth - src.Width*src.Depth
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			for z := 0; z < src.Depth; z++ {
				dst.Data[dstIdx] += src.Data[srcIdx] * scale
				dstIdx++
				srcIdx++
			}
		}
		dstIdx += dstStride
	}
}

// DeconvOutputSize gets the output dimensions from a
// transposed convolution operation.
func DeconvOutputSize(height, width, kernelSize, stride int) (heightOut, widthOut int) {
	if height > 0 {
		heightOut = (height-1)*stride + kernelSize
	}
	if width > 0 {
		widthOut = (width-1)*stride + kernelSize
	}
	return
}
