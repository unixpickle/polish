package nn

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
