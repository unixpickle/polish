package nn

import (
	"runtime"
	"sync"
)

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

	features := c.transposedFeatures()
	Patches(t, c.KernelSize, c.Stride, func(outIdx int, patch *Tensor) {
		for i, feature := range features {
			var dot float32
			for i, x := range patch.Data {
				dot += x * feature.Data[i]
			}
			out.Data[outIdx*len(features)+i] = dot
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

// SpatialConv is a spatial-only 2D convolution operator.
// Unlike Conv, it uses a separate depthwise filter for
// each channel of the input; channels are not mixed.
//
// It contains weights of the shape:
//
//     [depth x kernel_size x kernel_size]
//
type SpatialConv struct {
	Depth      int
	KernelSize int
	Stride     int
	Weights    []float32
}

// Apply applies the convolution to a Tensor.
//
// The resulting Tensor's size is determined by
// ConvOutputSize().
func (s *SpatialConv) Apply(t *Tensor) *Tensor {
	if t.Depth != s.Depth {
		panic("input Tensor does not have the correct number of channels")
	}
	outH, outW := ConvOutputSize(t.Height, t.Width, s.KernelSize, s.Stride)
	out := NewTensor(outH, outW, s.Depth)

	features := s.features()
	Patches(t, s.KernelSize, s.Stride, func(outIdx int, patch *Tensor) {
		for i, feature := range features {
			patchIdx := i
			var dot float32
			for _, x := range feature.Data {
				dot += x * patch.Data[patchIdx]
				patchIdx += s.Depth
			}
			out.Data[outIdx*len(features)+i] = dot
		}
	})

	return out
}

func (s *SpatialConv) features() []*Tensor {
	featureStride := s.KernelSize * s.KernelSize
	var featureIdx int
	var result []*Tensor
	for i := 0; i < s.Depth; i++ {
		feature := s.Weights[featureIdx : featureIdx+featureStride]
		featureIdx += featureStride
		result = append(result, &Tensor{
			Height: s.KernelSize,
			Width:  s.KernelSize,
			Depth:  1,
			Data:   feature,
		})
	}
	return result
}

// Patches extracts image patches for a convolution of the
// given kernel size and stride, and calls f with each
// patch.
//
// It may call f from multiple Goroutines concurrently.
//
// The patches may be passed to f in any order.
// The index passed as the first argument to f goes left
// to right, top to bottom, so that patches are indexed
// like the pixels of an output image.
func Patches(t *Tensor, kernelSize, stride int, f func(int, *Tensor)) {
	numGos := runtime.GOMAXPROCS(0)

	_, outCols := ConvOutputSize(t.Height, t.Width, kernelSize, stride)

	var wg sync.WaitGroup
	for i := 0; i < numGos; i++ {
		wg.Add(1)
		go func(goIdx int) {
			defer wg.Done()
			patch := NewTensor(kernelSize, kernelSize, t.Depth)
			idx := goIdx * outCols
			for y := goIdx * stride; y+kernelSize <= t.Height; y += stride * numGos {
				for x := 0; x+kernelSize <= t.Width; x += stride {
					copyPatch(patch, t, x, y)
					f(idx, patch)
					idx++
				}
				idx += (numGos - 1) * outCols
			}
		}(i)
	}
	wg.Wait()
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
