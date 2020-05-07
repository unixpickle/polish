package nn

import (
	"runtime"
	"sync"
)

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
	features := d.transposedFeatures()

	out := NewTensor(outH, outW, d.OutDepth)
	lock := sync.Mutex{}

	d.iteratePatches(t, func(tmp *Tensor, x, y int, data []float32) {
		for i, scale := range data {
			feature := features[i]
			if i == 0 {
				for j, x := range feature.Data {
					tmp.Data[j] = x * scale
				}
			} else {
				for j, x := range feature.Data {
					tmp.Data[j] += x * scale
				}
			}
		}
		outX := x * d.Stride
		outY := y * d.Stride

		// Adding to the global image must be synchronized,
		// but it takes up so little of the computation time
		// that it doesn't seem to matter.
		lock.Lock()
		addPatch(out, tmp, outX, outY)
		lock.Unlock()
	})

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

func (d *Deconv) iteratePatches(t *Tensor, f func(tmp *Tensor, x, y int, data []float32)) {
	numGos := runtime.GOMAXPROCS(0)
	var wg sync.WaitGroup
	for i := 0; i < numGos; i++ {
		wg.Add(1)
		go func(goIdx int) {
			defer wg.Done()
			tmp := NewTensor(d.KernelSize, d.KernelSize, d.OutDepth)
			idx := goIdx * t.Width * t.Depth
			for y := goIdx; y < t.Height; y += numGos {
				for x := 0; x < t.Width; x++ {
					f(tmp, x, y, t.Data[idx:idx+t.Depth])
					idx += t.Depth
				}
				idx += (numGos - 1) * t.Width
			}
		}(i)
	}
	wg.Wait()
}

func addPatch(dst, src *Tensor, outX, outY int) {
	var srcIdx int
	dstIdx := (outX + outY*dst.Width) * dst.Depth
	dstStride := dst.Width*dst.Depth - src.Width*src.Depth
	for y := 0; y < src.Height; y++ {
		for x := 0; x < src.Width; x++ {
			for z := 0; z < src.Depth; z++ {
				dst.Data[dstIdx] += src.Data[srcIdx]
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
