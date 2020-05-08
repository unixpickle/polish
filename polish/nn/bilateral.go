package nn

import "math"

// Bilateral is a bilateral filtering layer.
type Bilateral struct {
	KernelSize int
	SigmaBlur  float64
	SigmaDiff  float64
}

// Apply applies the bilateral filter and returns a Tensor
// of the same shape as t.
func (b *Bilateral) Apply(t *Tensor) *Tensor {
	distances := NewTensor(b.KernelSize, b.KernelSize, 1)
	center := b.KernelSize / 2
	for i := 0; i < b.KernelSize; i++ {
		for j := 0; j < b.KernelSize; j++ {
			*distances.At(i, j, 0) = float32((i-center)*(i-center) + (j-center)*(j-center))
		}
	}

	// Pad with very large negative numbers to prevent
	// the filter from incorporating the padding.
	padded := t.Add(100).Pad(center, center, center, center).Add(-100)
	out := NewTensor(t.Height, t.Width, t.Depth)
	Patches(padded, b.KernelSize, 1, func(idx int, patch *Tensor) {
		b.blurPatch(distances, patch, out.Data[idx*out.Depth:(idx+1)*out.Depth])
	})

	return out
}

func (b *Bilateral) blurPatch(dists, patch *Tensor, out []float32) {
	for i := 0; i < patch.Depth; i++ {
		out[i] = b.blurPatchChannel(dists, patch, i)
	}
}

func (b *Bilateral) blurPatchChannel(dists, patch *Tensor, z int) float32 {
	centerIdx := patch.Width / 2
	center := float64(*patch.At(centerIdx, centerIdx, z))

	weightedSum := 0.0
	weightSum := 0.0
	distsIdx := 0
	for patchIdx := z; patchIdx < len(patch.Data); patchIdx += patch.Depth {
		patchVal := float64(patch.Data[patchIdx])
		dist := dists.Data[distsIdx]
		distsIdx++
		weight := math.Exp(-(float64(dist)/(b.SigmaBlur*b.SigmaBlur) +
			math.Pow(patchVal-center, 2)/(b.SigmaDiff*b.SigmaDiff)))
		weightSum += weight
		weightedSum += weight * float64(patchVal)
	}

	return float32(weightedSum / weightSum)
}
