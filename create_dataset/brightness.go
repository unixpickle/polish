package main

import (
	"math"
	"math/rand"
	"sort"

	"github.com/unixpickle/model3d/render3d"
)

func RandomizeBrightness(img *render3d.Image) {
	target := math.Min(0.9, math.Max(0.1, rand.NormFloat64()*0.1+0.3))
	median := math.Max(1e-5, quantileBrightness(img))
	if median > target {
		// Don't darken images with very bright lights.
		return
	}
	img.Scale(target / median)
}

func quantileBrightness(img *render3d.Image) float64 {
	bs := make([]float64, len(img.Data))
	for i, c := range img.Data {
		bs[i] = c.Sum() / 3.0
	}
	sort.Float64s(bs)
	return bs[int(float64(len(bs))*0.8)]
}
