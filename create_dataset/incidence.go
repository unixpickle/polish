package main

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

// CreateIncidenceMap creates a grayscale image
// representing the dot product of the ray with the normal
// of the surface, for each pixel.
func CreateIncidenceMap(r *render3d.RecursiveRayTracer, obj render3d.Object) *image.Gray {
	caster := r.Camera.Caster(ImageSize-1, ImageSize-1)
	img := image.NewGray(image.Rect(0, 0, ImageSize, ImageSize))
	for y := 0; y < ImageSize; y++ {
		for x := 0; x < ImageSize; x++ {
			ray := &model3d.Ray{
				Origin:    r.Camera.Origin,
				Direction: caster(float64(x), float64(y)),
			}
			coll, _, ok := obj.Cast(ray)
			if ok {
				incidence := uint8(math.Abs(coll.Normal.Dot(ray.Direction.Normalize())) * 255.999)
				img.SetGray(x, y, color.Gray{Y: incidence})
			}
		}
	}
	return img
}

func SaveIncidenceMap(path string, img *image.Gray) {
	w, err := os.Create(path)
	essentials.Must(err)
	defer w.Close()
	essentials.Must(png.Encode(w, img))
}
