package polish

import (
	"image"
	"image/color"
	"image/png"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

// CreateIncidenceMap creates a feature image where each
// pixel indicates the dot product of the camera ray with
// the normal of the first ray collision.
func CreateIncidenceMap(r *render3d.RecursiveRayTracer, obj render3d.Object,
	width, height int) *image.Gray {
	caster := r.Camera.Caster(float64(width)-1, float64(height)-1)
	img := image.NewGray(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
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

// CreateAlbedoMap creates a feature image where each
// pixel indicates the albedo of the surface intersected
// by the camera ray.
//
// The bsdfSamples argument specifies how many times each
// BSDF is sampled to approximate the albedo.
// A higher value gives more accurate results for complex
// materials.
func CreateAlbedoMap(r *render3d.RecursiveRayTracer, obj render3d.Object,
	width, height, bsdfSamples int) *image.RGBA {
	caster := r.Camera.Caster(float64(width)-1, float64(height)-1)
	res := render3d.NewImage(width, height)
	gen := rand.New(rand.NewSource(rand.Int63()))
	var idx int
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ray := &model3d.Ray{
				Origin:    r.Camera.Origin,
				Direction: caster(float64(x), float64(y)),
			}
			coll, mat, ok := obj.Cast(ray)
			if ok {
				dest := ray.Direction.Scale(-1).Normalize()
				var colorSum render3d.Color
				for i := 0; i < bsdfSamples; i++ {
					source := mat.SampleSource(gen, coll.Normal, dest)
					density := mat.SourceDensity(coll.Normal, source, dest)
					bsdf := mat.BSDF(coll.Normal, source, dest)
					sourceDot := math.Abs(source.Dot(coll.Normal))
					colorSum = colorSum.Add(bsdf.Scale(sourceDot / density))
				}
				color := colorSum.Scale(1 / float64(bsdfSamples))
				res.Data[idx] = color
				idx++
			}
		}
	}
	return res.RGBA()
}

// SaveFeatureMap encodes a feature map image to a PNG
// file.
func SaveFeatureMap(path string, img image.Image) error {
	w, err := os.Create(path)
	if err != nil {
		return err
	}
	defer w.Close()
	return png.Encode(w, img)
}
