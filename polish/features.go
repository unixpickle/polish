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
	"github.com/unixpickle/polish/polish/nn"
)

// albedoMapSamples is the number of BSDF samples used to
// estimate a surface's albedo.
// This should roughly match the sample count used to
// train the model.
const albedoMapSamples = 400

// CreateAuxTensor creates a Tensor for a rendering with
// auxiliary feature channels.
//
// This Tensor can then be passed to PolishAux.
//
// The channels are ordered as follows:
//
//     1. Red
//     2. Green
//     3. Blue
//     4. Albedo red
//     5. Albedo green
//     6. Albedo blue
//     7. Ray-surface cosine map
//
func CreateAuxTensor(c *render3d.Camera, obj render3d.Object, img image.Image) *nn.Tensor {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	albedo := CreateAlbedoMap(c, obj, w, h, 400)
	incidence := CreateIncidenceMap(c, obj, w, h)
	return CreateAuxTensorImages(img, albedo, incidence)
}

// CreateAuxTensorImages creates an auxiliary Tensor using
// pre-constructed auxiliary images.
//
// See CreateAuxTensor for details on the channel order.
func CreateAuxTensorImages(img, albedo, incidence image.Image) *nn.Tensor {
	b := img.Bounds()
	w, h := b.Dx(), b.Dy()
	inTensor := nn.NewTensor(h, w, 7)
	for y := 0; y < h; y++ {
		for x := 0; x < w; x++ {
			red, green, blue, _ := img.At(x+b.Min.X, y+b.Min.Y).RGBA()
			for i, c := range []uint32{red, green, blue} {
				*inTensor.At(y, x, i) = float32(c) / 0xffff
			}
			red, green, blue, _ = albedo.At(x, y).RGBA()
			for i, c := range []uint32{red, green, blue} {
				*inTensor.At(y, x, i+3) = float32(c) / 0xffff
			}
			gray, _, _, _ := incidence.At(x, y).RGBA()
			*inTensor.At(y, x, 6) = float32(gray) / 0xffff
		}
	}
	return inTensor
}

// CreateIncidenceMap creates a feature image where each
// pixel indicates the dot product of the camera ray with
// the normal of the first ray collision.
func CreateIncidenceMap(c *render3d.Camera, obj render3d.Object,
	width, height int) *image.Gray {
	caster := c.Caster(float64(width)-1, float64(height)-1)
	img := image.NewGray(image.Rect(0, 0, width, height))
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ray := &model3d.Ray{
				Origin:    c.Origin,
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
func CreateAlbedoMap(c *render3d.Camera, obj render3d.Object,
	width, height, bsdfSamples int) *image.RGBA {
	caster := c.Caster(float64(width)-1, float64(height)-1)
	res := render3d.NewImage(width, height)
	gen := rand.New(rand.NewSource(rand.Int63()))
	var idx int
	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			ray := &model3d.Ray{
				Origin:    c.Origin,
				Direction: caster(float64(x), float64(y)),
			}
			coll, mat, ok := obj.Cast(ray)
			if ok {
				dest := ray.Direction.Scale(-1).Normalize()
				res.Data[idx] = estimateAlbedo(gen, mat, coll.Normal, dest, bsdfSamples)
				idx++
			}
		}
	}
	return res.RGBA()
}

func estimateAlbedo(gen *rand.Rand, mat render3d.Material, normal, dest model3d.Coord3D,
	bsdfSamples int) render3d.Color {
	switch mat := mat.(type) {
	case *render3d.LambertMaterial:
		if normal.Dot(dest) < 0 {
			return render3d.Color{}
		}
		return mat.DiffuseColor
	default:
		var colorSum render3d.Color
		for i := 0; i < bsdfSamples; i++ {
			source := mat.SampleSource(gen, normal, dest)
			density := mat.SourceDensity(normal, source, dest)
			bsdf := mat.BSDF(normal, source, dest)
			sourceDot := math.Abs(source.Dot(normal))
			colorSum = colorSum.Add(bsdf.Scale(sourceDot / density))
		}
		return colorSum.Scale(1 / float64(bsdfSamples))
	}
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
