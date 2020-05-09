package polish

import (
	"image"
	"image/color"
	"math/rand"
	"testing"

	"github.com/unixpickle/essentials"
)

func TestPatchEquivalence(t *testing.T) {
	img := image.NewRGBA(image.Rect(0, 0, 213, 192))
	for y := 0; y < img.Bounds().Dy(); y++ {
		for x := 0; x < img.Bounds().Dx(); x++ {
			img.SetRGBA(x, y, color.RGBA{
				R: uint8(rand.Intn(256)),
				G: uint8(rand.Intn(256)),
				B: uint8(rand.Intn(256)),
				A: 0xff,
			})
		}
	}

	// Use shallow model, since it has no global norms
	// like the deep model (which uses group norm).
	// Thus, the shallow model has a finite receptive
	// field.
	expected := PolishImage(ModelTypeShallow, img)

	actual := []image.Image{
		PolishImagePatches(ModelTypeShallow, img, 100, 50),
		PolishImagePatches(ModelTypeShallow, img, 100, 20),
		PolishImagePatches(ModelTypeShallow, img, 55, 18),
	}

CaseLoop:
	for i, a := range actual {
		for y := 0; y < a.Bounds().Dy(); y++ {
			for x := 0; x < a.Bounds().Dx(); x++ {
				r1, g1, b1, a1 := a.At(x, y).RGBA()
				r2, g2, b2, a2 := expected.At(x, y).RGBA()
				// Allow for small rounding errors.
				threshold := 0x200
				if essentials.AbsInt(int(r1-r2)) > threshold ||
					essentials.AbsInt(int(g1-g2)) > threshold ||
					essentials.AbsInt(int(b1-b2)) > threshold ||
					essentials.AbsInt(int(a1-a2)) > threshold {
					t.Errorf("case %d: mismatch at (%d, %d)", i, x, y)
					continue CaseLoop
				}
			}
		}
	}
}
