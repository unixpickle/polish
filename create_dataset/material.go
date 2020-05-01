package main

import (
	"image"
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

// ModelMaterial defines the material of an object.
type ModelMaterial interface {
	FixNormal(ray *model3d.Ray, normal model3d.Coord3D) model3d.Coord3D
	Material(coord model3d.Coord3D) render3d.Material
}

// RandomizeMaterial generates a random material for the
// mesh and returns a new mesh to use in the old mesh's
// place (which may be necessary for refracted objects).
//
// The material may be based on a random image from a list
// of images, or it may be some other kind of material
// chosen from a distribution.
func RandomizeMaterial(m *model3d.Mesh, images []string) (*model3d.Mesh, ModelMaterial) {
	n := rand.Intn(10)
	if n == 0 {
		m = RepairMesh(m)
	} else {
		m = RepairOrKeep(m)
	}
	switch n {
	case 0:
		return m, createTransparent()
	case 1:
		return m, createMirror()
	case 2, 3, 4, 5:
		return m, createColored()
	default:
		return m, createTextured(m, images)
	}
}

// RandomizeWallMaterial is like RandomizeMaterial, but
// with a restricted class of materials for boundaries of
// the scene.
func RandomizeWallMaterial(c model3d.Collider, images []string) ModelMaterial {
	switch rand.Intn(10) {
	case 0:
		return createMirror()
	case 1, 2, 3, 4, 5:
		return createColored()
	default:
		return createTextured(c, images)
	}
}

func createTransparent() ModelMaterial {
	reflectFraction := math.Pow(rand.Float64(), 5)

	var refractColor render3d.Color
	if rand.Intn(2) == 0 {
		refractColor = render3d.NewColor(1 - reflectFraction)
	} else {
		refractColor = render3d.NewColorRGB(
			rand.Float64(), rand.Float64(), rand.Float64(),
		).Scale(1 - reflectFraction)
	}

	refractIndex := rand.Float64() + 1

	return StaticModelMaterial{
		Mat: &render3d.JoinedMaterial{
			Materials: []render3d.Material{
				&render3d.RefractMaterial{
					IndexOfRefraction: refractIndex,
					RefractColor:      refractColor,
				},
				&render3d.PhongMaterial{
					Alpha:         200.0,
					SpecularColor: render3d.NewColor(reflectFraction),
				},
			},
			Probs: []float64{1 - reflectFraction, reflectFraction},
		},
	}
}

func createMirror() ModelMaterial {
	return StaticModelMaterial{
		ShouldFixNormal: true,
		Mat: &render3d.PhongMaterial{
			Alpha:         200.0,
			SpecularColor: render3d.NewColor(0.95 + rand.Float64()*0.05),
		},
	}
}

func createColored() ModelMaterial {
	color := render3d.NewColorRGB(rand.Float64(), rand.Float64(), rand.Float64())
	diffuse := rand.Float64()

	var mat render3d.Material
	if rand.Intn(2) == 0 {
		mat = &render3d.LambertMaterial{DiffuseColor: color.Scale(diffuse)}
	} else {
		specular := rand.Float64() * (1 - diffuse)
		alpha := math.Exp(rand.Float64()*5 + 1)
		mat = &render3d.PhongMaterial{
			Alpha:         alpha,
			DiffuseColor:  color.Scale(diffuse),
			SpecularColor: render3d.NewColor(specular),
		}
	}

	return StaticModelMaterial{
		ShouldFixNormal: true,
		Mat:             mat,
	}
}

func createTextured(obj model3d.Bounder, images []string) ModelMaterial {
	path := images[rand.Intn(len(images))]
	r, err := os.Open(path)
	essentials.Must(err)
	defer r.Close()
	img, _, err := image.Decode(r)
	essentials.Must(err)
	return NewTexturedModelMaterial(obj, img)
}

// StaticModelMaterial is a ModelMaterial with a constant
// value.
type StaticModelMaterial struct {
	ShouldFixNormal bool

	Mat render3d.Material
}

func (s StaticModelMaterial) FixNormal(r *model3d.Ray, normal model3d.Coord3D) model3d.Coord3D {
	if s.ShouldFixNormal && r.Direction.Dot(normal) > 0 {
		return normal.Scale(-1)
	}
	return normal
}

func (s StaticModelMaterial) Material(coord model3d.Coord3D) render3d.Material {
	return s.Mat
}

// A TexturedModelMaterial is a ModelMaterial that applies
// the orthographic projection of an image to the model.
type TexturedModelMaterial struct {
	Alpha    float64
	Specular float64
	Diffuse  float64
	Texture  image.Image
	XBasis   model3d.Coord3D
	YBasis   model3d.Coord3D
}

// NewTexturedModelMaterial creates an object with a
// texture randomly slapped on along some axis.
func NewTexturedModelMaterial(obj model3d.Bounder, texture image.Image) *TexturedModelMaterial {
	size := obj.Max().Sub(obj.Min())
	maxDim := math.Max(math.Max(size.X, size.Y), size.Z)

	xBasis := model3d.NewCoord3DRandUnit()
	yBasis := model3d.NewCoord3DRandUnit().ProjectOut(xBasis).Normalize()

	bounds := texture.Bounds()
	scale := math.Exp(rand.Float64()*5) * 0.5
	xBasis = xBasis.Scale(scale * float64(bounds.Dx()) / maxDim)
	yBasis = yBasis.Scale(scale * float64(bounds.Dy()) / maxDim)

	diffuse := rand.Float64()
	specular := rand.Float64() * (1 - diffuse)

	return &TexturedModelMaterial{
		Alpha:    math.Exp(rand.Float64()*5 - 1),
		Specular: specular,
		Diffuse:  diffuse,
		Texture:  texture,
		XBasis:   xBasis,
		YBasis:   yBasis,
	}
}

func (t *TexturedModelMaterial) FixNormal(r *model3d.Ray, normal model3d.Coord3D) model3d.Coord3D {
	if r.Direction.Dot(normal) > 0 {
		return normal.Scale(-1)
	}
	return normal
}

func (t *TexturedModelMaterial) Material(p model3d.Coord3D) render3d.Material {
	x := int(t.XBasis.Dot(p))
	y := int(t.YBasis.Dot(p))

	// Add a large offset to prevent the modulus from not
	// working.
	x += 1000000
	y += 1000000

	bounds := t.Texture.Bounds()
	if (x/bounds.Dx())%2 == 0 {
		x = bounds.Dx() - (x % bounds.Dx()) - 1
	} else {
		x = x % bounds.Dx()
	}
	if (y/bounds.Dy())%2 == 0 {
		y = bounds.Dy() - (y % bounds.Dy()) - 1
	} else {
		y = y % bounds.Dy()
	}

	r, g, b, _ := t.Texture.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
	color := render3d.NewColorRGB(float64(r)/0xffff, float64(g)/0xffff,
		float64(b)/0xffff)

	return &render3d.PhongMaterial{
		Alpha:         t.Alpha,
		SpecularColor: render3d.NewColor(t.Specular),
		DiffuseColor:  color.Scale(t.Diffuse),
	}
}
