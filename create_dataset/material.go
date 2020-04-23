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

// RandomizeMaterial adds a random material to the mesh to
// turn it into an object.
//
// The material may be based on a random image from a list
// of images, or it may be some other kind of material
// chosen from a distribution.
func RandomizeMaterial(m *model3d.Mesh, images []string) render3d.Object {
	switch rand.Intn(10) {
	case 0:
		return createTransparent(m)
	case 1:
		return createMirror(m)
	case 2, 3, 4, 5:
		return createColored(m)
	default:
		return createTextured(m, images)
	}
}

// RandomizeWallMaterial is like RandomizeMaterial, but
// with a restricted class of materials for boundaries of
// the scene.
func RandomizeWallMaterial(m *model3d.Mesh, images []string) render3d.Object {
	switch rand.Intn(10) {
	case 0:
		return createMirror(m)
	case 1, 2, 3, 4, 5:
		return createColored(m)
	default:
		return createTextured(m, images)
	}
}

func createTransparent(m *model3d.Mesh) render3d.Object {
	m = RepairMesh(m)

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

	return &render3d.ColliderObject{
		Collider: model3d.MeshToCollider(m),
		Material: &render3d.JoinedMaterial{
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

func createMirror(m *model3d.Mesh) render3d.Object {
	return &FixNormalsObject{
		Object: &render3d.ColliderObject{
			Collider: model3d.MeshToCollider(m),
			Material: &render3d.PhongMaterial{
				Alpha:         200.0,
				SpecularColor: render3d.NewColor(0.95 + rand.Float64()*0.05),
			},
		},
	}
}

func createColored(m *model3d.Mesh) render3d.Object {
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

	return &FixNormalsObject{
		Object: &render3d.ColliderObject{
			Collider: model3d.MeshToCollider(m),
			Material: mat,
		},
	}
}

func createTextured(m *model3d.Mesh, images []string) render3d.Object {
	path := images[rand.Intn(len(images))]
	r, err := os.Open(path)
	essentials.Must(err)
	defer r.Close()
	img, _, err := image.Decode(r)
	essentials.Must(err)
	return NewTexturedObject(&FixNormalsObject{
		Object: &render3d.ColliderObject{
			Collider: model3d.MeshToCollider(m),
		},
	}, img)
}

// A TexturedObject is an object with an image projected
// orthographically on top of it.
type TexturedObject struct {
	render3d.Object

	Alpha    float64
	Specular float64
	Diffuse  float64
	Texture  image.Image
	XBasis   model3d.Coord3D
	YBasis   model3d.Coord3D
}

// NewTexturedObject creates an object with a texture
// randomly slapped on along some axis.
func NewTexturedObject(obj render3d.Object, texture image.Image) *TexturedObject {
	size := obj.Max().Sub(obj.Min())
	maxDim := math.Max(math.Max(size.X, size.Y), size.Z)

	xBasis := model3d.NewCoord3DRandUnit()
	yBasis := model3d.NewCoord3DRandUnit().ProjectOut(xBasis).Normalize()

	bounds := texture.Bounds()
	xBasis = xBasis.Scale(float64(bounds.Dx()) / maxDim * (rand.Float64() + 0.5))
	yBasis = yBasis.Scale(float64(bounds.Dy()) / maxDim * (rand.Float64() + 0.5))

	diffuse := rand.Float64()
	specular := rand.Float64() * (1 - diffuse)

	return &TexturedObject{
		Object: obj,

		Alpha:    math.Exp(rand.Float64()*5 - 1),
		Specular: specular,
		Diffuse:  diffuse,
		Texture:  texture,
		XBasis:   xBasis,
		YBasis:   yBasis,
	}
}

func (t *TexturedObject) Cast(ray *model3d.Ray) (model3d.RayCollision, render3d.Material, bool) {
	rc, mat, ok := t.Object.Cast(ray)
	if !ok {
		return rc, mat, ok
	}

	p := ray.Origin.Add(ray.Direction.Scale(rc.Scale))

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

	return rc, &render3d.PhongMaterial{
		Alpha:         t.Alpha,
		SpecularColor: render3d.NewColor(t.Specular),
		DiffuseColor:  color.Scale(t.Diffuse),
	}, ok
}

// A FixNormalsObject wraps an object and makes sure the
// normals always face outward.
type FixNormalsObject struct {
	render3d.Object
}

func (f *FixNormalsObject) Cast(r *model3d.Ray) (model3d.RayCollision, render3d.Material, bool) {
	rc, mat, ok := f.Object.Cast(r)
	if ok {
		if rc.Normal.Dot(r.Direction) > 0 {
			rc.Normal = rc.Normal.Scale(-1)
		}
	}
	return rc, mat, ok
}
