package main

import (
	"math"
	"math/rand"
	"os"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

// RandomScene creates a random collection of objects and
// fills out a renderer to render them.
func RandomScene(models, images []string) (render3d.Object, *render3d.RecursiveRayTracer,
	*render3d.BidirPathTracer) {
	layout := RandomSceneLayout()
	numObjects := rand.Intn(10) + 1
	numLights := rand.Intn(10) + 1

	var objects render3d.JoinedObject
	var lights []render3d.AreaLight
	var focusPoints []render3d.FocusPoint
	var focusProbs []float64

	for _, wall := range layout.CreateBackdrop() {
		mat := RandomizeWallMaterial(wall, images)
		objects = append(objects, NewColliderObject(wall, mat))
	}

	var modelMeshes []*model3d.Mesh
	var modelMats []ModelMaterial
	for i := 0; i < numObjects; i++ {
		path := models[rand.Intn(len(models))]
		r, err := os.Open(path)
		essentials.Must(err)
		defer r.Close()
		tris, err := model3d.ReadOFF(r)
		essentials.Must(err)
		mesh := model3d.NewMeshTriangles(tris)
		mesh = randomRotation(mesh)
		mesh = layout.PlaceMesh(mesh)
		mesh, mat := RandomizeMaterial(mesh, images)
		modelMeshes = append(modelMeshes, mesh)
		modelMats = append(modelMats, mat)
	}
	objects = append(objects, NewMeshesObject(modelMeshes, modelMats))

	for i := 0; i < numLights; i++ {
		light, focusPoint := layout.CreateLight()
		objects = append(objects, light)
		lights = append(lights, light)
		focusPoints = append(focusPoints, focusPoint)
		focusProbs = append(focusProbs, 0.3/float64(numLights))
	}

	origin, target := layout.CameraInfo()
	fov := (rand.Float64()*0.5 + 0.5) * math.Pi / 3.0
	camera := render3d.NewCameraAt(origin, target, fov)
	return objects, &render3d.RecursiveRayTracer{
			Camera:          camera,
			FocusPoints:     focusPoints,
			FocusPointProbs: focusProbs,
		}, &render3d.BidirPathTracer{
			Camera: camera,
			Light:  render3d.JoinAreaLights(lights...),
		}
}

func randomRotation(m *model3d.Mesh) *model3d.Mesh {
	var rotation *model3d.Matrix3
	if rand.Intn(3) == 0 {
		// Completely random rotation.
		rotation = model3d.NewMatrix3Rotation(model3d.NewCoord3DRandUnit(),
			rand.Float64()*math.Pi*2)
	} else {
		// Axis swap rotation
		a1 := rand.Intn(3)
		a2 := rand.Intn(2)
		if a2 >= a1 {
			a2++
		}
		rotation = &model3d.Matrix3{}
		for i := 0; i < 3; i++ {
			if i == a1 {
				rotation[i*3+a2] = 1
			} else if i == a2 {
				rotation[i*3+a1] = 1
			} else {
				rotation[i*3+i] = 1
			}
		}
	}
	return m.MapCoords(rotation.MulColumn)
}

// RandomSceneLayout samples a SceneLayout from some
// distribution.
func RandomSceneLayout() SceneLayout {
	if rand.Intn(2) == 0 {
		return RoomLayout{
			Width: rand.Float64()*2.0 + 0.5,
			Depth: rand.Float64()*3.0 + 2.0,
		}
	} else {
		return WorldLayout{}
	}
}

type SceneLayout interface {
	// CameraInfo determines where the scene would like to
	// setup the camera for rendering.
	CameraInfo() (position, target model3d.Coord3D)

	// CreateLight creates a randomized light object that
	// makes sense in this kind of scene.
	CreateLight() (render3d.AreaLight, render3d.FocusPoint)

	// CreateBackdrop creates models which act as walls of
	// the scene.
	CreateBackdrop() []model3d.Collider

	// PlaceMesh translates and scales the mesh so that it
	// fits within the scene.
	PlaceMesh(m *model3d.Mesh) *model3d.Mesh
}

// RoomLayout is a simple scene in a room with lights on
// the walls and ceiling.
type RoomLayout struct {
	Width float64
	Depth float64
}

func (r RoomLayout) CameraInfo() (position, target model3d.Coord3D) {
	return model3d.Coord3D{Z: 0.5, Y: -r.Depth/2 + 1e-5}, model3d.Coord3D{Z: 0.5, Y: r.Depth / 2}
}

func (r RoomLayout) CreateLight() (render3d.AreaLight, render3d.FocusPoint) {
	var center model3d.Coord3D
	var axis model3d.Coord3D
	if rand.Intn(2) == 0 {
		// Place light on ceiling.
		center = model3d.Coord3D{
			X: (rand.Float64() - 0.5) * r.Width,
			Y: (rand.Float64() - 0.5) * r.Depth,
			Z: 1.0,
		}
		axis = model3d.Coord3D{Z: 1}
	} else {
		// Place light on side wall.
		x := r.Width / 2
		if rand.Intn(2) == 0 {
			x = -x
		}
		center = model3d.Coord3D{
			X: x,
			Y: (rand.Float64() - 0.5) * r.Depth,
			Z: rand.Float64() * 0.9,
		}
		axis = model3d.Coord3D{X: 1 / x}
	}

	var light render3d.AreaLight
	var focusRadius float64
	color := render3d.NewColor((rand.Float64() + 0.1) * 20)
	if rand.Intn(2) == 0 {
		focusRadius = rand.Float64()*0.2 + 0.05
		light = render3d.NewSphereAreaLight(
			&model3d.Sphere{Center: center, Radius: focusRadius},
			color,
		)
	} else {
		size := uniformRandom().Scale(0.1).Add(model3d.Coord3D{X: 0.05, Y: 0.05, Z: 0.05})
		light = render3d.NewMeshAreaLight(
			model3d.NewMeshRect(
				center.Sub(size),
				center.Add(size),
			),
			color,
		)
		focusRadius = size.Norm()
	}

	light = &HalfLight{
		AreaLight: light,
		Axis:      axis,
		MaxDot:    1,
	}

	return light, &render3d.SphereFocusPoint{
		Center: light.Min().Mid(light.Max()),
		Radius: focusRadius,
	}
}

func (r RoomLayout) CreateBackdrop() []model3d.Collider {
	min := model3d.Coord3D{X: -r.Width / 2, Y: -r.Depth / 2}
	max := model3d.Coord3D{X: r.Width / 2, Y: r.Depth / 2, Z: 1}
	mesh := model3d.NewMeshRect(min, max)

	var walls []model3d.Collider
	mesh.Iterate(func(t *model3d.Triangle) {
		var neighbor *model3d.Triangle
		for _, n := range mesh.Neighbors(t) {
			if n.Normal().Dot(t.Normal()) > 0.99 {
				neighbor = n
				break
			}
		}
		mesh.Remove(neighbor)
		mesh.Remove(t)
		walls = append(walls, model3d.NewJoinedCollider([]model3d.Collider{t, neighbor}))
	})

	return walls
}

func (r RoomLayout) PlaceMesh(m *model3d.Mesh) *model3d.Mesh {
	placeMin := model3d.Coord3D{X: -r.Width / 2, Y: -r.Depth / 4}
	placeMax := model3d.Coord3D{X: r.Width / 2, Y: r.Depth / 2, Z: 1}
	return placeInBounds(placeMin, placeMax, m)
}

func placeInBounds(placeMin, placeMax model3d.Coord3D, m *model3d.Mesh) *model3d.Mesh {
	min, max := m.Min(), m.Max()
	diff := max.Sub(min)
	pDiff := placeMax.Sub(placeMin)
	maxScale := math.Min(pDiff.X/diff.X, math.Min(pDiff.Y/diff.Y, pDiff.Z/diff.Z))
	scale := (rand.Float64()*0.9 + 0.1) * maxScale
	m = m.Scale(scale)

	min, max = m.Min(), m.Max()
	translateMin := placeMin.Sub(min)
	translateMax := placeMax.Sub(max)
	translate := uniformRandom().Mul(translateMax.Sub(translateMin)).Add(translateMin)

	// Drop Z to minimum.
	translate.Z = translateMin.Z

	return m.MapCoords(translate.Add)
}

func uniformRandom() model3d.Coord3D {
	return model3d.Coord3D{X: rand.Float64(), Y: rand.Float64(), Z: rand.Float64()}
}

// WorldLayout is a layout that places objects in a large
// hemisphere.
type WorldLayout struct{}

func (w WorldLayout) CameraInfo() (position, target model3d.Coord3D) {
	return model3d.Coord3D{Y: -20, Z: 5}, model3d.Coord3D{Y: 0, Z: 5}
}

func (w WorldLayout) CreateLight() (render3d.AreaLight, render3d.FocusPoint) {
	center := model3d.NewCoord3DRandUnit().Scale(70)
	if center.Z < 0 {
		center.Z = -center.Z
	}
	if center.Y > 0 {
		// Usually, we want the lights behind the camera.
		if rand.Intn(5) != 0 {
			center.Y *= -1
		}
	}
	shape := &model3d.Sphere{Center: center, Radius: rand.Float64()*5.0 + 2.0}
	r2 := shape.Radius * shape.Radius
	emission := render3d.NewColor((rand.Float64() + 0.5) * 200 / r2)
	return render3d.NewSphereAreaLight(shape, emission),
		&render3d.SphereFocusPoint{
			Center: shape.Center,
			Radius: shape.Radius,
		}
}

func (w WorldLayout) CreateBackdrop() []model3d.Collider {
	r := 100.0
	p1 := model3d.Coord3D{X: -r, Y: -r}
	p2 := model3d.Coord3D{X: -r, Y: r}
	p3 := model3d.Coord3D{X: r, Y: r}
	p4 := model3d.Coord3D{X: r, Y: -r}

	floor := model3d.NewMesh()
	floor.Add(&model3d.Triangle{p1, p2, p3})
	floor.Add(&model3d.Triangle{p1, p3, p4})

	dome := &model3d.Sphere{Radius: r}

	return []model3d.Collider{model3d.MeshToCollider(floor), dome}
}

func (w WorldLayout) PlaceMesh(m *model3d.Mesh) *model3d.Mesh {
	min := model3d.Coord3D{X: -7, Y: -7}
	max := model3d.Coord3D{X: 7, Y: 7, Z: 7}
	return placeInBounds(min, max, m)
}

type HalfLight struct {
	render3d.AreaLight

	Axis   model3d.Coord3D
	MaxDot float64
}

func (h *HalfLight) SampleLight(gen *rand.Rand) (point, normal model3d.Coord3D, c render3d.Color) {
	for {
		point, normal, c = h.AreaLight.SampleLight(gen)
		if h.Axis.Dot(point) < h.MaxDot {
			return
		}
	}
}

func (h *HalfLight) TotalEmission() float64 {
	return h.AreaLight.TotalEmission() / 2
}
