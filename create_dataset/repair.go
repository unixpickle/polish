package main

import (
	"math/rand"

	"github.com/unixpickle/model3d/model3d"
)

// RepairMesh creates a manifold, oriented mesh from a
// noisy mesh with incorrect normals, duplicate triangles,
// singularities, self-intersections, etc.
//
// If possible, this simply operates on the mesh. If a
// mesh-level repair does not work, a new mesh is created
// from scratch by thickening the surface of the original
// mesh.
func RepairMesh(m *model3d.Mesh) *model3d.Mesh {
	span := m.Max().Sub(m.Min()).Norm()

	// Fix small holes and duplicate triangles.
	if m.NeedsRepair() {
		m = m.Repair(span * 1e-5)
		m.Iterate(func(t *model3d.Triangle) {
			if len(m.Find(t[0], t[1], t[2])) > 1 {
				m.Remove(t)
			}
		})
		if m.NeedsRepair() {
			// Repair was unsuccessful.
			return createThicknessMesh(m)
		}
	}

	if !checkRayConsistency(m) {
		return createThicknessMesh(m)
	}

	if len(m.SingularVertices()) > 0 || m.SelfIntersections() != 0 {
		return createThicknessMesh(m)
	}

	m, _ = m.RepairNormals(span * 1e-5)
	return m
}

// checkRayConsistency makes sure the even-odd test is
// reliable for the mesh.
func checkRayConsistency(m *model3d.Mesh) bool {
	collider := model3d.MeshToCollider(m)

	min, max := m.Min(), m.Max()

	for i := 0; i < 1000; i++ {
		ray := &model3d.Ray{
			Origin: min.Add(model3d.Coord3D{
				X: rand.Float64(),
				Y: rand.Float64(),
				Z: rand.Float64(),
			}.Mul(max.Sub(min))),
			Direction: model3d.NewCoord3DRandUnit(),
		}

		c1 := collider.RayCollisions(ray, nil) % 2

		ray.Direction = model3d.NewCoord3DRandUnit()
		c2 := collider.RayCollisions(ray, nil) % 2

		if c1 != c2 {
			return false
		}
	}

	return true
}

// createThicknessMesh derives a new mesh from m based on
// the surface of m.
func createThicknessMesh(m *model3d.Mesh) *model3d.Mesh {
	delta := m.Max().Sub(m.Min()).Norm() / 100.0
	collider := model3d.MeshToCollider(m)
	solid := model3d.NewColliderSolidHollow(collider, delta*4)
	return model3d.MarchingCubesSearch(solid, delta, 8)
}
