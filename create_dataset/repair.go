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
	if m1 := RepairDirectly(m); m1 != nil {
		return m1
	}
	return createThicknessMesh(m)
}

// RepairOrKeep repairs the mesh on a triangle level as
// much as possible, and returns the new mesh.
func RepairOrKeep(m *model3d.Mesh) *model3d.Mesh {
	if m1 := RepairDirectly(m); m1 != nil {
		return m1
	}
	// At least eliminate some extra triangle overhead in
	// some 3D models.
	eliminateDuplicates(m)
	return m
}

// RepairDirectly attempts to repair the mesh by modifying
// its triangles.
// Returns nil if the mesh cannot be directly repaired.
func RepairDirectly(m *model3d.Mesh) *model3d.Mesh {
	span := m.Max().Sub(m.Min()).Norm()

	// Fix small holes and duplicate triangles.
	if m.NeedsRepair() {
		m = m.Repair(span * 1e-5)
		eliminateDuplicates(m)
		if m.NeedsRepair() {
			return nil
		}
	}

	if !checkRayConsistency(m) {
		return nil
	}

	if len(m.SingularVertices()) > 0 || m.SelfIntersections() != 0 {
		return nil
	}

	m, _ = m.RepairNormals(span * 1e-5)

	// Try to make the mesh smaller to speed things up.
	m = m.EliminateCoplanar(1e-5)

	return m
}

func eliminateDuplicates(m *model3d.Mesh) {
	m.Iterate(func(t *model3d.Triangle) {
		if len(m.Find(t[0], t[1], t[2])) > 1 {
			m.Remove(t)
		}
	})
}

// checkRayConsistency makes sure the even-odd test is
// reliable for the mesh.
func checkRayConsistency(m *model3d.Mesh) bool {
	collider := model3d.MeshToCollider(m)

	min, max := m.Min(), m.Max()

	evenOddAt := func(o model3d.Coord3D) bool {
		ray := &model3d.Ray{
			Origin:    o,
			Direction: model3d.NewCoord3DRandUnit(),
		}
		return collider.RayCollisions(ray, nil)%2 == 1
	}

	for i := 0; i < 1000; i++ {
		o := min.Add(model3d.Coord3D{
			X: rand.Float64(),
			Y: rand.Float64(),
			Z: rand.Float64(),
		}.Mul(max.Sub(min)))
		c1 := evenOddAt(o)
		c2 := evenOddAt(o)
		if c1 != c2 {
			return false
		}
	}

	// Check that the truly faces split the space.
	// This will catch infinitely thin meshes.
	for _, t := range m.TriangleSlice() {
		center := t[0].Add(t[1]).Add(t[2]).Scale(1.0 / 3)
		delta := t.Normal().Scale(1e-8)
		c1 := evenOddAt(center.Add(delta))
		c2 := evenOddAt(center.Sub(delta))
		if c1 == c2 {
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
	m = model3d.MarchingCubesSearch(solid, delta, 8)
	m = m.EliminateCoplanar(1e-5)
	return m
}
