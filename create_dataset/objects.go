package main

import (
	"github.com/unixpickle/model3d/model3d"
	"github.com/unixpickle/model3d/render3d"
)

type ColliderObject struct {
	render3d.Object
	Mat ModelMaterial
}

func NewColliderObject(c model3d.Collider, mat ModelMaterial) *ColliderObject {
	return &ColliderObject{
		Object: &render3d.ColliderObject{Collider: c},
		Mat:    mat,
	}
}

func (c *ColliderObject) Cast(r *model3d.Ray) (model3d.RayCollision, render3d.Material, bool) {
	rc, _, ok := c.Object.Cast(r)
	if !ok {
		return rc, nil, ok
	}
	mat := c.Mat.Material(r.Origin.Add(r.Direction.Scale(rc.Scale)))
	rc.Normal = c.Mat.FixNormal(r, rc.Normal)
	return rc, mat, ok
}

type MeshesObject struct {
	render3d.Object
	Mats map[*model3d.Triangle]ModelMaterial
}

func NewMeshesObject(meshes []*model3d.Mesh, mats []ModelMaterial) *MeshesObject {
	res := map[*model3d.Triangle]ModelMaterial{}
	fullMesh := model3d.NewMesh()
	for i, m := range meshes {
		m.Iterate(func(t *model3d.Triangle) {
			res[t] = mats[i]
		})
		fullMesh.AddMesh(m)
	}
	return &MeshesObject{
		Object: &render3d.ColliderObject{
			Collider: model3d.MeshToCollider(fullMesh),
		},
		Mats: res,
	}
}

func (m *MeshesObject) Cast(r *model3d.Ray) (model3d.RayCollision, render3d.Material, bool) {
	rc, _, ok := m.Object.Cast(r)
	if !ok {
		return rc, nil, ok
	}
	tri := rc.Extra.(*model3d.TriangleCollision).Triangle
	mmat := m.Mats[tri]
	mat := mmat.Material(r.Origin.Add(r.Direction.Scale(rc.Scale)))
	rc.Normal = mmat.FixNormal(r, rc.Normal)
	return rc, mat, ok
}
