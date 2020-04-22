package main

import (
	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/render3d"
)

func main() {
	var args Args
	args.Parse()

	trainModels, testModels, err := ScanModelNet(args.ModelNetPath)
	essentials.Must(err)

	// TODO: delete me.
	_ = testModels

	images, err := ScanImages(args.ImagesPath)
	essentials.Must(err)

	obj, rend := RandomScene(trainModels, images)
	rend.NumSamples = 10
	rend.Antialias = 1.0
	rend.MaxDepth = 10
	rend.Cutoff = 1e-4

	img := render3d.NewImage(300, 300)
	rend.Render(img, obj)
	img.Save("render.png")
}
