package main

import (
	"fmt"
	"math/rand"
	"os"
	"path/filepath"
	"time"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/model3d/render3d"
)

const ImageSize = 256

func main() {
	rand.Seed(time.Now().UnixNano())

	var args Args
	args.Parse()

	trainModels, testModels, err := ScanModelNet(args.ModelNetPath)
	essentials.Must(err)

	images, err := ScanImages(args.ImagesPath)
	essentials.Must(err)

	CreateOutput(args.OutputDir)

	for i := 0; true; i++ {
		outName := fmt.Sprintf("%05d.png", i)
		obj, rend := RandomScene(trainModels, images)
		SaveScene(filepath.Join(args.OutputDir, "train", outName), obj, rend)
		obj, rend = RandomScene(testModels, images)
		SaveScene(filepath.Join(args.OutputDir, "test", outName), obj, rend)
	}
}

func CreateOutput(outDir string) {
	if _, err := os.Stat(outDir); os.IsNotExist(err) {
		essentials.Must(os.Mkdir(outDir, 0755))
	} else {
		essentials.Must(err)
	}
	for _, dir := range []string{"train", "test"} {
		subDir := filepath.Join(outDir, dir)
		if _, err := os.Stat(subDir); os.IsNotExist(err) {
			essentials.Must(os.Mkdir(subDir, 0755))
		} else {
			essentials.Must(err)
		}
	}
}

func SaveScene(path string, obj render3d.Object, rend *render3d.RecursiveRayTracer) {
	rend.Antialias = 1.0
	rend.MaxDepth = 10
	rend.Cutoff = 1e-4

	grid := render3d.NewImage(ImageSize*2, ImageSize*2)

	renderAt := func(x, y, samples int) {
		rend.NumSamples = samples
		img := render3d.NewImage(ImageSize, ImageSize)
		rend.Render(img, obj)
		grid.CopyFrom(img, x, y)
	}

	renderAt(0, 0, 16)
	renderAt(ImageSize, 0, 32)
	renderAt(0, ImageSize, 128)
	renderAt(ImageSize, ImageSize, 1024)

	RandomizeBrightness(grid)
	grid.Save(path)
}
