package main

import (
	"fmt"
	"log"
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

	trainDir := filepath.Join(args.OutputDir, "train")
	testDir := filepath.Join(args.OutputDir, "test")
	for i := 0; true; i++ {
		obj, rend := RandomScene(trainModels, images)
		SaveScene(NextImagePath(trainDir), obj, rend)
		obj, rend = RandomScene(testModels, images)
		SaveScene(NextImagePath(testDir), obj, rend)
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

func NextImagePath(dir string) string {
	for i := 0; true; i++ {
		outName := fmt.Sprintf("%05d.png", i)
		newPath := filepath.Join(dir, outName)
		if _, err := os.Stat(newPath); os.IsNotExist(err) {
			return newPath
		}
	}
	essentials.Die("could not allocate a new output file")
	return ""
}

func SaveScene(path string, obj render3d.Object, rend *render3d.RecursiveRayTracer) {
	rend.Antialias = 1.0
	rend.MaxDepth = 10
	rend.Cutoff = 1e-4

	variance := rend.RayVariance(obj, 200, 200, 5)
	log.Printf("Creating scene (var=%f): %s", variance, path)

	grid := render3d.NewImage(ImageSize*2, ImageSize*5)

	renderAt := func(x, y, samples int) float64 {
		rend.NumSamples = samples
		img := render3d.NewImage(ImageSize, ImageSize)
		rend.Render(img, obj)
		grid.CopyFrom(img, x, y)
		return BrightnessScale(img)
	}

	for i, samples := range []int{1, 16, 32, 128} {
		for j := 0; j < 2; j++ {
			renderAt(j*ImageSize, i*ImageSize, samples)
		}
	}

	renderAt(0, ImageSize*4, 512)
	scale := renderAt(ImageSize, ImageSize*4, 2048)

	grid.Scale(scale)
	grid.Save(path)
}
