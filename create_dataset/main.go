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

	models, err := ScanModelNet(args.ModelNetPath)
	essentials.Must(err)

	images, err := ScanImages(args.ImagesPath)
	essentials.Must(err)

	CreateOutput(args.OutputDir)

	for i := 0; true; i++ {
		obj, rend := RandomScene(models, images)
		SaveScene(args.OutputDir, obj, rend)
	}
}

func CreateOutput(outDir string) {
	if _, err := os.Stat(outDir); os.IsNotExist(err) {
		essentials.Must(os.Mkdir(outDir, 0755))
	} else {
		essentials.Must(err)
	}
}

func CreateSceneDir(outDir string) string {
	for i := 0; i < 10; i++ {
		outName := fmt.Sprintf("%06x", rand.Intn(0x1000000))
		newPath := filepath.Join(outDir, outName)
		if _, err := os.Stat(newPath); os.IsNotExist(err) {
			essentials.Must(os.Mkdir(newPath, 0755))
			return newPath
		}
	}
	essentials.Die("could not allocate a new output file")
	return ""
}

func SaveScene(outDir string, obj render3d.Object, rend *render3d.RecursiveRayTracer) {
	rend.Antialias = 1.0
	rend.MaxDepth = 10
	rend.Cutoff = 1e-4

	variance := rend.RayVariance(obj, 200, 200, 5)
	log.Printf("Creating scene (var=%f) ...", variance)

	renderAtRes := func(samples int) *render3d.Image {
		rend.NumSamples = samples
		img1 := render3d.NewImage(ImageSize, ImageSize)
		img2 := render3d.NewImage(ImageSize, ImageSize)
		rend.Render(img1, obj)
		rend.Render(img2, obj)
		img := render3d.NewImage(ImageSize*2, ImageSize)
		img.CopyFrom(img1, 0, 0)
		img.CopyFrom(img2, ImageSize, 0)
		return img
	}

	images := map[string]*render3d.Image{}
	for _, samples := range []int{1, 16, 64, 128, 512} {
		images[fmt.Sprintf("input_%d.png", samples)] = renderAtRes(samples)
	}

	scale := BrightnessScale(images["input_512.png"])

	rend.NumSamples = 16384
	rend.MinSamples = 512
	rend.MaxStddev = 0.01 / scale
	rend.OversaturatedStddevs = 3
	target := render3d.NewImage(ImageSize, ImageSize)
	rend.Render(target, obj)
	images["target.png"] = target

	// Save all the outputs once we have created them
	// to avoid creating empty folders in the dataset
	// for a long period of time.
	sampleDir := CreateSceneDir(outDir)
	for name, img := range images {
		img.Scale(scale)
		img.Save(filepath.Join(sampleDir, name))
	}
}
