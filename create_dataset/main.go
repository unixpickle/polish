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
		obj, rend, bidir := RandomScene(models, images)
		SaveScene(args.OutputDir, obj, rend, bidir)
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

func SaveScene(outDir string, obj render3d.Object, rend *render3d.RecursiveRayTracer,
	bidir *render3d.BidirPathTracer) {
	rend.Antialias = 1.0
	rend.MaxDepth = 10
	rend.Cutoff = 1e-4
	bidir.Antialias = 1.0
	bidir.MinDepth = 2
	bidir.MaxDepth = 10
	bidir.Cutoff = 1e-5
	bidir.RouletteDelta = 0.05
	bidir.PowerHeuristic = 2

	variance := rend.RayVariance(obj, 200, 200, 10)
	bidirVariance := bidir.RayVariance(obj, 200, 200, 10)
	log.Printf("Creating scene (var=%f bidir_var=%f) ...", variance, bidirVariance)

	incidence := CreateIncidenceMap(rend, obj)

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

	bidir.NumSamples = 16384
	bidir.MinSamples = 1024
	bidir.MaxStddev = 0.005 / scale
	bidir.OversaturatedStddevs = 3
	bidir.Cutoff = 0.01 / scale
	bidir.RouletteDelta = 0.05 / scale

	var lastFrac float64
	bidir.LogFunc = func(frac, samples float64) {
		if frac-lastFrac > 0.1 {
			lastFrac = frac
			log.Printf("Progress %.1f (samples %d)", frac, int(samples))
		}
	}

	target := render3d.NewImage(ImageSize, ImageSize)
	bidir.Render(target, obj)
	images["target.png"] = target

	// Save all the outputs once we have created them
	// to avoid creating empty folders in the dataset
	// for a long period of time.
	sampleDir := CreateSceneDir(outDir)
	for name, img := range images {
		img.Scale(scale)
		img.Save(filepath.Join(sampleDir, name))
	}
	SaveIncidenceMap(filepath.Join(sampleDir, "incidence.png"), incidence)
}
