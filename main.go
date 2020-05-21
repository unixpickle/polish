// Command polish denoises images that were produced by a
// Monte Carlo rendering technique (e.g. path tracing).
package main

import (
	"flag"
	"fmt"
	"image"
	"image/png"
	"os"

	"github.com/unixpickle/polish/polish"

	"github.com/unixpickle/essentials"
)

func main() {
	var model string
	var patchSize int
	var patchBorder int
	var albedoPath string
	var incidencePath string
	flag.StringVar(&model, "model", "deep", "type of model to use "+
		"('shallow', 'deep', 'shallow-aux', 'deep-aux', 'bilateral')")
	flag.IntVar(&patchSize, "patch", 0, "image patch size to process at once (0 to disable)")
	flag.IntVar(&patchBorder, "patch-border", -1, "border for image patches (-1 uses default)")
	flag.StringVar(&albedoPath, "albedo", "", "path to albedo map image (for aux models)")
	flag.StringVar(&incidencePath, "incidence", "", "path to incidence map image (for aux models)")

	flag.Usage = func() {
		fmt.Fprintln(os.Stderr, "Usage: "+os.Args[0]+" [flags] <input.png> <output.png>")
		fmt.Fprintln(os.Stderr)
		flag.PrintDefaults()
		fmt.Fprintln(os.Stderr)
		os.Exit(1)
	}

	flag.Parse()
	if len(flag.Args()) != 2 {
		flag.Usage()
	}

	var modelType polish.ModelType
	if model == "shallow" {
		modelType = polish.ModelTypeShallow
	} else if model == "deep" {
		modelType = polish.ModelTypeDeep
	} else if model == "bilateral" {
		modelType = polish.ModelTypeBilateral
	} else if model == "shallow-aux" {
		modelType = polish.ModelTypeShallowAux
	} else if model == "deep-aux" {
		modelType = polish.ModelTypeDeepAux
	} else {
		flag.Usage()
	}

	if modelType.Aux() {
		if incidencePath == "" {
			fmt.Fprintln(os.Stderr, "auxiliary model requires -incidence flag")
		}
		if albedoPath == "" {
			fmt.Fprintln(os.Stderr, "auxiliary model requires -albedo flag")
		}
		if albedoPath == "" || incidencePath == "" {
			os.Exit(1)
		}
	}

	inPath := flag.Args()[0]
	outPath := flag.Args()[1]

	inImage := readPNG(inPath)

	var outImage image.Image
	if !modelType.Aux() {
		if patchSize != 0 {
			outImage = polish.PolishImagePatches(modelType, inImage, patchSize, patchBorder)
		} else {
			outImage = polish.PolishImage(modelType, inImage)
		}
	} else {
		albedo := readPNG(albedoPath)
		incidence := readPNG(incidencePath)
		inTensor := polish.CreateAuxTensorImages(inImage, albedo, incidence)
		if patchSize != 0 {
			outImage = polish.PolishAuxPatches(modelType, inTensor, patchSize, patchBorder)
		} else {
			outImage = polish.PolishAux(modelType, inTensor)
		}
	}

	w, err := os.Create(outPath)
	essentials.Must(err)
	essentials.Must(png.Encode(w, outImage))
}

func readPNG(path string) image.Image {
	r, err := os.Open(path)
	essentials.Must(err)
	defer r.Close()
	inImage, err := png.Decode(r)
	essentials.Must(err)
	return inImage
}
