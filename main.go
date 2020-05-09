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
	flag.StringVar(&model, "model", "deep", "type of model to use ('shallow', 'deep', 'bilateral')")
	flag.IntVar(&patchSize, "patch", 0, "image patch size to process at once (0 to disable)")
	flag.IntVar(&patchBorder, "patch-border", -1, "border for image patches (-1 uses default)")

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
	} else {
		flag.Usage()
	}

	inPath := flag.Args()[0]
	outPath := flag.Args()[1]

	r, err := os.Open(inPath)
	essentials.Must(err)
	defer r.Close()
	inImage, err := png.Decode(r)
	essentials.Must(err)

	var outImage image.Image
	if patchSize != 0 {
		outImage = polish.PolishImagePatches(modelType, inImage, patchSize, patchBorder)
	} else {
		outImage = polish.PolishImage(modelType, inImage)
	}

	w, err := os.Create(outPath)
	essentials.Must(err)
	essentials.Must(png.Encode(w, outImage))
}
