// Command polish denoises images that were produced by a
// Monte Carlo rendering technique (e.g. path tracing).
package main

import (
	"flag"
	"fmt"
	"image/png"
	"os"

	"github.com/unixpickle/polish/polish"

	"github.com/unixpickle/essentials"
)

func main() {
	var model string
	flag.StringVar(&model, "model", "deep", "type of model to use ('shallow' or 'deep')")
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

	outImage := polish.PolishImage(modelType, inImage)

	w, err := os.Create(outPath)
	essentials.Must(err)
	essentials.Must(png.Encode(w, outImage))
}
