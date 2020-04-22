package main

import (
	"flag"
	"fmt"
	"strings"

	"github.com/unixpickle/essentials"
)

type Args struct {
	ModelNetPath string
	ImagesPath   string

	OutputDir string
}

func (a *Args) Parse() {
	flag.StringVar(&a.ModelNetPath, "modelnet", "", "path to ModelNet-40 dataset")
	flag.StringVar(&a.ImagesPath, "images", "", "path to (recursive) texture library")
	flag.StringVar(&a.OutputDir, "outdir", "../data", "dataset output directory")
	flag.Parse()

	var missingArgs []string
	if a.ModelNetPath == "" {
		missingArgs = append(missingArgs, "-modelnet")
	}
	if a.ImagesPath == "" {
		missingArgs = append(missingArgs, "-images")
	}
	if len(missingArgs) > 0 {
		essentials.Die(fmt.Sprintf("missing required arguments: %s",
			strings.Join(missingArgs, ", ")))
	}
}
