package main

import (
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

// ScanImages finds all of the image paths in a directory,
// recursively.
func ScanImages(imageDir string) ([]string, error) {
	listing, err := ioutil.ReadDir(imageDir)
	if err != nil {
		return nil, errors.Wrap(err, "scan images")
	}

	var results []string

	for _, d := range listing {
		dPath := filepath.Join(imageDir, d.Name())
		if d.IsDir() {
			subResults, err := ScanImages(dPath)
			if err != nil {
				return nil, err
			}
			results = append(results, subResults...)
		} else {
			ext := strings.ToLower(filepath.Ext(d.Name()))
			if ext == ".jpg" || ext == ".jpeg" || ext == ".png" || ext == ".gif" {
				results = append(results, dPath)
			}
		}
	}

	return results, nil
}
