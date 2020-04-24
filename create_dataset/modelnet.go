package main

import (
	"io/ioutil"
	"path/filepath"

	"github.com/pkg/errors"
)

// ScanModelNet finds all of the .off model file paths.
func ScanModelNet(dir string) (paths []string, err error) {
	paths, err = scanModelNet(dir)
	if err != nil {
		err = errors.Wrap(err, "scan ModelNet")
	}
	return
}

func scanModelNet(dir string) (paths []string, err error) {
	dirs, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, err
	}
	for _, d := range dirs {
		if !d.IsDir() {
			if filepath.Ext(d.Name()) == ".off" {
				paths = append(paths, filepath.Join(dir, d.Name()))
			}
			continue
		}
		dPath := filepath.Join(dir, d.Name())
		subPaths, err := scanModelNet(dPath)
		if err != nil {
			return nil, err
		}
		paths = append(paths, subPaths...)
	}
	return
}
