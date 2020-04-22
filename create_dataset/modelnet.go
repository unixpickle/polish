package main

import (
	"io/ioutil"
	"path/filepath"
	"strings"

	"github.com/pkg/errors"
)

// ScanModelNet finds all of the train and test .off model
// file paths.
func ScanModelNet(dir string) (train, test []string, err error) {
	train, test, err = scanModelNet(dir)
	if err != nil {
		err = errors.Wrap(err, "scan ModelNet")
	}
	return
}

func scanModelNet(dir string) (train, test []string, err error) {
	dirs, err := ioutil.ReadDir(dir)
	if err != nil {
		return nil, nil, err
	}
	for _, d := range dirs {
		if !d.IsDir() || strings.HasPrefix(d.Name(), ".") {
			continue
		}
		dPath := filepath.Join(dir, d.Name())
		subTrains, err := scanModels(filepath.Join(dPath, "train"))
		if err != nil {
			return nil, nil, err
		}
		subTests, err := scanModels(filepath.Join(dPath, "train"))
		if err != nil {
			return nil, nil, err
		}
		train = append(train, subTrains...)
		test = append(test, subTests...)
	}
	return
}

func scanModels(splitDir string) ([]string, error) {
	listing, err := ioutil.ReadDir(splitDir)
	if err != nil {
		return nil, err
	}
	var names []string
	for _, d := range listing {
		if !d.IsDir() && filepath.Ext(d.Name()) == ".off" {
			names = append(names, filepath.Join(splitDir, d.Name()))
		}
	}
	return names, nil
}
