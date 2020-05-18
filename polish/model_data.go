package polish

import (
	"archive/zip"
	"bytes"
	"encoding/binary"
	"fmt"
	"io/ioutil"
	"math"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/polish/polish/nn"
)

func createShallow() nn.Layer {
	params := readParameterZip(shallowModelZipData)
	return nn.NN{
		loadConv(params, "conv1", 5, 1, 3, 32),
		nn.ReLU{},
		loadConv(params, "conv2", 5, 1, 32, 3),
	}
}

func createShallowAux() nn.Layer {
	params := readParameterZip(shallowAuxModelZipData)
	return nn.NN{
		loadConv(params, "conv1", 5, 1, 7, 32),
		nn.ReLU{},
		loadConv(params, "conv2", 5, 1, 32, 3),
	}
}

func createDeep() nn.Layer {
	params := readParameterZip(deepModelZipData)

	result := nn.NN{
		loadConv(params, "conv1", 5, 2, 3, 64),
		nn.ReLU{},
		loadDepthSepConv(params, "conv2", 5, 2, 64, 128),
	}

	for i := 0; i < 4; i++ {
		layer := fmt.Sprintf("residuals.%d", i)
		result = append(result, nn.Residual{
			loadBatchNorm(params, layer+".0"),
			nn.ReLU{},
			loadDepthSepConv(params, layer+".2", 3, 1, 128, 256),
			nn.ReLU{},
			loadDepthSepConv(params, layer+".4", 3, 1, 256, 128),
		})
	}

	result = append(result,
		loadDeconv(params, "deconv1", 4, 2, 128, 64),
		nn.ReLU{},
		loadDeconv(params, "deconv2", 4, 2, 64, 32),
		nn.ReLU{},
		loadConv(params, "conv3", 3, 1, 32, 3),
	)

	return result
}

func loadConv(p map[string][]float32, key string, kernel, stride, inDepth, outDepth int) nn.Layer {
	return nn.NN{
		nn.NewPad(kernel/2, kernel/2, kernel/2, kernel/2),
		&nn.Conv{
			InDepth:    inDepth,
			OutDepth:   outDepth,
			KernelSize: kernel,
			Stride:     stride,
			Weights:    p[key+".weight"],
		},
		&nn.Bias{Data: p[key+".bias"]},
	}
}

func loadDeconv(p map[string][]float32, key string, kernel, stride,
	inDepth, outDepth int) nn.Layer {
	s := (kernel - 1) / 2
	return nn.NN{
		&nn.Deconv{
			InDepth:    inDepth,
			OutDepth:   outDepth,
			KernelSize: kernel,
			Stride:     stride,
			Weights:    p[key+".weight"],
		},
		nn.NewUnpad(s, s, s, s),
		&nn.Bias{Data: p[key+".bias"]},
	}
}

func loadDepthSepConv(p map[string][]float32, key string,
	kernel, stride, inDepth, outDepth int) nn.Layer {
	return nn.NN{
		nn.NewPad(kernel/2, kernel/2, kernel/2, kernel/2),
		&nn.SpatialConv{
			Depth:      inDepth,
			KernelSize: kernel,
			Stride:     stride,
			Weights:    p[key+".spatial.weight"],
		},
		&nn.Bias{Data: p[key+".spatial.bias"]},
		nn.ReLU{},
		&nn.Conv{
			InDepth:    inDepth,
			OutDepth:   outDepth,
			KernelSize: 1,
			Stride:     1,
			Weights:    p[key+".depthwise.weight"],
		},
		&nn.Bias{Data: p[key+".depthwise.bias"]},
	}
}

func loadBatchNorm(p map[string][]float32, key string) nn.Layer {
	negMean := append([]float32{}, p[key+".running_mean"]...)
	for i, x := range negMean {
		negMean[i] = -x
	}
	variance := p[key+".running_var"]
	weight := p[key+".weight"]
	bias := p[key+".bias"]

	scale := make([]float32, len(variance))
	offset := make([]float32, len(variance))
	for i, x := range variance {
		invStd := float32(1.0 / math.Sqrt(float64(x+1e-5)))
		scale[i] = weight[i] * invStd
		offset[i] = bias[i]
	}
	return nn.NN{
		&nn.Bias{Data: negMean},
		&nn.Mul{Data: scale},
		&nn.Bias{Data: offset},
	}
}

func readParameterZip(rawZip string) map[string][]float32 {
	zipData := []byte(rawZip)
	byteReader := bytes.NewReader(zipData)
	zipReader, err := zip.NewReader(byteReader, int64(len(zipData)))
	essentials.Must(err)

	params := map[string][]float32{}
	for _, file := range zipReader.File {
		r, err := file.Open()
		essentials.Must(err)
		data, err := ioutil.ReadAll(r)
		essentials.Must(err)
		values := make([]float32, len(data)/4)
		binary.Read(bytes.NewReader(data), binary.LittleEndian, values)
		params[file.Name] = values
	}

	return params
}
