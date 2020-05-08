package polish

import "github.com/unixpickle/polish/polish/nn"

type ModelType int

const (
	// ModelTypeBilateral uses a tuned bilateral filter to
	// denoise the image without a neural network.
	ModelTypeBilateral ModelType = iota

	// ModelTypeShallow is a small convolutional network
	// that operates only on the pixels of an RGB image.
	//
	// It is fast, but the smoothing is very primitive.
	ModelTypeShallow

	// ModelTypeDeep is a large convolutional network that
	// operates only on the pixels of an RGB image.
	//
	// It is slow and performs a great deal of smoothing.
	ModelTypeDeep
)

// LCD gets a factor which must divide the dimensions of
// images fed to this type of model.
func (m ModelType) LCD() int {
	switch m {
	case ModelTypeShallow, ModelTypeBilateral:
		return 1
	case ModelTypeDeep:
		return 4
	default:
		panic("unknown model type")
	}
}

// Layer creates a pre-trained layer implementing this
// model.
func (m ModelType) Layer() nn.Layer {
	switch m {
	case ModelTypeBilateral:
		return &nn.Bilateral{
			// Parameters optimized on the training set
			// with SGD.
			SigmaBlur: 1.7016,
			SigmaDiff: 0.4821,

			KernelSize: 15,
		}
	case ModelTypeShallow:
		return createShallow()
	case ModelTypeDeep:
		return createDeep()
	default:
		panic("unknown model type")
	}
}
