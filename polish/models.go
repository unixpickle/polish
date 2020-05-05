package polish

import "github.com/unixpickle/polish/polish/nn"

type ModelType int

const (
	// ModelTypeShallow is a small convolutional network
	// that operates only on the pixels of an RGB image.
	//
	// It is fast, but the smoothing is very primitive.
	ModelTypeShallow ModelType = iota

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
	case ModelTypeShallow:
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
	case ModelTypeShallow:
		return createShallow()
	case ModelTypeDeep:
		return createDeep()
	default:
		panic("unknown model type")
	}
}
