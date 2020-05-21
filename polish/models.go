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

	// ModelTypeShallowAux is like ModelTypeShallow, but
	// the model expects albedo and ray incidence angles
	// as extra input channels.
	ModelTypeShallowAux

	// ModelTypeDeepAux is like ModelTypeDeep, but the
	// model expects albedo and ray incidence angles as
	// extra input channels.
	ModelTypeDeepAux
)

// LCD gets a factor which must divide the dimensions of
// images fed to this type of model.
func (m ModelType) LCD() int {
	switch m {
	case ModelTypeBilateral, ModelTypeShallow, ModelTypeShallowAux:
		return 1
	case ModelTypeDeep, ModelTypeDeepAux:
		return 4
	default:
		panic("unknown model type")
	}
}

// RF gets the radius of the receptive field of the model.
//
// The radius is the maximum number of pixels to the left,
// right, top, or bottom that the model can "see".
func (m ModelType) RF() int {
	switch m {
	case ModelTypeBilateral:
		return 7
	case ModelTypeShallow, ModelTypeShallowAux:
		return 4
	case ModelTypeDeep, ModelTypeDeepAux:
		return 42
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
		return createDeep(false)
	case ModelTypeShallowAux:
		return createShallowAux()
	case ModelTypeDeepAux:
		return createDeep(true)
	default:
		panic("unknown model type")
	}
}

// Aux checks if the model requires auxiliary features.
func (m ModelType) Aux() bool {
	return m == ModelTypeShallowAux || m == ModelTypeDeepAux
}
