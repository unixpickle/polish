package polish

import (
	"image"

	"github.com/unixpickle/polish/polish/nn"
)

// PolishImage applies a denoising network to an image.
//
// A model should be used which does not expect any extra
// feature channels besides RGB colors.
func PolishImage(t ModelType, img image.Image) image.Image {
	inTensor := nn.NewTensorRGB(img)
	inTensor = padInput(t, inTensor)
	outTensor := t.Layer().Apply(inTensor)
	outTensor = unpadInput(t, outTensor)
	return outTensor.RGB()
}

func padInput(t ModelType, in *nn.Tensor) *nn.Tensor {
	lcd := t.LCD()
	rightPad := in.Width % lcd
	bottomPad := in.Height % lcd
	return in.Pad(0, rightPad, bottomPad, 0)
}

func unpadInput(t ModelType, in *nn.Tensor) *nn.Tensor {
	lcd := t.LCD()
	rightPad := in.Width % lcd
	bottomPad := in.Height % lcd
	return in.Unpad(0, rightPad, bottomPad, 0)
}
