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
	pad, unpad := padAndUnpad(t, inTensor)
	outTensor := pad.Apply(inTensor)
	outTensor = t.Layer().Apply(outTensor)
	outTensor = unpad.Apply(outTensor)
	return outTensor.RGB()
}

func padAndUnpad(t ModelType, in *nn.Tensor) (pad, unpad nn.Layer) {
	lcd := t.LCD()
	rightPad := (lcd - in.Width%lcd) % lcd
	bottomPad := (lcd - in.Height%lcd) % lcd
	return nn.NewPad(0, rightPad, bottomPad, 0), nn.NewUnpad(0, rightPad, bottomPad, 0)
}
