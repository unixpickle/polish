package polish

import (
	"image"

	"github.com/unixpickle/essentials"
	"github.com/unixpickle/polish/polish/nn"
)

// PolishImage applies a denoising network to an image.
//
// A model should be used which does not expect any extra
// feature channels besides RGB colors.
func PolishImage(t ModelType, img image.Image) image.Image {
	patchSize := essentials.MaxInt(img.Bounds().Dx(), img.Bounds().Dy())
	return PolishImagePatches(t, img, patchSize, 0)
}

// PolishImagePatches is like PolishImage, but it applies
// the operation to patches of the image at a time to save
// memory.
//
// It is useful for very large images, where a neural
// network forward pass is too costly.
//
// The patchSize argument specifies how large the output
// patches should be (they are always square).
//
// The border argument specifies how many extra pixels are
// included on the side of each patch before it is fed
// into the network.
// A value of -1 will use a reasonable default border.
// Larger border values ensure more accuracy at the cost
// of redundant computation, while lower values may cause
// checkerboarding artifacts.
func PolishImagePatches(t ModelType, img image.Image, patchSize, border int) image.Image {
	inTensor := nn.NewTensorRGB(img)
	outTensor := operatePatches(inTensor, patchSize, border, func(in *nn.Tensor) *nn.Tensor {
		pad, unpad := padAndUnpad(t, in)
		outTensor := pad.Apply(in)
		outTensor = t.Layer().Apply(outTensor)
		outTensor = unpad.Apply(outTensor)
		return outTensor
	})
	return outTensor.RGB()
}

func padAndUnpad(t ModelType, in *nn.Tensor) (pad, unpad nn.Layer) {
	lcd := t.LCD()
	rightPad := (lcd - in.Width%lcd) % lcd
	bottomPad := (lcd - in.Height%lcd) % lcd
	return nn.NewPad(0, rightPad, bottomPad, 0), nn.NewUnpad(0, rightPad, bottomPad, 0)
}

func operatePatches(t *nn.Tensor, patchSize, border int, f func(*nn.Tensor) *nn.Tensor) *nn.Tensor {
	if patchSize >= t.Width && patchSize >= t.Height {
		// Special case when the patch fills the image.
		// This is utilized by PolishImage().
		return f(t)
	}

	if border == -1 {
		border = patchSize / 2
	}
	var output *nn.Tensor
	for y := 0; y < t.Height; y += patchSize {
		patchHeight := essentials.MinInt(patchSize, t.Height-y)
		extraTop := essentials.MinInt(y, border)
		extraBottom := essentials.MinInt(t.Height-(y+patchHeight), border)
		for x := 0; x < t.Width; x += patchSize {
			patchWidth := essentials.MinInt(patchSize, t.Width-x)
			extraLeft := essentials.MinInt(x, border)
			extraRight := essentials.MinInt(t.Width-(x+patchWidth), border)

			patch := nn.NewTensor(patchHeight+extraTop+extraBottom,
				patchWidth+extraLeft+extraRight, t.Depth)
			for subY := 0; subY < patch.Height; subY++ {
				for subX := 0; subX < patch.Width; subX++ {
					destIdx := (subX + subY*patch.Width) * patch.Depth
					dest := patch.Data[destIdx : destIdx+patch.Depth]
					sourceIdx := ((subX + x - extraLeft) + (subY+y-extraTop)*t.Width) * t.Depth
					source := t.Data[sourceIdx : sourceIdx+t.Depth]
					copy(dest, source)
				}
			}

			patchOut := f(patch)
			patchOut = patchOut.Unpad(extraTop, extraRight, extraBottom, extraLeft)
			if output == nil {
				output = nn.NewTensor(t.Height, t.Width, patchOut.Depth)
			}
			for subY := 0; subY < patchHeight; subY++ {
				for subX := 0; subX < patchWidth; subX++ {
					destIdx := ((subX + x) + (subY+y)*t.Width) * output.Depth
					dest := output.Data[destIdx : destIdx+output.Depth]
					sourceIdx := (subX + subY*patchOut.Width) * patchOut.Depth
					source := patchOut.Data[sourceIdx : sourceIdx+patchOut.Depth]
					copy(dest, source)
				}
			}
		}
	}
	return output
}
