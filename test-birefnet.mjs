import { pipeline, RawImage } from '@huggingface/transformers';
async function main() {
  const image = await RawImage.read('public/file.svg');
  console.log("Image loaded", image.width);
  try {
    const segmenter = await pipeline('image-segmentation', 'onnx-community/BiRefNet-ONNX');
    console.log("image-segmentation pipeline works");
    const out = await segmenter(image);
    console.log("Output:", out);
  } catch (e) {
    console.error("pipeline image-segmentation failed:", e);
  }
}
main();
