import { pipeline, RawImage } from '@huggingface/transformers';
async function main() {
  const url = 'https://images.pexels.com/photos/5965592/pexels-photo-5965592.jpeg?auto=compress&cs=tinysrgb&w=256';
  const image = await RawImage.fromURL(url);
  try {
    const segmenter = await pipeline('image-segmentation', 'onnx-community/BiRefNet-ONNX');
    console.log("image-segmentation pipeline works");
    const out = await segmenter(image);
    console.log("Output size:", out);
  } catch (e) {
    console.error("pipeline image-segmentation failed:", e.message);
  }
}
main();
