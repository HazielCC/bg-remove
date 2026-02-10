import path from 'path';
import { pipeline, RawImage } from '@huggingface/transformers';

async function run() {
  const modelPath = 'http://localhost:3002/models/modnet';
  console.log('Loading pipeline from', modelPath);
  const segmenter = await pipeline('background-removal', modelPath, { dtype: 'uint8' });
  console.log('Pipeline loaded');
  const imagePath = path.resolve('./public/example/person.jpg');
  const image = await RawImage.fromFile(imagePath);
  console.log('Image loaded: ', image.width, 'x', image.height);
  const output = await segmenter(image);
  console.log('Inference done', output.length);
  await output[0].save('mask_http.png');
  console.log('Mask saved to mask_http.png');
}

run().catch(err => { console.error(err); process.exit(1); });
