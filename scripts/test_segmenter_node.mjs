import fs from 'fs';
import path from 'path';
import { pipeline, RawImage } from '@huggingface/transformers';

async function run() {
  const modelPath = path.resolve('./public/models/modnet');
  console.log('Loading pipeline from', modelPath);
  const segmenter = await pipeline('background-removal', modelPath, { dtype: 'uint8' });
  console.log('Pipeline loaded');
  const imageUrl = 'http://localhost:3002/example/person.jpg';
  const image = await RawImage.fromURL(imageUrl);
  console.log('Image loaded: ', image.width, 'x', image.height);
  const output = await segmenter(image);
  console.log('Inference done', output.length);
  // Save mask
  await output[0].save('mask_node.png');
  console.log('Mask saved to mask_node.png');
}

run().catch((err) => { console.error('Error', err); process.exit(1); });
