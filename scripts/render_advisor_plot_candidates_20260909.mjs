import sharp from "file:///C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/node_modules/sharp/dist/index.mjs";
import { readdir } from "node:fs/promises";
import path from "node:path";

const outputDir = path.resolve("experiment_tracking/learning_curves/candidates_20260909");
for (const name of await readdir(outputDir)) {
  if (!name.endsWith(".svg")) continue;
  await sharp(path.join(outputDir, name), { density: 144 })
    .png()
    .toFile(path.join(outputDir, name.replace(/\.svg$/, ".png")));
}
