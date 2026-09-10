import sharp from "file:///C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/node_modules/sharp/dist/index.mjs";
import path from "node:path";

const folder = path.resolve("experiment_tracking/advisor_followup_20260910");
const names = [
  "rq1_stage2_training_vh_off",
  "rq2_mcts_vh_off",
  "rq3_value_head_training",
  "rq4_value_head_mcts",
];

for (const name of names) {
  await sharp(path.join(folder, `${name}.svg`)).png().toFile(path.join(folder, `${name}.png`));
}
