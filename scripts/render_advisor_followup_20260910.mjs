import sharp from "file:///C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/node_modules/sharp/dist/index.mjs";
import path from "node:path";
import fs from "node:fs";

const folder = path.resolve("experiment_tracking/advisor_followup_20260910");
const names = [
  "rq1_stage2_training_vh_off",
  "rq2_mcts_vh_off",
  "rq2_raw_means_by_stage",
  "rq3_value_head_training",
  "rq3_raw_means_and_interaction",
  "rq4_value_head_mcts",
  "rq4_direct",
  "rq4_cross_cell",
  "rq4_interaction",
  "rq4_raw_means_6h_by_stage",
  "rq2_rq4_pw70_final",
];

for (const name of names) {
  const target = path.join(folder, `${name}.png`);
  const temporary = path.join(folder, `${name}.next.png`);
  await sharp(path.join(folder, `${name}.svg`)).png().toFile(temporary);
  if (fs.existsSync(target)) fs.unlinkSync(target);
  fs.renameSync(temporary, target);
}
