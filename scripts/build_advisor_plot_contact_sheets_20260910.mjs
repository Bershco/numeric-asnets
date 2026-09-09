import sharp from "file:///C:/Users/roeeh/Desktop/School/Meitar/Thesis/Code/numeric-asnets/node_modules/sharp/dist/index.mjs";
import path from "node:path";

const root = path.resolve("experiment_tracking/advisor_meeting_20260910");

async function build(folder, files, output) {
  const width = 1800;
  const cellWidth = 880;
  const cellHeight = 545;
  const rows = Math.ceil(files.length / 2);
  const height = 70 + rows * cellHeight;
  const layers = [];
  for (let i = 0; i < files.length; i++) {
    const col = i % 2;
    const row = Math.floor(i / 2);
    const left = 20 + col * cellWidth;
    const top = 55 + row * cellHeight;
    const buffer = await sharp(path.join(root, folder, files[i][0]))
      .resize(850, 485, { fit: "contain", background: "white" })
      .png()
      .toBuffer();
    layers.push({ input: buffer, left, top: top + 32 });
    const label = `<svg width="850" height="32"><text x="8" y="23" font-family="Segoe UI,Arial" font-size="20" font-weight="700" fill="#17212b">${files[i][1]}</text></svg>`;
    layers.push({ input: Buffer.from(label), left, top });
  }
  const title = `<svg width="1800" height="55"><rect width="100%" height="100%" fill="white"/><text x="28" y="37" font-family="Segoe UI,Arial" font-size="29" font-weight="700" fill="#17212b">${folder === "before_review" ? "Before independent review" : "After independent review"}</text></svg>`;
  layers.unshift({ input: Buffer.from(title), left: 0, top: 0 });
  await sharp({ create: { width, height, channels: 3, background: "white" } })
    .composite(layers)
    .png()
    .toFile(path.join(root, output));
}

await build("before_review", [
  ["01_domain_scorecard.png", "Domain scorecard"],
  ["02_two_stage_learning_dynamics.png", "Learning dynamics"],
  ["03_mcts_cutoff_forest.png", "MCTS cutoff effects"],
  ["04_preserve3_validation_seed_robustness.png", "PRESERVE-3 robustness"],
  ["05_mprime_validation_problem.png", "MPrime validation"],
], "before_review_overview.png");

await build("after_review", [
  ["01_domain_scorecard.png", "Domain scorecard — corrected labels"],
  ["02_two_stage_learning_dynamics.png", "Full Stage-1 + Stage-2 curves"],
  ["03a_stage1_mcts_cutoff_forest.png", "Stage-1 MCTS effects"],
  ["03b_stage2_mcts_cutoff_forest.png", "Stage-2 MCTS effects"],
  ["04_preserve3_validation_seed_robustness.png", "PRESERVE-3 robustness"],
  ["05_mprime_validation_problem.png", "MPrime selection regret"],
], "after_review_overview.png");
