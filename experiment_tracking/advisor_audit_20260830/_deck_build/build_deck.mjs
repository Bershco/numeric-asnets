import fs from "node:fs/promises";
import path from "node:path";
import { Presentation, PresentationFile } from "@oai/artifact-tool";

const ROOT = path.resolve(process.cwd(), "..", "..", "..");
const OUT = path.resolve(process.cwd());
const TRACK = path.join(ROOT, "experiment_tracking");
const FIG = path.join(TRACK, "advisor_figures");
const MPRIME = path.join(TRACK, "mprime_validation_ipc_scale_v1");
const IMG = {
  mainstream: await fs.readFile(path.join(FIG,"mainstream_stage1_stage2_forest.png")),
  mprime: await fs.readFile(path.join(MPRIME,"corrected_learning_curves.png")),
  pwCoverage: await fs.readFile(path.join(FIG,"pw_kmin3_30min_coverage.png")),
  pwRuntime: await fs.readFile(path.join(FIG,"pw_kmin3_success_runtime_ecdf.png")),
};

const C = {
  bg: "#F7F7F5", ink: "#171717", muted: "#64645F", faint: "#E8E7E2",
  dark: "#1D252C", green: "#2F6B4F", mint: "#CFE4D5", orange: "#D9772D",
  paleOrange: "#F4DEC9", blue: "#426B8A", paleBlue: "#DDE8F0", red: "#A3473B",
  paleRed: "#F0D7D2", white: "#FFFFFF", gold: "#B58A2A", slate: "#4A5560",
};
const FONT = "Helvetica Neue";
const presentation = Presentation.create({ slideSize: { width: 1280, height: 720 } });

function rect(slide, x, y, w, h, fill, radius = "rounded-xl", line = "none") {
  return slide.shapes.add({ geometry: radius === 0 ? "rect" : "roundRect", position: { left:x, top:y, width:w, height:h }, fill, line: { style:"solid", fill: line, width: line === "none" ? 0 : 1 }, borderRadius: radius });
}
function txt(slide, text, x, y, w, h, size=18, color=C.ink, bold=false, align="left") {
  const s = slide.shapes.add({ geometry:"textbox", position:{left:x,top:y,width:w,height:h}, fill:"none", line:{style:"solid",fill:"none",width:0} });
  s.text = text;
  s.text.style = { fontFamily:FONT, fontSize:size, color, bold, alignment:align, verticalAlignment:"middle" };
  return s;
}
function line(slide, x, y, w, color=C.faint, width=2) {
  return slide.shapes.add({ geometry:"rect", position:{left:x,top:y,width:w,height:width}, fill:color, line:{style:"solid",fill:color,width:0} });
}
function base(title, section, source, page) {
  const slide = presentation.slides.add();
  slide.background.fill = C.bg;
  txt(slide, section.toUpperCase(), 62, 34, 600, 22, 12, C.green, true);
  txt(slide, title, 62, 62, 1130, 54, 36, C.ink, true);
  line(slide, 62, 123, 1156, C.faint, 2);
  txt(slide, `NUMERIC ASNETS · ADVISOR STATUS · 30 AUG 2026`, 62, 684, 620, 18, 10, C.muted, true);
  txt(slide, String(page).padStart(2,"0"), 1170, 681, 48, 20, 11, C.muted, true, "right");
  slide.speakerNotes.textFrame.setText(`[Sources]\n${source}\n[/Sources]`);
  return slide;
}
function callout(slide, number, label, x, y, w, color=C.green, tint=C.mint) {
  rect(slide,x,y,w,112,tint,"rounded-2xl",tint);
  txt(slide,number,x+18,y+15,w-36,46,34,color,true);
  txt(slide,label,x+18,y+65,w-36,32,14,C.ink,true);
}
function bulletList(slide, items, x, y, w, size=18, gap=46, color=C.ink) {
  items.forEach((item,i)=>{
    rect(slide,x,y+i*gap+8,8,8,C.orange,"rounded-full",C.orange);
    txt(slide,item,x+20,y+i*gap,w-20,gap-2,size,color,false);
  });
}
function simpleTable(slide, headers, rows, x, y, widths, rowH=40, headerFill=C.dark, font=14) {
  let cx=x;
  headers.forEach((h,i)=>{rect(slide,cx,y,widths[i],rowH,headerFill,0,headerFill);txt(slide,h,cx+8,y+2,widths[i]-16,rowH-4,font,C.white,true);cx+=widths[i];});
  rows.forEach((row,r)=>{cx=x;const fill=r%2===0?C.white:"#F0F0EC";row.forEach((v,i)=>{rect(slide,cx,y+rowH*(r+1),widths[i],rowH,fill,0,C.faint);txt(slide,String(v),cx+8,y+rowH*(r+1)+2,widths[i]-16,rowH-4,font,C.ink,i===0);cx+=widths[i];});});
}
function image(slide, blob, x, y, w, h, alt) { slide.images.add({ blob, contentType:"image/png", alt, fit:"contain", position:{left:x,top:y,width:w,height:h} }); }
function tag(slide, text, x, y, w, fill, color=C.ink) { rect(slide,x,y,w,28,fill,"rounded-full",fill);txt(slide,text,x+8,y+1,w-16,25,12,color,true,"center"); }

let p=1;
{
  const s=presentation.slides.add(); s.background.fill=C.dark;
  tag(s,"ADVISOR MEETING · STATUS + DECISIONS",72,54,310,C.orange,C.white);
  txt(s,"Numeric ASNets",72,148,720,64,52,C.white,true);
  txt(s,"What the experiments now say —\nand what still has to close",72,222,820,112,40,"#DCE6E0",true);
  txt(s,"Nine domains · 10-seed mainstream replication · policy, MCTS, training and search-correctness extensions",72,382,720,64,19,"#BBC7C0");
  callout(s,"3","stable domains",72,520,220,C.green,C.mint);
  callout(s,"6","imperfect domains",310,520,220,C.orange,C.paleOrange);
  callout(s,"65","jobs running",548,520,220,C.blue,C.paleBlue);
  callout(s,"5.93 TiB","requested live RAM",786,520,270,C.red,C.paleRed);
  txt(s,"Roee Hersco · 30 August 2026",72,668,500,20,12,"#AEBAB3",true);
  s.speakerNotes.textFrame.setText(`[Sources]\n${path.join(TRACK,"advisor_audit_20260830","cluster_workload.csv")}\n${path.join(TRACK,"experiments.csv")}\n[/Sources]`); p++;
}
{
  const s=base("The thesis story in five sentences","Executive summary",path.join(TRACK,"advisor_audit_20260830","advisor_meeting_brief.md"),p++);
  const items=[
    ["1","Stage-2 training is not uniformly beneficial","Counters can improve strongly; Block Grouping and FO Counters often regress."],
    ["2","MCTS helps selectively","Strong Stage-1 gains in Drone/VH-on and FO Counters; not a universal policy upgrade."],
    ["3","Efficiency is now an explicit result","PW Kmin=3 keeps most coverage while sharply reducing long-tail runtime."],
    ["4","Search correctness matters","SAFE-1 fixes 2/4 failures; action-history aliasing is common enough to warrant SAFE-CONTEXT."],
    ["5","The domain split changed","MPrime is not stably solved: three stable domains, six imperfect domains."],
  ];
  items.forEach((v,i)=>{const y=150+i*96;txt(s,v[0],72,y,40,40,26,C.orange,true,"center");txt(s,v[1],126,y-3,420,30,20,C.ink,true);txt(s,v[2],126,y+30,1000,45,16,C.muted);if(i<4)line(s,126,y+82,1030,C.faint,1);});
}
{
  const s=base("Four RQs, three evidence layers","Research map",path.join(ROOT,"thesis","draft","research_questions.md"),p++);
  const rows=[
    ["RQ1","Does continued MCTS-guided training improve policy?","VH-off S2 policy − S1 policy","Complete"],
    ["RQ2","Does inference-time MCTS improve policy?","VH-off S2 MCTS − same policy","Primary incomplete"],
    ["RQ3","Does the value head improve refinement?","(on S2−S1) − (off S2−S1)","Complete"],
    ["RQ4","Does the value head improve inference-time MCTS?","VH-on S2 MCTS − same policy","Primary incomplete"],
  ];
  simpleTable(s,["RQ","Question","Registered estimand","State"],rows,70,160,[90,430,420,205],70,C.dark,16);
  tag(s,"PRIMARY",80,510,100,C.green,C.white);txt(s,"Stage-2 comparisons answer the registered thesis questions.",194,506,880,34,18,C.ink,true);
  tag(s,"SECONDARY",80,557,100,C.blue,C.white);txt(s,"Stage-1 policy→MCTS is already complete and informative, but not a substitute for RQ2/RQ4.",194,553,930,40,17,C.muted);
}
{
  const s=base("Checkpoint choice is itself an experimental factor","Two mainstream routes",path.join(TRACK,"experiments.csv"),p++);
  rect(s,70,160,530,390,C.white,"rounded-2xl",C.faint);tag(s,"MAIN-VAL · COMPLETE",94,184,176,C.green,C.white);txt(s,"Validation-led",94,232,460,40,28,C.ink,true);txt(s,"S1 validation-selected checkpoint",94,286,450,30,18,C.green,true);txt(s,"→ Stage-2 training",94,328,450,30,18,C.ink,true);txt(s,"→ S2 validation-selected checkpoint",94,370,450,30,18,C.green,true);txt(s,"Ten matched seeds × five domains × two VH modes",94,438,450,50,16,C.muted);txt(s,"Policy endpoints: complete",94,503,260,26,15,C.ink,true);
  rect(s,638,160,530,390,C.white,"rounded-2xl",C.faint);tag(s,"MAIN-TERM · COMPLETE",662,184,185,C.orange,C.white);txt(s,"Terminal-led",662,232,460,40,28,C.ink,true);txt(s,"S1 final checkpoint",662,286,450,30,18,C.orange,true);txt(s,"→ Stage-2 training",662,328,450,30,18,C.ink,true);txt(s,"→ S2 validation-selected checkpoint",662,370,450,30,18,C.orange,true);txt(s,"Terminal sensitivity, not strict paper replication",662,438,450,50,16,C.muted);txt(s,"Strict stopping fidelity: STOP-ORIG held",662,503,400,26,15,C.ink,true);
  rect(s,70,584,1098,55,C.paleRed,"rounded-xl",C.paleRed);txt(s,"Interpretation rule: checkpoint selection can change results dramatically; never merge these routes into one aggregate.",92,596,1040,30,17,C.red,true);
}
{
  const s=base("The domain landscape is now 3 stable + 6 imperfect","Domain framing",path.join(TRACK,"advisor_audit_20260830","advisor_meeting_brief.md"),p++);
  txt(s,"STABLE / PRESERVATION",78,158,420,28,14,C.green,true);txt(s,"IMPERFECT / IMPROVEMENT",550,158,520,28,14,C.orange,true);
  ["Delivery  · S1 selected 19.8 / 19.2","TPP · S1 selected 20.0 / 20.0","Zenotravel · S1 selected 20.0 / 20.0"].forEach((t,i)=>{rect(s,72,202+i*88,450,64,C.mint,"rounded-xl",C.mint);txt(s,t,92,213+i*88,410,40,18,C.green,true);});
  ["Block Grouping","Drone","FO Counters","Rover","Counters","MPrime · corrected selected 15.0 / 14.6"].forEach((t,i)=>{const col=i%2,row=Math.floor(i/2);rect(s,550+col*310,202+row*88,286,64,C.paleOrange,"rounded-xl",C.paleOrange);txt(s,t,568+col*310,213+row*88,250,40,17,C.ink,true);});
  rect(s,72,502,1074,112,C.white,"rounded-2xl",C.faint);txt(s,"Why MPrime moved",94,520,240,26,19,C.red,true);txt(s,"The corrected validation set removes epoch-1 bias, yet validation/test rank agreement remains weak and selected test coverage is far below the historic 19/20 claim.",94,554,1010,48,17,C.muted);
}
{
  const s=base("RQ1: Stage-2 policy gains are domain-dependent","Mainstream policy",path.join(TRACK,"advisor_audit_20260830","rq_results.csv"),p++);
  image(s,IMG.mainstream,68,145,720,500,"Forest plot of Stage-1 to Stage-2 policy changes");
  rect(s,825,154,360,164,C.paleOrange,"rounded-2xl",C.paleOrange);txt(s,"No domain survives Holm correction",850,176,310,54,24,C.orange,true);txt(s,"RQ1 uses VH-off and corrects across five domains within the RQ.",850,240,310,55,16,C.ink);
  rect(s,825,344,360,230,C.white,"rounded-2xl",C.faint);txt(s,"Signals worth discussing",850,364,310,28,19,C.ink,true);bulletList(s,["Counters terminal-led: +16.4, raw p=.027","FO Counters: consistent negative point estimates","Drone: small positive, high uncertainty"],850,410,300,16,52,C.ink);
}
{
  const s=base("RQ3: value-head effect is a difference-in-differences","Value head",path.join(TRACK,"advisor_audit_20260830","rq_results.csv"),p++);
  const rows=[
    ["Block Grouping","−2.8 [−4.61, −.99] · pH=.088","−3.4 [−5.13, −1.67] · pH=.0195"],
    ["Drone","−.9 [−2.88, 1.08]","−1.1 [−3.57, 1.37]"],
    ["FO Counters","+.7 [−.26, 1.66]","+1.3 [.08, 2.52] · pH=.188"],
    ["Rover","+.1 [−.31, .51]","−.1 [−.51, .31]"],
    ["Counters","−1.2 [−25.87, 23.47]","−16.8 [−29.64, −3.96] · pH=.078"],
  ];
  simpleTable(s,["Domain","MAIN-VAL: on-change minus off-change","MAIN-TERM: on-change minus off-change"],rows,72,160,[240,450,450],66,C.dark,15);
  rect(s,72,565,1138,70,C.paleRed,"rounded-xl",C.paleRed);txt(s,"Only terminal-led Block Grouping is significant after Holm: the value head worsens refinement relative to VH-off.",94,579,1080,42,18,C.red,true);
}
{
  const s=base("Stage-1 MCTS: strong gains, but only in selected cells","Secondary RQ2/RQ4",path.join(TRACK,"advisor_audit_20260830","rq_results.csv"),p++);
  const domains=["BG","Drone","FO","Rover","Counters"];
  const off=[-0.9,1.0,3.6,1.0,-10.6], on=[0.3,5.3,2.0,0.6,-1.9];
  const x0=455, zero=x0+210, scale=16;
  txt(s,"VH-off",72,162,150,26,17,C.green,true);txt(s,"VH-on",72,382,150,26,17,C.blue,true);
  line(s,zero,158,2,C.slate,450);
  [off,on].forEach((vals,g)=>{vals.forEach((v,i)=>{const y=198+g*220+i*35;txt(s,domains[i],72,y-8,90,25,14,C.ink,true);const x=v>=0?zero:zero+v*scale;const w=Math.max(3,Math.abs(v)*scale);rect(s,x,y,w,16,v>=0?(g?C.blue:C.green):C.red,0,v>=0?(g?C.blue:C.green):C.red);txt(s,`${v>0?"+":""}${v.toFixed(1)}`,v>=0?zero+w+8:x-68,y-7,64,26,13,C.ink,true,v>=0?"left":"right");});});
  rect(s,840,164,340,164,C.mint,"rounded-2xl",C.mint);txt(s,"Holm-significant",865,183,280,30,19,C.green,true);txt(s,"VH-off: FO Counters +3.6\nVH-on: Drone +5.3; FO +2.0",865,225,280,74,18,C.ink,true);
  rect(s,840,354,340,220,C.white,"rounded-2xl",C.faint);txt(s,"Important caveat",865,374,280,28,19,C.orange,true);txt(s,"These are validation-selected Stage-1 checkpoints. The registered primary RQ2/RQ4 estimand is Stage-2 policy versus Stage-2 MCTS.",865,416,280,122,17,C.muted);
}
{
  const s=base("Primary RQ2/RQ4 evidence is the biggest open mainstream gap","Stage-2 MCTS",path.join(TRACK,"main_val_stage2_drone_mcts_gap.csv"),p++);
  callout(s,"20/20","Drone Stage-2 policy endpoints available",76,160,320,C.green,C.mint);
  callout(s,"4/20","matching MCTS endpoints already recovered",414,160,320,C.blue,C.paleBlue);
  callout(s,"16","gap jobs submitted and memory-pending",752,160,320,C.orange,C.paleOrange);
  txt(s,"What closes this hole",76,322,330,34,22,C.ink,true);
  bulletList(s,["Finish the 16 submitted Drone selected-checkpoint MCTS jobs","Then release domain-matched Stage-2 MCTS by information value and resource risk","Keep Stage-2-final MCTS excluded unless the protocol changes"],76,372,1040,18,64);
  rect(s,76,584,1030,52,C.paleRed,"rounded-xl",C.paleRed);txt(s,"Until this closes, Stage-1 MCTS is secondary evidence—not the final RQ2/RQ4 answer.",98,595,980,30,18,C.red,true);
}
{
  const s=base("Stable-domain preservation is live, not yet a finished claim","Delivery · TPP · Zenotravel",path.join(TRACK,"four_domain_preservation"),p++);
  const rows=[
    ["Zenotravel","Complete","S1 selected 20.0/off, 20.0/on","S2 selected 20.0/off, 19.9/on","No material loss"],
    ["Delivery","13/16 held-out terminal; 3 live","S1 selected 19.8/off, 19.2/on","Stage-2 endpoints accumulating","~5–6 h for live training"],
    ["TPP","13/16 held-out terminal; 3 live","S1 selected 20.0 / 20.0","Stage-2 endpoints accumulating","~7.5 h, 23 h, ≤32 h"],
  ];
  simpleTable(s,["Domain","Confirmation state","Stage 1","Stage 2","Estimate"],rows,68,170,[190,260,250,250,190],72,C.dark,14);
  rect(s,68,496,1140,112,C.white,"rounded-2xl",C.faint);txt(s,"Required reporting discipline",90,516,300,26,19,C.orange,true);txt(s,"The winning coefficient uses two tuning seeds per VH. Present those separately from the eight held-out confirmation seeds; all ten are not equally held out.",90,552,1070,42,17,C.muted);
}
{
  const s=base("MPrime correction worked—but ranking remains weak","Validation redesign",path.join(MPRIME,"corrected_learning_curves.png"),p++);
  image(s,IMG.mprime,70,154,720,480,"Corrected MPrime validation and test learning curves");
  callout(s,"15.0 / 14.6","selected test mean · off / on",824,160,350,C.green,C.mint);
  callout(s,"13.5 / 13.2","final test mean · off / on",824,292,350,C.orange,C.paleOrange);
  rect(s,824,424,350,168,C.white,"rounded-2xl",C.faint);txt(s,"Checkpoint-ranking signal",846,444,306,28,19,C.ink,true);txt(s,"Pooled Spearman ≈ .25\nSelected regret to retrospective best: 3.1/off, 2.5/on",846,482,306,82,17,C.muted);
  txt(s,"Conclusion: MPrime joins the six imperfect domains; anchor tuning is live.",824,610,350,42,16,C.red,true);
}
{
  const s=base("Progressive widening exposes an efficiency–coverage frontier","Search efficiency",path.join(FIG,"pw_kmin3_30min_coverage.png"),p++);
  image(s,IMG.pwCoverage,68,150,550,470,"Thirty-minute coverage comparison for policy, fixed top-20, and PW Kmin3");
  image(s,IMG.pwRuntime,650,150,550,470,"Successful-instance runtime ECDF for fixed top-20 and PW Kmin3");
  tag(s,"COVERAGE",80,604,100,C.orange,C.white);txt(s,"Policy 7.0 · fixed 10.5 · PW Kmin=3 9.5",190,599,390,32,16,C.ink,true);
  tag(s,"TAIL",662,604,72,C.green,C.white);txt(s,"p90: 893 s → 318 s · max: 6,127 s → 594 s",744,599,430,32,16,C.ink,true);
}
{
  const s=base("Cross-domain PW is a screen, not yet confirmation","PW live extension",path.join(TRACK,"mcts_progressive_widening_cross_domain"),p++);
  const rows=[
    ["Block Grouping","4 running","11–13/20 classified; all successes so far","Likely 1–4 h"],
    ["Counters Stage 1","4 running","12–21 successes; 21–26 classified","Hours to a day"],
    ["Counters Stage 2","3 running + 1 pending","18–52 lower-bound successes","Hours to a day"],
    ["FO Counters / Rover","0 running / 8 pending","Awaiting memory","No honest start estimate"],
  ];
  simpleTable(s,["Domain","State","Current evidence","Estimate"],rows,72,165,[230,190,480,220],68,C.dark,15);
  rect(s,72,518,1120,96,C.paleOrange,"rounded-2xl",C.paleOrange);txt(s,"Decision rule",94,536,180,26,19,C.orange,true);txt(s,"Use two seeds to screen. Expand only a promising domain/arm to five matched seeds; do not turn every arm into a large grid.",250,530,900,54,18,C.ink,true);
}
{
  const s=base("SAFE-1 fixes choices; SAFE-CONTEXT fixes identity","Search correctness",path.join(TRACK,"mcts_safe_context","README.md"),p++);
  rect(s,72,160,350,420,C.white,"rounded-2xl",C.faint);tag(s,"SAFE-1 · COMPLETE",94,184,150,C.green,C.white);txt(s,"2 / 4",94,242,280,62,46,C.green,true);txt(s,"known terminal-choice failures repaired",94,306,280,58,18,C.ink,true);bulletList(s,["Hard-mask known non-goal terminal children","Restore safe duplicates before unsafe actions","Two failures still wandered into failure"],94,386,278,15,58);
  rect(s,448,160,740,420,C.white,"rounded-2xl",C.faint);tag(s,"SAFE-CONTEXT · LIVE",470,184,190,C.blue,C.white);txt(s,"47.1%",470,238,220,60,44,C.blue,true);txt(s,"physical revisits had a different action-history context",470,302,330,52,18,C.ink,true);callout(s,"5.44×","maximum diagnostic node multiplier",824,224,320,C.red,C.paleRed);bulletList(s,["Independent node/prior/value/visit cache per context digest","Physical-state cycle detection retained","Full 20-job Drone match must report coverage + RSS + nodes"],470,386,650,15,58);
}
{
  const s=base("Binding horizon is nearly a non-effect in Drone","Finite-horizon search",path.join(TRACK,"mcts_horizon_binding","results.csv"),p++);
  callout(s,"8 pairs","aware/unaware terminal",76,164,250,C.green,C.mint);callout(s,"7 ties","exact coverage equality",344,164,250,C.blue,C.paleBlue);callout(s,"+1","one aware improvement",612,164,250,C.orange,C.paleOrange);callout(s,"1 cutoff","mechanism actually triggered",880,164,250,C.red,C.paleRed);
  txt(s,"Two aware arms remain live",76,324,350,32,22,C.ink,true);txt(s,"16–17/20 instances classified; expected within roughly 6–8 hours.",76,362,780,34,18,C.muted);
  rect(s,76,430,1054,142,C.white,"rounded-2xl",C.faint);txt(s,"If cutoffs remain rare",100,452,260,30,20,C.orange,true);txt(s,"Move the efficacy test to cutoff-rich Counters instances with a preregistered action limit. Do not interpret run-to-run coverage variation as a horizon effect when the cutoff never fires.",100,494,980,62,18,C.ink);
}
{
  const s=base("Counters shows why width and failure mode must be explicit","Narrow search",path.join(TRACK,"mcts_counters_width_sensitivity"),p++);
  const rows=[
    ["Stage 1 · VH-off","32.5","21.9","≥25.7","7 partial lower-bound rows"],
    ["Stage 1 · VH-on","18.6","16.7","≥22.5","6 partial lower-bound rows"],
    ["Stage 2 · VH-off","44.83","not declared","43.67 · n=6","11/12 misses diverged to 10k"],
    ["Stage 2 · VH-on","29.4","not declared","33.8 · n=5","remaining jobs live"],
  ];
  simpleTable(s,["Cell","Policy","Normal 20/70","Narrow 5/20","Interpretation"],rows,68,166,[230,150,190,190,370],70,C.dark,15);
  rect(s,68,535,1130,78,C.paleRed,"rounded-xl",C.paleRed);txt(s,"Interrupted jobs are conservative fixed-budget lower bounds—not invalid plans. Every printed plan in the audited partial logs was VAL-valid.",90,552,1080,43,17,C.red,true);
}
{
  const s=base("The queue is full on memory, not on job count","Cluster workload",path.join(TRACK,"advisor_audit_20260830","cluster_workload.csv"),p++);
  callout(s,"65","running jobs",72,158,250,C.green,C.mint);callout(s,"390","requested CPUs",342,158,250,C.blue,C.paleBlue);callout(s,"5.93 TiB","running RAM request",612,158,250,C.orange,C.paleOrange);callout(s,"43 + 20","ordinary + held pending",882,158,300,C.red,C.paleRed);
  const rows=[
    ["SAFE-CONTEXT","19","114","2,280 GiB"],
    ["PW cross-domain","11 + 9 pending","66 + 54","1,320 + 1,080 GiB"],
    ["MPrime anchor","18","108","864 GiB"],
    ["Counters S2 narrow","6","36","720 GiB"],
    ["Delivery + TPP","6","36","288 GiB"],
    ["Horizon + Rover","5 + 18 pending","30 + 108","600 + 2,160 GiB"],
  ];
  simpleTable(s,["Experiment","Jobs","CPU","Requested RAM"],rows,150,324,[310,230,180,330],48,C.dark,14);
}
{
  const s=base("What should finish next","Live ETA board",path.join(TRACK,"live_jobs.csv"),p++);
  const rows=[
    ["MPrime anchor","18 live","≈0.3–9 h","Freeze coefficient; refresh policy curves"],
    ["Delivery Stage 2","3 live","≈5–6 h","Evaluate every-five + selected/final"],
    ["Binding Horizon","2 live","≈6–8 h","Finalize matched aware/unaware"],
    ["TPP Stage 2","3 live","≈7.5 h / 23 h / ≤32 h","One likely scheduler-limited"],
    ["Counters S2 narrow","6 live","≤19.4 h hard bound","Conservative lower bounds if interrupted"],
    ["SAFE-CONTEXT","19 live","many 2–12 h; heavy 12–30 h","Coverage/runtime/nodes/peak RSS"],
    ["Rover MCTS","3 live + 18 pending","historically ~30 h; 72 h cap","High OOM risk"],
  ];
  simpleTable(s,["Experiment","Queue","Estimate","Next evidence"],rows,70,150,[260,210,250,420],64,C.dark,14);
}
{
  const s=base("The biggest story holes are actionable","Threats to inference",path.join(TRACK,"advisor_audit_20260830","story_holes.csv"),p++);
  const rows=[
    ["Primary RQ2/RQ4 incomplete","Finish selected-checkpoint Stage-2 MCTS","Compute","High"],
    ["MAIN-TERM ≠ strict paper stopping","Pilot STOP-ORIG on 3 seeds","Training","High"],
    ["OOM/time censoring","2 workers / 160 GiB lifecycle-safe resumption","Compute","High"],
    ["MPrime weak ranking","Report regret; consider frozen larger validation","Analysis","High"],
    ["PW cross-domain n=2","Expand only promising arms to n=5","Conditional","Medium"],
    ["SAFE-CONTEXT memory risk","Make peak RSS and node multiplier co-primary","Instrumentation","High"],
  ];
  simpleTable(s,["Hole","Defensible cover","Work","Priority"],rows,66,154,[340,500,170,120],60,C.dark,14);
  txt(s,"Full 18-item register: story_holes.csv",70,610,520,25,14,C.muted,true);
}
{
  const s=base("Improvements are prioritized against overlap","Experiment portfolio",path.join(TRACK,"advisor_audit_20260830","improvement_priorities.csv"),p++);
  const rows=[
    ["Finish Stage-2 MCTS gaps","5 / 5","High RQ closure","No duplicate"],
    ["STOP-ORIG 3-seed pilot","5 / 5","High fidelity","Covers MAIN-TERM caveat"],
    ["Lifecycle-safe resource resumption","5 / 5","High reliability","Covers Rover/FO legacy"],
    ["SAFE-CONTEXT joint coverage/RSS analysis","5 / 5","High correctness","Extends SAFE-1"],
    ["Automated anchor→policy→confirmation chain","5 / 4","High operational","Supports PRESERVE-4"],
    ["PW screen→selective n=5 expansion","4 / 5","Medium-high","Avoids duplicate grid"],
    ["SAFE-2 state×h profiling","2 / 4","Uncertain","Held; distinct from cutoff-only"],
  ];
  simpleTable(s,["Improvement","Priority / relevance","Predicted benefit","Overlap check"],rows,68,150,[400,210,260,260],56,C.dark,14);
  rect(s,68,612,1130,36,C.mint,"rounded-xl",C.mint);txt(s,"Every registered experiment has a scored improvement in improvement_priorities.csv (22/22).",88,616,1080,26,15,C.green,true);
}
{
  const s=base("Recommended continuation order","Decisions for the next meeting",path.join(TRACK,"advisor_audit_20260830","advisor_meeting_brief.md"),p++);
  const items=[
    ["NOW","Let running primary policy/training and the 16 Drone Stage-2 MCTS gaps finish."],
    ["FREEZE","Adopt the RQ-aligned statistics and three-stable/six-imperfect framing."],
    ["CONFIRM","Finish Delivery/TPP; separate tuning seeds from eight held-out seeds."],
    ["SELECT","Use PW cross-domain n=2 as a screen; expand only promising arms."],
    ["REPAIR","Deploy lifecycle-safe Rover/FO resource continuation before broad release."],
    ["REPLICATE","Pilot STOP-ORIG on three seeds before a full strict-paper campaign."],
  ];
  items.forEach((it,i)=>{const y=150+i*78;tag(s,it[0],76,y,110,i<2?C.green:(i<4?C.blue:C.orange),C.white);txt(s,it[1],208,y-2,950,42,18,C.ink,true);if(i<5)line(s,208,y+57,930,C.faint,1);});
  rect(s,76,625,1080,34,C.paleOrange,"rounded-xl",C.paleOrange);txt(s,"Defer SAFE-2, path-batched PW, PUCT/estimator sweeps and Stage-2-final MCTS until the primary RQs close.",96,628,1040,25,15,C.orange,true);
}
{
  const s=base("Experiment register — completed and live","Appendix",path.join(TRACK,"experiments.csv"),p++);
  const rows=[
    ["MAIN-VAL","Completed","Policy mainstream"],["MAIN-TERM","Completed","Terminal sensitivity"],["PRESERVE-4","Live","Stable-three confirmation"],["MPRIME-VAL","Completed / reclassified","Corrected validation"],["ANCHOR-4","Live","Coefficient tuning"],["MCTS-WIDTH","Live","Counters narrow"],["MCTS-PW","Completed","Efficiency trade-off"],["MCTS-PW-SAFE","Completed extension","Kmin=3"],["MCTS-SAFE","Completed diagnostic","2/4 repairs"],["MCTS-HORIZON","Live","Binding 750-action"],["MCTS-SAFE-CONTEXT","Live","Contextual nodes"],
  ];
  simpleTable(s,["ID","Status","Role"],rows,180,142,[330,300,430],44,C.dark,14);
}
{
  const s=base("Experiment register — held and secondary","Appendix",path.join(TRACK,"experiments.csv"),p++);
  const rows=[
    ["MAIN-VAL-S2-MCTS","Live/pending","Primary RQ2/RQ4 gap"],["MCTS-PW-CROSS-DOMAIN","Live","Two-seed screen"],["MCTS-RESOURCE","Held","Lifecycle/resource sensitivity"],["MCTS-LEGACY-ROVER","Resource-pending","Endpoint release"],["LONG-DRONE","Live/static follow-up","Training duration"],["STOP-ORIG","Held","Strict stopping fidelity"],["MCTS-SAFE2","Held design","State × horizon"],["MCTS-PW-PATHBATCH","Held design","Multi-node widening"],["PUCT-EST","Held","Search coefficients"],["ENHSP-LEAF","Completed exploratory","Leaf estimator"],["BG-HIST","Completed historical","Provenance only"],
  ];
  simpleTable(s,["ID","Status","Purpose"],rows,180,142,[330,300,430],44,C.dark,14);
}
{
  const s=base("Statistical interpretation","Appendix",path.join(TRACK,"advisor_audit_20260830","rq_results.csv"),p++);
  bulletList(s,[
    "Independent unit: network seed, not individual planning instance.",
    "Effect: paired per-seed coverage difference; RQ3 uses paired difference-in-differences.",
    "95% confidence interval: t interval over ten seed-level effects.",
    "p-value: exact sign-flip enumeration over all 2¹⁰ sign assignments.",
    "Multiplicity: Holm correction over five domains within each RQ.",
    "A non-significant result means insufficient evidence—not proof of no effect.",
    "Paper baselines lack seed-level variance; compare descriptively, never with a fabricated p-value.",
  ],86,154,1050,18,64);
  rect(s,86,615,1040,40,C.paleBlue,"rounded-xl",C.paleBlue);txt(s,"All corrected estimands and both raw/Holm p-values are in rq_results.csv.",106,619,1000,28,16,C.blue,true);
}

await fs.mkdir(path.join(OUT,"rendered"),{recursive:true});
for (const [index,slide] of presentation.slides.items.entries()) {
  const stem=`slide-${String(index+1).padStart(2,"0")}`;
  const png=await presentation.export({slide,format:"png",scale:1});
  await fs.writeFile(path.join(OUT,"rendered",`${stem}.png`),new Uint8Array(await png.arrayBuffer()));
  const layout=await slide.export({format:"layout"});
  await fs.writeFile(path.join(OUT,"rendered",`${stem}.layout.json`),await layout.text());
}
const montage=await presentation.export({format:"webp",montage:true,scale:0.5});
await fs.writeFile(path.join(OUT,"rendered","deck-montage.webp"),new Uint8Array(await montage.arrayBuffer()));
const pptx=await PresentationFile.exportPptx(presentation);
await pptx.save(path.join(OUT,"..","numeric_asnets_advisor_status_20260830.pptx"));
console.log(`slides=${presentation.slides.items.length}`);
