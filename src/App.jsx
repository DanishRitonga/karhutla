import { useState, useMemo } from "react";
import { Flame, TrendingUp, MapPin, Calendar, Search, ChevronRight, Check } from "lucide-react";

const GRID = {"minx":-213782,"miny":-234545,"cell":5000.0,"cols":108,"rows":81,"rowsRLE":[[0,[[57,57]]],[1,[[53,58]]],[2,[[51,59]]],[3,[[37,39],[50,60]]],[4,[[35,41],[45,45],[47,61]]],[5,[[34,62]]],[6,[[33,63]]],[7,[[31,64],[72,73]]],[8,[[30,74]]],[9,[[29,73]]],[10,[[28,73],[95,95]]],[11,[[26,73],[95,95],[97,98]]],[12,[[25,73],[95,98]]],[13,[[23,75],[93,99]]],[14,[[22,76],[93,99]]],[15,[[21,77],[93,93],[96,99]]],[16,[[21,78],[97,98]]],[17,[[17,18],[21,81]]],[18,[[17,80],[105,107]]],[19,[[16,78],[97,100],[104,106]]],[20,[[15,75],[97,103],[106,106]]],[21,[[15,75],[98,102]]],[22,[[15,77],[99,102]]],[23,[[15,77],[99,101]]],[24,[[15,80],[99,100]]],[25,[[15,82]]],[26,[[15,81],[97,97]]],[27,[[16,81],[100,100]]],[28,[[16,81],[99,99]]],[29,[[9,81]]],[30,[[8,80]]],[31,[[7,80]]],[32,[[5,5],[7,78],[95,95]]],[33,[[5,77]]],[34,[[4,75]]],[35,[[4,66],[70,74]]],[36,[[3,65],[68,70],[72,73]]],[37,[[3,65],[68,71]]],[38,[[3,64],[68,70]]],[39,[[4,63],[68,70],[74,75]]],[40,[[4,62],[64,65],[73,75],[92,92]]],[41,[[3,53],[63,65],[73,75],[91,92]]],[42,[[1,51],[54,65],[73,76],[82,82],[91,91]]],[43,[[1,50],[52,64],[66,67],[73,75],[81,84],[91,92],[99,100]]],[44,[[1,49],[52,68],[73,75],[89,90],[98,100]]],[45,[[1,48],[51,67],[85,85],[89,89],[97,101]]],[46,[[2,47],[50,52],[54,59],[62,66],[84,85],[97,100]]],[47,[[1,47],[49,55],[57,57],[60,65],[72,73],[86,88],[93,95],[97,101]]],[48,[[1,46],[48,55],[57,64],[72,73],[85,89],[92,101]]],[49,[[1,46],[48,53],[58,62],[86,90],[93,100]]],[50,[[2,46],[48,52],[87,87],[89,89],[95,99]]],[51,[[1,46],[48,52]]],[52,[[1,45],[48,52]]],[53,[[1,45],[47,51],[53,53]]],[54,[[0,45],[48,53]]],[55,[[0,45],[48,53]]],[56,[[6,43],[49,53]]],[57,[[7,41],[45,53]]],[58,[[8,40],[44,52]]],[59,[[8,39],[43,46]]],[60,[[8,38]]],[61,[[8,31],[35,36]]],[62,[[8,29],[32,34]]],[63,[[8,28],[30,35]]],[64,[[6,28],[30,36]]],[65,[[5,27],[30,36]]],[66,[[5,27],[29,36]]],[67,[[5,27],[29,37]]],[68,[[5,27],[29,37]]],[69,[[5,16],[18,26],[29,36]]],[70,[[6,15],[17,26],[33,36]]],[71,[[6,13],[16,24],[34,35]]],[72,[[6,10],[16,23]]],[73,[[6,9],[15,22]]],[74,[[6,8],[16,21]]],[75,[[6,7],[18,19]]],[76,[[5,6]]],[77,[[5,5]]],[78,[[5,5]]],[79,[[5,5]]]],"regions":{"Bengkalis":[5561,60852],"Siak":[5562,-16596],"Pelalawan":[38941,-77445],"Rokan Hilir":[-116767,94061],"Indragiri Hilir":[127973,-149330],"Kampar":[-105694,-71900],"Dumai":[-61167,71921],"Rokan Hulu":[-172417,-5491]},"outline":[[[[314016,-132443],[317095,-127614],[326165,-141697],[321027,-146983],[317324,-146231],[306626,-138670],[300822,-133388],[280382,-141217],[277667,-137505],[271966,-137643],[270077,-131484],[278851,-124955],[279316,-117487],[284307,-108247],[293147,-114564],[302071,-124886],[306847,-132003],[314016,-132443]]],[[[289105,-100810],[278932,-89768],[281219,-85600],[289654,-93580],[296634,-102692],[289105,-100810]]],[[[246819,6658],[249931,13359],[259277,15131],[259017,19592],[267923,22229],[278379,18510],[286921,21211],[289416,13775],[293342,11846],[296016,3290],[293006,-1512],[296015,-9429],[291548,-10861],[288806,-20908],[280349,-19251],[274748,-11487],[271257,-10335],[268515,-2321],[271613,104],[272843,5899],[267576,5546],[264265,945],[250986,135],[246819,6658]]],[[[285704,-159825],[287997,-161219],[281950,-170430],[277819,-182339],[271501,-175288],[266440,-177374],[265611,-184520],[260974,-183078],[259108,-172011],[251475,-165709],[253380,-154631],[259317,-159742],[262724,-158587],[270448,-153315],[271176,-148762],[281679,-151624],[285704,-159825]]],[[[274458,-105051],[266089,-99774],[270611,-95607],[277124,-103248],[274458,-105051]]],[[[269206,-74058],[267416,-78218],[261606,-75784],[261815,-70505],[269206,-74058]]],[[[242455,-24081],[238879,-16196],[232988,-14220],[233498,-4567],[238179,-6559],[239413,-10700],[242761,-14441],[247470,-13359],[252248,-15344],[249624,-20010],[242455,-24081]]],[[[250430,-22100],[251780,-27790],[249222,-33265],[243474,-29147],[243264,-24398],[250430,-22100]]],[[[223619,20230],[228069,17430],[233948,21761],[239497,15093],[237906,3920],[232239,2887],[223334,-1734],[218212,230],[211833,7693],[215780,8876],[213615,15053],[221734,15995],[223619,20230]]],[[[204532,-3266],[206426,3267],[210864,56],[215199,168],[218754,-3835],[213232,-7563],[204532,-3266]]],[[[212020,-17371],[208792,-22044],[203393,-16796],[209630,-14947],[212020,-17371]]],[[[201707,-18132],[204251,-25949],[197167,-22898],[190880,-14317],[194297,-12678],[201707,-18132]]],[[[-147221,125030],[-134075,117982],[-125498,107940],[-124340,115326],[-134609,124206],[-137232,136090],[-133004,140881],[-118356,143813],[-104889,142024],[-103840,136010],[-92932,126127],[-81891,119959],[-75559,110105],[-74164,99096],[-71222,91195],[-71001,84241],[-65811,77547],[-58130,75926],[-50396,70945],[-44041,70924],[-37827,73861],[-27042,73508],[-19270,66352],[5320,47705],[16183,42189],[16170,32712],[17706,25930],[22521,22290],[20555,11583],[26948,-2902],[35692,-9089],[43644,-19805],[54814,-27062],[64706,-28808],[77052,-27018],[85497,-29585],[97438,-29336],[101078,-31636],[112795,-45369],[116895,-52217],[123807,-59719],[130980,-59205],[137480,-55272],[143803,-54029],[147633,-49887],[156645,-52351],[166385,-60579],[175831,-66643],[178263,-71967],[183778,-72893],[192044,-79684],[196430,-88899],[197071,-98325],[200574,-110923],[186675,-112955],[177458,-112944],[174985,-117420],[176805,-121631],[166904,-123887],[166926,-131282],[170393,-135273],[180570,-135994],[195936,-144295],[197304,-148494],[188434,-149505],[179311,-153893],[177808,-159057],[167355,-166612],[158643,-168339],[157582,-185048],[155978,-191232],[162391,-192652],[164977,-195923],[149451,-197518],[125321,-193176],[112290,-194148],[100700,-209523],[89987,-217539],[86884,-223846],[76541,-229919],[71014,-234545],[68655,-231695],[55143,-230603],[49254,-224049],[43784,-224570],[42988,-220504],[36105,-216902],[33868,-212054],[28306,-212561],[24160,-215304],[18111,-210059],[12979,-212521],[6453,-209066],[-1831,-208014],[-8428,-214766],[-17220,-219310],[-28061,-217658],[-38521,-214068],[-44973,-204781],[-55646,-199986],[-65967,-192132],[-71245,-183550],[-82736,-179953],[-91115,-171406],[-103754,-165732],[-108520,-154694],[-107752,-148326],[-111684,-145393],[-117808,-145178],[-121250,-151626],[-129789,-145831],[-129220,-142758],[-135646,-136100],[-139267,-117824],[-136851,-114271],[-138712,-105240],[-134387,-97244],[-135950,-91885],[-141624,-89239],[-145990,-90932],[-155129,-90306],[-170888,-86015],[-173880,-83175],[-178610,-73004],[-182245,-71392],[-191177,-73751],[-191100,-65441],[-198076,-52509],[-199589,-44950],[-194041,-32101],[-206876,-22328],[-207726,-11940],[-212151,-9335],[-207270,-5571],[-204520,-36],[-210379,8014],[-204572,16818],[-207946,19377],[-208717,28265],[-212812,37405],[-213782,42904],[-195445,43496],[-191852,46646],[-187313,46125],[-176969,52084],[-174165,56611],[-176181,59472],[-171796,69649],[-177008,86039],[-189936,88816],[-190565,91913],[-186587,101725],[-186517,112234],[-182342,135061],[-187958,148367],[-188774,156433],[-185593,169778],[-182024,153798],[-174065,141063],[-159573,128758],[-147221,125030]]],[[[149171,-21526],[152666,-11147],[161277,-15777],[166445,-17706],[170255,-22469],[166870,-32415],[166873,-39306],[165488,-41595],[155738,-39005],[150206,-28896],[149171,-21526]]],[[[164085,-10476],[165131,-15719],[161271,-15420],[156183,-11293],[159583,-8152],[164085,-10476]]],[[[157971,4820],[159457,-1251],[147672,1954],[145342,8366],[150133,14681],[155024,6394],[157971,4820]]],[[[134646,-52960],[128757,-55197],[125880,-46221],[128271,-34869],[133626,-32670],[141375,-37192],[144575,-44447],[142404,-52124],[134646,-52960]]],[[[83731,17783],[99671,13872],[108752,9947],[128777,-11151],[128508,-16082],[124403,-18854],[119351,-18861],[105766,-10542],[98226,-3657],[88518,12],[82046,4515],[74029,2215],[71600,3323],[75924,12877],[83731,17783]]],[[[116091,-31488],[109904,-34438],[102114,-25341],[95929,-23189],[72237,-22928],[65250,-25686],[61091,-25240],[48677,-19576],[44610,-12683],[48968,-6554],[55005,-6173],[50655,8965],[53131,14307],[61565,11776],[67943,4840],[67492,2],[71779,-835],[79750,2418],[82529,9],[97332,-5434],[105355,-13180],[116177,-19938],[116091,-31488]]],[[[50807,57775],[54953,51891],[54514,41705],[56139,28183],[42167,37504],[32536,47411],[14553,50483],[3213,58788],[3,65709],[4,67378],[7930,66945],[17028,63749],[43532,59838],[50807,57775]]],[[[42609,-9586],[29974,1261],[25557,11124],[27198,19437],[25808,25316],[22317,31091],[24994,41109],[22883,44750],[32211,45819],[43429,34460],[53878,25557],[48387,9088],[53398,-5281],[47946,-5395],[42609,-9586]]],[[[-24302,103993],[-27934,98098],[-30478,84742],[-42880,76475],[-50674,77485],[-60874,80499],[-68920,100750],[-66249,113719],[-60245,118010],[-53981,114922],[-50038,116100],[-46148,120971],[-38845,124909],[-33286,121880],[-24977,109304],[-24302,103993]]]]};

const RISK_LEVELS = [
  { key: "low", label: "Rendah", color: "#bbf7d0", text: "#166534", bg: "#f0fdf4" },
  { key: "mid", label: "Sedang", color: "#fde047", text: "#854d0e", bg: "#fefce8" },
  { key: "high", label: "Tinggi", color: "#fb923c", text: "#9a3412", bg: "#fff7ed" },
  { key: "vhigh", label: "Sangat tinggi", color: "#ef4444", text: "#991b1b", bg: "#fef2f2" },
];

function levelInfo(key) {
  return RISK_LEVELS.find((l) => l.key === key);
}

function decodeRLE() {
  const cells = [];
  GRID.rowsRLE.forEach(([r, ranges]) => {
    ranges.forEach(([a, b]) => {
      for (let c = a; c <= b; c++) cells.push({ r, c });
    });
  });
  return cells;
}
const ALL_CELLS = decodeRLE();

const REGION_NAMES = Object.keys(GRID.regions);

function cellCenter(r, c) {
  const x = GRID.minx + (c + 0.5) * GRID.cell;
  const y = GRID.miny + (r + 0.5) * GRID.cell;
  return [x, y];
}

function nearestRegion(x, y) {
  let best = REGION_NAMES[0];
  let bestD = Infinity;
  for (const name of REGION_NAMES) {
    const [rx, ry] = GRID.regions[name];
    const d = (rx - x) ** 2 + (ry - y) ** 2;
    if (d < bestD) {
      bestD = d;
      best = name;
    }
  }
  return best;
}

function hash(a, b, c) {
  let h = a * 374761393 + b * 668265263 + c * 2246822519;
  h = (h ^ (h >>> 13)) * 1274126177;
  h = h ^ (h >>> 16);
  return ((h >>> 0) % 10000) / 10000;
}

const HOT_ANCHORS = [
  { name: "Bengkalis", w: 1.0 },
  { name: "Siak", w: 0.75 },
];

function riskForCell(cell, day) {
  const [x, y] = cellCenter(cell.r, cell.c);
  let heat = 0;
  HOT_ANCHORS.forEach((a) => {
    const [ax, ay] = GRID.regions[a.name];
    const dist = Math.sqrt((ax - x) ** 2 + (ay - y) ** 2) / 1000;
    const falloff = Math.exp(-dist / 55);
    heat += a.w * falloff;
  });
  const noise = hash(cell.r, cell.c, day * 97 + 11);
  const dayBoost = (day - 1) * 0.015;
  const score = Math.min(0.97, heat * 0.75 + noise * 0.28 + dayBoost);
  let level = "low";
  if (score > 0.72) level = "vhigh";
  else if (score > 0.5) level = "high";
  else if (score > 0.3) level = "mid";
  return { level, score: Math.round(score * 100) / 100 };
}

function buildDay(day) {
  return ALL_CELLS.map((cell) => {
    const [x, y] = cellCenter(cell.r, cell.c);
    const region = nearestRegion(x, y);
    const { level, score } = riskForCell(cell, day);
    return { ...cell, x, y, region, level, score, id: `RIAU_${cell.r}_${cell.c}` };
  });
}

const DAY_CACHE = {};
function getDay(day) {
  if (!DAY_CACHE[day]) DAY_CACHE[day] = buildDay(day);
  return DAY_CACHE[day];
}

function summarize(cells) {
  const total = cells.length;
  const high = cells.filter((c) => c.level === "high" || c.level === "vhigh").length;
  const byRegion = {};
  cells.forEach((c) => {
    if (!byRegion[c.region]) byRegion[c.region] = { high: 0, total: 0, sum: 0 };
    byRegion[c.region].total += 1;
    byRegion[c.region].sum += c.score;
    if (c.level === "high" || c.level === "vhigh") byRegion[c.region].high += 1;
  });
  const ranking = Object.entries(byRegion)
    .map(([name, v]) => ({ name, avg: v.sum / v.total, high: v.high }))
    .sort((a, b) => b.avg - a.avg);
  return { total, high, ranking };
}

const AI_WEEKLY = {
  1: "Risiko masih terkonsentrasi di bagian utara Bengkalis dan Siak, dengan sebagian besar wilayah lain berada pada kategori rendah hingga sedang.",
  3: "Risiko meningkat pada horizon hari ke-3, terutama di area gambut Bengkalis. Siak mulai menunjukkan pola serupa dengan minggu sebelumnya.",
  5: "Pada hari ke-5, sejumlah grid di Pelalawan mulai masuk kategori sedang. Bengkalis tetap menjadi wilayah dengan konsentrasi risiko tertinggi.",
  7: "Proyeksi 7 hari menunjukkan risiko tetap tinggi di Bengkalis dan Siak apabila kondisi kering berlanjut tanpa hujan signifikan.",
};
function nearestInsight(day) {
  const keys = [1, 3, 5, 7];
  const closest = keys.reduce((a, b) => (Math.abs(b - day) < Math.abs(a - day) ? b : a));
  return AI_WEEKLY[closest];
}

const MAPW = GRID.cols * GRID.cell;
const MAPH = GRID.rows * GRID.cell;
const VB_W = 640;
const VB_H = Math.round((MAPH / MAPW) * VB_W);
const SCALE = VB_W / MAPW;
const CELL_PX = GRID.cell * SCALE;

function toScreen(x, y) {
  const sx = (x - GRID.minx) * SCALE;
  const sy = VB_H - (y - GRID.miny) * SCALE;
  return [sx, sy];
}

function outlinePath() {
  let d = "";
  GRID.outline.forEach((rings) => {
    rings.forEach((ring) => {
      ring.forEach(([x, y], i) => {
        const [sx, sy] = toScreen(x, y);
        d += (i === 0 ? "M" : "L") + sx.toFixed(1) + "," + sy.toFixed(1) + " ";
      });
      d += "Z ";
    });
  });
  return d;
}
const OUTLINE_D = outlinePath();

function RiauMap({ cells, size, onCellClick, selectedId }) {
  const isSmall = size === "small";
  const w = isSmall ? 340 : "auto";
  const h = isSmall ? Math.round((VB_H / VB_W) * 340) : "auto";
  return (
    <svg
      viewBox={`0 0 ${VB_W} ${VB_H}`}
      width={w}
      height={h}
      style={
        isSmall
          ? { display: "block", maxWidth: 340 }
          : { display: "block", maxWidth: "100%", maxHeight: "56vh", width: "auto", height: "auto" }
      }
    >
      <path d={OUTLINE_D} fill="#f8fafc" stroke="#cbd5e1" strokeWidth={1.2} fillRule="evenodd" />
      <g clipPath="url(#riau-clip)">
        {cells.map((cell) => {
          const [sx, sy] = toScreen(cell.x, cell.y);
          const info = levelInfo(cell.level);
          const isSel = selectedId === cell.id;
          return (
            <rect
              key={cell.id}
              x={sx - CELL_PX / 2}
              y={sy - CELL_PX / 2}
              width={Math.max(CELL_PX, 1)}
              height={Math.max(CELL_PX, 1)}
              fill={info.color}
              stroke={isSel ? "#1e293b" : "rgba(255,255,255,0.35)"}
              strokeWidth={isSel ? 1.6 : 0.3}
              onClick={() => onCellClick(cell)}
              style={{ cursor: "pointer" }}
            >
              <title>{`${cell.id} · ${cell.region} · ${levelInfo(cell.level).label}`}</title>
            </rect>
          );
        })}
      </g>
      <path d={OUTLINE_D} fill="none" stroke="#64748b" strokeWidth={1.2} fillRule="evenodd" />
    </svg>
  );
}

export default function Dashboard() {
  const [tab, setTab] = useState("overview");
  const [day, setDay] = useState(1);
  const [selectedCell, setSelectedCell] = useState(null);
  const [question, setQuestion] = useState("");
  const [asked, setAsked] = useState(null);

  const cells = useMemo(() => getDay(day), [day]);
  const { total, high, ranking } = useMemo(() => summarize(cells), [cells]);
  const prevHigh = day > 1 ? summarize(getDay(day - 1)).high : high;
  const delta = prevHigh === 0 ? 0 : Math.round(((high - prevHigh) / Math.max(prevHigh, 1)) * 100);
  const trend7 = useMemo(() => {
    const d1High = summarize(getDay(1)).high;
    const d7High = summarize(getDay(7)).high;
    if (d1High === 0) return d7High > 0 ? 100 : 0;
    return Math.round(((d7High - d1High) / d1High) * 100);
  }, []);

  const openExplorer = (cell) => {
    setSelectedCell(cell);
    setTab("explorer");
    setAsked(null);
    setQuestion("");
  };

  const handleAsk = () => {
    if (!question.trim()) return;
    setAsked(question.trim());
  };

  return (
    <div className="min-h-screen bg-gray-50 text-slate-800">
      <div className="w-full px-6 lg:px-10 py-6">
        <header className="flex items-center justify-between mb-6">
          <div className="flex items-center gap-2">
            <Flame className="text-orange-500" size={22} />
            <div>
              <h1 className="text-base font-semibold leading-tight">Karhutla early warning</h1>
              <p className="text-xs text-slate-500 leading-tight">Prediksi hotspot 7 hari · Provinsi Riau</p>
            </div>
          </div>
          <nav className="flex bg-white border border-slate-200 rounded-lg p-1 gap-1">
            <button
              onClick={() => setTab("overview")}
              className={`px-4 py-1.5 text-sm rounded-md transition ${
                tab === "overview" ? "bg-slate-800 text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              Overview
            </button>
            <button
              onClick={() => setTab("explorer")}
              className={`px-4 py-1.5 text-sm rounded-md transition ${
                tab === "explorer" ? "bg-slate-800 text-white" : "text-slate-600 hover:bg-slate-100"
              }`}
            >
              Explorer
            </button>
          </nav>
        </header>

        {tab === "overview" && (
          <div className="space-y-4">
            <div className="grid grid-cols-3 gap-4">
              <div className="bg-white border border-slate-200 rounded-xl p-4 col-span-1">
                {high === 0 ? (
                  <>
                    <div className="flex items-center gap-2 text-emerald-600">
                      <Check size={18} />
                      <span className="text-sm font-medium">Tidak ada grid risiko tinggi</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">Mayoritas wilayah Riau berada pada kategori rendah hingga sedang.</p>
                  </>
                ) : (
                  <>
                    <div className="flex items-baseline gap-2">
                      <Flame className="text-red-500 mt-1" size={20} />
                      <span className="text-2xl font-semibold">{high} grid risiko tinggi</span>
                    </div>
                    <p className="text-xs text-slate-500 mt-1">dari {total} grid di Riau</p>
                    {day > 1 && (
                      <p className={`text-xs mt-1 flex items-center gap-1 ${delta >= 0 ? "text-red-600" : "text-emerald-600"}`}>
                        <TrendingUp size={13} className={delta < 0 ? "rotate-180" : ""} />
                        {delta >= 0 ? "+" : ""}
                        {delta}% dibanding hari sebelumnya
                      </p>
                    )}
                  </>
                )}
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-4 col-span-2">
                <div className="flex items-center gap-2 text-xs text-slate-500 mb-2">
                  <Calendar size={13} />
                  Horizon prediksi
                </div>
                <div className="flex gap-1.5">
                  {[1, 2, 3, 4, 5, 6, 7].map((d) => (
                    <button
                      key={d}
                      onClick={() => setDay(d)}
                      className={`flex-1 py-1.5 text-sm rounded-md border transition ${
                        day === d
                          ? "bg-slate-800 text-white border-slate-800"
                          : "bg-white text-slate-600 border-slate-200 hover:border-slate-400"
                      }`}
                    >
                      +{d}
                    </button>
                  ))}
                </div>
              </div>
            </div>

            <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 items-stretch">
              <div className="lg:col-span-9 bg-white border border-slate-200 rounded-xl p-4">
                <div className="flex items-center justify-between mb-3">
                  <span className="text-sm font-medium">Peta risiko · hari +{day}</span>
                  <div className="flex gap-3 text-xs text-slate-500">
                    {RISK_LEVELS.map((l) => (
                      <span key={l.key} className="flex items-center gap-1">
                        <span style={{ width: 8, height: 8, borderRadius: 2, backgroundColor: l.color, display: "inline-block" }} />
                        {l.label}
                      </span>
                    ))}
                  </div>
                </div>
                <div className="flex justify-center py-2">
                  <RiauMap cells={cells} size="large" onCellClick={openExplorer} selectedId={selectedCell?.id} />
                </div>
                <p className="text-xs text-slate-400 text-center mt-2">Klik satu grid untuk membuka detail di Explorer</p>
              </div>

              <div className="lg:col-span-3 flex flex-col gap-4">
                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
                  <span className="text-sm font-medium">Wilayah paling perlu diperhatikan</span>
                  <div className="flex-1 flex flex-col justify-center space-y-1.5">
                    {ranking.slice(0, 4).map((r, i) => (
                      <div key={r.name} className="flex items-center justify-between text-sm py-1 border-b border-slate-100 last:border-0">
                        <span className="flex items-center gap-2">
                          <span className="text-xs text-slate-400 w-3">{i + 1}</span>
                          {r.name}
                        </span>
                        <span className="text-xs text-slate-500">{levelInfo(r.avg > 0.65 ? "high" : r.avg > 0.4 ? "mid" : "low").label}</span>
                      </div>
                    ))}
                  </div>
                </div>

                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
                  <span className="text-sm font-medium">AI weekly insight</span>
                  <div className="flex-1 flex items-center">
                    <p className="text-sm text-slate-600 leading-relaxed">{nearestInsight(day)}</p>
                  </div>
                </div>

                <div className="flex-1 bg-white border border-slate-200 rounded-xl p-4 flex flex-col">
                  <span className="text-sm font-medium">Ringkasan</span>
                  <div className="flex-1 flex flex-col justify-center space-y-2.5">
                    <div className="flex items-center gap-2 text-sm">
                      <Flame size={14} className="text-red-500 shrink-0" />
                      <span className="text-slate-500">Grid risiko tinggi</span>
                      <span className="ml-auto font-semibold text-slate-800">{high}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <MapPin size={14} className="text-slate-400 shrink-0" />
                      <span className="text-slate-500">Wilayah tertinggi</span>
                      <span className="ml-auto font-semibold text-slate-800">{ranking[0]?.name ?? "-"}</span>
                    </div>
                    <div className="flex items-center gap-2 text-sm">
                      <TrendingUp size={14} className={`shrink-0 ${trend7 < 0 ? "rotate-180 text-emerald-500" : "text-red-500"}`} />
                      <span className="text-slate-500">Tren 7 hari</span>
                      <span className={`ml-auto font-semibold ${trend7 >= 0 ? "text-red-600" : "text-emerald-600"}`}>
                        {trend7 >= 0 ? "+" : ""}
                        {trend7}%
                      </span>
                    </div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        )}

        {tab === "explorer" && (
          <div className="space-y-4">
            <div className="bg-white border border-slate-200 rounded-xl px-4 py-2.5 flex items-center gap-4 text-sm">
              <span className="flex items-center gap-1.5 text-slate-700">
                <MapPin size={14} className="text-slate-400" />
                {selectedCell ? selectedCell.region : "Belum ada grid dipilih"}
              </span>
              <span className="flex items-center gap-1.5 text-slate-700">
                <Calendar size={14} className="text-slate-400" />
                Hari +{day}
              </span>
            </div>

            <div className="grid grid-cols-2 gap-4">
              <div className="bg-white border border-slate-200 rounded-xl p-4">
                <span className="text-sm font-medium mb-2 block">Peta · klik grid lain untuk berpindah</span>
                <div className="flex justify-center py-2">
                  <RiauMap cells={cells} size="small" onCellClick={setSelectedCell} selectedId={selectedCell?.id} />
                </div>
              </div>

              <div className="bg-white border border-slate-200 rounded-xl p-4">
                {!selectedCell ? (
                  <p className="text-sm text-slate-400 py-8 text-center">Pilih grid pada peta untuk melihat detail risiko.</p>
                ) : (
                  <>
                    <div className="flex items-center justify-between">
                      <span className="text-xs text-slate-400">{selectedCell.id}</span>
                      <span
                        className="text-xs font-medium px-2 py-0.5 rounded"
                        style={{ backgroundColor: levelInfo(selectedCell.level).bg, color: levelInfo(selectedCell.level).text }}
                      >
                        {levelInfo(selectedCell.level).label}
                      </span>
                    </div>
                    <p className="text-2xl font-semibold mt-1">{selectedCell.score.toFixed(2)}</p>
                    <p className="text-xs text-slate-500 mb-3">risk score</p>

                    <div className="border-t border-slate-100 pt-3">
                      <p className="text-xs font-medium text-slate-500 mb-1.5">AI summary</p>
                      <p className="text-sm text-slate-700 leading-relaxed">
                        Risiko {levelInfo(selectedCell.level).label.toLowerCase()} di {selectedCell.region} dipengaruhi
                        oleh curah hujan rendah selama sepuluh hari terakhir dan dominasi lahan gambut di area ini.
                      </p>
                    </div>

                    <div className="border-t border-slate-100 pt-3 mt-3">
                      <p className="text-xs font-medium text-slate-500 mb-1.5">Evidence</p>
                      <div className="text-xs text-slate-600 space-y-1 font-mono">
                        <div className="flex justify-between"><span>Rainfall anomaly</span><span>-58%</span></div>
                        <div className="flex justify-between"><span>Soil moisture pct</span><span>18</span></div>
                        <div className="flex justify-between"><span>Peat fraction</span><span>0.66</span></div>
                      </div>
                    </div>
                  </>
                )}
              </div>
            </div>

            <div className="bg-white border border-slate-200 rounded-xl p-4">
              <span className="text-sm font-medium flex items-center gap-1.5">
                <Search size={14} className="text-slate-400" />
                Ask about this region
              </span>
              <div className="flex gap-2 mt-2">
                <input
                  value={question}
                  onChange={(e) => setQuestion(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && handleAsk()}
                  placeholder="mis. Kabupaten mana yang risikonya naik minggu ini?"
                  className="flex-1 text-sm border border-slate-200 rounded-md px-3 py-2 outline-none focus:border-slate-400"
                />
                <button
                  onClick={handleAsk}
                  className="px-3 py-2 bg-slate-800 text-white text-sm rounded-md flex items-center gap-1"
                >
                  Tanya <ChevronRight size={14} />
                </button>
              </div>
              {asked && (
                <div className="mt-3 bg-slate-50 rounded-md p-3 text-sm text-slate-700">
                  <p className="text-xs text-slate-400 mb-1">Anda bertanya: "{asked}"</p>
                  Berdasarkan data 7 hari, {ranking[0].name} menunjukkan peningkatan risiko paling konsisten,
                  dengan {ranking[0].high} grid berada pada kategori tinggi atau sangat tinggi.
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
