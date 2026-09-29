import fs from "node:fs/promises";
import { FileBlob, SpreadsheetFile } from "@oai/artifact-tool";

const inputPath = "/Users/madhubhatt/Desktop/Weekly Buy Tracker and Technical Analysis.xlsx";
const outputDir = "/tmp/stock_signals_roi_preview";

await fs.mkdir(outputDir, { recursive: true });

let workbook;
try {
  const input = await FileBlob.load(inputPath);
  workbook = await SpreadsheetFile.importXlsx(input);
} catch (error) {
  console.error("IMPORT_ERROR");
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
}

const summary = await workbook.inspect({
  kind: "workbook,sheet,table",
  maxChars: 8000,
  tableMaxRows: 8,
  tableMaxCols: 16,
  tableMaxCellChars: 80,
});
console.log(summary.ndjson);

try {
  const render = await workbook.render({
    sheetName: "ROI Journal",
    range: "A1:N12",
    scale: 2,
    format: "png",
  });
  await fs.writeFile(`${outputDir}/roi_journal_A1_N12.png`, new Uint8Array(await render.arrayBuffer()));
  console.log(`rendered=${outputDir}/roi_journal_A1_N12.png`);
} catch (error) {
  console.error("RENDER_ERROR");
  console.error(error?.stack || error?.message || String(error));
  process.exit(1);
}
