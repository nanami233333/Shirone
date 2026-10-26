import { existsSync, statSync } from "node:fs";
import { resolve } from "node:path";

// Pages static requests do not invoke Functions. Reject both supported runtime
// entry modes before upload so a theme update cannot silently change that.
const root = resolve(process.argv[2] ?? ".");
const forbidden = ["functions", "dist/functions", "dist/_worker.js"];
const runtimes = forbidden.filter((entry) => existsSync(resolve(root, entry)));

if (runtimes.length > 0) {
	console.error(`[static-deploy] Refusing server runtime: ${runtimes.join(", ")}`);
	process.exit(1);
}

const index = resolve(root, "dist/index.html");
if (!existsSync(index) || !statSync(index).isFile()) {
	console.error("[static-deploy] Missing dist/index.html; build the static site first.");
	process.exit(1);
}

console.log("[static-deploy] PASS: static index exists; no Pages Functions or _worker.js.");
