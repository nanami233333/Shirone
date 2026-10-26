import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { spawnSync } from "node:child_process";
import test from "node:test";

const script = fileURLToPath(new URL("../scripts/check-static-deploy.mjs", import.meta.url));

function fixture(t) {
	const root = mkdtempSync(join(tmpdir(), "shirone-static-"));
	t.after(() => rmSync(root, { recursive: true, force: true }));
	mkdirSync(join(root, "dist"));
	writeFileSync(join(root, "dist", "index.html"), "<!doctype html><title>Static</title>");
	return root;
}

function check(root) {
	return spawnSync(process.execPath, [script, root], { encoding: "utf8" });
}

test("accepts a static HTML build", (t) => {
	const result = check(fixture(t));
	assert.equal(result.status, 0, result.stderr);
});

for (const path of ["functions", "dist/functions", "dist/_worker.js"]) {
	test(`rejects executable Pages entry point ${path}`, (t) => {
		const root = fixture(t);
		mkdirSync(join(root, path), { recursive: true });
		writeFileSync(join(root, path, "index.js"), "export default {};");
		const result = check(root);
		assert.equal(result.status, 1);
		assert.match(result.stderr, /server runtime/i);
	});
}

test("rejects a bundled _worker.js file", (t) => {
	const root = fixture(t);
	writeFileSync(join(root, "dist", "_worker.js"), "export default {};");
	const result = check(root);
	assert.equal(result.status, 1);
	assert.match(result.stderr, /server runtime/i);
});

test("rejects a missing index instead of approving an empty output", (t) => {
	const root = fixture(t);
	rmSync(join(root, "dist", "index.html"));
	const result = check(root);
	assert.equal(result.status, 1);
	assert.match(result.stderr, /index.html/);
});
