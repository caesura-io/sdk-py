// Run after building the local JS SDK: node scripts/check_js_parity.mjs ../sdk-js
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';

const sdk = resolve(process.argv[2] ?? '../sdk-js');
const coreUrl = pathToFileURL(`${sdk}/packages/core/dist/index.js`).href;
const core = await import(coreUrl);
const require = createRequire(`${sdk}/package.json`);
const ts = require('typescript');
// Transpile the current adapter in memory; do not edit either JS source or dist.
const adapter = ts.transpileModule(
  readFileSync(`${sdk}/packages/openai/src/adapters.ts`, 'utf8'),
  { compilerOptions: { module: ts.ModuleKind.ESNext, target: ts.ScriptTarget.ES2022 } },
).outputText.replaceAll("'@caesura-io/core'", JSON.stringify(coreUrl));
const { getMessageText } = await import(`data:text/javascript;base64,${Buffer.from(adapter).toString('base64')}`);
const fixtures = JSON.parse(readFileSync(new URL('../packages/caesura-core/tests/fixtures/shared-policies.json', import.meta.url)));
for (const c of fixtures.multipart) assert.equal(getMessageText(c.message.content), c.text, c.name);
for (const c of fixtures.templates) assert.equal(core.renderAnalysis(c.analysis, c.template), c.expected, c.name);
for (const c of fixtures.history) {
  const dialogue = c.dialogue.map(m => ({
    speakerRole: m.role, speakerName: m.role === 'assistant' ? 'Agent' : 'Customer',
    speakerIndex: m.role === 'assistant' ? 0 : 1, text: m.text,
  }));
  const anchors = core.dialogueAnchors(dialogue);
  const state = new core.MemoryCaesuraStore().get('fixture');
  state.recommendations = c.analyses.map((r, i) => ({
    id: String(i), analysis: r.value, afterMessageHash: 'unused',
    afterMessageAnchor: r.afterOccurrence ? anchors[r.afterOccurrence - 1] : 'old-sha-anchor',
    createdAtMs: 0, createdAtTurn: i,
  }));
  const send = {};
  if ('max_messages' in c.send) send.maxMessages = c.send.max_messages;
  if ('max_input_chars' in c.send) send.maxInputChars = c.send.max_input_chars;
  assert.deepEqual(core.buildAnalyzeMessages(dialogue, state, send), c.expected, c.name);
}
console.log(`Python/JS parity: ${fixtures.multipart.length + fixtures.templates.length + fixtures.history.length} shared policy vectors passed.`);
