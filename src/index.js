import { buildBigrams, entropy, getTopN, mergeModels, recordKeypress, toProbabilities } from './bobbi.js';

// ─── 1. Corpus pré-entraîné ───────────────────────────────────────────────────

const corpus =
  'le chat mange la souris sur le tapis le soleil brille sur la ville les enfants jouent dans la cour';

const pretrained = buildBigrams(corpus);
console.log('=== Modèle pré-entraîné ===');
console.log('Top 5 après "le" :', getTopN(5, pretrained, 'le'));

// ─── 2. Modèle live (frappes utilisateur) ─────────────────────────────────────

let live = {};
const frappes = ['b', 'o', 'n', 'j', 'o', 'u', 'r', ' ', 'l', 'e', ' ', 'm', 'o', 'n', 'd', 'e'];

// On simule des frappes : chaque paire (chars[i]+chars[i+1]) → chars[i+2]
for (let i = 0; i < frappes.length - 2; i++) {
  const context = frappes[i] + frappes[i + 1];
  const next = frappes[i + 2];
  live = recordKeypress(context, next, live);
}

console.log('\n=== Modèle live ===');
console.log('Top 3 après "on" :', getTopN(3, live, 'on'));

// ─── 3. Fusion des deux modèles ───────────────────────────────────────────────

const merged = mergeModels(0.6, pretrained, live);
console.log('\n=== Modèle fusionné (60% live) ===');
console.log('Top 5 après "le" :', getTopN(5, merged, 'le'));

// ─── 4. Entropie ──────────────────────────────────────────────────────────────

const probs = toProbabilities(pretrained['le'] ?? {});
console.log('\n=== Entropie après "le" ===');
console.log(entropy(probs).toFixed(4), 'bits');

