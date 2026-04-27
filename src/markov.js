import * as R from 'ramda';

// ─── Nettoyage du texte ──────────────────────────────────────────────────────

/**
 * Met en minuscules, enlève les accents, caractères spéciaux
 * et espaces multiples.
 */
const sanitizeText = (text) =>
  text
    .toLowerCase()
    .normalize('NFD')
    .replace(/[\u0300-\u036f]/g, '')
    .replace(/[^a-z ]/g, '')
    .replace(/\s+/g, ' ')
    .trim();

// ─── Entraînement ────────────────────────────────────────────────────────────

/**
 * Construit une table de bigrammes à partir d'un texte.
 * { "ab": { "c": 3, "d": 1 }, "bc": { "a": 2 } ... }
 *
 * Utilise R.reduce pour parcourir les caractères,
 * R.assocPath pour mettre à jour les comptes.
 */
export const buildBigrams = (text) => {
  const cleanText = sanitizeText(text);
  const chars = R.split('', cleanText);

  return R.reduce(
    (acc, i) => {
      const context = chars[i] + chars[i + 1];
      const next = chars[i + 2];
      if (!next) return acc;

      const current = R.pathOr(0, [context, next], acc);
      return R.assocPath([context, next], current + 1, acc);
    },
    {},
    R.range(0, chars.length - 2),
  );
};

// ─── Calcul des probabilités ─────────────────────────────────────────────────

/**
 * Convertit les comptes bruts d'un contexte en probabilités.
 * { "c": 3, "d": 1 } → { "c": 0.75, "d": 0.25 }
 *
 * Utilise R.map, R.values, R.sum pour normaliser.
 */
export const toProbabilities = (counts) => {
  const total = R.sum(R.values(counts));
  return R.map((count) => count / total, counts);
};

// ─── Prédiction ──────────────────────────────────────────────────────────────

/**
 * Retourne le top N des touches les plus probables pour un contexte donné.
 * Résultat : [{ key: 'e', prob: 0.42 }, { key: 's', prob: 0.18 }, ...]
 *
 * Utilise R.pipe, R.toPairs, R.sortWith, R.descend, R.take.
 */
export const getTopN = R.curry((n, bigrams, context) => {
  const cleanContext = sanitizeText(context).replace(/ /g, '');
  const counts = R.propOr({}, cleanContext, bigrams);

  if (R.isEmpty(counts)) return [];

  const probs = toProbabilities(counts);

  return R.pipe(
    R.toPairs,
    R.sortWith([R.descend(R.prop(1))]),
    R.take(n),
    R.map(([key, prob]) => ({ key, prob: +prob.toFixed(4) })),
  )(probs);
});

// ─── Fusion pré-entraîné + live ──────────────────────────────────────────────

/**
 * Fusionne deux tables de bigrammes avec un poids pour le live.
 * liveWeight = 0.7 → 70% live, 30% pré-entraîné.
 *
 * Utilise R.mergeWith pour combiner les comptes des deux modèles.
 */
export const mergeModels = R.curry((liveWeight, pretrained, live) => {
  const scale = (weight) => R.map(R.map((v) => v * weight));

  const weightedPretrained = scale(1 - liveWeight)(pretrained);
  const weightedLive = scale(liveWeight)(live);

  return R.mergeWith(
    R.mergeWith(R.add),
    weightedPretrained,
    weightedLive,
  );
});

// ─── Mise à jour live ────────────────────────────────────────────────────────

/**
 * Enregistre une nouvelle frappe dans le modèle live.
 * context = les 2 dernières touches, next = la touche actuelle.
 *
 * Utilise R.assocPath, R.pathOr.
 */
export const recordKeypress = R.curry((context, next, liveModel) => {
  const cleanContext = sanitizeText(context).replace(/ /g, '');
  const cleanNext = sanitizeText(next).replace(/ /g, '');

  if (cleanContext.length !== 2 || cleanNext.length !== 1) {
    return liveModel;
  }

  const current = R.pathOr(0, [cleanContext, cleanNext], liveModel);
  return R.assocPath([cleanContext, cleanNext], current + 1, liveModel);
});

// ─── Entropie ────────────────────────────────────────────────────────────────

/**
 * Calcule l'entropie de Shannon d'une distribution de probabilités.
 * Plus l'entropie est haute, plus le modèle est incertain.
 *
 * Utilise R.pipe, R.values, R.reduce.
 */
export const entropy = R.pipe(
  R.values,
  R.reduce((acc, p) => (p > 0 ? acc - p * Math.log2(p) : acc), 0),
);

// ─── Helper : hello ──────────────────────────────────────────────────────────

export const hello = (name = 'world') => `Hello, ${name}!`;
