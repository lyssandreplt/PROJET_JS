import * as R from 'ramda';
import { CORPUS as fichier } from './data.js';

//------NETTOYAGE----------
const dico= R.pipe(
  R.invoker(1, 'normalize')('NFD'), // On enlève les accents
  R.replace(/[\u0300-\u036f]/g, ''),
  R.replace(/[#^w+$#'-]/g,' '), // on ne fait pas (/-/g,' ', fichier), car on appel fichier a la fin
  R.replace(/[^a-zA-Z\s$]/g, ''), //Enlever les caractères spéciaux mais garder les espaces
  R.replace(/[A-Z]/g, R.toLower), //Majuscules en minuscules
  R.split(' ') //Faire le dico
);

const dictionnaire = dico(fichier);

//...........MARKOV.....................
const ajouterProb = (dico, lettre1, lettre2) =>
  R.assocPath(
    [lettre1, lettre2],
    R.pathOr(0, [lettre1, lettre2], dico) + 1, //nouveau score (ancien score + 1)
    dico); //renvoi le new dico

const listeLettre = (dico, word) => {
  const lettres = R.split('', word);

  //acceder a la position de la lettre (addIndex)
  return R.addIndex(R.reduce)(
    (tempDico, lettreNow, idx) => {
      const nextLettre = R.nth(idx + 1, lettres);
      return nextLettre ? ajouterProb(tempDico, lettreNow, nextLettre) : tempDico; //if condition ? vrai : faux
    }, //fonction
    dico, //depart
    lettres);//données
};

const dicLettres = R.reduce(listeLettre, {})(dictionnaire);
console.log(dicLettres);

// ───── Probabilités ─────

const toProbabilities = (counts) => {
  const total = R.sum(R.values(counts));
  return total === 0 ? {} : R.map((count) => count / total, counts);
};

const dicProbabilites = R.map(toProbabilities, dicLettres);

// ───── Nettoyage de l'entrée utilisateur ─────

const nettoyerEntree = R.pipe(
  R.invoker(1, 'normalize')('NFD'),
  R.replace(/[\u0300-\u036f]/g, ''),
  R.replace(/[#^w+$#'-]/g, ' '),
  R.replace(/[^a-zA-Z\s$]/g, ''),
  R.replace(/[A-Z]/g, R.toLower),
  R.replace(/\s+/g, ''),
);

// ───── Top N des lettres probables ─────

const getTopN = (dicoProb, lettre, n = 5) => {
  const probs = R.propOr({}, lettre, dicoProb);

  if (R.isEmpty(probs)) {
    return [];
  }

  return R.pipe(
    R.toPairs,
    R.sortWith([R.descend(R.prop(1))]),
    R.take(n),
    R.map(([key, prob]) => ({ key, prob: +prob.toFixed(4) })),
  )(probs);
};

// ───── Prédiction de la lettre suivante ─────

const predictNextLetter = (texte, n = 5) => {
  const texteNettoye = nettoyerEntree(texte);

  if (texteNettoye.length < 1) {
    return [];
  }

  const derniereLettre = R.last(R.split('', texteNettoye));
  return getTopN(dicProbabilites, derniereLettre, n);
};

// ───── Test ─────

const typedText = 'b';
const predictions = predictNextLetter(typedText, 5);

console.log(`Lettre entrée : "${typedText}"`);

if (predictions.length === 0) {
  console.log('Aucune prédiction disponible.');
} else {
  console.log('Lettres suivantes probables :');

  predictions.forEach(({ key, prob }, index) => {
    console.log(`${index + 1}. ${key} -> ${(prob * 100).toFixed(2)} %`);
  });

  console.log(
    `\nLettre prédite : ${predictions[0].key} (${(predictions[0].prob * 100).toFixed(2)} %)`
  );
}
