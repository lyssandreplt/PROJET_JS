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
