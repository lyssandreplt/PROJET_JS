# PROJET_JS
NOM : Clara Herbert & Lyssandre Poulet

## Sujet
Prédiction de lettre sur un clavier de téléphone à l’aide d’une chaîne de Markov.

## Objectif
Le programme apprend les transitions entre lettres à partir d’un corpus de texte, puis estime la probabilité de la touche suivante selon la touche actuelle.

## Technologies
- JavaScript
- Node.js
- Ramda
- Biome

## Structure
- `src/data.js` : corpus d’entraînement
- `src/markov.js` : logique Markov
- `src/index.js` : exécution principale

## Lancer le projet
```bash
npm install
npm start
