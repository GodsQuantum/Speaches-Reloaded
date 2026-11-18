# ASR français — benchmark indépendant Google FLEURS (18 novembre 2026)

## Source et sélection

Corpus officiel Google FLEURS, configuration fr_fr, partition test, commit 70bb2e84b976b7e960aa89f1c648e09c59f894dd, indépendant des moteurs comparés. Les transcriptions sont celles du corpus, non celles d'un SRT AutoSubs, d'un film ni de Whisper.

- TSV source : SHA-256 5d06d338b242e00786fcf12c4c92008b9f399d5a5c872c91dca90572e7869c0d.
- Archive audio officielle : SHA-256 d23690e102f373554d1b544cd2ff1e76e4fedeb04953c0b72751a1b7c518cfdd.
- Sélection pseudo-aléatoire fixée avant les mesures (graines 20261118 et 20261119), 60 phrases uniques : 30 voix féminines et 30 masculines. Durée totale 594,72 secondes, 1 531 mots de référence.
- Les transcriptions et les empreintes de chaque enregistrement sont conservées dans manifest.json. Les binaires audio officiels ne sont pas dupliqués dans Git.
- Métrique : WER micro corpus après normalisation Unicode NFKC, minuscules, uniformisation des apostrophes/tirets, tokenisation Unicode, Levenshtein des mots ; comparaisons appariées. Les références officielles peuvent avoir des imperfections ; le corpus ne fournit pas d'annotations temporelles humaines mot par mot.

## Comparatif qualité (mêmes 60 enregistrements)

| Mesure | Parakeet Ultra GGUF | Whisper Turbo CT2 DeepML |
|---|---:|---:|
| Transcriptions textuelles obtenues | 60/60 | 60/60 |
| Erreurs / 1531 mots officiels | 80 | 82 |
| WER | 5,23 % | 5,36 % |
| Clips où il est strictement meilleur | 16 | 10 |
| Clips égaux | 34 | 34 |
| Demandes HTTP de timestamps réussies | 60/60 | 59/60 |
| Timestamps structurellement invalides parmi les succès | 0 | 0 |

Écart de WER CT2 moins Parakeet : +0,131 point. Intervalle de confiance bootstrap apparié à 95 % : [-1,601 ; +1,597] point (50 000 rééchantillonnages, graine 20261118). La différence de qualité n'est donc pas statistiquement démontrée sur ce corpus.

Comparatif de latence sur les 59 mêmes requêtes avec timestamps réussies des deux côtés : Parakeet 69,43 secondes cumulées ; CT2 575,51 secondes ; Parakeet est environ 8,29 fois plus rapide. Ce chiffre inclut le temps HTTP local et l'alignement acoustique. Les modèles étaient présents dans Speaches CT410 (pas de temps de téléchargement des poids).

Le corpus contient de la parole française lue : ne pas extrapoler le WER à des conversations de stand-up ou à une consultation dentaire. Aucun classement qualitatif n'est calculé à partir des SRT historiques AutoSubs ; Teddy Riner n'est pas un Reel corrigé manuellement.

## Échec unique de timestamp CT2 et correction source vérifiée

Échantillon 1960, fichier 6350573990637191461.wav : source officielle WAV IEEE float32, 16 kHz mono. Le lecteur Python wave de l'aligneur strict rejette le format audio 3. Sa conversion à débit et durée inchangés en PCM16 révèle aussi un deuxième problème : le moteur CrispASR retourne le guillemet fermant isolé comme pseudo-mot terminal, en plus des 40 vrais mots.

- La transcription CT2 de l'original est valide via réponse HTTP 200 sans word timestamps : 3 erreurs sur 39 mots officiels. Elle est conservée séparément dans ct2_sample_1960_text_only.json et contribue au WER 60/60.
- La requête de timestamps CT2 de production a renvoyé HTTP 503, sans aucune fausse attribution de temps. Ce statut 59/60 est conservé dans results.jsonl.
- Patch du code source Cloud9-Speaches : normalisation locale bornée float32 -> PCM16 sans rééchantillonnage ; les seules ponctuations finales isolées peuvent être ignorées comme non acoustiques ; les mots supplémentaires restent rejetés.
- Validation en sandbox avec le module modifié et le même original float32 : 40/40 mots CT2 horodatés, aucune durée nulle, aucune violation monotone, interpolation désactivée ; environ 0,74 s. Parakeet sur le même échantillon converti : 40/40 mots, aucune anomalie.
- Ce patch est vérifié hors production, mais ne figure pas encore dans l'image Docker Speaches en cours : ne pas prétendre que le correctif HTTP a déjà été déployé.

## Fichiers

- manifest.json : références officielles, choix fixes, identifiants, origines et SHA des audios.
- results.jsonl : données brutes des 120 requêtes de timestamps (dont un HTTP 503 CT2 documenté).
- ct2_sample_1960_text_only.json : sortie CT2 indépendante, sans timestamps, pour l'analyse WER complète.
- summary.json : mesures finales, comparaison appariée, contrôle des durées et résultat de confiance.

Modèles : handy-computer/parakeet-ultra-gguf et deepdml/faster-whisper-large-v3-turbo-ct2. API OpenAI-compatible /v1/audio/transcriptions, fr, verbose_json avec timestamp_granularities[]=word. Les contrôles de timestamps portent sur la validité structurelle, pas une précision temporelle absolue sans annotations humaines.
