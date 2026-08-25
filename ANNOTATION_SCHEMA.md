# Annotationsschema Version 1

## Zweck

`annotation_<UUID>.json` dokumentiert beobachtbare Ground Truth. Die Datei enthält keine Fenster-, Feature-, Modell- oder Fusionsparameter.

Die kanonische Zeitreferenz lautet:

```text
AUDIO_SECONDS_FROM_WAV_START
```

Sekunde `0.0` entspricht dem Beginn der Smartphone-WAV-Datei. Watch- und Videozeiten werden ausschließlich über `synchronization_report_<UUID>.json` auf diese Achse transformiert.

## Trennung der Ebenen

1. **Ground Truth:** `drink_events`, Mundkontakt, negative, ausgeschlossene und unsichere Intervalle.
2. **Abgeleitete Modelllabels:** beispielsweise `FULL_EVENT`, `MOUTH_CONTACT` oder `MOUTH_CONTACT_WITH_CONTEXT`; noch nicht Bestandteil dieses Schemas.
3. **Experimentkonfiguration:** Fensterlänge, Überlappung, Features, Modelle und Fusion; nicht Bestandteil dieses Schemas.

## Intervallsemantik

Alle Intervalle sind halb offen:

```text
[start_seconds, end_seconds)
```

Der Start gehört zum Intervall, das Ende nicht. Direkt aufeinanderfolgende Intervalle überlappen dadurch nicht.

## DRINK_EVENT

Ein bestätigtes `DRINK_EVENT` enthält die äußeren beobachtbaren Bewegungsgrenzen und mindestens ein sichtbares `mouth_contact_interval`. Der Begriff dokumentiert Mundkontakt, nicht den physiologischen Schluckzeitpunkt.

Mehrere sichtbare Mundkontakte in einem durchgehenden Bewegungszyklus werden als mehrere Kontaktintervalle innerhalb eines Ereignisses gespeichert. Zwei getrennte Bewegungszyklen zum Mund sind zwei Ereignisse.

## Fachliche Zustände

- `drink_events`: bestätigte positive Ereignisse.
- `negative_intervals`: explizit beobachtete Nicht-Trink-Aktivitäten.
- `excluded_intervals`: technisch oder methodisch unbrauchbare Abschnitte.
- `uncertain_intervals`: überprüfte, aber nicht sicher einzuordnende Abschnitte.
- `UNLABELED`: impliziter Status aller nicht explizit annotierten Zeiten; kein eigenes Array.

Nicht annotierte Zeit wird niemals automatisch zu `NOT_DRINKING`.

## Disjunktheitsregel

Die vier expliziten fachlichen Strukturen müssen auf der gemeinsamen Audiozeitachse disjunkt sein. Diese technische Regel erhält die fachliche Aussage, weil ein Zeitraum dadurch genau einen primären Verarbeitungsstatus besitzt. Überlagerte oder nicht eindeutig trennbare Aktivitäten werden als `UNCERTAIN` mit `OVERLAPPING_ACTIVITIES` dokumentiert. Mundkontaktintervalle liegen als Unterstruktur innerhalb ihres `DRINK_EVENT` und sind von dieser Top-Level-Regel ausgenommen.

## Ground Truth und Video

Für `SYNCHRONIZED_VIDEO` müssen gelten:

- Referenzvideo mit derselben Session-ID vorhanden,
- Video nicht im Repository,
- Video-zu-Audio-Abbildung im Synchronisationsbericht vorhanden,
- lineare Abbildung aus Anfangs- und Endmarker,
- Annotation enthält keine duplizierten Offset- oder Driftwerte.

## Personengetrennte Evaluation

`participant_id` wird in jeder späteren Fensterzeile erhalten. Trainings- und Testdaten derselben Person dürfen nicht vermischt werden. Fensterbildung kann technisch vor oder nach der Fold-Zuordnung erfolgen; die gruppierte Trennung und alle lernenden Verarbeitungsschritte müssen testpersonenfrei bleiben.
