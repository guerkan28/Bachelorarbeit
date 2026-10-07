# Trinkerkennung – Smartphone-Audio und Smartwatch-Sensordaten

Dieses Repository enthält den Quellcode des im Rahmen einer Bachelorarbeit entwickelten prototypischen Erfassungssystems sowie die Python-basierte Analysepipeline zur Erkennung von Trink- und Nicht-Trink-Aktivitäten.

Das System kombiniert:

- Smartphone-Audioaufnahmen,
- Beschleunigungs- und Gyroskopdaten einer Wear-OS-Smartwatch,
- eine nachgelagerte zeitliche Synchronisation,
- manuell annotierte Ground Truth auf Grundlage eines Referenzvideos,
- fensterbasierte Merkmalsextraktion,
- Audio-only-, Watch-only- und Feature-Level-Fusionsmodelle.

Die mobile Anwendung dient ausschließlich der Datenerfassung. Training und Evaluation der Modelle erfolgen offline in Python.

## Repository-Struktur

| Pfad | Inhalt |
| --- | --- |
| `Trinkerkennung/` | Android-Studio-Projekt mit `app/` für das Smartphone und `wear/` für die Wear-OS-Smartwatch |
| `trinkerkennung_analysis/` | Python-Analysepipeline und Evaluationsmodule |
| `tests/` | Automatisierte Tests |
| `build_windows.py` | Startskript für die 1-s- und 2-s-Fensterdatensätze |
| `build_features.py` | Startskript für Audio-, Watch- und Fusionsfeatures |
| `annotation_schema.json` | Maschinenlesbares Annotationsschema |
| `annotation_template.json` | Vorlage für Sitzungsannotation |
| `ANNOTATION_SCHEMA.md` | Beschreibung des Annotationsschemas |
| `WORKFLOW.md` | Vollständiger technischer Ablauf einschließlich Skriptparametern |
| `requirements.txt` | Python-Abhängigkeiten |

Alle für die Nutzung des Codes notwendigen technischen Schritte von der Installation der mobilen Anwendungen bis zur Modellbewertung sind in [`WORKFLOW.md`](WORKFLOW.md) beschrieben. Der reguläre Weg für eigene Aufnahmen steht in den Abschnitten 1–18; die Forschungsdaten der Bachelorarbeit werden dafür nicht benötigt.

Die Dateien `build_windows.py`, `build_features.py` und `WORKFLOW.md` liegen im Repository-Stamm neben `requirements.txt`. Die Startskripte verwenden die vorhandenen Verarbeitungsfunktionen aus `trinkerkennung_analysis`. Alle folgenden Python-Befehle werden aus dem Repository-Stamm ausgeführt.

## Voraussetzungen

### Mobile Erfassung

- Android Studio
- Android-Smartphone
- Wear-OS-Smartwatch
- gekoppelte Smartphone-/Smartwatch-Verbindung
- Android Debug Bridge (`adb`) für den späteren Datenexport

Das Android-Projekt verwendet zwei Module:

- `app` für das Smartphone
- `wear` für die Smartwatch

Die Gradle-Konfiguration verwendet Android SDK 36. Das Smartphone-Modul besitzt `minSdk 28`, das Wear-OS-Modul `minSdk 30`.

### Python-Analyse

Die Analyse wurde mit Python 3.13 verwendet. Die benötigten Bibliotheken sind in `requirements.txt` festgelegt.

Beispiel unter Windows PowerShell:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Mobile Anwendungen installieren

1. Den Ordner `Trinkerkennung` in Android Studio öffnen.
2. Gradle-Synchronisation vollständig abschließen lassen.
3. Smartphone mit Android Studio beziehungsweise ADB verbinden.
4. Das Modul `app` auf dem Smartphone ausführen.
5. Smartwatch mit Android Studio beziehungsweise ADB verbinden.
6. Das Modul `wear` auf der Smartwatch ausführen.

Die grundlegende Kopplung von Smartphone und Smartwatch ist Voraussetzung und wird nicht durch dieses Projekt eingerichtet.

## Kurzablauf einer Aufnahmesitzung

Die Smartphone-Anwendung steuert den regulären gemeinsamen Aufnahmeablauf.

1. Optional `PING an Smartwatch senden`, um die Verbindung zu prüfen.
2. `Neue Sitzung vorbereiten`.
3. Auf die bestätigte Bereitschaft der Smartwatch warten.
4. Erst nach erfolgreicher Vorbereitung das Referenzvideo starten.
5. `Gemeinsame Aufnahme starten`.
6. Synchronisationsmarker und die vorgesehene Aktivität durchführen.
7. `Gemeinsame Aufnahme stoppen`.
8. Den erfolgreichen Abschluss der Sitzung abwarten.
9. Referenzvideo stoppen.

Die manuellen Buttons `Aufnahme starten` und `Aufnahme stoppen` auf der Smartwatch dienen Entwicklungs- und Testzwecken. Im regulären Ablauf wird die Sensoraufnahme vom Smartphone gesteuert.

Die vollständige Beschreibung von Datenexport, Synchronisation, Annotation, Fensterbildung, Merkmalsextraktion und Evaluation befindet sich in [`WORKFLOW.md`](WORKFLOW.md).

## Erzeugte Dateien

Eine regulär abgeschlossene mobile Sitzung erzeugt:

```text
smartphone_<Session-UUID>_<Zeitstempel>.wav
watch_<Session-UUID>_<Zeitstempel>.csv
session_<Session-UUID>.json
```

Das Referenzvideo wird separat aufgezeichnet. Für die weitere Verarbeitung wird sein Dateiname aus dem zugehörigen Smartphone-WAV-Dateinamen abgeleitet. Dabei wird der Präfix `smartphone_` durch `reference_video_` ersetzt. Session-UUID und Smartphone-Zeitstempel bleiben unverändert:

```text
Smartphone-WAV:
smartphone_<Session-UUID>_<Smartphone-Zeitstempel>.wav

Referenzvideo:
reference_video_<Session-UUID>_<Smartphone-Zeitstempel>.mp4
```

Der Zeitstempel im Namen des Referenzvideos stammt somit aus dem Dateinamen der Smartphone-Audioaufnahme und nicht aus einem von der Kamera erzeugten Zeitstempel. Dadurch ist die Zuordnung des Referenzvideos zur zugehörigen Smartphone-Aufnahme unmittelbar am Dateinamen erkennbar.

## Analysepipeline

Die wesentlichen Verarbeitungsschritte sind:

1. Rohdaten exportieren und die Sitzung technisch validieren.
2. Watch- und Videozeiten auf die Audiozeitachse synchronisieren.
3. Ground Truth annotieren und die Annotation validieren.
4. Fensterdatensätze mit `build_windows.py` erzeugen.
5. Audio-, Watch- und Fusionsfeatures mit `build_features.py` berechnen.
6. Die Modelle personengetrennt trainieren und bewerten.

### Eigene Aufnahmen auswerten

Nachdem Export, Synchronisation und Annotation gemäß `WORKFLOW.md` abgeschlossen sind, können die Fenster und Features für die Standardablage erzeugt und die Modelle ausgewertet werden:

```powershell
python .\build_windows.py --data-root ".\data"
python .\build_features.py --data-root ".\data"
python -m trinkerkennung_analysis.pilot_model_evaluation `
    --feature-root ".\ML\02_Feature_Datasets" `
    --output-root ".\ML\03_Model_Evaluation"
```

Bei dieser Ablage liegen die finalen Annotationen direkt unter `data/annotations`; Rohdateien, Referenzvideos und Synchronisationsberichte müssen unter `data` eindeutig auffindbar sein. Die Fenster werden nach `ML/01_Window_Datasets` geschrieben, die sechs Featuredateien nach `ML/02_Feature_Datasets`.

Für andere Ablageorte werden die Startparameter angepasst, ohne den Python-Code zu bearbeiten. Abschnitt 16 des Workflows beschreibt `--data-root`, `--annotations-root` und `--output-root` für die Fensterbildung. Abschnitt 17 beschreibt `--data-root`, `--window-root`, `--output-root` und die optionale Zuordnung über `--participant-root` für die Featurebildung. Bei abweichenden Feature- und Ergebnisordnern entsprechend auch `--feature-root` und `--output-root` im Evaluationsaufruf anpassen.

Der Runner `pilot_model_evaluation` verwendet alle Teilnehmer-IDs der eigenen Featuredateien. Sein historischer Name und die Dateipräfixe `pilot_` erfordern keine separate Pilotphase. Die vorhandene Leave-One-Subject-Out-Auswertung benötigt mindestens drei verschiedene Personen; für die logistische Regression müssen beide Klassen `DRINK` und `NON_DRINK` in jedem Trainingsfold vorkommen. Aufnahmen und Annotationen so planen, dass auch nach der Fensterbildung beide Klassen ausreichend vertreten sind.

Die Auswertung erzeugt unter `ML/03_Model_Evaluation`:

```text
pilot_fold_results.csv
pilot_summary_results.csv
pilot_window_predictions.csv
pilot_window_errors.csv
```

Die Modelle werden in diesem Lauf trainiert und bewertet. Es wird dabei kein fertig trainiertes Modell für eine spätere direkte Klassifikation neuer Rohaufnahmen gespeichert.

### Studienspezifische Hauptauswertung

Der zusätzliche Hauptstudienrunner verwendet das Evaluationsdesign der Bachelorarbeit mit den festgelegten Testpersonen P104–P108. Die benötigte Dateibenennung lässt sich mit `build_features.py --naming main-study` erzeugen; die vollständigen Aufrufe und Voraussetzungen stehen in Abschnitt 19 des Workflows.

Die finale Hauptstudienauswertung wird mit folgendem Modul gestartet:

```powershell
python -m trinkerkennung_analysis.main_model_evaluation `
    --feature-root <PFAD_ZU_FEATURE_DATEIEN> `
    --output-root <AUSGABEORDNER>
```

Dabei entstehen:

```text
main_fold_results.csv
main_summary_results.csv
main_window_predictions.csv
main_window_errors.csv
```

## Tests

Die Python-Tests können aus dem Repository-Stamm ausgeführt werden:

```powershell
python -m unittest discover -s tests
```

## Forschungsdaten und Datenschutz

Das Repository enthält bewusst keine personenbezogenen oder pseudonymisiert zuordenbaren Forschungsrohdaten. Insbesondere werden nicht veröffentlicht:

- Referenzvideos,
- WAV-Aufnahmen,
- reale Smartwatch-CSV-Dateien,
- reale Sitzungsmetadaten,
- sitzungsspezifische Annotationen,
- sitzungsspezifische Synchronisationsberichte,
- session- oder teilnehmerbezogene Fensterdatensätze,
- session- oder teilnehmerbezogene Featuredateien,
- Einzelvorhersagen und Fehlklassifikationsdateien mit Teilnehmer-/Sessionbezug,
- Einwilligungserklärungen,
- Zuordnungen zwischen Teilnehmerkennungen und realen Personen.

Die im Code verwendeten Teilnehmerkennungen wie `P101` bis `P108` sind pseudonymisierte Kennungen ohne Namenszuordnung.

## Reproduzierbarkeit der berichteten Ergebnisse

Für die exakte Reproduktion der in der Bachelorarbeit berichteten Modellkennzahlen sind die separat aufbewahrten archivierten finalen Feature- und Ergebnisdateien des jeweiligen Auswertungsstands maßgeblich. Diese Forschungsdateien sind nicht Bestandteil des öffentlichen GitHub-Repositories.

Bei stochastischen Lernverfahren kann eine veränderte Zeilenreihenfolge trotz identischer Beobachtungen und festem `random_state` zu geringfügig anderen Modellergebnissen führen. Deshalb sollten für eine exakte Ergebnisreproduktion die archivierten finalen Featuretabellen unverändert verwendet werden.
