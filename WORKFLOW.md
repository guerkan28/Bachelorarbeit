# Technischer Workflow – von der Datenerfassung bis zur Modellbewertung

Dieses Dokument beschreibt, wie eigene Aufnahmen mit dem vorliegenden GitHub-Code erfasst, exportiert, synchronisiert, anhand eines Referenzvideos annotiert und anschließend für Training und Bewertung der ML-Modelle verwendet werden. Vorhandene Forschungsdaten der Bachelorarbeit werden für diesen Ablauf nicht benötigt.

Die folgenden Schritte beschreiben den regulären Analysepfad des vorliegenden Codes. Historische Arbeitsstand-Dokumente sind keine Voraussetzung für seine Ausführung. Das Annotationsschema und das Template ergänzen die Anleitung.

Die beiden Hilfsskripte `build_windows.py` und `build_features.py` liegen im Repository-Stamm neben `requirements.txt`. Ihre Datenpfade werden über Startparameter gesetzt; die Verarbeitungsschritte stehen in Abschnitt 16–17.

Die Shell-Beispiele verwenden **Windows und PowerShell** und werden aus dem Repository-Stamm ausgeführt. Platzhalter in `<...>` müssen vor dem Ausführen ersetzt werden. Zeitwerte werden in Sekunden mit Dezimalpunkt angegeben, beispielsweise `12.345`.

Der Hauptweg für eigene Daten sind die Abschnitte 1–18. Dabei entstehen die WAV-Dateien, Watch-CSVs und Sitzungsmetadaten durch die eigenen Aufnahmen; Referenzvideos werden selbst aufgenommen, Synchronisationsberichte erzeugt und Annotationen selbst erstellt. Die studienspezifische Hauptauswertung und das vorhandene Ergebnisarchiv werden ergänzend in Abschnitt 19–20 beschrieben.

Für die vorhandene personengetrennte ML-Auswertung werden **mindestens drei verschiedene Personen** benötigt. Jede Person erhält ein eigenes Pseudonym wie `P001`, `P002` oder `P003`, das über ihre Sessions hinweg gleich bleibt. Eine Person darf nicht zur Erfüllung dieser Mindestzahl unter mehreren Pseudonymen geführt werden. Trink- und Nicht-Trink-Aktivitäten aufnehmen und annotieren; für ein sinnvolles Evaluationsdesign beide Klassen bei jeder Person erheben. Für die logistische Regression müssen in jedem Trainingsfold beide Klassen vorhanden sein. Auch nach der 2-s-Fensterbildung müssen dafür ausreichend lange positive und negative Intervalle übrig bleiben.

Die Modelle werden innerhalb der Evaluation auf den eigenen annotierten Daten trainiert und an jeweils zurückgehaltenen Personen bewertet. Der Code enthält keinen beschriebenen Ablauf, bei dem ein bereits fertiges Modell unmittelbar eine einzelne neue Rohaufnahme klassifiziert. Aufnahmen einer einzelnen Person lassen sich vorbereiten und in Features überführen, erfüllen aber nicht die Mindestanforderung der implementierten LOSO-Auswertung.

## 1. Voraussetzungen

Benötigt werden:

- Android-Smartphone mit mindestens API-Level 28
- Wear-OS-Smartwatch mit mindestens API-Level 30
- gekoppelte Smartphone-/Smartwatch-Verbindung
- Android Studio
- Android Debug Bridge (`adb`)
- Python 3.13 und die in `requirements.txt` festgeschriebenen Paketversionen
- das vorliegende Repository
- eine Kamera beziehungsweise ein weiteres Gerät für das Referenzvideo

Die Smartwatch muss Beschleunigungssensor und Gyroskop bereitstellen. Die Android-Module konfigurieren `compileSdk` API 36 mit Minor-API-Level 1. Für den Gradle-Daemon ist im Projekt eine Java-21-Toolchain konfiguriert. Diese Projekteinstellungen bei der Android-Studio-Synchronisation berücksichtigen; die Java-11-Quell-/Zielkompatibilität der Module ist davon zu unterscheiden.

## 2. Python-Umgebung vorbereiten

Aus dem Repository-Stamm:

```powershell
py -3.13 -m venv .venv
.\.venv\Scripts\Activate.ps1

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Anschließend die Installation und die Testsuite prüfen:

```powershell
python --version
python -m pip check
python -m unittest discover -s tests
```

Die Ausgabe von `python --version` muss zur vorgesehenen Python-3.13-Umgebung gehören. Importfehler, etwa wegen fehlendem `librosa`, sind keine erfolgreiche Testsuite. Für den vollständigen Analyselauf müssen alle in `requirements.txt` genannten Pakete installiert sein.

## 3. Smartphone- und Watch-App installieren

### 3.1 Android-Projekt öffnen

In Android Studio den Ordner

```text
Trinkerkennung
```

öffnen und die Gradle-Synchronisation abschließen lassen.

Das Projekt enthält zwei Android-Module:

```text
app     Smartphone-Anwendung
wear    Wear-OS-Anwendung
```

### 3.2 Smartphone-App installieren

1. Smartphone per USB oder unterstützter ADB-Verbindung mit dem Entwicklungsrechner verbinden.
2. In Android Studio das Modul `app` auswählen.
3. Das Smartphone als Zielgerät auswählen.
4. Die Anwendung über `Run` installieren und starten.

Beim ersten Start muss der Mikrofonzugriff erlaubt werden.

### 3.3 Wear-OS-App installieren

1. Smartwatch für ADB-Debugging verfügbar machen.
2. Die Smartwatch in Android Studio beziehungsweise `adb devices -l` als Gerät verfügbar machen.
3. Das Modul `wear` auswählen.
4. Die Smartwatch als Zielgerät auswählen.
5. Die Anwendung über `Run` installieren und starten.

Die Kopplung zwischen Smartphone und Smartwatch selbst wird nicht von diesem Projekt durchgeführt. Beide Geräte müssen bereits miteinander verbunden sein.

## 4. Bedienung der mobilen Anwendungen

### 4.1 Smartphone-App

Die Smartphone-App enthält im regulären Aufnahmebereich vier zentrale Aktionen:

```text
PING an Smartwatch senden
Neue Sitzung vorbereiten
Gemeinsame Aufnahme starten
Gemeinsame Aufnahme stoppen
```

`PING an Smartwatch senden` ist eine optionale technische Verbindungsprüfung und kein notwendiger Bestandteil jeder Aufnahmesitzung.

### 4.2 Smartwatch-App

Die Wear-OS-App zeigt aktuelle Sensorwerte sowie manuelle Start- und Stoppbuttons.

```text
Aufnahme starten
Aufnahme stoppen
```

Diese beiden Buttons dienen Entwicklungs- und Testzwecken. Im regulären Erhebungsablauf wird die Watch-Aufzeichnung durch die Smartphone-App gesteuert.

## 5. Durchführung einer kontrollierten Aufnahmesitzung

Dieser Abschnitt beschreibt den Grundablauf einer eigenen Aufnahme. Die unten aufgeführten A–I-Sitzungen dokumentieren das Erhebungsdesign der Bachelorarbeit und können als Vorlage für eine neue Erhebung verwendet werden. Für neue Personen eigene Pseudonyme und die neu erzeugten Session-UUIDs verwenden. Die mobile Anwendung kennt die späteren Klassen `DRINK` und `NON_DRINK` nicht. Sie zeichnet lediglich Smartphone-Audio und Watch-Sensordaten unter einer gemeinsamen Session-UUID auf. Die fachliche Zuordnung erfolgt erst nachträglich anhand des Referenzvideos.

### 5.1 Versuchsaufbau

Für die kontrollierte Datenerhebung wurde der Aufbau vor Beginn einer Person möglichst konstant gehalten:

- Die Smartwatch wurde an der Hand getragen, mit der die jeweilige Aktivität ausgeführt wurde.
- Das Smartphone lag bei den regulären kontrollierten Sitzungen an einer festgelegten Position rechts auf dem Tisch.
- Die Smartphoneposition wurde nicht abhängig von der Trinkhand verschoben.
- Für die Hauptuntersuchung wurden dasselbe Trinkglas sowie pro Person eine neue 1,5-l-PET-Flasche desselben Typs verwendet.
- Die Kamera wurde so ausgerichtet, dass Oberkörper, Aktivitätshand, Gefäß und Mundbereich für die spätere Annotation erkennbar waren.
- Während der kontrollierten Aufnahme wurde außerhalb der vorgesehenen Aktivitäten nach Möglichkeit nicht gesprochen.

Vor jeder Aufnahme wurden Smartphone- und Watch-App geöffnet und die Verbindung geprüft. Bei Bedarf konnte über

```text
PING an Smartwatch senden
```

die Kommunikation getestet werden.

### 5.2 Exakter Grundablauf einer Aufnahme

Jede kontrollierte Sitzung folgte demselben technischen Grundschema:

1. Auf dem Smartphone **`Neue Sitzung vorbereiten`** auswählen.
2. Die bestätigte Bereitschaft der Smartwatch abwarten. Erst nach der READY-Bestätigung wird mit der eigentlichen Aufzeichnung fortgefahren.
3. **Referenzvideo starten.**
4. Auf dem Smartphone **`Gemeinsame Aufnahme starten`** auswählen.
5. **3–5 s ruhig warten.**
6. Mit der Watch-Hand **drei Synchronisationsklopfer** erzeugen.
7. **2–3 s ruhig warten.**
8. Die für die jeweilige Sitzung vorgesehene Aktivität beziehungsweise Aktivitätsfolge durchführen.
9. Bei getrennten Wiederholungen zwischen den einzelnen Aktivitäten nach Möglichkeit **5–8 s neutrale Pause** einhalten.
10. Nach der letzten fachlichen Aktivität **2–3 s ruhig warten.**
11. Mit der Watch-Hand erneut **drei Synchronisationsklopfer** erzeugen.
12. **3–5 s ruhig warten.**
13. Auf dem Smartphone **`Gemeinsame Aufnahme stoppen`** auswählen und den erfolgreichen Sitzungsabschluss abwarten.
14. **Referenzvideo stoppen.**
15. Die erzeugte Session-UUID für die weitere Zuordnung der Dateien dokumentieren.

Das Referenzvideo wird erst nach erfolgreicher Vorbereitung der Sitzung gestartet und läuft anschließend während der vollständigen Smartphone-/Watch-Aufnahme einschließlich der Start- und Endklopfmarker.

### 5.3 Technischer Ablauf beim Start

Nach Auswahl von

```text
Gemeinsame Aufnahme starten
```

läuft intern folgender Ablauf ab:

```text
Smartphone-Audio startet
→ START wird an die Smartwatch gesendet
→ Smartwatch startet die Sensoraufnahme
→ STARTED wird bestätigt
→ gemeinsame Sitzung läuft
```

Die Smartphone-Audioaufnahme beginnt technisch vor der Watch-Aufnahme. Die spätere zeitliche Ausrichtung erfolgt deshalb nicht über den absoluten Startzeitpunkt der Dateien, sondern über die aufgezeichneten Synchronisationsmarker.

### 5.4 Synchronisationsmarker

Die drei Klopfimpulse am Anfang und die drei Klopfimpulse am Ende einer Sitzung werden mit der Watch-Hand erzeugt. Sie müssen sowohl

- im Smartphone-Audiosignal als auch
- im Beschleunigungssignal der Smartwatch

erkennbar sein.

Die Marker dienen ausschließlich der späteren zeitlichen Ausrichtung der Datenquellen. Sie werden als ausgeschlossene Bereiche behandelt und nicht als Trainings- oder Testfenster für die Klassifikation verwendet.

### 5.5 In der Hauptuntersuchung verwendete Sitzungen

Die App selbst erzwingt keine bestimmte Versuchsanordnung. Für die Bachelorarbeit wurde der oben beschriebene technische Grundablauf jedoch mit den folgenden kontrollierten Sitzungen verwendet:

| Sitzung | Inhalt |
| --- | --- |
| A | `DRINK_FROM_GLASS`: 5 vollständige Trinkvorgänge mit Glas |
| B | `DRINK_FROM_BOTTLE`: 5 vollständige Trinkvorgänge mit Flasche |
| C | `LIFT_CONTAINER_WITHOUT_DRINKING`: 3 × Glas und 3 × Flasche anheben, ohne zu trinken |
| D | `MOVE_CONTAINER_WITHOUT_DRINKING`: 3 × Glas und 3 × Flasche seitlich versetzen, ohne zu trinken |
| E | `HAND_TO_FACE_WITHOUT_CONTAINER`: 5 Hand-zu-Gesicht-Bewegungen ohne Gefäß |
| F | `REST`: ungefähr 30–45 s kontrollierte Ruhe |
| G | `CONTINUOUS_MIXED_ACTIVITY`: Trink- und Nicht-Trink-Aktivitäten innerhalb einer kontinuierlichen Aufnahme |
| H | Grenzfall mit gehaltener Flasche und 3 getrennten Trinkvorgängen |
| I | zusätzliche Negativaktivitäten `SMARTPHONE_USE`, `COUGHING` und `ARM_MOVEMENT_WITHOUT_CONTAINER` |

Die Sitzungen A bis H bildeten das ursprüngliche Hauptstudienprotokoll. Sitzung I wurde im Verlauf der Hauptdatenerhebung als zusätzliche Negativsitzung ergänzt. Sie wurde für alle fünf Hauptstudienpersonen P104 bis P108 erhoben. Die Pilotpersonen P101 bis P103 wurden dafür nicht erneut aufgenommen.

#### Sitzung G – kontinuierlich gemischte Aktivität

Sitzung G kombinierte positive und negative Aktivitäten innerhalb einer einzelnen laufenden Aufnahme. Verwendet wurden insbesondere:

```text
DRINK_FROM_GLASS
DRINK_FROM_BOTTLE
LIFT_CONTAINER_WITHOUT_DRINKING mit Flasche
MOVE_CONTAINER_WITHOUT_DRINKING mit Glas
HAND_TO_FACE_WITHOUT_CONTAINER
```

Die Reihenfolge war pro Hauptstudienperson vorab festgelegt und wurde nicht während der Aufnahme spontan angepasst. Zu Beginn und am Ende der fachlichen Aktivitätsfolge wurden jeweils ungefähr **8–10 s REST** eingehalten. Zwischen den einzelnen Aktivitäten lagen ungefähr **5–8 s neutrale Ruhe**.

Die festgelegten Reihenfolgen waren:

| Person | Reihenfolge zwischen den beiden REST-Abschnitten |
| --- | --- |
| P104 | Glas trinken → Flasche anheben ohne Trinken → Hand zum Gesicht ohne Gefäß → Flasche trinken → Glas versetzen ohne Trinken |
| P105 | Glas versetzen ohne Trinken → Flasche trinken → Hand zum Gesicht ohne Gefäß → Flasche anheben ohne Trinken → Glas trinken |
| P106 | Hand zum Gesicht ohne Gefäß → Glas trinken → Glas versetzen ohne Trinken → Flasche trinken → Flasche anheben ohne Trinken |
| P107 | Flasche anheben ohne Trinken → Glas versetzen ohne Trinken → Glas trinken → Hand zum Gesicht ohne Gefäß → Flasche trinken |
| P108 | Flasche trinken → Hand zum Gesicht ohne Gefäß → Flasche anheben ohne Trinken → Glas versetzen ohne Trinken → Glas trinken |

In den archivierten Annotationen verwenden G und I `session_mode="CONTROLLED_MIXED_ACTIVITY"`, A bis F `"CONTROLLED_SINGLE_ACTIVITY"` und H `"EDGE_CASE"`. Die Beschreibung „kontinuierlich“ bei G bezeichnet die durchgehende Aufnahme; `CONTINUOUS_MIXED_ACTIVITY` ist zusätzlich ein zulässiger Schemawert, wurde für diese finalen G-Annotationen jedoch nicht verwendet.

#### Sitzung H – gehaltene Flasche

In Sitzung H wurde die Flasche nur einmal aufgenommen und über drei getrennte Trinkvorgänge hinweg in der Hand behalten:

```text
Flasche aufnehmen
→ 3–5 s ruhig halten
→ Trinkvorgang D001
→ Flasche vom Mund entfernen und abgesenkt weiter halten
→ 4–6 s ruhig halten
→ Trinkvorgang D002
→ Flasche vom Mund entfernen und abgesenkt weiter halten
→ 4–6 s ruhig halten
→ Trinkvorgang D003
→ Flasche vom Mund entfernen und abgesenkt weiter halten
→ 3–5 s ruhig halten
→ Flasche abstellen
```

D001 bis D003 wurden als drei getrennte Bewegungs- und Trinkzyklen ausgeführt. Das anfängliche Aufnehmen und das abschließende Abstellen der Flasche gehören nicht zu den drei Trinkereignissen.

#### Sitzung I – zusätzliche Negativaktivitäten

Sitzung I enthielt ausschließlich zusätzliche Nicht-Trink-Aktivitäten. Der verwendete Aufbau war:

```text
SMARTPHONE_USE
→ Smartphone etwa 20–30 s mit der Nicht-Watch-Hand halten
→ Smartphone mit der Watch-Hand bedienen

COUGHING
→ 3 einzelne Hustenepisoden

ARM_MOVEMENT_WITHOUT_CONTAINER
→ 5 gezielte Armbewegungen ohne Gefäß
→ Bewegung ungefähr bis Brust- beziehungsweise Schulterhöhe
```

In den final erhobenen Sitzungen wurde diese Reihenfolge als `SMARTPHONE_USE` → `COUGHING` → `ARM_MOVEMENT_WITHOUT_CONTAINER` umgesetzt.

### 5.6 Technischer Ablauf beim Stoppen

Nach Auswahl von

```text
Gemeinsame Aufnahme stoppen
```

läuft intern folgender Ablauf ab:

```text
STOP wird an die Smartwatch gesendet
→ Smartwatch beendet die Sensoraufnahme
→ STOPPED wird bestätigt
→ Smartphone beendet die Audioaufnahme
→ Metadatendatei wird finalisiert
```

Wenn die erwartete STARTED- oder STOPPED-Bestätigung nach einem erfolgreich übertragenen Befehl ausbleibt, verwendet die Smartphone-Anwendung eine maximale Wartezeit von zehn Sekunden und behandelt die Sitzung anschließend als technisch fehlgeschlagen.

### 5.7 Umgang mit Abweichungen

Abgebrochene, missverstandene oder technisch unvollständige Aktivitäten werden nicht nachträglich als reguläre Ereignisse behandelt. Auffälligkeiten werden bei der späteren Qualitätskontrolle und Annotation berücksichtigt.

Nicht eindeutig zuordenbare fachliche Bereiche werden als `UNCERTAIN` mit einem zulässigen semantischen Grund dokumentiert, beispielsweise `ABORTED_ACTIVITY`, `EVENT_BOUNDARY_NOT_VISIBLE` oder `OTHER_SEMANTIC_UNCERTAINTY`. Technisch oder methodisch ausgeschlossene Bereiche werden als `EXCLUDED` mit einem dafür vorgesehenen Grund gekennzeichnet, beispielsweise `INVALID_SIGNAL`, `EXPERIMENT_INSTRUCTION` oder `SYNCHRONIZATION_MARKER`. Ein semantisch unsicherer Vorgang wird nicht allein deshalb zu einem technischen Artefakt.

Natürliche Unterschiede in Bewegungsablauf, Bewegungsgeschwindigkeit, Körperhaltung oder Trinkhand stellen innerhalb des vorgesehenen Versuchsrahmens nicht automatisch einen Ausschlussgrund dar.

## 6. Erzeugte mobile Dateien

Eine erfolgreich abgeschlossene Sitzung erzeugt drei zentrale mobile Dateien.

### Smartphone

```text
smartphone_<Session-UUID>_<Zeitstempel>.wav
session_<Session-UUID>.json
```

### Smartwatch

```text
watch_<Session-UUID>_<Zeitstempel>.csv
```

Die Session-UUID verbindet alle Dateien derselben Sitzung.

Die Smartphone-Anwendung zeichnet Mono-Audio mit 48 kHz und 16-Bit-PCM auf. Die Smartwatch speichert Beschleunigungs- und Gyroskopereignisse mit ihren tatsächlichen Zeitstempeln.

## 7. Dateien per ADB auf den Analysecomputer übertragen

### 7.1 Geräte anzeigen

```powershell
$adb = "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe"

& $adb start-server
& $adb devices -l
```

Smartphone und Watch müssen als `device` erscheinen.

Bei einer Smartwatch mit drahtlosem ADB kann gegebenenfalls zunächst eine ADB-Kopplung beziehungsweise ADB-Verbindung erforderlich sein.

### 7.2 App-spezifische Remote-Pfade

Smartphone-Audio:

```text
/sdcard/Android/data/de.bachelorarbeit.trinkerkennung/files/Music/recordings
```

Smartphone-Metadaten:

```text
/sdcard/Android/data/de.bachelorarbeit.trinkerkennung/files/Documents/session_metadata
```

Watch-Sensordaten:

```text
/sdcard/Android/data/de.bachelorarbeit.trinkerkennung/files/Documents/sensor_recordings
```

### 7.3 Beispielvariablen

```powershell
$phone = "<ADB-ID-DES-SMARTPHONES>"
$watch = "<ADB-ID-DER-SMARTWATCH>"
$sessionId = "<SESSION-UUID>"

$phoneRemote = "/sdcard/Android/data/de.bachelorarbeit.trinkerkennung/files/Music/recordings"
$metadataRemote = "/sdcard/Android/data/de.bachelorarbeit.trinkerkennung/files/Documents/session_metadata"
$watchRemote = "/sdcard/Android/data/de.bachelorarbeit.trinkerkennung/files/Documents/sensor_recordings"
```

### 7.4 Dateien einer Sitzung suchen

```powershell
& $adb -s $phone shell "ls -1 '$phoneRemote'/smartphone_$sessionId*.wav"
& $adb -s $phone shell "ls -1 '$metadataRemote'/session_$sessionId.json"
& $adb -s $watch shell "ls -1 '$watchRemote'/watch_$sessionId*.csv"
```

### 7.5 Dateien herunterladen

Beispiel für eine lokale Forschungsdatenstruktur:

```text
data/
├── phone/
├── watch/
├── metadata/
├── camera/
├── synchronization/
└── annotations/
```

Die Zielordner vor dem ersten Export anlegen:

```powershell
New-Item -ItemType Directory -Force -Path @(
    ".\data\phone",
    ".\data\watch",
    ".\data\metadata",
    ".\data\camera",
    ".\data\synchronization",
    ".\data\annotations"
) | Out-Null
```

Die Dateien können anschließend mit `adb pull` übertragen werden. Für jede Dateisuche muss genau ein Treffer vorliegen.

```powershell
$phoneMatches = @(
    & $adb -s $phone shell "ls -1 '$phoneRemote'/smartphone_$sessionId*.wav"
)
if ($LASTEXITCODE -ne 0 -or $phoneMatches.Count -ne 1) {
    throw "Smartphone-WAV fehlt oder ist nicht eindeutig."
}
$phoneMatch = $phoneMatches[0].Trim()

$metadataMatches = @(
    & $adb -s $phone shell "ls -1 '$metadataRemote'/session_$sessionId.json"
)
if ($LASTEXITCODE -ne 0 -or $metadataMatches.Count -ne 1) {
    throw "Sitzungsmetadaten fehlen oder sind nicht eindeutig."
}
$metadataMatch = $metadataMatches[0].Trim()

$watchMatches = @(
    & $adb -s $watch shell "ls -1 '$watchRemote'/watch_$sessionId*.csv"
)
if ($LASTEXITCODE -ne 0 -or $watchMatches.Count -ne 1) {
    throw "Watch-CSV fehlt oder ist nicht eindeutig."
}
$watchMatch = $watchMatches[0].Trim()

& $adb -s $phone pull $phoneMatch ".\data\phone\"
if ($LASTEXITCODE -ne 0) { throw "WAV-Export fehlgeschlagen." }
& $adb -s $phone pull $metadataMatch ".\data\metadata\"
if ($LASTEXITCODE -ne 0) { throw "Metadatenexport fehlgeschlagen." }
& $adb -s $watch pull $watchMatch ".\data\watch\"
if ($LASTEXITCODE -ne 0) { throw "Watch-Export fehlgeschlagen." }
```

Gerätedaten sollten erst gelöscht werden, wenn die lokalen Dateien auf Lesbarkeit und Vollständigkeit geprüft wurden und die vorgesehene Sicherung vorhanden ist.

## 8. Referenzvideo ablegen

Das Referenzvideo wird manuell nach

```text
data/camera/
```

kopiert.

Für die weitere Pipeline wird der Dateiname des Referenzvideos aus dem zugehörigen Smartphone-WAV-Dateinamen abgeleitet. Dabei wird der Präfix `smartphone_` durch `reference_video_` ersetzt. Session-UUID und Smartphone-Zeitstempel bleiben unverändert.

Beispiel:

```text
Smartphone-WAV:
smartphone_<Session-UUID>_<Smartphone-Zeitstempel>.wav

Referenzvideo:
reference_video_<Session-UUID>_<Smartphone-Zeitstempel>.mp4
```

Der im Namen des Referenzvideos enthaltene Zeitstempel ist damit ausdrücklich der Zeitstempel aus dem Smartphone-WAV-Dateinamen. Er beschreibt nicht den Aufnahmezeitpunkt der Kamera. Diese Benennung dient der eindeutigen Zuordnung des Referenzvideos zur zugehörigen Smartphone-Aufnahme.

## 9. Rohsession technisch validieren

Vor der Synchronisation werden WAV, Watch-CSV und Metadaten geprüft.

```powershell
python -m trinkerkennung_analysis.inspect_session `
    --session-id $sessionId `
    --data-root ".\data" `
    --report ".\data\synchronization\$sessionId\inspect_session_report_$sessionId.json"
```

Geprüft werden unter anderem:

- eindeutige Dateizuordnung über die Session-UUID,
- Metadatenstatus,
- WAV-Struktur,
- Watch-CSV-Struktur,
- vorhandene Sensorarten,
- Zeitwerte und Aufnahmedauern.

Fehler oder unerklärte Warnungen müssen vor der weiteren Verarbeitung geklärt werden. In PowerShell kann der Exitcode direkt nach einem Python-Aufruf mit `$LASTEXITCODE` geprüft werden; `0` bedeutet, dass der Aufruf ohne gemeldeten Fehler beendet wurde. Warnungen und die fachliche Plausibilität sind zusätzlich zu kontrollieren.

Alle Dateisuchen erfolgen rekursiv unter `data_root`. Pro Session dürfen dort jeweils nur die eindeutigen kanonischen Rohdateien und Synchronisationsberichte auffindbar sein. Sicherungskopien mit denselben Dateinamen außerhalb dieses Suchroots ablegen. Insbesondere ältere Berichte aus einem Ordner wie `P101_pre_video_alignment_backup` dürfen nicht zusätzlich unter dem aktiven Datenroot liegen.

## 10. Signale optional visualisieren

Für eine visuelle Kontrolle von Audio- und Watch-Signalen kann verwendet werden:

```powershell
python -m trinkerkennung_analysis.visualize_session `
    --session-id $sessionId `
    --data-root ".\data" `
    --output-root ".\visualizations"
```

## 11. Smartwatch auf die Audiozeitachse synchronisieren

Die Smartwatch-zu-Audio-Synchronisation wird aus den Start- und Endklopfmarkern bestimmt.

```powershell
python -m trinkerkennung_analysis.estimate_synchronization `
    --session-id $sessionId `
    --data-root ".\data" `
    --output-root ".\data\synchronization"
```

Für jede Session wird unter

```text
data/synchronization/<Session-UUID>/
```

unter anderem erzeugt:

```text
synchronization_report_<Session-UUID>.json
session_<Session-UUID>_aligned_overview.png
```

Im regulären Verfahren werden jeweils drei Start- und drei Endmarker automatisch bestimmt.

Falls die reguläre Gruppenerkennung bei eindeutig sichtbaren realen Markern nicht zuverlässig funktioniert, unterstützt das Modul bestätigte enge Peak-Suchfenster. Diese Fallback-Parameter dürfen nur anhand der Synchronisationssignale und nicht anhand späterer Ground-Truth- oder Modellergebnisse gewählt werden.

### 11.1 Qualität des konstanten Watch-Offsets prüfen

Im Bericht sind insbesondere zu kontrollieren:

```text
offset_consistent
absolute_offset_change_seconds
offset_consistency_tolerance_seconds
warnings
validation_warnings
```

Der Code prüft die Änderung zwischen Start- und Endoffset gegen `max(0.040 s, 2 × medianer Watch-Abtastabstand)`. Maßgeblich ist der im Bericht gespeicherte Toleranzwert. Bei `offset_consistent=false` oder ungeklärten Warnungen die Markerzuordnung bzw. die Session prüfen, bevor der Bericht für die folgenden Schritte verwendet wird. Eine erzeugte JSON-Datei allein ist keine Qualitätsfreigabe.

### 11.2 Bestätigte Peak-Suchfenster verwenden

Die vier optionalen Argumente sind:

```text
--start-audio-peak-window START ENDE
--start-watch-peak-window START ENDE
--end-audio-peak-window START ENDE
--end-watch-peak-window START ENDE
```

Bei den regulären drei Klopfimpulsen wird ein verwendetes Argument **dreimal** angegeben: ein enges Suchfenster pro Impuls. Die Fenster derselben Gruppe müssen zeitlich geordnet sein und dürfen sich nicht überlappen. Nur die angegebenen Gruppen werden überschrieben; die übrigen bleiben automatisch.

- Audio-Suchfenster: Sekunden ab Beginn der Smartphone-WAV-Datei.
- Watch-Suchfenster: lokale Sekunden ab Beginn der Sensoraufnahme, vor der Watch-zu-Audio-Abbildung.

Beispielsyntax für manuell bestätigte Startmarker; alle Platzhalter durch die aus den Signalen bestimmten Grenzen ersetzen:

```powershell
python -m trinkerkennung_analysis.estimate_synchronization `
    --session-id $sessionId `
    --data-root ".\data" `
    --output-root ".\data\synchronization" `
    --start-audio-peak-window <A1_START> <A1_ENDE> `
    --start-audio-peak-window <A2_START> <A2_ENDE> `
    --start-audio-peak-window <A3_START> <A3_ENDE> `
    --start-watch-peak-window <W1_START> <W1_ENDE> `
    --start-watch-peak-window <W2_START> <W2_ENDE> `
    --start-watch-peak-window <W3_START> <W3_ENDE>
```

Für Endmarker werden entsprechend `--end-audio-peak-window` und/oder `--end-watch-peak-window` verwendet. Die eingesetzten Fenster und Gruppenmodi werden unter `marker_detection` im Bericht dokumentiert. Sechs finale P108-Sessions verwenden diese bestätigten Suchfenster; die konkreten Werte stehen in den zugehörigen archivierten Berichten.

Ein erneuter Aufruf von `estimate_synchronization` schreibt den Synchronisationsbericht neu und übernimmt eine zuvor ergänzte Videoabbildung nicht. Nach Änderungen an dieser Synchronisation deshalb Abschnitt 12 erneut durchführen und die davon abhängigen Annotationen, Fenster, Features und Ergebnisse erneut prüfen bzw. erzeugen.

## 12. Referenzvideo auf die Audiozeitachse abbilden

Die drei Start- und drei Endklopfmarker werden zusätzlich manuell im Referenzvideo bestimmt.

Als Beobachtungsregel wurde im Projekt der erste sichtbare Frame verwendet, in dem die Watch-Hand beim jeweiligen Klopfimpuls die Oberfläche erreicht.

Beispiel:

```powershell
$syncReport = ".\data\synchronization\$sessionId\synchronization_report_$sessionId.json"
$videoFile = "<PFAD_ZUM_REFERENZVIDEO>"

python -m trinkerkennung_analysis.add_video_alignment `
    --synchronization-report $syncReport `
    --reference-video-file $videoFile `
    --video-start-markers <S1> <S2> <S3> `
    --video-end-markers <E1> <E2> <E3>
```

Die Werte `<S1>` bis `<E3>` sind Sekunden ab Beginn des Referenzvideos.

Der Synchronisationsbericht wird dabei um die lineare Video-zu-Audio-Abbildung ergänzt.

Die lineare Abbildung verwendet die mittleren Impulse der Start- und Endmarkergruppen als zwei Anker. Für die vier übrigen Kontrollmarker wird ein maximales absolutes Residuum von 0,10 s als Akzeptanzgrenze verwendet. Der Code speichert das Maximum über alle sechs Residuen; die beiden Ankerresiduen sind bis auf Rundungsfehler null.

Im Bericht unter `time_mappings.video_to_audio` kontrollieren:

```text
maximum_absolute_marker_residual_seconds
marker_residual_tolerance_seconds
warnings
```

Eine Überschreitung muss fachlich geprüft werden. Fehlerhafte Video-Marker werden korrigiert und die Abbildung anschließend neu berechnet. Solange die Abbildung die vorgesehene Qualitätsgrenze nicht erfüllt, wird sie nicht für die Annotation verwendet. Der CLI-Aufruf kann auch bei einer Residualwarnung erfolgreich enden und den Bericht schreiben; Exitcode 0 ersetzt diese Kontrolle nicht.

Der optionale Methodenvergleich `trinkerkennung_analysis.compare_video_alignment_methods` ist ein eigener Analysebaustein für technische Videoalignment-Sessions. Er vergleicht `LINEAR_TWO_POINT` mit `LINEAR_ALL_SIX_OLS` und ersetzt die hier verwendete Zwei-Anker-Abbildung nicht.

## 13. Fachliche Videozeitpunkte in Audiozeit umrechnen

Manuell im Referenzvideo bestimmte Aktivitätsgrenzen werden nicht per Kopfrechnung übertragen.

Beispiel:

```powershell
python -m trinkerkennung_analysis.convert_video_times `
    --synchronization-report $syncReport `
    --video-times <EVENT_START> <CONTACT_START> <CONTACT_END> <EVENT_END> `
    --labels EVENT_START MOUTH_CONTACT_START MOUTH_CONTACT_END EVENT_END
```

Die Ausgabe enthält die transformierten Zeitpunkte in:

```text
AUDIO_SECONDS_FROM_WAV_START
```

Die vier Platzhalter im Beispiel bezeichnen in dieser Reihenfolge die beobachteten Videozeiten für Ereignisbeginn, Mundkontaktbeginn, Mundkontaktende und Ereignisende. Bei mehreren Mundkontakten werden entsprechend zusätzliche Start-/Endwerte umgerechnet.

Diese gemeinsame Audiozeitreferenz wird in den Annotationen und späteren Fenstern verwendet.

## 14. Ground-Truth-Annotation erstellen

Als Ausgangspunkt dient `annotation_template.json`. Für jede Session eine eigene Kopie erstellen und unter

```text
data/annotations/annotation_<Session-UUID>.json
```

speichern. Das Schema wird durch `ANNOTATION_SCHEMA.md`, `annotation_schema.json` und den Validator `annotation.py` beschrieben.

### 14.1 Metadaten und Template ausfüllen

Die Beispielwerte des Templates müssen auf die konkrete Session angepasst werden:

| Feld | Inhalt |
| --- | --- |
| `schema_version` | `1` |
| `session_id` | Tatsächliche Session-UUID; muss zum Dateinamen und zu den mobilen Dateien passen |
| `participant_id` | Eigenes Pseudonym der aufgenommenen Person, beispielsweise `P001`; über deren Sessions hinweg unverändert |
| `recording_protocol_version` | `"1.0"`; derzeit der vom Schema unterstützte Protokollwert |
| `session_mode` | Passender Modus: `CONTROLLED_SINGLE_ACTIVITY`, `CONTROLLED_MIXED_ACTIVITY`, `CONTINUOUS_MIXED_ACTIVITY` oder `EDGE_CASE`; Zuordnung der A–I-Vorlage siehe Abschnitt 5.5 |
| `time_reference` | `"AUDIO_SECONDS_FROM_WAV_START"` |
| `synchronization_report_file` | Dateiname `synchronization_report_<Session-UUID>.json` |
| `ground_truth.source` | `"SYNCHRONIZED_VIDEO"` |
| `ground_truth.reference_video_file` | Dateiname des zugehörigen Videos gemäß Abschnitt 8, ohne Verzeichnispfad |
| `ground_truth.annotator_id` | Pseudonym der annotierenden Person, beispielsweise `A001` |
| `ground_truth.annotation_coverage` | `"PARTIAL"` oder `"EXHAUSTIVE"`, entsprechend der tatsächlichen Annotation |
| `reviewed_intervals` | Tatsächlich überprüfte Bereiche mit `start_seconds` und `end_seconds` auf der Audiozeitachse |

Alle 69 archivierten finalen Annotationen verwenden `PARTIAL`. Für diesen Ablauf die überprüften Bereiche in `reviewed_intervals` dokumentieren. Die fachlichen positiven, negativen und unsicheren Intervalle müssen innerhalb der dokumentierten überprüften Bereiche liegen. `EXHAUSTIVE` nur verwenden, wenn die tatsächliche Annotation diesem Anspruch entspricht; nicht zum Unterdrücken fehlender Angaben auswählen.

Sämtliche Beispielereignisse, Beispielintervalle und Beispielzeiten entfernen oder durch die aus dem synchronisierten Video bestimmten Werte ersetzen. Den JSON-Wert `recording_protocol_version="1.0"` beibehalten; der vorhandene Validator erwartet diesen Schemawert.

### 14.2 Fachliche Intervalle annotieren

Die vier fachlichen Strukturen sind:

```text
drink_events
negative_intervals
excluded_intervals
uncertain_intervals
```

- `drink_events`: bestätigte Trinkereignisse. `event_start_seconds` und `event_end_seconds` beschreiben die äußeren sichtbaren Grenzen des vollständigen Bewegungs- und Trinkzyklus. Innerhalb jedes Ereignisses wird mindestens ein Mundkontaktintervall mit eigener Start- und Endzeit gespeichert. Zwei getrennte Bewegungszyklen zum Mund sind zwei Ereignisse; mehrere Mundkontakte in einem durchgehenden Zyklus können zu einem Ereignis gehören. Für Sitzung H gilt die besondere Abgrenzung aus Abschnitt 5.5.
- `negative_intervals`: ausdrücklich beobachtete Nicht-Trink-Aktivitäten mit `scenario` und Start-/Endzeit. Bei LIFT/MOVE den Gefäßtyp `container_type` nach Möglichkeit als `GLASS` oder `BOTTLE` angeben.
- `excluded_intervals`: technisch oder methodisch ausgeschlossene Abschnitte mit einem zulässigen `reason`, beispielsweise Synchronisationsmarker oder ungültiges Signal.
- `uncertain_intervals`: überprüfte, aber fachlich unsichere Abschnitte mit einem zulässigen Unsicherheitsgrund.

`notes` ist bei Trinkereignissen und negativen Intervallen ein Pflichtfeld und darf `null` sein. Ereignis- und Intervall-IDs, beispielsweise `D001`, `N001`, `X001` und `U001`, müssen innerhalb einer Annotation eindeutig sein.

Alle Zeiten sind Audiosekunden. Intervalle sind halb offen (`[Start, Ende)`); der Start gehört dazu, das Ende nicht. Es muss `Start < Ende` gelten, und alle Zeiten müssen innerhalb der WAV-Dauer liegen. Die vier fachlichen Strukturen dürfen sich auf oberster Ebene nicht überlappen. Mundkontakte liegen als Unterstruktur vollständig innerhalb ihres Trinkereignisses.

Ein sichtbarer Mundkontakt ist kein direkter Nachweis des physiologischen Schluckzeitpunkts. Für die spätere Fensterbildung werden die vollständigen äußeren Ereignisgrenzen verwendet, nicht nur die Mundkontaktintervalle.

Nicht explizit annotierte Zeitbereiche bleiben `UNLABELED` und werden nicht automatisch zu `NON_DRINK`. Das gilt auch für neutrale Pausen: Nur ausdrücklich überprüfte und als negativ annotierte Bereiche liefern negative ML-Fenster.

### 14.3 Synchronisationsmarker ausschließen

Für Start- und Endklopfmarker `excluded_intervals` mit `reason="SYNCHRONIZATION_MARKER"` anlegen. Die Ausschlussgrenzen aus `audio_marker_exclusion_intervals_seconds` des finalen Synchronisationsberichts übernehmen bzw. vollständig abdecken; nicht die Beispielwerte des Templates stehen lassen. Der Validator überprüft die Abdeckung mit einer standardmäßigen Markertoleranz von 0,050 s. Diese Toleranz ist eine technische Prüfung und kein Ersatz für fachlich korrekte Intervallgrenzen.

Die ausgeschlossenen Synchronisationsmarker werden später nicht für ML-Fenster verwendet.

## 15. Annotation validieren

Für die in Abschnitt 14 gezeigte Ablage unter `data/annotations`:

```powershell
python -m trinkerkennung_analysis.validate_annotation `
    --session-id $sessionId `
    --data-root ".\data" `
    --report ".\data\synchronization\$sessionId\annotation_validation_report_$sessionId.json"
```

Liegt die Annotation außerhalb dieses Datenroots, wird stattdessen der konkrete Dateipfad übergeben:

```powershell
python -m trinkerkennung_analysis.validate_annotation `
    --annotation "<PFAD_ZUR_ANNOTATION_JSON>" `
    --data-root "<DATENROOT_MIT_ROHDATEN_VIDEO_UND_SYNC>" `
    --report "<PFAD_ZUM_VALIDIERUNGSBERICHT>"
```

`--session-id` und `--annotation` sind alternative Selektoren; nicht gleichzeitig angeben. Nur erfolgreich validierte Annotationen werden für die Fensterbildung verwendet. Eventuelle Warnungen zusätzlich prüfen.

Der Validator sucht die Rohdateien, das im Annotationseintrag genannte Referenzvideo und den Synchronisationsbericht rekursiv unter `data_root`. Diese Dateien müssen dort eindeutig auffindbar sein; die Annotation selbst darf beim expliziten `--annotation`-Aufruf außerhalb liegen. Für die spätere Fenster-API wird diese Trennung über `annotations_root` angegeben. Ein Datenroot, der ausschließlich WAV/Watch/Metadaten enthält, reicht ohne Video und Synchronisationsbericht nicht für diese Annotationvalidierung aus.

## 16. Fensterdatensätze erzeugen

Das Repository enthält das Hilfsskript `build_windows.py`. Es liegt im Repository-Stamm neben `requirements.txt` und ruft die vorhandenen Funktionen `generate_windows_for_sessions` und `write_window_dataset_csv` auf. Das Skript muss nicht aus einem Codeblock neu angelegt werden.

Es erzeugt beide vorgesehenen Konfigurationen:

```text
windows_1s_stride1s.csv: 1 s Fenster, 1 s Schrittweite
windows_2s_stride2s.csv: 2 s Fenster, 2 s Schrittweite
```

Fenster entstehen ausschließlich vollständig innerhalb expliziter positiver oder negativer Quellintervalle. Für `DRINK` zählen die äußeren Trinkereignisgrenzen. Nicht passende Reststücke am Intervallende erzeugen kein Fenster; `UNLABELED`, `EXCLUDED` und `UNCERTAIN` liefern keine ML-Fenster.

Nur die final freigegebenen Annotationen in den jeweiligen kanonischen Annotationsordner legen. Backups, verworfene Sessions und andere Testannotationen nicht in diese Auswahl aufnehmen. Das Skript verwendet alle `annotation_*.json` direkt in diesem Ordner und sortiert deren Dateinamen für eine feste Reihenfolge. Jede Annotation wird mit den vorhandenen Verarbeitungsfunktionen erneut gegen die benötigten Dateien validiert.

### 16.1 Gemeinsamer Datenroot

Für die Struktur aus Abschnitt 7–15 genügt aus dem Repository-Stamm:

```powershell
python .\build_windows.py --data-root ".\data"
```

Dabei werden die Annotationen unter `data/annotations` gesucht und die beiden CSV-Dateien unter `ML/01_Window_Datasets` erzeugt. Diese Pfade sind die Standardwerte des Skripts.

Bei einer anderen Ablage die Pfade über Startparameter übergeben. Der Python-Code muss dafür nicht bearbeitet werden:

```powershell
python .\build_windows.py `
    --data-root "<DATENROOT_MIT_ROHDATEN_VIDEO_UND_SYNC>" `
    --annotations-root "<ORDNER_MIT_FINALEN_ANNOTATIONEN>" `
    --output-root "<AUSGABEORDNER_FUER_FENSTER>"
```

`data_root` bleibt der rekursive Suchroot für Rohdateien, Referenzvideo und Synchronisationsbericht. Der Annotationsordner darf außerhalb dieses Roots liegen.

### 16.2 Mehrere getrennte Datenroots

Diese Variante ist nur nötig, wenn die eigenen Daten in getrennten Roots liegen. Die Datenroots können beispielsweise getrennte Pilot- und Hauptstudienordner sein.

```powershell
python .\build_windows.py `
    --data-root "<DATENROOT_1>" `
    --data-root "<DATENROOT_2>" `
    --annotations-root "<ANNOTATIONSORDNER_1>" `
    --annotations-root "<ANNOTATIONSORDNER_2>" `
    --output-root ".\ML\01_Window_Datasets"
```

Bei expliziten Annotationsordnern muss für jeden Datenroot genau ein `--annotations-root` angegeben werden, jeweils in derselben Reihenfolge. Werden keine Annotationsordner angegeben, verwendet das Skript für jeden Root dessen Unterordner `annotations`.

Die Sessions werden pro Root validiert und als `WindowRecord`-Listen zusammengeführt. Pro Konfiguration wird eine gemeinsame CSV geschrieben; Kopfzeilen werden nicht mehrfach übernommen. Doppelte Session-UUIDs über mehrere Roots führen zu einem Fehler. Wenn für eine der Konfigurationen überhaupt keine vollständig passenden Fenster entstehen, wird der Lauf ebenfalls abgebrochen.

Für den historischen vollständigen Bestand der Bachelorarbeit sind 24 Pilot- und 45 Hauptstudien-Sessions enthalten. Dessen Datensätze enthalten 1.449 Fenster bei 1 s/1 s und 633 Fenster bei 2 s/2 s. Bei eigenen Aufnahmen ergeben sich andere Zahlen; diese Kontrollwerte sind dafür keine Voraussetzung.

Hilfe zu allen Parametern:

```powershell
python .\build_windows.py --help
```

Die feste Reihenfolge einer Neuberechnung muss nicht die ursprüngliche archivierte Zeilenreihenfolge reproduzieren. Für eine zusätzliche Wiederholung der historischen Modellkennzahlen gilt Abschnitt 20.

## 17. Feature-Datensätze erzeugen

Das Repository enthält das Hilfsskript `build_features.py` im Repository-Stamm. Es verwendet die vorhandenen Funktionen `build_feature_datasets_from_window_csv` und `write_feature_datasets`, liest beide Fensterdateien aus Abschnitt 16 und erzeugt alle sechs Featuredateien für Abschnitt 18.

### 17.1 Gemeinsamer Datenroot

Für die Standardablage aus den bisherigen Abschnitten:

```powershell
python .\build_features.py --data-root ".\data"
```

Standardmäßig werden die Fenster aus `ML/01_Window_Datasets` gelesen und die Features nach `ML/02_Feature_Datasets` geschrieben. Das Skript erwartet dort die Dateinamen `windows_1s_stride1s.csv` und `windows_2s_stride2s.csv`.

Bei anderen Daten-/Ausgabeordnern die Pfade als Startparameter setzen:

```powershell
python .\build_features.py `
    --data-root "<DATENROOT_MIT_ROHDATEN_UND_SYNC>" `
    --window-root "<FENSTERORDNER_AUS_ABSCHNITT_16>" `
    --output-root "<FEATURE_AUSGABEORDNER>"
```

Die Datenpfade werden damit beim Start angepasst; Änderungen am Python-Code sind nicht erforderlich. Im jeweiligen Datenroot müssen die betreffenden Rohdateien und der kanonische Synchronisationsbericht eindeutig auffindbar sein.

### 17.2 Getrennte Datenroots je Person

Für Personen, deren Dateien außerhalb des Standard-Datenroots liegen, kann `--participant-root` mehrfach angegeben werden. Jedes Argument besteht aus der eigenen Teilnehmer-ID und dem zugehörigen Datenroot:

```powershell
python .\build_features.py `
    --data-root "<STANDARD_DATENROOT>" `
    --participant-root P001 "<DATENROOT_FUER_P001>" `
    --participant-root P002 "<DATENROOT_FUER_P002>" `
    --window-root ".\ML\01_Window_Datasets" `
    --output-root ".\ML\02_Feature_Datasets"
```

`P001` und `P002` sind Beispiele und müssen den IDs der eigenen Annotationen entsprechen. Personen ohne eigenen Eintrag werden über `--data-root` geladen. Eine ID darf nicht mehrfach als `--participant-root` angegeben werden.

Die vorhandene Feature-API ordnet pro Person einen Datenroot zu. Alle benötigten Sessions derselben Person müssen unter diesem Root auffindbar sein. Wenn Sessions derselben Person auf mehrere Ordner verteilt sind, einen gemeinsamen übergeordneten Root mit eindeutig auffindbaren Dateien verwenden oder die Daten kontrolliert in einen gemeinsamen Root konsolidieren.

### 17.3 Ausgabe und Dateinamen

Ohne weitere Optionen verwendet das Skript `--naming general` und erzeugt:

```text
pilot_audio_features_1s.csv
pilot_watch_features_1s.csv
pilot_fusion_features_1s.csv
pilot_audio_features_2s.csv
pilot_watch_features_2s.csv
pilot_fusion_features_2s.csv
```

Das Wort `pilot` ist die historische Namenskonvention des vorhandenen allgemeinen Evaluationsrunners. Die Dateien enthalten die eigenen aufgenommenen Personen und Sessions; eine separate Pilotphase ist für diesen Weg nicht nötig. Die tatsächlichen Konfigurationen bleiben 1 s/1 s und 2 s/2 s.

Die Merkmalsdimensionen ohne Metadatenspalten sind:

```text
Audio-only:            38
Watch-only:            48
Feature-Level-Fusion:  86
```

Die Featurebildung erhält die Zeilenreihenfolge des Fensterdatensatzes. Die Paketversionen aus `requirements.txt` werden für die Berechnung benötigt, insbesondere `librosa` für die Audiofeatures.

Hilfe zu allen Parametern:

```powershell
python .\build_features.py --help
```

Die zusätzliche Option `--naming main-study` ist ausschließlich für die Dateibenennung des speziellen Hauptstudienrunners aus Abschnitt 19 vorgesehen. Für eigene Daten im allgemeinen Weg die Standardoption `general` verwenden.

## 18. ML-Auswertung eigener Aufnahmen

Für die eigenen Daten aus Abschnitt 16–17 den vorhandenen Runner `pilot_model_evaluation` verwenden. Trotz seines historischen Namens ist er nicht auf P101–P103 beschränkt: Er verwendet alle `participant_id`-Werte in seinen Eingabedateien und führt mit diesen Personen eine Leave-One-Subject-Out-Auswertung aus. Die Teilnehmer-IDs der Bachelorarbeit und deren Daten sind dafür nicht nötig.

### 18.1 Eingaben prüfen

Erwartet werden diese sechs Dateien aus Abschnitt 17:

```text
pilot_audio_features_1s.csv
pilot_watch_features_1s.csv
pilot_fusion_features_1s.csv
pilot_audio_features_2s.csv
pilot_watch_features_2s.csv
pilot_fusion_features_2s.csv
```

Es müssen mindestens drei unterschiedliche Personen vorkommen. Die Klassen sind `DRINK` und `NON_DRINK`; für die logistische Regression müssen beide in jedem Trainingsfold vorkommen. Audio-, Watch- und Fusionstabellen müssen je Konfiguration dieselben Fenster in derselben Reihenfolge und mit denselben Teilnehmer-IDs und Labels enthalten. Die Erzeugung aus Abschnitt 17 und die Runnerprüfung unterstützen diese Konsistenz.

### 18.2 Auswertung starten

```powershell
python -m trinkerkennung_analysis.pilot_model_evaluation `
    --feature-root ".\ML\02_Feature_Datasets" `
    --output-root ".\ML\03_Model_Evaluation"
```

Für jede Fensterkonfiguration werden ausgewertet:

```text
DummyClassifier
Audio-only + logistische Regression
Audio-only + Random Forest
Watch-only + logistische Regression
Watch-only + Random Forest
Fusion + logistische Regression
Fusion + Random Forest
```

In jedem Fold ist eine Person ausschließlich Testperson; die übrigen bilden die Trainingsdaten. Jede vorhandene Person wird einmal als Testperson verwendet. Bei der logistischen Regression wird auch die Standardisierung ausschließlich anhand des jeweiligen Trainingsfolds angepasst.

### 18.3 Ergebnisse lesen

Die Ausgabe verwendet ebenfalls die historischen Namenspräfixe:

```text
pilot_fold_results.csv
pilot_summary_results.csv
pilot_window_predictions.csv
pilot_window_errors.csv
```

- `pilot_fold_results.csv`: Kennzahlen je zurückgehaltener Person, Modell, Modalität und Fensterkonfiguration.
- `pilot_summary_results.csv`: personenweise gemittelte Kennzahlen und zusammengezählte Konfusionsmatrix-Zählwerte.
- `pilot_window_predictions.csv`: wahres und vorhergesagtes Label jedes Testfensters, mit Person und Session für die Rückverfolgung.
- `pilot_window_errors.csv`: die falsch klassifizierten Testfenster.

Bei `N` Personen, zwei Konfigurationen und sieben Modell-/Modalitätskombinationen entstehen `14 × N` Fold-Zeilen und 14 Summary-Zeilen. Für drei eigene Personen sind es 42 Fold-Zeilen. Die Kennzahlen beziehen sich auf die eigenen Aufnahmen; sie müssen nicht den Ergebnissen der Bachelorarbeit entsprechen.

Dieser Lauf trainiert und bewertet die Modelle. Die Ergebnis-CSVs sind keine gespeicherten, später wieder ladbaren trainierten Modelle. Ein separater Ablauf für Modellpersistierung und Anwendung auf weitere Einzelaufnahmen ist in diesem Workflow nicht implementiert.

Damit ist der reguläre Weg für eigene Aufnahmen abgeschlossen. Die nächsten Abschnitte behandeln zusätzlich das spezielle Evaluationsdesign und die Archive der Bachelorarbeit.

## 19. Studienspezifische Hauptauswertung der Bachelorarbeit

Dieser zusätzliche Weg ist für das festgelegte Evaluationsdesign der Bachelorarbeit mit den Testpersonen P104–P108 vorgesehen. Für eigene Daten mit anderen Teilnehmer-IDs den allgemeinen Weg aus Abschnitt 18 verwenden. Der finale Evaluationsrunner erwartet die sechs Featuredateien mit den für die Bachelorarbeit festgelegten Namen:

```text
primary_P101_P108_audio_features_1s_stride1s.csv
primary_P101_P108_watch_features_1s_stride1s.csv
primary_P101_P108_fusion_features_1s_stride1s.csv

sensitivity_P101_P108_audio_features_2s_stride2s.csv
sensitivity_P101_P108_watch_features_2s_stride2s.csv
sensitivity_P101_P108_fusion_features_2s_stride2s.csv
```

Bei einer Neuberechnung dieser speziellen Hauptauswertung kann das enthaltene Feature-Skript die erforderlichen Namen erzeugen:

```powershell
python .\build_features.py `
    --data-root "<DATENROOT_MIT_ROHDATEN_UND_SYNC>" `
    --window-root "<FENSTERORDNER_AUS_ABSCHNITT_16>" `
    --output-root ".\30_ML\02_Feature_Datasets" `
    --naming main-study
```

Bei getrennten Datenroots die passenden `--participant-root`-Argumente wie in Abschnitt 17 ergänzen. Für diese Benennung verwendet das Skript folgende Werte des vorhandenen Feature-Writers:

| Fensterkonfiguration | `file_prefix` | `configuration_name` |
| --- | --- | --- |
| 1 s/1 s | `"primary_P101_P108"` | `"1s_stride1s"` |
| 2 s/2 s | `"sensitivity_P101_P108"` | `"2s_stride2s"` |

Die dort genannten Fenster- und Featurebestände gehören zum P101–P108-Design. Alternativ die archivierten Features aus Abschnitt 20 direkt verwenden. Die Auswertung wird gestartet mit:

```powershell
python -m trinkerkennung_analysis.main_model_evaluation `
    --feature-root ".\30_ML\02_Feature_Datasets" `
    --output-root ".\30_ML\03_Model_Evaluation"
```

Die Implementierung verwendet die Hauptstudienpersonen `P104` bis `P108` als fünf zurückgehaltene Testpersonen. Die Pilotpersonen `P101` bis `P103` verbleiben auf der Trainingsseite.

Pro Fensterkonfiguration werden ausgewertet:

```text
DummyClassifier
Audio-only + logistische Regression
Audio-only + Random Forest
Watch-only + logistische Regression
Watch-only + Random Forest
Fusion + logistische Regression
Fusion + Random Forest
```

Für beide Konfigurationen zusammen entstehen beim vollständigen Hauptstudienbestand 70 Fold-Zeilen und 14 Summary-Zeilen. Die Summary-Kennzahlen sind personenweise Mittelwerte über die fünf Testfolds; sie sind nicht einfach aus allen Testfenstern gemeinsam berechnet.

Die erzeugten Ergebnisdateien sind:

```text
main_fold_results.csv
main_summary_results.csv
main_window_predictions.csv
main_window_errors.csv
```

## 20. Zuordnung zu den finalen Ergebnisdateien der Bachelorarbeit

Das nachfolgend beschriebene `30_ML`-Archiv dokumentiert den separat aufbewahrten finalen ML-Ergebnisstand der Bachelorarbeit und ist nicht Bestandteil des öffentlichen GitHub-Repositories.

Der finale archivierte ML-Stand der Bachelorarbeit besitzt folgende Struktur:

```text
30_ML/
├── 01_Window_Datasets/
│   ├── window_dataset_primary_1s_stride1s_P101-P108.csv
│   └── window_dataset_sensitivity_2s_stride2s_P101-P108.csv
├── 02_Feature_Datasets/
│   ├── primary_P101_P108_audio_features_1s_stride1s.csv
│   ├── primary_P101_P108_watch_features_1s_stride1s.csv
│   ├── primary_P101_P108_fusion_features_1s_stride1s.csv
│   ├── sensitivity_P101_P108_audio_features_2s_stride2s.csv
│   ├── sensitivity_P101_P108_watch_features_2s_stride2s.csv
│   └── sensitivity_P101_P108_fusion_features_2s_stride2s.csv
└── 03_Model_Evaluation/
    ├── main_fold_results.csv
    ├── main_summary_results.csv
    ├── main_window_predictions.csv
    └── main_window_errors.csv
```

Für die exakte Reproduktion der in der Bachelorarbeit berichteten Modellkennzahlen sind die archivierten finalen Featuredateien maßgeblich. Außerdem dieselbe Modellkonfiguration sowie Python 3.13 und die in `requirements.txt` festgeschriebenen Paketversionen verwenden. Ein Lauf mit anderen Bibliotheksversionen kann funktionsfähig sein und trotzdem abweichende Vorhersagen liefern.

Die sechs archivierten Featuredateien enthalten bereits die Beobachtungen von P101–P108. Für eine reine Wiederholung der Hauptauswertung sind weder eine neue Aufnahme noch Windowing oder Featureextraktion nötig. Beispiel mit separat entpacktem Archiv:

```powershell
python -m trinkerkennung_analysis.main_model_evaluation `
    --feature-root "<ENTPACKTES_ARCHIV>\30_ML\02_Feature_Datasets" `
    --output-root ".\reproduction_results"
```

Als Referenz dienen die vier archivierten CSV-Dateien unter `30_ML/03_Model_Evaluation`. Die neue Ausgabe in einen eigenen Ordner schreiben, damit die Referenzergebnisse beim Vergleich erhalten bleiben.

Bei Random Forests ist ein fester Zufallsstartwert reproduzierbar für eine unveränderte Eingabereihenfolge. Eine reine Zeilenpermutation kann trotz identischer Beobachtungen eine andere Zuordnung der zufällig gezogenen Bootstrap-Positionen zu konkreten Trainingsfenstern bewirken. Deshalb sollte die Reihenfolge der archivierten finalen Featuredateien für eine exakte Reproduktion nicht verändert werden.

## 21. Qualitätskontrolle

Vor jedem nachfolgenden Verarbeitungsschritt sollte der vorherige Schritt erfolgreich abgeschlossen sein.

```text
Rohdateien vollständig
→ Sessionvalidierung erfolgreich
→ Watch-zu-Audio-Synchronisation plausibel
→ Video-zu-Audio-Synchronisation innerhalb der Qualitätsgrenzen
→ vorgesehene Annotation und überprüfte Bereiche dokumentiert
→ Annotationvalidierung erfolgreich
→ Windowing
→ Featureextraktion
→ Evaluation
```

Warnungen sollten nicht allein deshalb ignoriert werden, weil eine Ausgabedatei technisch erzeugt wurde. Eine erfolgreiche Annotationvalidierung bestätigt außerdem die technischen Konsistenzregeln, nicht automatisch die inhaltliche Richtigkeit jeder beobachteten Aktivitätsgrenze.

Nach einer Änderung von Synchronisationsmarkern oder Ground Truth die davon abhängigen Verarbeitungsschritte erneut durchgehen; vorhandene Fenster-, Feature- und Ergebnisdateien aktualisieren. Die Referenzarchive für Vergleiche erhalten.

## 22. Datenschutz

Forschungsrohdaten werden nicht in Git versioniert.

Nicht in ein öffentliches Repository gehören insbesondere:

```text
Referenzvideos
WAV-Aufnahmen
reale Watch-CSV-Dateien
reale Sitzungsmetadaten
reale Annotationen
reale Synchronisationsberichte
session- oder teilnehmerbezogene Fensterdatensätze
session- oder teilnehmerbezogene Featuredateien
Einzelvorhersagen und Fehlklassifikationsdateien mit Teilnehmer-/Sessionbezug
Einwilligungserklärungen
Namens- oder Kontaktzuordnungen
```

Die technische Dokumentation und der Quellcode können unabhängig davon veröffentlicht werden, solange diese Dateien und sonstige sensible Informationen nicht enthalten sind.
