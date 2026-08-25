# Aufnahmeprotokoll 1.0

## Technischer Schema-Test

Der erste Test prüft Dateistruktur, Synchronisation, manuelle Videoannotation, Schema und Validator. Er ist noch kein belastbarer ML-Datensatz.

Ablauf je kontrollierter Sitzung:

1. Referenzvideo starten.
2. Smartphone- und Watch-Aufnahme starten.
3. Drei bis fünf Sekunden ruhig warten.
4. Drei Anfangsklopfer mit der Watch-Hand ausführen.
5. Zwei bis drei Sekunden ruhig warten.
6. Drei Wiederholungen der festgelegten Aktivität ausführen.
7. Zwischen Wiederholungen fünf bis acht Sekunden Ruhe einhalten.
8. Zwei bis drei Sekunden ruhig warten.
9. Drei Endklopfer ausführen.
10. Drei bis fünf Sekunden ruhig warten.
11. Smartphone- und Watch-Aufnahme stoppen.
12. Referenzvideo stoppen.

Während der kontrollierten Sitzung wird nicht gesprochen. Instruktionen erfolgen vor der Aufnahme oder über vereinbarte lautlose Zeichen.

Mindestsitzungen:

- `DRINK_FROM_GLASS`
- `DRINK_FROM_BOTTLE`
- `LIFT_CONTAINER_WITHOUT_DRINKING`
- `MOVE_CONTAINER_WITHOUT_DRINKING`
- `HAND_TO_FACE_WITHOUT_CONTAINER`
- `REST`

## Referenzvideo

Das Video dient ausschließlich der manuellen Ground-Truth-Annotation, wird nicht als Modellmodalität verwendet, nicht veröffentlicht und nicht im Git-Repository gespeichert. Es erhält dieselbe Session-ID wie WAV, CSV und Annotation. Einwilligung und Löschkonzept werden vor der Datenerhebung verbindlich dokumentiert.

## Pilotstudie

Nach erfolgreichem Schema-Test sind vorläufig zwei bis drei volljährige Personen, ungefähr fünf Wiederholungen zentraler Aktivitäten je Person, Glas- und Flaschentrinken, schwierige Negativaktivitäten, mindestens eine gemischte kontinuierliche Sitzung und mindestens eine Randfallsitzung vorgesehen. Die endgültigen Zahlen werden erst nach der technischen und pilotierenden Auswertung festgelegt.
