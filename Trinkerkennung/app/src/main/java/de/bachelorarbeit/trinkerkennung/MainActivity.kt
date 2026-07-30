package de.bachelorarbeit.trinkerkennung

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import de.bachelorarbeit.trinkerkennung.ui.theme.TrinkerkennungTheme

class MainActivity : ComponentActivity() {

    override fun onCreate(
        savedInstanceState: Bundle?
    ) {
        super.onCreate(savedInstanceState)

        setContent {
            TrinkerkennungTheme {
                RecordingSessionScreen()
            }
        }
    }
}

@Composable
private fun RecordingSessionScreen() {
    val context =
        LocalContext.current

    val mainHandler = remember {
        Handler(
            Looper.getMainLooper()
        )
    }

    val audioRecorder = remember {
        WavAudioRecorder(
            context.applicationContext
        )
    }

    var wearStatus by remember {
        mutableStateOf(
            "Noch keine Sitzung vorbereitet."
        )
    }

    var audioStatus by remember {
        mutableStateOf(
            "Smartphone-Audioaufnahme ist bereit."
        )
    }

    var canStartWatchRecording by remember {
        mutableStateOf(false)
    }

    var canStopWatchRecording by remember {
        mutableStateOf(false)
    }

    var isPhoneAudioRecording by remember {
        mutableStateOf(false)
    }

    var audioRecordingPath by remember {
        mutableStateOf<String?>(null)
    }

    var permissionGranted by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) ==
                    PackageManager.PERMISSION_GRANTED
        )
    }

    val wearCommunication = remember(
        audioRecorder,
        mainHandler
    ) {
        PhoneWearCommunication(
            context =
                context.applicationContext,

            onStatusChanged = { newStatus ->
                wearStatus =
                    newStatus
            },

            onSessionControlsChanged = {
                    canStart,
                    canStop ->

                canStartWatchRecording =
                    canStart

                canStopWatchRecording =
                    canStop
            },

            onSessionStartFailed = {
                    sessionId,
                    reason ->

                Thread {
                    audioRecorder.abortRecording()

                    mainHandler.post {
                        isPhoneAudioRecording =
                            false

                        audioRecordingPath =
                            null

                        audioStatus =
                            "Smartphone-Audioaufnahme für " +
                                    "Sitzung ${sessionId.take(8)} " +
                                    "wurde verworfen, da die " +
                                    "Watch-Aufnahme nicht gestartet " +
                                    "werden konnte: $reason"
                    }
                }.apply {
                    name =
                        "AbortPhoneAudioRecording"

                    start()
                }
            },

            onSessionStopped = {
                    sessionId,
                    watchRecordingSucceeded,
                    watchFileName ->

                Thread {
                    val completedAudioFile =
                        audioRecorder.stopRecording()

                    mainHandler.post {
                        isPhoneAudioRecording =
                            false

                        if (
                            completedAudioFile != null &&
                            completedAudioFile.exists()
                        ) {
                            audioRecordingPath =
                                completedAudioFile.absolutePath

                            audioStatus =
                                buildString {
                                    append(
                                        "Gemeinsame Sitzung abgeschlossen."
                                    )

                                    append(
                                        "\nSession: "
                                    )
                                    append(
                                        sessionId.take(8)
                                    )

                                    append(
                                        "\nSmartphone-Audio gespeichert: ja"
                                    )

                                    append(
                                        "\nSmartphone-Datei: "
                                    )
                                    append(
                                        completedAudioFile.name
                                    )

                                    append(
                                        "\nWatch gespeichert: "
                                    )
                                    append(
                                        if (
                                            watchRecordingSucceeded
                                        ) {
                                            "ja"
                                        } else {
                                            "nein"
                                        }
                                    )

                                    append(
                                        "\nWatch-Datei: "
                                    )
                                    append(
                                        watchFileName
                                            ?: "keine Datei"
                                    )
                                }
                        } else {
                            audioRecordingPath =
                                null

                            audioStatus =
                                buildString {
                                    append(
                                        "Die Watch-Sitzung wurde beendet, "
                                    )

                                    append(
                                        "aber die Smartphone-Audiodatei "
                                    )

                                    append(
                                        "konnte nicht gespeichert werden."
                                    )

                                    append(
                                        "\nSession: "
                                    )
                                    append(
                                        sessionId.take(8)
                                    )

                                    append(
                                        "\nWatch gespeichert: "
                                    )
                                    append(
                                        if (
                                            watchRecordingSucceeded
                                        ) {
                                            "ja"
                                        } else {
                                            "nein"
                                        }
                                    )
                                }
                        }
                    }
                }.apply {
                    name =
                        "FinalizePhoneAudioRecording"

                    start()
                }
            }
        )
    }

    DisposableEffect(
        wearCommunication
    ) {
        wearCommunication.startListening()

        onDispose {
            wearCommunication.stopListening()
        }
    }

    DisposableEffect(
        audioRecorder
    ) {
        onDispose {
            audioRecorder.release()
        }
    }

    val permissionLauncher =
        rememberLauncherForActivityResult(
            contract =
                ActivityResultContracts
                    .RequestPermission()
        ) { granted ->
            permissionGranted =
                granted

            audioStatus =
                if (granted) {
                    "Mikrofonberechtigung wurde erteilt."
                } else {
                    "Mikrofonberechtigung wurde verweigert."
                }
        }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .verticalScroll(
                rememberScrollState()
            )
            .padding(24.dp),
        horizontalAlignment =
            Alignment.CenterHorizontally,
        verticalArrangement =
            Arrangement.Center
    ) {
        Text(
            text =
                "Gemeinsame Aufnahmesitzung",
            style =
                MaterialTheme.typography
                    .headlineMedium
        )

        Spacer(
            modifier =
                Modifier.height(24.dp)
        )

        Text(
            text =
                "Smartwatch-Kommunikation",
            style =
                MaterialTheme.typography
                    .titleMedium
        )

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        Text(
            text = wearStatus
        )

        Spacer(
            modifier =
                Modifier.height(12.dp)
        )

        Button(
            onClick = {
                wearCommunication.sendPing()
            },
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "PING an Smartwatch senden"
            )
        }

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        Button(
            onClick = {
                wearCommunication
                    .prepareSession()
            },
            enabled =
                !isPhoneAudioRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Neue Sitzung vorbereiten"
            )
        }

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        Button(
            onClick = {
                val sessionId =
                    wearCommunication
                        .getPreparedSessionId()

                if (
                    sessionId.isNullOrBlank()
                ) {
                    audioStatus =
                        "Es ist keine gültige vorbereitete " +
                                "Sitzung vorhanden."

                    return@Button
                }

                try {
                    val audioFile =
                        audioRecorder.startRecording(
                            sessionId = sessionId
                        )

                    isPhoneAudioRecording =
                        true

                    audioRecordingPath =
                        null

                    audioStatus =
                        buildString {
                            append(
                                "Smartphone-Audioaufnahme läuft."
                            )

                            append(
                                "\nSession: "
                            )
                            append(
                                sessionId.take(8)
                            )

                            append(
                                "\nVorläufige Datei: "
                            )
                            append(
                                audioFile.name
                            )

                            append(
                                "\nWarte auf STARTED der Watch."
                            )
                        }

                    val startRequested =
                        wearCommunication
                            .startPreparedSession()

                    if (!startRequested) {
                        audioRecorder
                            .abortRecording()

                        isPhoneAudioRecording =
                            false

                        audioRecordingPath =
                            null

                        audioStatus =
                            "Die gemeinsame Aufnahme " +
                                    "konnte nicht gestartet werden."
                    }
                } catch (
                    exception: Exception
                ) {
                    isPhoneAudioRecording =
                        false

                    audioRecordingPath =
                        null

                    audioStatus =
                        "Smartphone-Audioaufnahme konnte " +
                                "nicht gestartet werden: " +
                                (
                                        exception.message
                                            ?: "Unbekannter Fehler"
                                        )
                }
            },
            enabled =
                permissionGranted &&
                        canStartWatchRecording &&
                        !isPhoneAudioRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Gemeinsame Aufnahme starten"
            )
        }

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        Button(
            onClick = {
                val stopRequested =
                    wearCommunication
                        .stopCurrentSession()

                if (stopRequested) {
                    audioStatus =
                        "Watch wird gestoppt. " +
                                "Die Smartphone-Audioaufnahme " +
                                "läuft bis zur STOPPED-Bestätigung weiter."
                }
            },
            enabled =
                canStopWatchRecording &&
                        isPhoneAudioRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Gemeinsame Aufnahme stoppen"
            )
        }

        Spacer(
            modifier =
                Modifier.height(24.dp)
        )

        Text(
            text =
                "Smartphone-Audio",
            style =
                MaterialTheme.typography
                    .titleMedium
        )

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        Text(
            text =
                audioStatus
        )

        audioRecordingPath?.let { path ->
            Spacer(
                modifier =
                    Modifier.height(12.dp)
            )

            Text(
                text =
                    "Gespeicherter Pfad:\n$path",
                style =
                    MaterialTheme.typography
                        .bodySmall
            )
        }

        if (!permissionGranted) {
            Spacer(
                modifier =
                    Modifier.height(16.dp)
            )

            Button(
                onClick = {
                    permissionLauncher.launch(
                        Manifest.permission
                            .RECORD_AUDIO
                    )
                },
                modifier =
                    Modifier.fillMaxWidth()
            ) {
                Text(
                    "Mikrofonzugriff erlauben"
                )
            }
        }

        Spacer(
            modifier =
                Modifier.height(24.dp)
        )
    }
}