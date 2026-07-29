package de.bachelorarbeit.trinkerkennung

import android.Manifest
import android.content.pm.PackageManager
import android.os.Bundle
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
                AudioRecordingScreen()
            }
        }
    }
}

@Composable
private fun AudioRecordingScreen() {
    val context =
        LocalContext.current

    var wearStatus by remember {
        mutableStateOf(
            "Noch keine Testnachricht gesendet."
        )
    }

    var canStartWatchRecording by remember {
        mutableStateOf(false)
    }

    var canStopWatchRecording by remember {
        mutableStateOf(false)
    }

    val wearCommunication = remember {
        PhoneWearCommunication(
            context =
                context.applicationContext,
            onStatusChanged = { newStatus ->
                wearStatus = newStatus
            },
            onSessionControlsChanged = {
                    canStart,
                    canStop ->

                canStartWatchRecording =
                    canStart

                canStopWatchRecording =
                    canStop
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

    val audioRecorder = remember {
        WavAudioRecorder(
            context.applicationContext
        )
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

    var isRecording by remember {
        mutableStateOf(false)
    }

    var statusText by remember {
        mutableStateOf(
            if (permissionGranted) {
                "Bereit für eine Testaufnahme."
            } else {
                "Die Mikrofonberechtigung wurde " +
                        "noch nicht erteilt."
            }
        )
    }

    var recordingPath by remember {
        mutableStateOf<String?>(null)
    }

    val permissionLauncher =
        rememberLauncherForActivityResult(
            contract =
                ActivityResultContracts
                    .RequestPermission()
        ) { granted ->
            permissionGranted = granted

            statusText =
                if (granted) {
                    "Mikrofonberechtigung wurde erteilt."
                } else {
                    "Mikrofonberechtigung wurde verweigert."
                }
        }

    DisposableEffect(audioRecorder) {
        onDispose {
            audioRecorder.stopRecording()
            audioRecorder.release()
        }
    }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .padding(24.dp),
        horizontalAlignment =
            Alignment.CenterHorizontally,
        verticalArrangement =
            Arrangement.Center
    ) {
        Text(
            text = "Audioaufnahme",
            style =
                MaterialTheme.typography
                    .headlineMedium
        )

        Spacer(
            modifier =
                Modifier.height(24.dp)
        )

        Text(
            text = "Smartwatch-Kommunikation",
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
                wearCommunication.prepareSession()
            },
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
                wearCommunication
                    .startPreparedSession()
            },
            enabled =
                canStartWatchRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Smartwatch-Sensoraufnahme starten"
            )
        }

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        Button(
            onClick = {
                wearCommunication
                    .stopCurrentSession()
            },
            enabled =
                canStopWatchRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Smartwatch-Sensoraufnahme stoppen"
            )
        }

        Spacer(
            modifier =
                Modifier.height(24.dp)
        )

        Text(
            text = statusText
        )

        Spacer(
            modifier =
                Modifier.height(8.dp)
        )

        recordingPath?.let { path ->
            Spacer(
                modifier =
                    Modifier.height(16.dp)
            )

            Text(
                text =
                    "Gespeicherte Datei:\n$path",
                style =
                    MaterialTheme.typography
                        .bodySmall
            )
        }

        Spacer(
            modifier =
                Modifier.height(24.dp)
        )

        if (!permissionGranted) {
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

            Spacer(
                modifier =
                    Modifier.height(12.dp)
            )
        }

        Button(
            onClick = {
                try {
                    audioRecorder.startRecording()

                    isRecording = true
                    recordingPath = null

                    statusText =
                        "Aufnahme läuft: Bitte einige " +
                                "Sekunden sprechen."
                } catch (
                    exception: Exception
                ) {
                    isRecording = false

                    statusText =
                        "Aufnahme konnte nicht gestartet " +
                                "werden: " +
                                (
                                        exception.message
                                            ?: "Unbekannter Fehler"
                                        )
                }
            },
            enabled =
                permissionGranted &&
                        !isRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Aufnahme starten"
            )
        }

        Spacer(
            modifier =
                Modifier.height(12.dp)
        )

        Button(
            onClick = {
                val file =
                    audioRecorder.stopRecording()

                isRecording = false

                if (
                    file != null &&
                    file.exists()
                ) {
                    recordingPath =
                        file.absolutePath

                    statusText =
                        "Aufnahme wurde erfolgreich " +
                                "gespeichert."
                } else {
                    recordingPath = null

                    statusText =
                        "Die Aufnahme konnte nicht " +
                                "gespeichert werden."
                }
            },
            enabled = isRecording,
            modifier =
                Modifier.fillMaxWidth()
        ) {
            Text(
                "Aufnahme stoppen"
            )
        }
    }
}