package de.bachelorarbeit.trinkerkennung.wear.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.wear.compose.foundation.lazy.ScalingLazyColumn
import androidx.wear.compose.material3.Button
import androidx.wear.compose.material3.MaterialTheme
import androidx.wear.compose.material3.Text
import java.util.Locale

private const val SENSOR_PERIOD_US = 20_000
private const val UI_UPDATE_INTERVAL_NS = 100_000_000L

private data class AxisValues(
    val x: Float = 0f,
    val y: Float = 0f,
    val z: Float = 0f
)

class MainActivity : ComponentActivity() {

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContent {
            MaterialTheme {
                Box(
                    modifier = Modifier
                        .fillMaxSize()
                        .background(Color.Black)
                ) {
                    MotionSensorScreen()
                }
            }
        }
    }
}

@Composable
private fun MotionSensorScreen() {
    val context = LocalContext.current

    val sensorManager = remember {
        context.getSystemService(Context.SENSOR_SERVICE) as SensorManager
    }

    val sensorRecorder = remember {
        WatchSensorRecorder(context.applicationContext)
    }

    val accelerometer = remember(sensorManager) {
        sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    }

    val gyroscope = remember(sensorManager) {
        sensorManager.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
    }

    var accelerometerValues by remember {
        mutableStateOf(AxisValues())
    }

    var gyroscopeValues by remember {
        mutableStateOf(AxisValues())
    }

    var accelerometerReceived by remember {
        mutableStateOf(false)
    }

    var gyroscopeReceived by remember {
        mutableStateOf(false)
    }

    var isRecording by remember {
        mutableStateOf(false)
    }

    var statusText by remember {
        mutableStateOf("Bereit für eine Sensoraufnahme.")
    }

    var savedFileName by remember {
        mutableStateOf<String?>(null)
    }

    /*
     * Dieser Listener ist nur für die sichtbare Live-Anzeige zuständig.
     * Die CSV-Aufzeichnung erfolgt getrennt im WatchSensorRecorder.
     */
    DisposableEffect(
        sensorManager,
        accelerometer,
        gyroscope
    ) {
        var lastAccelerometerUpdateNs = 0L
        var lastGyroscopeUpdateNs = 0L

        val listener = object : SensorEventListener {

            override fun onSensorChanged(event: SensorEvent) {
                when (event.sensor.type) {
                    Sensor.TYPE_ACCELEROMETER -> {
                        if (
                            event.timestamp - lastAccelerometerUpdateNs >=
                            UI_UPDATE_INTERVAL_NS
                        ) {
                            accelerometerValues = AxisValues(
                                x = event.values[0],
                                y = event.values[1],
                                z = event.values[2]
                            )

                            accelerometerReceived = true
                            lastAccelerometerUpdateNs = event.timestamp
                        }
                    }

                    Sensor.TYPE_GYROSCOPE -> {
                        if (
                            event.timestamp - lastGyroscopeUpdateNs >=
                            UI_UPDATE_INTERVAL_NS
                        ) {
                            gyroscopeValues = AxisValues(
                                x = event.values[0],
                                y = event.values[1],
                                z = event.values[2]
                            )

                            gyroscopeReceived = true
                            lastGyroscopeUpdateNs = event.timestamp
                        }
                    }
                }
            }

            override fun onAccuracyChanged(
                sensor: Sensor?,
                accuracy: Int
            ) {
                // Für die Live-Anzeige ist keine Reaktion erforderlich.
            }
        }

        accelerometer?.let { sensor ->
            sensorManager.registerListener(
                listener,
                sensor,
                SENSOR_PERIOD_US
            )
        }

        gyroscope?.let { sensor ->
            sensorManager.registerListener(
                listener,
                sensor,
                SENSOR_PERIOD_US
            )
        }

        onDispose {
            sensorManager.unregisterListener(listener)
        }
    }

    /*
     * Beendet eine laufende CSV-Aufzeichnung,
     * falls die Oberfläche verlassen wird.
     */
    DisposableEffect(sensorRecorder) {
        onDispose {
            sensorRecorder.release()
        }
    }

    ScalingLazyColumn(
        modifier = Modifier.fillMaxSize(),
        contentPadding = PaddingValues(
            horizontal = 18.dp,
            vertical = 28.dp
        ),
        horizontalAlignment = Alignment.CenterHorizontally
    ) {
        item {
            Text(
                text = "Sensoraufnahme",
                style = MaterialTheme.typography.titleMedium,
                textAlign = TextAlign.Center,
                color = Color.White
            )
        }

        item {
            Text(
                text = statusText,
                modifier = Modifier
                    .fillMaxWidth()
                    .padding(vertical = 8.dp),
                style = MaterialTheme.typography.bodySmall,
                textAlign = TextAlign.Center,
                color = Color.White
            )
        }

        savedFileName?.let { fileName ->
            item {
                Text(
                    text = "Datei:\n$fileName",
                    modifier = Modifier
                        .fillMaxWidth()
                        .padding(bottom = 8.dp),
                    style = MaterialTheme.typography.bodySmall,
                    textAlign = TextAlign.Center,
                    color = Color.White
                )
            }
        }

        item {
            Button(
                onClick = {
                    try {
                        val file = sensorRecorder.startRecording()

                        isRecording = true
                        savedFileName = file.name
                        statusText =
                            "Aufzeichnung läuft. Bewege den Arm."
                    } catch (exception: Exception) {
                        isRecording = false
                        savedFileName = null
                        statusText =
                            "Start fehlgeschlagen: " +
                                    (
                                            exception.message
                                                ?: "Unbekannter Fehler"
                                            )
                    }
                },
                enabled = !isRecording,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Aufnahme starten")
            }
        }

        item {
            Button(
                onClick = {
                    val file = sensorRecorder.stopRecording()

                    isRecording = false

                    if (file != null && file.exists()) {
                        savedFileName = file.name
                        statusText =
                            "Sensoraufnahme wurde gespeichert."
                    } else {
                        savedFileName = null
                        statusText =
                            "Sensoraufnahme konnte nicht gespeichert werden."
                    }
                },
                enabled = isRecording,
                modifier = Modifier.fillMaxWidth()
            ) {
                Text("Aufnahme stoppen")
            }
        }

        item {
            SensorValuesText(
                title = "Beschleunigung",
                unit = "m/s²",
                sensorAvailable = accelerometer != null,
                valuesReceived = accelerometerReceived,
                values = accelerometerValues
            )
        }

        item {
            SensorValuesText(
                title = "Gyroskop",
                unit = "rad/s",
                sensorAvailable = gyroscope != null,
                valuesReceived = gyroscopeReceived,
                values = gyroscopeValues
            )
        }
    }
}

@Composable
private fun SensorValuesText(
    title: String,
    unit: String,
    sensorAvailable: Boolean,
    valuesReceived: Boolean,
    values: AxisValues
) {
    val content = when {
        !sensorAvailable -> {
            "$title\nSensor nicht verfügbar"
        }

        !valuesReceived -> {
            "$title\nWarte auf Messwerte …"
        }

        else -> {
            buildString {
                append(title)

                append("\nX: ")
                append(formatSensorValue(values.x))
                append(" ")
                append(unit)

                append("\nY: ")
                append(formatSensorValue(values.y))
                append(" ")
                append(unit)

                append("\nZ: ")
                append(formatSensorValue(values.z))
                append(" ")
                append(unit)
            }
        }
    }

    Text(
        text = content,
        modifier = Modifier
            .fillMaxWidth()
            .padding(vertical = 14.dp),
        style = MaterialTheme.typography.bodySmall,
        textAlign = TextAlign.Center,
        color = Color.White
    )
}

private fun formatSensorValue(value: Float): String {
    return String.format(
        Locale.US,
        "%.3f",
        value
    )
}