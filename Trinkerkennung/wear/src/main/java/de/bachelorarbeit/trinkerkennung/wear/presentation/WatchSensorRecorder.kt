package de.bachelorarbeit.trinkerkennung.wear.presentation

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import android.os.Environment
import android.os.Handler
import android.os.HandlerThread
import android.os.SystemClock
import java.io.BufferedWriter
import java.io.File
import java.io.FileOutputStream
import java.io.OutputStreamWriter
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.concurrent.CountDownLatch
import java.util.concurrent.TimeUnit

class WatchSensorRecorder(
    private val context: Context
) : SensorEventListener {

    companion object {
        private const val SENSOR_PERIOD_US = 20_000
        private const val MAX_REPORT_LATENCY_US = 0
    }

    private val sensorManager = checkNotNull(
        context.getSystemService(SensorManager::class.java)
    ) {
        "SensorManager ist nicht verfügbar."
    }

    private val accelerometer =
        sensorManager.getDefaultSensor(
            Sensor.TYPE_ACCELEROMETER
        )

    private val gyroscope =
        sensorManager.getDefaultSensor(
            Sensor.TYPE_GYROSCOPE
        )

    @Volatile
    private var acceptsSensorEvents = false

    @Volatile
    private var recordingFailure: Throwable? = null

    private var activeSessionId: String? = null

    private var outputFile: File? = null
    private var writer: BufferedWriter? = null

    private var recordingThread: HandlerThread? = null
    private var recordingHandler: Handler? = null

    private var sessionStartEpochMs = 0L
    private var sessionStartElapsedRealtimeNs = 0L
    private var recordedEventCount = 0L

    /*
     * Diese Variante bleibt für den bisherigen manuellen
     * Start-Button auf der Watch erhalten.
     */
    fun startRecording(): File {
        return startRecording(
            sessionId = "manual"
        )
    }

    /*
     * Diese Variante wird später vom Foreground Service
     * mit der Session-ID des Smartphones verwendet.
     */
    fun startRecording(
        sessionId: String
    ): File {
        require(sessionId.isNotBlank()) {
            "Die Session-ID darf nicht leer sein."
        }

        check(outputFile == null) {
            "Es läuft bereits eine Sensoraufzeichnung."
        }

        val accelerationSensor =
            checkNotNull(accelerometer) {
                "Der Beschleunigungssensor ist nicht verfügbar."
            }

        val rotationSensor =
            checkNotNull(gyroscope) {
                "Das Gyroskop ist nicht verfügbar."
            }

        val recordingDirectory =
            createRecordingDirectory()

        val timestamp = SimpleDateFormat(
            "yyyyMMdd_HHmmss_SSS",
            Locale.US
        ).format(Date())

        val safeSessionId =
            sanitizeForFileName(sessionId)

        val newOutputFile = File(
            recordingDirectory,
            "watch_${safeSessionId}_$timestamp.csv"
        )

        val newWriter = BufferedWriter(
            OutputStreamWriter(
                FileOutputStream(newOutputFile),
                Charsets.UTF_8
            )
        )

        newWriter.write(
            "session_id," +
                    "session_start_epoch_ms," +
                    "session_start_elapsed_realtime_ns," +
                    "sensor," +
                    "event_timestamp_ns," +
                    "relative_time_ns," +
                    "estimated_epoch_ns," +
                    "x," +
                    "y," +
                    "z," +
                    "accuracy"
        )
        newWriter.newLine()
        newWriter.flush()

        val newRecordingThread =
            HandlerThread(
                "WatchSensorCsvRecorder"
            ).apply {
                start()
            }

        val newRecordingHandler =
            Handler(newRecordingThread.looper)

        sessionStartEpochMs =
            System.currentTimeMillis()

        sessionStartElapsedRealtimeNs =
            SystemClock.elapsedRealtimeNanos()

        activeSessionId = sessionId
        outputFile = newOutputFile
        writer = newWriter
        recordingThread = newRecordingThread
        recordingHandler = newRecordingHandler
        recordingFailure = null
        recordedEventCount = 0L
        acceptsSensorEvents = true

        try {
            val accelerometerRegistered =
                sensorManager.registerListener(
                    this,
                    accelerationSensor,
                    SENSOR_PERIOD_US,
                    MAX_REPORT_LATENCY_US,
                    newRecordingHandler
                )

            val gyroscopeRegistered =
                sensorManager.registerListener(
                    this,
                    rotationSensor,
                    SENSOR_PERIOD_US,
                    MAX_REPORT_LATENCY_US,
                    newRecordingHandler
                )

            if (
                !accelerometerRegistered ||
                !gyroscopeRegistered
            ) {
                stopRecording()

                error(
                    "Mindestens ein Sensor konnte " +
                            "nicht registriert werden."
                )
            }
        } catch (exception: Exception) {
            stopRecording()
            throw exception
        }

        return newOutputFile
    }

    fun stopRecording(): File? {
        val completedFile =
            outputFile ?: return null

        val handler = recordingHandler
        val thread = recordingThread

        acceptsSensorEvents = false

        sensorManager.unregisterListener(this)

        val writerClosed =
            CountDownLatch(1)

        if (handler != null) {
            val taskAccepted = handler.post {
                try {
                    writer?.flush()
                    writer?.close()
                } catch (exception: Exception) {
                    if (recordingFailure == null) {
                        recordingFailure = exception
                    }
                } finally {
                    writerClosed.countDown()
                }
            }

            if (!taskAccepted) {
                writerClosed.countDown()
            }
        } else {
            writerClosed.countDown()
        }

        writerClosed.await(
            3,
            TimeUnit.SECONDS
        )

        thread?.quitSafely()
        thread?.join(3000)

        val recordingSucceeded =
            recordingFailure == null &&
                    recordedEventCount > 0 &&
                    completedFile.exists()

        activeSessionId = null
        writer = null
        outputFile = null
        recordingHandler = null
        recordingThread = null
        recordingFailure = null
        recordedEventCount = 0L

        return if (recordingSucceeded) {
            completedFile
        } else {
            completedFile.delete()
            null
        }
    }

    fun release() {
        if (outputFile != null) {
            stopRecording()
        }
    }

    override fun onSensorChanged(
        event: SensorEvent
    ) {
        if (
            !acceptsSensorEvents ||
            recordingFailure != null
        ) {
            return
        }

        val sessionId =
            activeSessionId ?: return

        val sensorName =
            when (event.sensor.type) {
                Sensor.TYPE_ACCELEROMETER ->
                    "ACCELEROMETER"

                Sensor.TYPE_GYROSCOPE ->
                    "GYROSCOPE"

                else ->
                    return
            }

        val relativeTimeNs =
            event.timestamp -
                    sessionStartElapsedRealtimeNs

        /*
         * Einzelne unmittelbar nach der Registrierung gelieferte
         * Sensorereignisse können noch vor dem gespeicherten
         * Sitzungsstart liegen. Diese Werte werden nicht gespeichert.
         */
        if (relativeTimeNs < 0L) {
            return
        }

        val estimatedEpochNs =
            sessionStartEpochMs * 1_000_000L +
                    relativeTimeNs

        try {
            val activeWriter =
                writer ?: return

            activeWriter.write(
                buildString {
                    append(sessionId)
                    append(",")

                    append(sessionStartEpochMs)
                    append(",")

                    append(
                        sessionStartElapsedRealtimeNs
                    )
                    append(",")

                    append(sensorName)
                    append(",")

                    append(event.timestamp)
                    append(",")

                    append(relativeTimeNs)
                    append(",")

                    append(estimatedEpochNs)
                    append(",")

                    append(event.values[0])
                    append(",")

                    append(event.values[1])
                    append(",")

                    append(event.values[2])
                    append(",")

                    append(event.accuracy)
                }
            )

            activeWriter.newLine()
            recordedEventCount++
        } catch (exception: Exception) {
            recordingFailure = exception
            acceptsSensorEvents = false
        }
    }

    override fun onAccuracyChanged(
        sensor: Sensor?,
        accuracy: Int
    ) {
        /*
         * Der aktuelle Genauigkeitsstatus wird bereits
         * mit jedem Sensorereignis gespeichert.
         */
    }

    private fun createRecordingDirectory(): File {
        val baseDirectory =
            context.getExternalFilesDir(
                Environment.DIRECTORY_DOCUMENTS
            ) ?: context.filesDir

        val recordingDirectory = File(
            baseDirectory,
            "sensor_recordings"
        )

        check(
            recordingDirectory.exists() ||
                    recordingDirectory.mkdirs()
        ) {
            "Der Aufnahmeordner konnte nicht erstellt werden."
        }

        return recordingDirectory
    }

    private fun sanitizeForFileName(
        value: String
    ): String {
        return value.replace(
            regex = Regex("[^A-Za-z0-9_-]"),
            replacement = "_"
        )
    }
}