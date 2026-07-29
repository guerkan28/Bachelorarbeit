package de.bachelorarbeit.trinkerkennung.wear.presentation

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.PendingIntent
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.pm.ServiceInfo
import android.os.Build
import android.os.Handler
import android.os.IBinder
import android.os.Looper
import android.os.PowerManager
import android.os.VibrationEffect
import android.os.Vibrator
import android.os.VibratorManager
import android.util.Log
import com.google.android.gms.wearable.Wearable
import de.bachelorarbeit.trinkerkennung.wear.R
import java.io.File
import java.nio.charset.StandardCharsets

class WatchRecordingService : Service() {

    companion object {
        private const val LOG_TAG =
            "WatchRecordingService"

        private const val ACTION_PREPARE =
            "de.bachelorarbeit.trinkerkennung.action.PREPARE"

        private const val ACTION_START =
            "de.bachelorarbeit.trinkerkennung.action.START"

        private const val ACTION_STOP =
            "de.bachelorarbeit.trinkerkennung.action.STOP"

        private const val EXTRA_SESSION_ID =
            "extra_session_id"

        private const val EXTRA_SOURCE_NODE_ID =
            "extra_source_node_id"

        private const val READY_PATH =
            "/trinkerkennung/session/ready"

        private const val STARTED_PATH =
            "/trinkerkennung/session/started"

        private const val STOPPED_PATH =
            "/trinkerkennung/session/stopped"

        private const val NOTIFICATION_CHANNEL_ID =
            "sensor_recording"

        private const val NOTIFICATION_ID =
            1001

        private const val PREPARATION_TIMEOUT_MS =
            120_000L

        private const val MAX_WAKE_LOCK_DURATION_MS =
            30L * 60L * 1000L

        fun createPrepareIntent(
            context: Context,
            sessionId: String,
            sourceNodeId: String
        ): Intent {
            return createCommandIntent(
                context = context,
                action = ACTION_PREPARE,
                sessionId = sessionId,
                sourceNodeId = sourceNodeId
            )
        }

        fun createStartIntent(
            context: Context,
            sessionId: String,
            sourceNodeId: String
        ): Intent {
            return createCommandIntent(
                context = context,
                action = ACTION_START,
                sessionId = sessionId,
                sourceNodeId = sourceNodeId
            )
        }

        fun createStopIntent(
            context: Context,
            sessionId: String,
            sourceNodeId: String
        ): Intent {
            return createCommandIntent(
                context = context,
                action = ACTION_STOP,
                sessionId = sessionId,
                sourceNodeId = sourceNodeId
            )
        }

        private fun createCommandIntent(
            context: Context,
            action: String,
            sessionId: String,
            sourceNodeId: String
        ): Intent {
            return Intent(
                context,
                WatchRecordingService::class.java
            ).apply {
                this.action = action

                putExtra(
                    EXTRA_SESSION_ID,
                    sessionId
                )

                putExtra(
                    EXTRA_SOURCE_NODE_ID,
                    sourceNodeId
                )
            }
        }
    }

    private val messageClient by lazy {
        Wearable.getMessageClient(this)
    }

    private val notificationManager by lazy {
        getSystemService(
            NotificationManager::class.java
        )
    }

    private val sensorRecorder by lazy {
        WatchSensorRecorder(applicationContext)
    }

    private val timeoutHandler =
        Handler(Looper.getMainLooper())

    private var currentSessionId: String? = null
    private var currentSourceNodeId: String? = null
    private var currentOutputFile: File? = null

    private var isRecording = false

    private var wakeLock: PowerManager.WakeLock? = null

    private val preparationTimeoutTask =
        Runnable {
            Log.i(
                LOG_TAG,
                "Vorbereitungszeit abgelaufen. Dienst wird beendet."
            )

            stopPreparedService()
        }

    override fun onCreate() {
        super.onCreate()

        createNotificationChannel()

        Log.i(
            LOG_TAG,
            "WatchRecordingService wurde erstellt."
        )
    }

    override fun onBind(intent: Intent?): IBinder? {
        return null
    }

    override fun onStartCommand(
        intent: Intent?,
        flags: Int,
        startId: Int
    ): Int {
        val action = intent?.action

        val sessionId =
            intent?.getStringExtra(
                EXTRA_SESSION_ID
            )

        val sourceNodeId =
            intent?.getStringExtra(
                EXTRA_SOURCE_NODE_ID
            )

        if (
            action.isNullOrBlank() ||
            sessionId.isNullOrBlank() ||
            sourceNodeId.isNullOrBlank()
        ) {
            Log.e(
                LOG_TAG,
                "Service-Befehl enthält keine gültigen Daten."
            )

            stopSelf(startId)

            return START_NOT_STICKY
        }

        /*
         * startForegroundService() verlangt, dass der Dienst
         * kurzfristig tatsächlich in den Vordergrund wechselt.
         */
        ensureForeground(
            statusText = "Verarbeite Sitzungsbefehl …"
        )

        when (action) {
            ACTION_PREPARE -> {
                prepareSession(
                    sessionId = sessionId,
                    sourceNodeId = sourceNodeId
                )
            }

            ACTION_START -> {
                startSensorRecording(
                    sessionId = sessionId,
                    sourceNodeId = sourceNodeId
                )
            }

            ACTION_STOP -> {
                stopSensorRecording(
                    sessionId = sessionId,
                    sourceNodeId = sourceNodeId
                )
            }

            else -> {
                Log.w(
                    LOG_TAG,
                    "Unbekannte Service-Aktion: $action"
                )

                finishService()
            }
        }

        return START_NOT_STICKY
    }

    private fun prepareSession(
        sessionId: String,
        sourceNodeId: String
    ) {
        if (isRecording) {
            Log.w(
                LOG_TAG,
                "Neue Vorbereitung abgelehnt, " +
                        "da bereits eine Aufnahme läuft."
            )
            return
        }

        currentSessionId = sessionId
        currentSourceNodeId = sourceNodeId
        currentOutputFile = null

        updateNotification(
            statusText =
                "Sitzung ${sessionId.take(8)} ist vorbereitet."
        )

        timeoutHandler.removeCallbacks(
            preparationTimeoutTask
        )

        vibrateReady()

        sendReady(
            sessionId = sessionId,
            sourceNodeId = sourceNodeId
        )

        timeoutHandler.postDelayed(
            preparationTimeoutTask,
            PREPARATION_TIMEOUT_MS
        )

        Log.i(
            LOG_TAG,
            "Sitzung ${sessionId.take(8)} wurde vorbereitet."
        )
    }

    private fun startSensorRecording(
        sessionId: String,
        sourceNodeId: String
    ) {
        if (isRecording) {
            Log.w(
                LOG_TAG,
                "START ignoriert: Eine Aufnahme läuft bereits."
            )
            return
        }

        if (
            sessionId != currentSessionId ||
            sourceNodeId != currentSourceNodeId
        ) {
            Log.e(
                LOG_TAG,
                "START gehört nicht zur vorbereiteten Sitzung."
            )

            /*
             * Wurde der Dienst nur durch einen ungültigen START-Befehl
             * neu erzeugt, darf er nicht ohne aktive Sitzung weiterlaufen.
             * Eine bereits laufende andere Sitzung wird dagegen nicht beendet.
             */
            if (
                currentSessionId == null &&
                !isRecording
            ) {
                finishService()
            }

            return
        }

        timeoutHandler.removeCallbacks(
            preparationTimeoutTask
        )

        try {
            acquireWakeLock()

            val outputFile =
                sensorRecorder.startRecording(
                    sessionId = sessionId
                )

            currentOutputFile = outputFile
            isRecording = true

            updateNotification(
                statusText =
                    "Sensoraufnahme ${sessionId.take(8)} läuft."
            )

            sendStarted(
                sessionId = sessionId,
                sourceNodeId = sourceNodeId,
                outputFile = outputFile
            )

            Log.i(
                LOG_TAG,
                "Sensoraufnahme für Sitzung " +
                        "${sessionId.take(8)} gestartet."
            )
        } catch (exception: Exception) {
            releaseWakeLock()

            currentOutputFile = null
            isRecording = false

            Log.e(
                LOG_TAG,
                "Sensoraufnahme konnte nicht gestartet werden.",
                exception
            )
        }
    }

    private fun stopSensorRecording(
        sessionId: String,
        sourceNodeId: String
    ) {
        if (
            sessionId != currentSessionId ||
            sourceNodeId != currentSourceNodeId
        ) {
            Log.e(
                LOG_TAG,
                "STOP gehört nicht zur aktiven Sitzung."
            )

            /*
             * Ein allein durch einen ungültigen STOP-Befehl erzeugter
             * Dienst wird sofort wieder beendet.
             */
            if (
                currentSessionId == null &&
                !isRecording
            ) {
                finishService()
            }

            return
        }

        timeoutHandler.removeCallbacks(
            preparationTimeoutTask
        )

        val completedFile =
            if (isRecording) {
                sensorRecorder.stopRecording()
            } else {
                null
            }

        isRecording = false
        currentOutputFile = completedFile

        releaseWakeLock()

        updateNotification(
            statusText =
                "Sensoraufnahme wird abgeschlossen."
        )

        sendStopped(
            sessionId = sessionId,
            sourceNodeId = sourceNodeId,
            completedFile = completedFile
        )
    }

    private fun sendReady(
        sessionId: String,
        sourceNodeId: String
    ) {
        val readyPayload = buildString {
            append("session_id=")
            append(sessionId)

            append(";watch_ready_epoch_ms=")
            append(System.currentTimeMillis())
        }.toByteArray(StandardCharsets.UTF_8)

        messageClient
            .sendMessage(
                sourceNodeId,
                READY_PATH,
                readyPayload
            )
            .addOnSuccessListener {
                Log.i(
                    LOG_TAG,
                    "READY für Sitzung " +
                            "${sessionId.take(8)} gesendet."
                )
            }
            .addOnFailureListener { exception ->
                Log.e(
                    LOG_TAG,
                    "READY konnte nicht gesendet werden.",
                    exception
                )
            }
    }

    private fun sendStarted(
        sessionId: String,
        sourceNodeId: String,
        outputFile: File
    ) {
        val startedPayload = buildString {
            append("session_id=")
            append(sessionId)

            append(";watch_start_epoch_ms=")
            append(System.currentTimeMillis())

            append(";watch_file_name=")
            append(outputFile.name)
        }.toByteArray(StandardCharsets.UTF_8)

        messageClient
            .sendMessage(
                sourceNodeId,
                STARTED_PATH,
                startedPayload
            )
            .addOnSuccessListener {
                Log.i(
                    LOG_TAG,
                    "STARTED für Sitzung " +
                            "${sessionId.take(8)} gesendet."
                )
            }
            .addOnFailureListener { exception ->
                Log.e(
                    LOG_TAG,
                    "STARTED konnte nicht gesendet werden.",
                    exception
                )
            }
    }

    private fun sendStopped(
        sessionId: String,
        sourceNodeId: String,
        completedFile: File?
    ) {
        val recordingSucceeded =
            completedFile != null &&
                    completedFile.exists()

        val stoppedPayload = buildString {
            append("session_id=")
            append(sessionId)

            append(";watch_stop_epoch_ms=")
            append(System.currentTimeMillis())

            append(";recording_success=")
            append(recordingSucceeded)

            append(";watch_file_name=")
            append(
                completedFile?.name
                    ?: "none"
            )
        }.toByteArray(StandardCharsets.UTF_8)

        messageClient
            .sendMessage(
                sourceNodeId,
                STOPPED_PATH,
                stoppedPayload
            )
            .addOnSuccessListener {
                Log.i(
                    LOG_TAG,
                    "STOPPED für Sitzung " +
                            "${sessionId.take(8)} gesendet."
                )
            }
            .addOnFailureListener { exception ->
                Log.e(
                    LOG_TAG,
                    "STOPPED konnte nicht gesendet werden.",
                    exception
                )
            }
            .addOnCompleteListener {
                finishService()
            }
    }

    private fun ensureForeground(
        statusText: String
    ) {
        val notification =
            createNotification(
                statusText = statusText
            )

        startForeground(
            NOTIFICATION_ID,
            notification,
            ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH
        )
    }

    private fun updateNotification(
        statusText: String
    ) {
        startForeground(
            NOTIFICATION_ID,
            createNotification(
                statusText = statusText
            ),
            ServiceInfo.FOREGROUND_SERVICE_TYPE_HEALTH
        )
    }

    private fun createNotification(
        statusText: String
    ): Notification {
        val openAppIntent = Intent(
            this,
            MainActivity::class.java
        ).apply {
            flags =
                Intent.FLAG_ACTIVITY_CLEAR_TOP or
                        Intent.FLAG_ACTIVITY_SINGLE_TOP
        }

        val openAppPendingIntent =
            PendingIntent.getActivity(
                this,
                0,
                openAppIntent,
                PendingIntent.FLAG_UPDATE_CURRENT or
                        PendingIntent.FLAG_IMMUTABLE
            )

        return Notification.Builder(
            this,
            NOTIFICATION_CHANNEL_ID
        )
            .setSmallIcon(R.mipmap.ic_launcher)
            .setContentTitle("Trinkerkennung")
            .setContentText(statusText)
            .setContentIntent(openAppPendingIntent)
            .setCategory(Notification.CATEGORY_SERVICE)
            .setOngoing(true)
            .setOnlyAlertOnce(true)
            .build()
    }

    private fun createNotificationChannel() {
        val channel = NotificationChannel(
            NOTIFICATION_CHANNEL_ID,
            "Sensoraufzeichnung",
            NotificationManager.IMPORTANCE_LOW
        ).apply {
            description =
                "Status der Smartwatch-Sensoraufzeichnung"
        }

        notificationManager.createNotificationChannel(
            channel
        )
    }

    private fun acquireWakeLock() {
        if (wakeLock?.isHeld == true) {
            return
        }

        val powerManager =
            getSystemService(
                PowerManager::class.java
            )

        wakeLock = powerManager.newWakeLock(
            PowerManager.PARTIAL_WAKE_LOCK,
            "$packageName:WatchSensorRecording"
        ).apply {
            setReferenceCounted(false)

            acquire(
                MAX_WAKE_LOCK_DURATION_MS
            )
        }

        Log.i(
            LOG_TAG,
            "Partial Wake Lock wurde aktiviert."
        )
    }

    private fun releaseWakeLock() {
        val activeWakeLock =
            wakeLock

        if (activeWakeLock?.isHeld == true) {
            activeWakeLock.release()

            Log.i(
                LOG_TAG,
                "Partial Wake Lock wurde freigegeben."
            )
        }

        wakeLock = null
    }

    private fun vibrateReady() {
        val vibrator: Vibrator? =
            if (
                Build.VERSION.SDK_INT >=
                Build.VERSION_CODES.S
            ) {
                getSystemService(
                    VibratorManager::class.java
                )?.defaultVibrator
            } else {
                @Suppress("DEPRECATION")
                getSystemService(
                    Context.VIBRATOR_SERVICE
                ) as? Vibrator
            }

        if (vibrator?.hasVibrator() != true) {
            Log.w(
                LOG_TAG,
                "Kein Vibrationsmotor verfügbar."
            )
            return
        }

        val effect =
            VibrationEffect.createOneShot(
                150L,
                VibrationEffect.DEFAULT_AMPLITUDE
            )

        vibrator.vibrate(effect)

        Log.i(
            LOG_TAG,
            "Bereitschaftsvibration ausgelöst."
        )
    }

    private fun stopPreparedService() {
        if (isRecording) {
            Log.w(
                LOG_TAG,
                "Vorbereitungs-Timeout ignoriert, " +
                        "da eine Aufnahme läuft."
            )
            return
        }

        finishService()
    }

    private fun finishService() {
        timeoutHandler.removeCallbacks(
            preparationTimeoutTask
        )

        if (isRecording) {
            sensorRecorder.release()
            isRecording = false
        }

        releaseWakeLock()

        currentSessionId = null
        currentSourceNodeId = null
        currentOutputFile = null

        stopForeground(STOP_FOREGROUND_REMOVE)
        stopSelf()
    }

    override fun onDestroy() {
        timeoutHandler.removeCallbacks(
            preparationTimeoutTask
        )

        if (isRecording) {
            sensorRecorder.release()
            isRecording = false
        }

        releaseWakeLock()

        currentSessionId = null
        currentSourceNodeId = null
        currentOutputFile = null

        Log.i(
            LOG_TAG,
            "WatchRecordingService wurde beendet."
        )

        super.onDestroy()
    }
}