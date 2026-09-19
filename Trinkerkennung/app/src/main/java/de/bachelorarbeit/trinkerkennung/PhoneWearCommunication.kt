package de.bachelorarbeit.trinkerkennung

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import java.nio.charset.StandardCharsets
import java.util.UUID

class PhoneWearCommunication(
    context: Context,
    private val sessionMetadataStore:
    SessionMetadataStore,
    private val onStatusChanged: (String) -> Unit,
    private val onSessionControlsChanged: (
        canStart: Boolean,
        canStop: Boolean
    ) -> Unit,
    private val onSessionStartFailed: (
        sessionId: String,
        reason: String
    ) -> Unit,
    private val onSessionStopped: (
        sessionId: String,
        watchRecordingSucceeded: Boolean,
        watchFileName: String?
    ) -> Unit
) : MessageClient.OnMessageReceivedListener {

    companion object {
        private const val PING_PATH =
            "/trinkerkennung/test/ping"

        private const val ACK_PATH =
            "/trinkerkennung/test/ack"

        private const val PREPARE_PATH =
            "/trinkerkennung/session/prepare"

        private const val READY_PATH =
            "/trinkerkennung/session/ready"

        private const val START_PATH =
            "/trinkerkennung/session/start"

        private const val STARTED_PATH =
            "/trinkerkennung/session/started"

        private const val STOP_PATH =
            "/trinkerkennung/session/stop"

        private const val STOPPED_PATH =
            "/trinkerkennung/session/stopped"

        private const val SESSION_ACK_TIMEOUT_MS =
            10_000L
    }

    private enum class SessionState {
        IDLE,
        PREPARING,
        READY,
        STARTING,
        RECORDING,
        STOPPING
    }

    private val applicationContext =
        context.applicationContext

    private val messageClient =
        Wearable.getMessageClient(
            applicationContext
        )

    private val nodeClient =
        Wearable.getNodeClient(
            applicationContext
        )

    private val mainHandler =
        Handler(Looper.getMainLooper())

    @Volatile
    private var currentSessionId: String? =
        null

    @Volatile
    private var currentWatchNodeId: String? =
        null

    @Volatile
    private var sessionState =
        SessionState.IDLE

    private var startAckTimeoutRunnable: Runnable? =
        null

    private var stopAckTimeoutRunnable: Runnable? =
        null

    fun startListening() {
        notifySessionControls()

        messageClient
            .addListener(this)
            .addOnFailureListener { exception ->
                updateStatus(
                    "Nachrichtenempfang konnte nicht " +
                            "aktiviert werden: " +
                            errorMessage(exception)
                )
            }
    }

    fun stopListening() {
        messageClient.removeListener(this)
    }

    fun getPreparedSessionId(): String? {
        return if (
            sessionState ==
            SessionState.READY
        ) {
            currentSessionId
        } else {
            null
        }
    }

    fun sendPing() {
        updateStatus(
            "Suche verbundene Smartwatch …"
        )

        nodeClient.connectedNodes
            .addOnSuccessListener { nodes ->
                if (nodes.isEmpty()) {
                    updateStatus(
                        "Keine verbundene Smartwatch gefunden."
                    )
                    return@addOnSuccessListener
                }

                val payload =
                    System.currentTimeMillis()
                        .toString()
                        .toByteArray(
                            StandardCharsets.UTF_8
                        )

                val node =
                    nodes.first()

                messageClient
                    .sendMessage(
                        node.id,
                        PING_PATH,
                        payload
                    )
                    .addOnSuccessListener {
                        updateStatus(
                            "PING an ${node.displayName} " +
                                    "gesendet. Warte auf ACK …"
                        )
                    }
                    .addOnFailureListener { exception ->
                        updateStatus(
                            "PING konnte nicht gesendet " +
                                    "werden: " +
                                    errorMessage(exception)
                        )
                    }
            }
            .addOnFailureListener { exception ->
                updateStatus(
                    "Gerätesuche fehlgeschlagen: " +
                            errorMessage(exception)
                )
            }
    }

    fun prepareSession() {
        if (
            sessionState ==
            SessionState.STARTING ||
            sessionState ==
            SessionState.RECORDING ||
            sessionState ==
            SessionState.STOPPING
        ) {
            updateStatus(
                "Eine laufende Sitzung muss zuerst " +
                        "beendet werden."
            )
            return
        }

        currentSessionId?.let { previousSessionId ->
            sessionMetadataStore
                .discardSession(previousSessionId)
        }

        if (
            sessionState !=
            SessionState.IDLE
        ) {
            resetSession()
        }

        val sessionId =
            UUID.randomUUID().toString()

        val phonePrepareEpochMs =
            System.currentTimeMillis()

        sessionMetadataStore.beginSession(
            sessionId =
                sessionId,
            phonePrepareEpochMs =
                phonePrepareEpochMs
        )

        currentSessionId =
            sessionId

        currentWatchNodeId =
            null

        updateSessionState(
            SessionState.PREPARING
        )

        updateStatus(
            "Bereite Sitzung " +
                    "${sessionId.take(8)} vor …"
        )

        nodeClient.connectedNodes
            .addOnSuccessListener { nodes ->
                if (nodes.isEmpty()) {
                    if (
                        currentSessionId ==
                        sessionId
                    ) {
                        sessionMetadataStore
                            .discardSession(sessionId)

                        resetSession()
                    }

                    updateStatus(
                        "Keine verbundene Smartwatch gefunden."
                    )
                    return@addOnSuccessListener
                }

                val node =
                    nodes.first()

                if (
                    currentSessionId !=
                    sessionId
                ) {
                    return@addOnSuccessListener
                }

                currentWatchNodeId =
                    node.id

                val payload = buildString {
                    append("session_id=")
                    append(sessionId)

                    append(
                        ";phone_prepare_epoch_ms="
                    )
                    append(
                        phonePrepareEpochMs
                    )
                }.toByteArray(
                    StandardCharsets.UTF_8
                )

                messageClient
                    .sendMessage(
                        node.id,
                        PREPARE_PATH,
                        payload
                    )
                    .addOnSuccessListener {
                        if (
                            currentSessionId ==
                            sessionId &&
                            sessionState ==
                            SessionState.PREPARING
                        ) {
                            updateStatus(
                                "PREPARE für Sitzung " +
                                        "${sessionId.take(8)} " +
                                        "gesendet. Warte auf READY …"
                            )
                        }
                    }
                    .addOnFailureListener { exception ->
                        if (
                            currentSessionId ==
                            sessionId
                        ) {
                            sessionMetadataStore
                                .discardSession(sessionId)

                            resetSession()
                        }

                        updateStatus(
                            "PREPARE konnte nicht gesendet " +
                                    "werden: " +
                                    errorMessage(exception)
                        )
                    }
            }
            .addOnFailureListener { exception ->
                if (
                    currentSessionId ==
                    sessionId
                ) {
                    sessionMetadataStore
                        .discardSession(sessionId)

                    resetSession()
                }

                updateStatus(
                    "Gerätesuche fehlgeschlagen: " +
                            errorMessage(exception)
                )
            }
    }

    fun startPreparedSession(): Boolean {
        if (
            sessionState !=
            SessionState.READY
        ) {
            updateStatus(
                "Die Smartwatch ist noch nicht für " +
                        "eine Aufnahme bereit."
            )
            return false
        }

        val sessionId =
            currentSessionId

        val watchNodeId =
            currentWatchNodeId

        if (
            sessionId.isNullOrBlank() ||
            watchNodeId.isNullOrBlank()
        ) {
            if (!sessionId.isNullOrBlank()) {
                sessionMetadataStore
                    .recordFailure(
                        sessionId,
                        "Ungültige Verbindungsdaten " +
                                "beim Sitzungsstart."
                    )
            }

            resetSession()

            updateStatus(
                "Die vorbereitete Sitzung enthält " +
                        "keine gültigen Verbindungsdaten."
            )
            return false
        }

        val phoneStartCommandEpochMs =
            System.currentTimeMillis()

        sessionMetadataStore
            .recordStartCommand(
                sessionId =
                    sessionId,
                phoneStartCommandEpochMs =
                    phoneStartCommandEpochMs
            )

        updateSessionState(
            SessionState.STARTING
        )

        scheduleStartAckTimeout(
            sessionId = sessionId,
            watchNodeId = watchNodeId
        )

        val payload = buildString {
            append("session_id=")
            append(sessionId)

            append(
                ";phone_start_command_epoch_ms="
            )
            append(
                phoneStartCommandEpochMs
            )
        }.toByteArray(
            StandardCharsets.UTF_8
        )

        updateStatus(
            "Starte gemeinsame Aufnahme für " +
                    "Sitzung ${sessionId.take(8)} …"
        )

        messageClient
            .sendMessage(
                watchNodeId,
                START_PATH,
                payload
            )
            .addOnSuccessListener {
                if (
                    currentSessionId ==
                    sessionId &&
                    sessionState ==
                    SessionState.STARTING
                ) {
                    updateStatus(
                        "START für Sitzung " +
                                "${sessionId.take(8)} " +
                                "gesendet. Warte auf STARTED …"
                    )
                }
            }
            .addOnFailureListener { exception ->
                if (
                    currentSessionId !=
                    sessionId ||
                    sessionState !=
                    SessionState.STARTING
                ) {
                    return@addOnFailureListener
                }

                val reason =
                    errorMessage(exception)

                sessionMetadataStore
                    .recordFailure(
                        sessionId,
                        "START konnte nicht gesendet " +
                                "werden: $reason"
                    )

                resetSession()

                updateStatus(
                    "START konnte nicht gesendet werden: " +
                            reason
                )

                onSessionStartFailed(
                    sessionId,
                    reason
                )
            }

        return true
    }

    fun stopCurrentSession(): Boolean {
        if (
            sessionState !=
            SessionState.RECORDING
        ) {
            updateStatus(
                "Es läuft derzeit keine bestätigte " +
                        "gemeinsame Aufnahme."
            )
            return false
        }

        val sessionId =
            currentSessionId

        val watchNodeId =
            currentWatchNodeId

        if (
            sessionId.isNullOrBlank() ||
            watchNodeId.isNullOrBlank()
        ) {
            resetSession()

            updateStatus(
                "Die laufende Sitzung enthält keine " +
                        "gültigen Verbindungsdaten."
            )
            return false
        }

        val phoneStopCommandEpochMs =
            System.currentTimeMillis()

        sessionMetadataStore
            .recordStopCommand(
                sessionId =
                    sessionId,
                phoneStopCommandEpochMs =
                    phoneStopCommandEpochMs
            )

        updateSessionState(
            SessionState.STOPPING
        )

        scheduleStopAckTimeout(
            sessionId = sessionId
        )

        val payload = buildString {
            append("session_id=")
            append(sessionId)

            append(
                ";phone_stop_command_epoch_ms="
            )
            append(
                phoneStopCommandEpochMs
            )
        }.toByteArray(
            StandardCharsets.UTF_8
        )

        updateStatus(
            "Beende gemeinsame Aufnahme für " +
                    "Sitzung ${sessionId.take(8)} …"
        )

        messageClient
            .sendMessage(
                watchNodeId,
                STOP_PATH,
                payload
            )
            .addOnSuccessListener {
                if (
                    currentSessionId ==
                    sessionId &&
                    sessionState ==
                    SessionState.STOPPING
                ) {
                    updateStatus(
                        "STOP für Sitzung " +
                                "${sessionId.take(8)} " +
                                "gesendet. Warte auf STOPPED …"
                    )
                }
            }
            .addOnFailureListener { exception ->
                if (
                    currentSessionId ==
                    sessionId &&
                    sessionState ==
                    SessionState.STOPPING
                ) {
                    cancelStopAckTimeout()

                    updateSessionState(
                        SessionState.RECORDING
                    )
                }

                updateStatus(
                    "STOP konnte nicht gesendet werden: " +
                            errorMessage(exception)
                )
            }

        return true
    }

    override fun onMessageReceived(
        messageEvent: MessageEvent
    ) {
        when (messageEvent.path) {
            ACK_PATH ->
                handlePingAcknowledgement(
                    messageEvent
                )

            READY_PATH ->
                handleSessionReady(
                    messageEvent
                )

            STARTED_PATH ->
                handleSessionStarted(
                    messageEvent
                )

            STOPPED_PATH ->
                handleSessionStopped(
                    messageEvent
                )
        }
    }

    private fun handlePingAcknowledgement(
        messageEvent: MessageEvent
    ) {
        val phoneReceiveTimestampMs =
            System.currentTimeMillis()

        val payload =
            messageEvent.data.toString(
                StandardCharsets.UTF_8
            )

        val values =
            parsePayload(payload)

        val phoneSendTimestampMs =
            values["phone_timestamp_ms"]
                ?.toLongOrNull()

        val watchTimestampMs =
            values["watch_timestamp_ms"]
                ?.toLongOrNull()

        if (
            phoneSendTimestampMs == null ||
            watchTimestampMs == null
        ) {
            updateStatus(
                "ACK empfangen, aber Zeitstempel " +
                        "konnten nicht ausgewertet werden:\n" +
                        payload
            )
            return
        }

        val roundTripTimeMs =
            phoneReceiveTimestampMs -
                    phoneSendTimestampMs

        updateStatus(
            buildString {
                append(
                    "ACK von der Smartwatch empfangen."
                )
                append(
                    "\nRound-Trip-Time: "
                )
                append(roundTripTimeMs)
                append(" ms")
                append(
                    "\nSmartphone Start: "
                )
                append(phoneSendTimestampMs)
                append(
                    "\nWatch Verarbeitung: "
                )
                append(watchTimestampMs)
                append(
                    "\nSmartphone Empfang: "
                )
                append(phoneReceiveTimestampMs)
            }
        )
    }

    private fun handleSessionReady(
        messageEvent: MessageEvent
    ) {
        val phoneReadyReceivedEpochMs =
            System.currentTimeMillis()

        val values =
            parsePayload(
                messageEvent.data.toString(
                    StandardCharsets.UTF_8
                )
            )

        val receivedSessionId =
            values["session_id"]

        if (
            !isExpectedSessionMessage(
                messageEvent,
                receivedSessionId,
                "READY"
            )
        ) {
            return
        }

        if (
            sessionState !=
            SessionState.PREPARING &&
            sessionState !=
            SessionState.READY
        ) {
            return
        }

        val confirmedSessionId =
            receivedSessionId
                ?: return

        val watchReadyEpochMs =
            values["watch_ready_epoch_ms"]
                ?.toLongOrNull()

        sessionMetadataStore.recordReady(
            sessionId =
                confirmedSessionId,
            phoneReadyReceivedEpochMs =
                phoneReadyReceivedEpochMs,
            watchReadyEpochMs =
                watchReadyEpochMs
        )

        updateSessionState(
            SessionState.READY
        )

        updateStatus(
            buildString {
                append(
                    "Smartwatch ist bereit."
                )
                append("\nSession: ")
                append(
                    confirmedSessionId.take(8)
                )
                append(
                    "\nWatch READY: "
                )
                append(
                    watchReadyEpochMs
                        ?: "unbekannt"
                )
            }
        )
    }

    private fun handleSessionStarted(
        messageEvent: MessageEvent
    ) {
        val phoneStartedReceivedEpochMs =
            System.currentTimeMillis()

        val values =
            parsePayload(
                messageEvent.data.toString(
                    StandardCharsets.UTF_8
                )
            )

        val receivedSessionId =
            values["session_id"]

        if (
            !isExpectedSessionMessage(
                messageEvent,
                receivedSessionId,
                "STARTED"
            )
        ) {
            return
        }

        if (
            sessionState !=
            SessionState.STARTING &&
            sessionState !=
            SessionState.RECORDING
        ) {
            return
        }

        val confirmedSessionId =
            receivedSessionId
                ?: return

        val watchStartEpochMs =
            values["watch_start_epoch_ms"]
                ?.toLongOrNull()

        val watchFileName =
            values["watch_file_name"]
                ?.takeUnless {
                    it == "none" ||
                            it == "unbekannt"
                }

        cancelStartAckTimeout()

        sessionMetadataStore.recordStarted(
            sessionId =
                confirmedSessionId,
            phoneStartedReceivedEpochMs =
                phoneStartedReceivedEpochMs,
            watchStartEpochMs =
                watchStartEpochMs,
            watchFileName =
                watchFileName
        )

        updateSessionState(
            SessionState.RECORDING
        )

        updateStatus(
            buildString {
                append(
                    "Gemeinsame Aufnahme läuft."
                )
                append("\nSession: ")
                append(
                    confirmedSessionId.take(8)
                )
                append(
                    "\nWatch STARTED: "
                )
                append(
                    watchStartEpochMs
                        ?: "unbekannt"
                )
                append(
                    "\nWatch-Datei: "
                )
                append(
                    watchFileName
                        ?: "unbekannt"
                )
            }
        )
    }

    private fun handleSessionStopped(
        messageEvent: MessageEvent
    ) {
        val phoneStoppedReceivedEpochMs =
            System.currentTimeMillis()

        val values =
            parsePayload(
                messageEvent.data.toString(
                    StandardCharsets.UTF_8
                )
            )

        val receivedSessionId =
            values["session_id"]

        if (
            !isExpectedSessionMessage(
                messageEvent,
                receivedSessionId,
                "STOPPED"
            )
        ) {
            return
        }

        if (
            sessionState !=
            SessionState.STOPPING &&
            sessionState !=
            SessionState.RECORDING
        ) {
            return
        }

        val confirmedSessionId =
            receivedSessionId
                ?: return

        val watchStopEpochMs =
            values["watch_stop_epoch_ms"]
                ?.toLongOrNull()

        val watchRecordingSucceeded =
            values["recording_success"] ==
                    "true"

        val watchFileName =
            values["watch_file_name"]
                ?.takeUnless {
                    it == "none"
                }

        cancelStopAckTimeout()

        sessionMetadataStore.recordStopped(
            sessionId =
                confirmedSessionId,
            phoneStoppedReceivedEpochMs =
                phoneStoppedReceivedEpochMs,
            watchStopEpochMs =
                watchStopEpochMs,
            watchRecordingSuccess =
                watchRecordingSucceeded,
            watchFileName =
                watchFileName
        )

        resetSession()

        updateStatus(
            buildString {
                append(
                    "Smartwatch-Aufnahme beendet."
                )
                append("\nSession: ")
                append(
                    confirmedSessionId.take(8)
                )
                append(
                    "\nWatch erfolgreich gespeichert: "
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
                    "\nWatch STOPPED: "
                )
                append(
                    watchStopEpochMs
                        ?: "unbekannt"
                )
                append(
                    "\nWatch-Datei: "
                )
                append(
                    watchFileName
                        ?: "keine Datei"
                )
            }
        )

        onSessionStopped(
            confirmedSessionId,
            watchRecordingSucceeded,
            watchFileName
        )
    }

    private fun isExpectedSessionMessage(
        messageEvent: MessageEvent,
        receivedSessionId: String?,
        messageName: String
    ): Boolean {
        val expectedSessionId =
            currentSessionId

        val expectedWatchNodeId =
            currentWatchNodeId

        if (
            receivedSessionId.isNullOrBlank() ||
            expectedSessionId.isNullOrBlank()
        ) {
            updateStatus(
                "$messageName ohne gültige " +
                        "Session-ID empfangen."
            )
            return false
        }

        if (
            receivedSessionId !=
            expectedSessionId
        ) {
            updateStatus(
                "$messageName gehört zu einer " +
                        "anderen Sitzung."
            )
            return false
        }

        if (
            expectedWatchNodeId != null &&
            messageEvent.sourceNodeId !=
            expectedWatchNodeId
        ) {
            updateStatus(
                "$messageName wurde von einer " +
                        "anderen Watch empfangen."
            )
            return false
        }

        return true
    }

    private fun scheduleStartAckTimeout(
        sessionId: String,
        watchNodeId: String
    ) {
        cancelStartAckTimeout()

        val timeoutRunnable =
            Runnable {
                if (
                    currentSessionId != sessionId ||
                    currentWatchNodeId != watchNodeId ||
                    sessionState !=
                    SessionState.STARTING
                ) {
                    return@Runnable
                }

                val reason =
                    "START_ACK_TIMEOUT: Keine STARTED-" +
                            "Best?tigung innerhalb von " +
                            "${SESSION_ACK_TIMEOUT_MS / 1_000} " +
                            "Sekunden."

                sessionMetadataStore
                    .recordFailure(
                        sessionId,
                        reason
                    )

                /*
                 * START kann die Watch erreicht haben, obwohl die
                 * STARTED-Best?tigung verloren gegangen ist.
                 * Deshalb wird bestm?glich noch ein STOP gesendet,
                 * bevor die Smartphone-Seite die Sitzung verwirft.
                 */
                sendBestEffortStopAfterStartTimeout(
                    sessionId = sessionId,
                    watchNodeId = watchNodeId
                )

                resetSession()

                updateStatus(
                    "STARTED-Timeout f?r Sitzung " +
                            "${sessionId.take(8)}. " +
                            "Die Sitzung wird als technisch " +
                            "fehlgeschlagen beendet."
                )

                onSessionStartFailed(
                    sessionId,
                    reason
                )
            }

        startAckTimeoutRunnable =
            timeoutRunnable

        mainHandler.postDelayed(
            timeoutRunnable,
            SESSION_ACK_TIMEOUT_MS
        )
    }

    private fun scheduleStopAckTimeout(
        sessionId: String
    ) {
        cancelStopAckTimeout()

        val timeoutRunnable =
            Runnable {
                if (
                    currentSessionId != sessionId ||
                    sessionState !=
                    SessionState.STOPPING
                ) {
                    return@Runnable
                }

                val reason =
                    "STOP_ACK_TIMEOUT: Keine STOPPED-" +
                            "Best?tigung innerhalb von " +
                            "${SESSION_ACK_TIMEOUT_MS / 1_000} " +
                            "Sekunden."

                sessionMetadataStore
                    .recordFailure(
                        sessionId,
                        reason
                    )

                resetSession()

                updateStatus(
                    "STOPPED-Timeout f?r Sitzung " +
                            "${sessionId.take(8)}. " +
                            "Die Smartphone-Aufnahme wird " +
                            "kontrolliert abgeschlossen; die " +
                            "Sitzung bleibt technisch fehlerhaft."
                )

                /*
                 * Der vorhandene Callback beendet die lokale
                 * Audioaufnahme und finalisiert die Metadaten.
                 * Aufgrund von recordFailure() erh?lt die Sitzung
                 * dabei den finalen Status FAILED.
                 */
                onSessionStopped(
                    sessionId,
                    false,
                    null
                )
            }

        stopAckTimeoutRunnable =
            timeoutRunnable

        mainHandler.postDelayed(
            timeoutRunnable,
            SESSION_ACK_TIMEOUT_MS
        )
    }

    private fun cancelStartAckTimeout() {
        startAckTimeoutRunnable
            ?.let { runnable ->
                mainHandler.removeCallbacks(
                    runnable
                )
            }

        startAckTimeoutRunnable =
            null
    }

    private fun cancelStopAckTimeout() {
        stopAckTimeoutRunnable
            ?.let { runnable ->
                mainHandler.removeCallbacks(
                    runnable
                )
            }

        stopAckTimeoutRunnable =
            null
    }

    private fun cancelSessionAckTimeouts() {
        cancelStartAckTimeout()
        cancelStopAckTimeout()
    }

    private fun sendBestEffortStopAfterStartTimeout(
        sessionId: String,
        watchNodeId: String
    ) {
        val phoneStopCommandEpochMs =
            System.currentTimeMillis()

        val payload = buildString {
            append("session_id=")
            append(sessionId)

            append(
                ";phone_stop_command_epoch_ms="
            )
            append(
                phoneStopCommandEpochMs
            )
        }.toByteArray(
            StandardCharsets.UTF_8
        )

        /*
         * Best-Effort-Recovery: Es wird bewusst nicht auf das
         * Ergebnis dieses STOP-Kommandos gewartet. Die Session
         * ist bereits als FAILED markiert.
         */
        messageClient.sendMessage(
            watchNodeId,
            STOP_PATH,
            payload
        )
    }

    private fun resetSession() {
        cancelSessionAckTimeouts()
        currentSessionId =
            null

        currentWatchNodeId =
            null

        updateSessionState(
            SessionState.IDLE
        )
    }

    private fun updateSessionState(
        newState: SessionState
    ) {
        sessionState =
            newState

        notifySessionControls()
    }

    private fun notifySessionControls() {
        val canStart =
            sessionState ==
                    SessionState.READY

        val canStop =
            sessionState ==
                    SessionState.RECORDING

        mainHandler.post {
            onSessionControlsChanged(
                canStart,
                canStop
            )
        }
    }

    private fun parsePayload(
        payload: String
    ): Map<String, String> {
        return payload
            .split(";")
            .mapNotNull { part ->
                val separatorIndex =
                    part.indexOf("=")

                if (
                    separatorIndex <= 0
                ) {
                    null
                } else {
                    val key =
                        part.substring(
                            0,
                            separatorIndex
                        ).trim()

                    val value =
                        part.substring(
                            separatorIndex + 1
                        ).trim()

                    key to value
                }
            }
            .toMap()
    }

    private fun updateStatus(
        text: String
    ) {
        mainHandler.post {
            onStatusChanged(text)
        }
    }

    private fun errorMessage(
        exception: Exception
    ): String {
        return exception.message
            ?: "Unbekannter Fehler"
    }
}