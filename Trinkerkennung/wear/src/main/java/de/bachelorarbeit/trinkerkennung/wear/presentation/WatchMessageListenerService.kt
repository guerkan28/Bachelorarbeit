package de.bachelorarbeit.trinkerkennung.wear.presentation

import android.content.Intent
import android.util.Log
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import com.google.android.gms.wearable.WearableListenerService
import java.nio.charset.StandardCharsets

class WatchMessageListenerService :
    WearableListenerService() {

    companion object {
        private const val LOG_TAG =
            "WatchMessageService"

        private const val PING_PATH =
            "/trinkerkennung/test/ping"

        private const val ACK_PATH =
            "/trinkerkennung/test/ack"

        private const val PREPARE_PATH =
            "/trinkerkennung/session/prepare"

        private const val START_PATH =
            "/trinkerkennung/session/start"

        private const val STOP_PATH =
            "/trinkerkennung/session/stop"
    }

    private val messageClient by lazy {
        Wearable.getMessageClient(this)
    }

    override fun onMessageReceived(
        messageEvent: MessageEvent
    ) {
        super.onMessageReceived(messageEvent)

        Log.i(
            LOG_TAG,
            "Nachricht empfangen: ${messageEvent.path}"
        )

        when (messageEvent.path) {
            PING_PATH -> {
                handlePing(messageEvent)
            }

            PREPARE_PATH -> {
                handlePrepare(messageEvent)
            }

            START_PATH -> {
                handleStart(messageEvent)
            }

            STOP_PATH -> {
                handleStop(messageEvent)
            }

            else -> {
                Log.w(
                    LOG_TAG,
                    "Unbekannter Nachrichtenpfad: " +
                            messageEvent.path
                )
            }
        }
    }

    private fun handlePing(
        messageEvent: MessageEvent
    ) {
        val phoneTimestamp = messageEvent.data
            .toString(StandardCharsets.UTF_8)

        val acknowledgement = buildString {
            append("phone_timestamp_ms=")
            append(phoneTimestamp)

            append(";watch_timestamp_ms=")
            append(System.currentTimeMillis())
        }

        messageClient
            .sendMessage(
                messageEvent.sourceNodeId,
                ACK_PATH,
                acknowledgement.toByteArray(
                    StandardCharsets.UTF_8
                )
            )
            .addOnSuccessListener {
                Log.i(
                    LOG_TAG,
                    "PING empfangen und ACK gesendet."
                )
            }
            .addOnFailureListener { exception ->
                Log.e(
                    LOG_TAG,
                    "ACK konnte nicht gesendet werden.",
                    exception
                )
            }
    }

    private fun handlePrepare(
        messageEvent: MessageEvent
    ) {
        val sessionId =
            readSessionId(
                messageEvent = messageEvent,
                commandName = "PREPARE"
            ) ?: return

        val serviceIntent =
            WatchRecordingService.createPrepareIntent(
                context = this,
                sessionId = sessionId,
                sourceNodeId =
                    messageEvent.sourceNodeId
            )

        startRecordingService(
            serviceIntent = serviceIntent,
            commandName = "PREPARE",
            sessionId = sessionId
        )
    }

    private fun handleStart(
        messageEvent: MessageEvent
    ) {
        val sessionId =
            readSessionId(
                messageEvent = messageEvent,
                commandName = "START"
            ) ?: return

        val serviceIntent =
            WatchRecordingService.createStartIntent(
                context = this,
                sessionId = sessionId,
                sourceNodeId =
                    messageEvent.sourceNodeId
            )

        startRecordingService(
            serviceIntent = serviceIntent,
            commandName = "START",
            sessionId = sessionId
        )
    }

    private fun handleStop(
        messageEvent: MessageEvent
    ) {
        val sessionId =
            readSessionId(
                messageEvent = messageEvent,
                commandName = "STOP"
            ) ?: return

        val serviceIntent =
            WatchRecordingService.createStopIntent(
                context = this,
                sessionId = sessionId,
                sourceNodeId =
                    messageEvent.sourceNodeId
            )

        startRecordingService(
            serviceIntent = serviceIntent,
            commandName = "STOP",
            sessionId = sessionId
        )
    }

    private fun startRecordingService(
        serviceIntent: Intent,
        commandName: String,
        sessionId: String
    ) {
        try {
            startForegroundService(
                serviceIntent
            )

            Log.i(
                LOG_TAG,
                "$commandName für Sitzung " +
                        "${sessionId.take(8)} an " +
                        "WatchRecordingService übergeben."
            )
        } catch (exception: Exception) {
            Log.e(
                LOG_TAG,
                "$commandName konnte nicht an den " +
                        "WatchRecordingService übergeben werden.",
                exception
            )
        }
    }

    private fun readSessionId(
        messageEvent: MessageEvent,
        commandName: String
    ): String? {
        val payload = messageEvent.data
            .toString(StandardCharsets.UTF_8)

        val values =
            parsePayload(payload)

        val sessionId =
            values["session_id"]

        if (sessionId.isNullOrBlank()) {
            Log.e(
                LOG_TAG,
                "$commandName ohne gültige Session-ID empfangen."
            )

            return null
        }

        return sessionId
    }

    private fun parsePayload(
        payload: String
    ): Map<String, String> {
        return payload
            .split(";")
            .mapNotNull { part ->
                val separatorIndex =
                    part.indexOf("=")

                if (separatorIndex <= 0) {
                    null
                } else {
                    val key = part
                        .substring(
                            startIndex = 0,
                            endIndex = separatorIndex
                        )
                        .trim()

                    val value = part
                        .substring(
                            startIndex = separatorIndex + 1
                        )
                        .trim()

                    key to value
                }
            }
            .toMap()
    }
}