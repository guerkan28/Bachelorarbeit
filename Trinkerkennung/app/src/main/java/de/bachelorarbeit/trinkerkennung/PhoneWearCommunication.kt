package de.bachelorarbeit.trinkerkennung

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import java.nio.charset.StandardCharsets

class PhoneWearCommunication(
    context: Context,
    private val onStatusChanged: (String) -> Unit
) : MessageClient.OnMessageReceivedListener {

    companion object {
        private const val PING_PATH =
            "/trinkerkennung/test/ping"

        private const val ACK_PATH =
            "/trinkerkennung/test/ack"
    }

    private val applicationContext =
        context.applicationContext

    private val messageClient =
        Wearable.getMessageClient(applicationContext)

    private val nodeClient =
        Wearable.getNodeClient(applicationContext)

    private val mainHandler =
        Handler(Looper.getMainLooper())

    fun startListening() {
        messageClient
            .addListener(this)
            .addOnFailureListener { exception ->
                updateStatus(
                    "Nachrichtenempfang konnte nicht aktiviert werden: " +
                            (exception.message ?: "Unbekannter Fehler")
                )
            }
    }

    fun stopListening() {
        messageClient.removeListener(this)
    }

    fun sendPing() {
        updateStatus("Suche verbundene Smartwatch …")

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
                        .toByteArray(StandardCharsets.UTF_8)

                val node = nodes.first()

                messageClient
                    .sendMessage(
                        node.id,
                        PING_PATH,
                        payload
                    )
                    .addOnSuccessListener {
                        updateStatus(
                            "PING an ${node.displayName} gesendet. " +
                                    "Warte auf ACK …"
                        )
                    }
                    .addOnFailureListener { exception ->
                        updateStatus(
                            "PING konnte nicht gesendet werden: " +
                                    (
                                            exception.message
                                                ?: "Unbekannter Fehler"
                                            )
                        )
                    }
            }
            .addOnFailureListener { exception ->
                updateStatus(
                    "Gerätesuche fehlgeschlagen: " +
                            (exception.message ?: "Unbekannter Fehler")
                )
            }
    }

    override fun onMessageReceived(
        messageEvent: MessageEvent
    ) {
        if (messageEvent.path != ACK_PATH) {
            return
        }

        val phoneReceiveTimestampMs =
            System.currentTimeMillis()

        val payload = messageEvent.data
            .toString(StandardCharsets.UTF_8)

        val values = payload
            .split(";")
            .map { it.trim() }
            .mapNotNull { part ->
                val separatorIndex = part.indexOf("=")

                if (separatorIndex <= 0) {
                    null
                } else {
                    val key = part
                        .substring(0, separatorIndex)
                        .trim()

                    val value = part
                        .substring(separatorIndex + 1)
                        .trim()

                    key to value
                }
            }
            .toMap()

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
                "ACK empfangen, aber Zeitstempel konnten " +
                        "nicht ausgewertet werden:\n$payload"
            )
            return
        }

        val roundTripTimeMs =
            phoneReceiveTimestampMs -
                    phoneSendTimestampMs

        updateStatus(
            buildString {
                append("ACK von der Smartwatch empfangen.")
                append("\nRound-Trip-Time: ")
                append(roundTripTimeMs)
                append(" ms")
                append("\nSmartphone Start: ")
                append(phoneSendTimestampMs)
                append("\nWatch Verarbeitung: ")
                append(watchTimestampMs)
                append("\nSmartphone Empfang: ")
                append(phoneReceiveTimestampMs)
            }
        )
    }

    private fun updateStatus(text: String) {
        mainHandler.post {
            onStatusChanged(text)
        }
    }
}