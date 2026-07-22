package de.bachelorarbeit.trinkerkennung.wear.presentation

import android.content.Context
import android.os.Handler
import android.os.Looper
import com.google.android.gms.wearable.MessageClient
import com.google.android.gms.wearable.MessageEvent
import com.google.android.gms.wearable.Wearable
import java.nio.charset.StandardCharsets

class WatchWearCommunication(
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

    private val mainHandler =
        Handler(Looper.getMainLooper())

    fun startListening() {
        messageClient
            .addListener(this)
            .addOnFailureListener { exception ->
                updateStatus(
                    "Empfang konnte nicht aktiviert werden: " +
                            (exception.message ?: "Unbekannter Fehler")
                )
            }
    }

    fun stopListening() {
        messageClient.removeListener(this)
    }

    override fun onMessageReceived(
        messageEvent: MessageEvent
    ) {
        if (messageEvent.path != PING_PATH) {
            return
        }

        updateStatus("PING vom Smartphone empfangen.")

        val phoneTimestamp = messageEvent.data
            .toString(StandardCharsets.UTF_8)

        val acknowledgement =
            "ACK; phone_timestamp_ms=$phoneTimestamp;" +
                    " watch_timestamp_ms=${System.currentTimeMillis()}"

        messageClient
            .sendMessage(
                messageEvent.sourceNodeId,
                ACK_PATH,
                acknowledgement.toByteArray(
                    StandardCharsets.UTF_8
                )
            )
            .addOnSuccessListener {
                updateStatus(
                    "PING empfangen und ACK gesendet."
                )
            }
            .addOnFailureListener { exception ->
                updateStatus(
                    "ACK konnte nicht gesendet werden: " +
                            (
                                    exception.message
                                        ?: "Unbekannter Fehler"
                                    )
                )
            }
    }

    private fun updateStatus(text: String) {
        mainHandler.post {
            onStatusChanged(text)
        }
    }
}