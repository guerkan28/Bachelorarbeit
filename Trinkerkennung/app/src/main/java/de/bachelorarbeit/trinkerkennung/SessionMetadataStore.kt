package de.bachelorarbeit.trinkerkennung

import android.content.Context
import android.os.Environment
import org.json.JSONObject
import java.io.File

class SessionMetadataStore(
    context: Context
) {

    private val applicationContext =
        context.applicationContext

    private val sessions =
        mutableMapOf<String, MutableSessionMetadata>()

    @Synchronized
    fun beginSession(
        sessionId: String,
        phonePrepareEpochMs: Long
    ) {
        require(sessionId.isNotBlank()) {
            "Die Session-ID darf nicht leer sein."
        }

        sessions[sessionId] =
            MutableSessionMetadata(
                sessionId = sessionId,
                phonePrepareEpochMs =
                    phonePrepareEpochMs
            )
    }

    @Synchronized
    fun discardSession(
        sessionId: String
    ) {
        sessions.remove(sessionId)
    }

    @Synchronized
    fun recordReady(
        sessionId: String,
        phoneReadyReceivedEpochMs: Long,
        watchReadyEpochMs: Long?
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.status = "READY"
        metadata.phoneReadyReceivedEpochMs =
            phoneReadyReceivedEpochMs
        metadata.watchReadyEpochMs =
            watchReadyEpochMs
    }

    @Synchronized
    fun recordPhoneAudioStarted(
        sessionId: String,
        fileName: String,
        startEpochMs: Long,
        startElapsedRealtimeNs: Long
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.phoneAudioFileName =
            fileName
        metadata.phoneAudioStartEpochMs =
            startEpochMs
        metadata.phoneAudioStartElapsedRealtimeNs =
            startElapsedRealtimeNs
    }

    @Synchronized
    fun recordStartCommand(
        sessionId: String,
        phoneStartCommandEpochMs: Long
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.status = "STARTING"
        metadata.phoneStartCommandEpochMs =
            phoneStartCommandEpochMs
    }

    @Synchronized
    fun recordStarted(
        sessionId: String,
        phoneStartedReceivedEpochMs: Long,
        watchStartEpochMs: Long?,
        watchFileName: String?
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.status = "RECORDING"
        metadata.phoneStartedReceivedEpochMs =
            phoneStartedReceivedEpochMs
        metadata.watchStartEpochMs =
            watchStartEpochMs

        if (!watchFileName.isNullOrBlank()) {
            metadata.watchSensorFileName =
                watchFileName
        }
    }

    @Synchronized
    fun recordStopCommand(
        sessionId: String,
        phoneStopCommandEpochMs: Long
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.status = "STOPPING"
        metadata.phoneStopCommandEpochMs =
            phoneStopCommandEpochMs
    }

    @Synchronized
    fun recordStopped(
        sessionId: String,
        phoneStoppedReceivedEpochMs: Long,
        watchStopEpochMs: Long?,
        watchRecordingSuccess: Boolean,
        watchFileName: String?
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.phoneStoppedReceivedEpochMs =
            phoneStoppedReceivedEpochMs
        metadata.watchStopEpochMs =
            watchStopEpochMs
        metadata.watchRecordingSuccess =
            watchRecordingSuccess

        if (!watchFileName.isNullOrBlank()) {
            metadata.watchSensorFileName =
                watchFileName
        }
    }

    @Synchronized
    fun recordPhoneAudioStopped(
        sessionId: String,
        stopEpochMs: Long,
        stopElapsedRealtimeNs: Long,
        success: Boolean,
        fileName: String?
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.phoneAudioStopEpochMs =
            stopEpochMs
        metadata.phoneAudioStopElapsedRealtimeNs =
            stopElapsedRealtimeNs
        metadata.phoneAudioSuccess =
            success

        if (!fileName.isNullOrBlank()) {
            metadata.phoneAudioFileName =
                fileName
        }
    }

    @Synchronized
    fun recordFailure(
        sessionId: String,
        reason: String
    ) {
        val metadata =
            getOrCreateSession(sessionId)

        metadata.status = "FAILED"
        metadata.failureReason =
            reason
    }

    @Synchronized
    fun finalizeSession(
        sessionId: String
    ): File {
        val metadata =
            sessions[sessionId]
                ?: error(
                    "Für die Sitzung $sessionId " +
                            "liegen keine Metadaten vor."
                )

        metadata.status =
            when {
                metadata.failureReason != null ->
                    "FAILED"

                metadata.phoneAudioSuccess == true &&
                        metadata.watchRecordingSuccess == true ->
                    "COMPLETED"

                else ->
                    "COMPLETED_WITH_ERRORS"
            }

        metadata.metadataFinalizedEpochMs =
            System.currentTimeMillis()

        val directory =
            createMetadataDirectory()

        val safeSessionId =
            sanitizeForFileName(sessionId)

        val finalFile = File(
            directory,
            "session_$safeSessionId.json"
        )

        val temporaryFile = File(
            directory,
            ".session_$safeSessionId.tmp"
        )

        temporaryFile.writeText(
            metadata.toJson().toString(2),
            Charsets.UTF_8
        )

        if (
            finalFile.exists() &&
            !finalFile.delete()
        ) {
            temporaryFile.delete()

            error(
                "Eine bestehende Metadatendatei " +
                        "konnte nicht ersetzt werden."
            )
        }

        if (!temporaryFile.renameTo(finalFile)) {
            temporaryFile.delete()

            error(
                "Die Metadatendatei konnte nicht " +
                        "atomar abgeschlossen werden."
            )
        }

        sessions.remove(sessionId)

        return finalFile
    }

    private fun getOrCreateSession(
        sessionId: String
    ): MutableSessionMetadata {
        return sessions.getOrPut(sessionId) {
            MutableSessionMetadata(
                sessionId = sessionId
            )
        }
    }

    private fun createMetadataDirectory(): File {
        val baseDirectory =
            applicationContext.getExternalFilesDir(
                Environment.DIRECTORY_DOCUMENTS
            ) ?: applicationContext.filesDir

        val metadataDirectory = File(
            baseDirectory,
            "session_metadata"
        )

        check(
            metadataDirectory.exists() ||
                    metadataDirectory.mkdirs()
        ) {
            "Der Metadatenordner konnte nicht " +
                    "erstellt werden."
        }

        return metadataDirectory
    }

    private fun sanitizeForFileName(
        value: String
    ): String {
        return value.replace(
            Regex("[^A-Za-z0-9_-]"),
            "_"
        )
    }

    private data class MutableSessionMetadata(
        val sessionId: String,
        var status: String = "PREPARING",
        var phonePrepareEpochMs: Long? = null,
        var phoneReadyReceivedEpochMs: Long? = null,
        var watchReadyEpochMs: Long? = null,
        var phoneAudioFileName: String? = null,
        var phoneAudioStartEpochMs: Long? = null,
        var phoneAudioStartElapsedRealtimeNs: Long? = null,
        var phoneStartCommandEpochMs: Long? = null,
        var phoneStartedReceivedEpochMs: Long? = null,
        var watchStartEpochMs: Long? = null,
        var watchSensorFileName: String? = null,
        var phoneStopCommandEpochMs: Long? = null,
        var phoneStoppedReceivedEpochMs: Long? = null,
        var watchStopEpochMs: Long? = null,
        var phoneAudioStopEpochMs: Long? = null,
        var phoneAudioStopElapsedRealtimeNs: Long? = null,
        var phoneAudioSuccess: Boolean? = null,
        var watchRecordingSuccess: Boolean? = null,
        var failureReason: String? = null,
        var metadataFinalizedEpochMs: Long? = null
    ) {

        fun toJson(): JSONObject {
            return JSONObject().apply {
                put("schema_version", 1)
                put("session_id", sessionId)
                put("status", status)

                putNullable(
                    "phone_prepare_epoch_ms",
                    phonePrepareEpochMs
                )
                putNullable(
                    "phone_ready_received_epoch_ms",
                    phoneReadyReceivedEpochMs
                )
                putNullable(
                    "watch_ready_epoch_ms",
                    watchReadyEpochMs
                )
                putNullable(
                    "phone_audio_file_name",
                    phoneAudioFileName
                )
                putNullable(
                    "phone_audio_start_epoch_ms",
                    phoneAudioStartEpochMs
                )
                putNullable(
                    "phone_audio_start_elapsed_realtime_ns",
                    phoneAudioStartElapsedRealtimeNs
                )
                putNullable(
                    "phone_start_command_epoch_ms",
                    phoneStartCommandEpochMs
                )
                putNullable(
                    "phone_started_received_epoch_ms",
                    phoneStartedReceivedEpochMs
                )
                putNullable(
                    "watch_start_epoch_ms",
                    watchStartEpochMs
                )
                putNullable(
                    "watch_sensor_file_name",
                    watchSensorFileName
                )
                putNullable(
                    "phone_stop_command_epoch_ms",
                    phoneStopCommandEpochMs
                )
                putNullable(
                    "phone_stopped_received_epoch_ms",
                    phoneStoppedReceivedEpochMs
                )
                putNullable(
                    "watch_stop_epoch_ms",
                    watchStopEpochMs
                )
                putNullable(
                    "phone_audio_stop_epoch_ms",
                    phoneAudioStopEpochMs
                )
                putNullable(
                    "phone_audio_stop_elapsed_realtime_ns",
                    phoneAudioStopElapsedRealtimeNs
                )
                putNullable(
                    "phone_audio_success",
                    phoneAudioSuccess
                )
                putNullable(
                    "watch_recording_success",
                    watchRecordingSuccess
                )
                putNullable(
                    "failure_reason",
                    failureReason
                )
                putNullable(
                    "metadata_finalized_epoch_ms",
                    metadataFinalizedEpochMs
                )
            }
        }
    }
}

private fun JSONObject.putNullable(
    key: String,
    value: Any?
) {
    put(
        key,
        value ?: JSONObject.NULL
    )
}