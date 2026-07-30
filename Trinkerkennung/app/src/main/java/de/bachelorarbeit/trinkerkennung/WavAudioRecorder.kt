package de.bachelorarbeit.trinkerkennung

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.media.AudioFormat
import android.media.AudioRecord
import android.media.MediaRecorder
import android.os.Environment
import androidx.core.content.ContextCompat
import java.io.File
import java.io.RandomAccessFile
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class WavAudioRecorder(
    private val context: Context
) {

    companion object {
        private const val SAMPLE_RATE = 48_000
        private const val CHANNEL_COUNT = 1
        private const val BITS_PER_SAMPLE = 16
        private const val WAV_HEADER_SIZE = 44
    }

    @Volatile
    private var isRecording = false

    private var audioRecord: AudioRecord? = null
    private var recordingThread: Thread? = null
    private var outputFile: File? = null
    private var recordingFailure: Throwable? = null

    /*
     * Diese Methode wird später für die gemeinsame
     * Smartphone-Watch-Sitzung verwendet.
     */
    fun startRecording(
        sessionId: String
    ): File {
        require(sessionId.isNotBlank()) {
            "Die Session-ID darf nicht leer sein."
        }

        check(audioRecord == null) {
            "Es läuft bereits eine Audioaufnahme."
        }

        if (
            ContextCompat.checkSelfPermission(
                context,
                Manifest.permission.RECORD_AUDIO
            ) != PackageManager.PERMISSION_GRANTED
        ) {
            throw SecurityException(
                "Die Mikrofonberechtigung wurde nicht erteilt."
            )
        }

        val minimumBufferSize =
            AudioRecord.getMinBufferSize(
                SAMPLE_RATE,
                AudioFormat.CHANNEL_IN_MONO,
                AudioFormat.ENCODING_PCM_16BIT
            )

        check(minimumBufferSize > 0) {
            "Die Audiokonfiguration wird vom Gerät " +
                    "nicht unterstützt."
        }

        val bufferSize = maxOf(
            minimumBufferSize * 2,
            4096
        )

        val recorder = AudioRecord.Builder()
            .setAudioSource(
                MediaRecorder.AudioSource.MIC
            )
            .setAudioFormat(
                AudioFormat.Builder()
                    .setSampleRate(SAMPLE_RATE)
                    .setEncoding(
                        AudioFormat.ENCODING_PCM_16BIT
                    )
                    .setChannelMask(
                        AudioFormat.CHANNEL_IN_MONO
                    )
                    .build()
            )
            .setBufferSizeInBytes(bufferSize)
            .build()

        check(
            recorder.state ==
                    AudioRecord.STATE_INITIALIZED
        ) {
            recorder.release()

            "AudioRecord konnte nicht initialisiert werden."
        }

        val directory =
            createRecordingDirectory()

        val timestamp = SimpleDateFormat(
            "yyyyMMdd_HHmmss_SSS",
            Locale.US
        ).format(Date())

        val safeSessionId =
            sanitizeForFileName(sessionId)

        val newOutputFile = File(
            directory,
            "smartphone_${safeSessionId}_$timestamp.wav"
        )

        val randomAccessFile =
            RandomAccessFile(
                newOutputFile,
                "rw"
            )

        try {
            randomAccessFile.setLength(0)

            /*
             * Platzhalter für den später geschriebenen
             * WAV-Header.
             */
            randomAccessFile.write(
                ByteArray(WAV_HEADER_SIZE)
            )

            recorder.startRecording()

            check(
                recorder.recordingState ==
                        AudioRecord.RECORDSTATE_RECORDING
            ) {
                "Die Audioaufnahme konnte nicht " +
                        "gestartet werden."
            }
        } catch (exception: Exception) {
            randomAccessFile.close()
            recorder.release()
            newOutputFile.delete()

            throw exception
        }

        audioRecord = recorder
        outputFile = newOutputFile
        recordingFailure = null
        isRecording = true

        recordingThread = Thread {
            val buffer =
                ByteArray(bufferSize)

            try {
                while (isRecording) {
                    val bytesRead =
                        recorder.read(
                            buffer,
                            0,
                            buffer.size
                        )

                    when {
                        bytesRead > 0 -> {
                            randomAccessFile.write(
                                buffer,
                                0,
                                bytesRead
                            )
                        }

                        bytesRead < 0 &&
                                isRecording -> {
                            throw IllegalStateException(
                                "Fehler beim Lesen der " +
                                        "Audiodaten: $bytesRead"
                            )
                        }
                    }
                }
            } catch (exception: Throwable) {
                if (isRecording) {
                    recordingFailure =
                        exception
                }
            } finally {
                randomAccessFile.close()
            }
        }.apply {
            name =
                "WavAudioRecordingThread"

            start()
        }

        return newOutputFile
    }

    fun stopRecording(): File? {
        val recorder =
            audioRecord ?: return null

        val completedFile =
            outputFile

        isRecording = false

        runCatching {
            recorder.stop()
        }

        recordingThread?.join(3000)

        val threadStillRunning =
            recordingThread?.isAlive == true

        recorder.release()

        audioRecord = null
        recordingThread = null
        outputFile = null

        if (
            threadStillRunning ||
            recordingFailure != null ||
            completedFile == null ||
            !completedFile.exists() ||
            completedFile.length() <=
            WAV_HEADER_SIZE
        ) {
            completedFile?.delete()
            recordingFailure = null

            return null
        }

        writeWavHeader(completedFile)

        recordingFailure = null

        return completedFile
    }

    fun abortRecording() {
        val abortedFile =
            stopRecording()

        if (
            abortedFile != null &&
            abortedFile.exists()
        ) {
            abortedFile.delete()
        }
    }

    fun release() {
        if (audioRecord != null) {
            stopRecording()
        }
    }

    private fun createRecordingDirectory(): File {
        val baseDirectory =
            context.getExternalFilesDir(
                Environment.DIRECTORY_MUSIC
            ) ?: context.filesDir

        val recordingDirectory = File(
            baseDirectory,
            "recordings"
        )

        check(
            recordingDirectory.exists() ||
                    recordingDirectory.mkdirs()
        ) {
            "Der Aufnahmeordner konnte nicht " +
                    "erstellt werden."
        }

        return recordingDirectory
    }

    private fun sanitizeForFileName(
        value: String
    ): String {
        return value.replace(
            regex =
                Regex("[^A-Za-z0-9_-]"),
            replacement = "_"
        )
    }

    private fun writeWavHeader(
        file: File
    ) {
        val audioDataLength =
            file.length() -
                    WAV_HEADER_SIZE

        val completeFileLength =
            audioDataLength + 36

        val byteRate =
            SAMPLE_RATE *
                    CHANNEL_COUNT *
                    BITS_PER_SAMPLE / 8

        val blockAlignment =
            CHANNEL_COUNT *
                    BITS_PER_SAMPLE / 8

        RandomAccessFile(
            file,
            "rw"
        ).use { wavFile ->
            wavFile.seek(0)

            wavFile.writeBytes("RIFF")

            writeLittleEndianInt(
                wavFile,
                completeFileLength.toInt()
            )

            wavFile.writeBytes("WAVE")
            wavFile.writeBytes("fmt ")

            writeLittleEndianInt(
                wavFile,
                16
            )

            writeLittleEndianShort(
                wavFile,
                1
            )

            writeLittleEndianShort(
                wavFile,
                CHANNEL_COUNT
            )

            writeLittleEndianInt(
                wavFile,
                SAMPLE_RATE
            )

            writeLittleEndianInt(
                wavFile,
                byteRate
            )

            writeLittleEndianShort(
                wavFile,
                blockAlignment
            )

            writeLittleEndianShort(
                wavFile,
                BITS_PER_SAMPLE
            )

            wavFile.writeBytes("data")

            writeLittleEndianInt(
                wavFile,
                audioDataLength.toInt()
            )
        }
    }

    private fun writeLittleEndianInt(
        file: RandomAccessFile,
        value: Int
    ) {
        file.write(
            value and 0xFF
        )

        file.write(
            value shr 8 and 0xFF
        )

        file.write(
            value shr 16 and 0xFF
        )

        file.write(
            value shr 24 and 0xFF
        )
    }

    private fun writeLittleEndianShort(
        file: RandomAccessFile,
        value: Int
    ) {
        file.write(
            value and 0xFF
        )

        file.write(
            value shr 8 and 0xFF
        )
    }
}