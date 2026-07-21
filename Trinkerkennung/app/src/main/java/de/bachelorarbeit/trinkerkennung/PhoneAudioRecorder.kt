package de.bachelorarbeit.trinkerkennung

import android.content.Context
import android.media.MediaRecorder
import android.os.Build
import android.os.Environment
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

class PhoneAudioRecorder(
    private val context: Context
) {
    private var mediaRecorder: MediaRecorder? = null
    private var outputFile: File? = null

    fun startRecording(): File {
        check(mediaRecorder == null) {
            "Es läuft bereits eine Aufnahme."
        }

        val baseDirectory =
            context.getExternalFilesDir(Environment.DIRECTORY_MUSIC)
                ?: context.filesDir

        val recordingDirectory = File(baseDirectory, "recordings")

        check(recordingDirectory.exists() || recordingDirectory.mkdirs()) {
            "Der Aufnahmeordner konnte nicht erstellt werden."
        }

        val timestamp = SimpleDateFormat(
            "yyyyMMdd_HHmmss",
            Locale.US
        ).format(Date())

        val newOutputFile = File(
            recordingDirectory,
            "smartphone_$timestamp.m4a"
        )

        val recorder = createMediaRecorder()

        try {
            recorder.apply {
                setAudioSource(MediaRecorder.AudioSource.MIC)
                setOutputFormat(MediaRecorder.OutputFormat.MPEG_4)
                setAudioEncoder(MediaRecorder.AudioEncoder.AAC)
                setOutputFile(newOutputFile.absolutePath)
                prepare()
                start()
            }

            mediaRecorder = recorder
            outputFile = newOutputFile

            return newOutputFile
        } catch (exception: Exception) {
            recorder.release()
            newOutputFile.delete()
            throw exception
        }
    }

    fun stopRecording(): File? {
        val recorder = mediaRecorder ?: return null
        val completedFile = outputFile

        return try {
            recorder.stop()
            completedFile
        } catch (exception: RuntimeException) {
            // Sehr kurze oder fehlerhafte Aufnahmen können ungültig sein.
            completedFile?.delete()
            null
        } finally {
            recorder.release()
            mediaRecorder = null
            outputFile = null
        }
    }

    fun release() {
        mediaRecorder?.release()
        mediaRecorder = null
        outputFile = null
    }

    @Suppress("DEPRECATION")
    private fun createMediaRecorder(): MediaRecorder {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            MediaRecorder(context)
        } else {
            MediaRecorder()
        }
    }
}