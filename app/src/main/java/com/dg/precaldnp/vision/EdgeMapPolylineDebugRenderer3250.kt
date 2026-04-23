package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.os.Environment
import android.provider.MediaStore
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import java.io.OutputStream
import kotlin.math.max

object EdgeMapPolylineDebugRenderer3250 {
    private const val TAG = "EdgePolyDbg3250"

    fun logSummary3250(
        result: EdgeMapPolylineDebug3250.Result3250,
        maxItems: Int = 30
    ) {
        Log.d(
            TAG,
            "extract polylines: count=${result.polylines.size} roi=${result.width}x${result.height}"
        )

        result.polylines.take(maxItems).forEachIndexed { i, p ->
            Log.d(
                TAG,
                "poly[$i] pts=${p.pointCount} per=${"%.1f".format(p.perimeterPx)} area=${"%.1f".format(p.areaPx)} " +
                        "bboxLocal=[x=${p.bboxLocal.x},y=${p.bboxLocal.y},w=${p.bboxLocal.width},h=${p.bboxLocal.height}]"
            )
        }
    }

    fun renderToBitmap3250(
        edgeU8: ByteArray,
        w: Int,
        h: Int,
        result: EdgeMapPolylineDebug3250.Result3250,
        drawIndex: Boolean = true,
        strokePx: Float = 2f,
        pointRadiusPx: Float = 0f
    ): Bitmap {
        val base = edgeU8ToBitmap3250(edgeU8, w, h).copy(Bitmap.Config.ARGB_8888, true)
        val canvas = Canvas(base)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = strokePx
        }

        val pointPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.FILL
        }

        val textPaint = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.YELLOW
            textSize = max(16f, w / 40f)
        }

        result.polylines.forEachIndexed { i, poly ->
            val color = debugColor3250(i)
            paint.color = color
            pointPaint.color = color

            val pts = poly.pointsLocal
            if (pts.size >= 2) {
                for (k in 1 until pts.size) {
                    val p0 = pts[k - 1]
                    val p1 = pts[k]
                    canvas.drawLine(p0.x, p0.y, p1.x, p1.y, paint)
                }
            }

            if (pointRadiusPx > 0f) {
                for (p in pts) {
                    canvas.drawCircle(p.x, p.y, pointRadiusPx, pointPaint)
                }
            }

            if (drawIndex && pts.isNotEmpty()) {
                val anchor = pts[0]
                canvas.drawText(
                    "#$i",
                    anchor.x + 4f,
                    anchor.y - 4f,
                    textPaint
                )
            }
        }

        return base
    }

    fun saveBitmapToGallery3250(
        context: Context,
        bitmap: Bitmap,
        name: String
    ): String? {
        val filename = if (name.lowercase().endsWith(".png")) name else "$name.png"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, filename)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(
                MediaStore.Images.Media.RELATIVE_PATH,
                Environment.DIRECTORY_PICTURES + "/medirDNP"
            )
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val resolver = context.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null

        return try {
            val out: OutputStream = resolver.openOutputStream(uri) ?: return null
            out.use { bitmap.compress(Bitmap.CompressFormat.PNG, 100, it) }

            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)

            uri.toString()
        } catch (t: Throwable) {
            Log.e(TAG, "saveBitmapToGallery3250 failed", t)
            null
        }
    }

    fun dump3250(
        context: Context,
        edgeU8: ByteArray,
        w: Int,
        h: Int,
        result: EdgeMapPolylineDebug3250.Result3250,
        name: String,
        drawIndex: Boolean = true,
        strokePx: Float = 2f,
        pointRadiusPx: Float = 0f,
        maxItemsToLog: Int = 30
    ): String? {
        logSummary3250(result, maxItems = maxItemsToLog)

        val bmp = renderToBitmap3250(
            edgeU8 = edgeU8,
            w = w,
            h = h,
            result = result,
            drawIndex = drawIndex,
            strokePx = strokePx,
            pointRadiusPx = pointRadiusPx
        )

        val uri = saveBitmapToGallery3250(
            context = context,
            bitmap = bmp,
            name = name
        )

        Log.d(TAG, "dump uri=$uri")
        return uri
    }

    private fun edgeU8ToBitmap3250(edgeU8: ByteArray, w: Int, h: Int): Bitmap {
        val mat = Mat(h, w, CvType.CV_8UC1)
        mat.put(0, 0, edgeU8)

        val rgba = Mat()
        org.opencv.imgproc.Imgproc.cvtColor(mat, rgba, org.opencv.imgproc.Imgproc.COLOR_GRAY2RGBA)

        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        Utils.matToBitmap(rgba, bmp)
        return bmp
    }

    private fun debugColor3250(i: Int): Int {
        val palette = intArrayOf(
            Color.RED,
            Color.GREEN,
            Color.CYAN,
            Color.MAGENTA,
            Color.YELLOW,
            Color.BLUE,
            Color.rgb(255, 128, 0),
            Color.rgb(0, 255, 128),
            Color.rgb(255, 0, 128),
            Color.rgb(128, 255, 0),
            Color.rgb(0, 128, 255),
            Color.WHITE
        )
        return palette[i % palette.size]
    }
}