package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Path
import android.graphics.PointF
import android.graphics.RectF
import android.net.Uri
import android.os.SystemClock
import android.provider.MediaStore
import android.util.Log
import androidx.core.graphics.createBitmap
import java.io.OutputStream
import kotlin.math.max
import kotlin.math.min

object DebugPaths3250 {
    const val REL_PATH = "Pictures/PrecalDNP/DEBUG"
}

object DebugDump3250 {

    private const val TAG = "DebugDump3250"
    private const val REL_PATH = DebugPaths3250.REL_PATH

    private const val FIL_LABEL_DETECTOR = "FIL DETECTOR"
    private const val FIL_LABEL_ARCFIT = "FIL ARC FIT"

    private const val FIL_COLOR_DETECTOR = 0xFFFF4DFF.toInt() // magenta-violeta
    private const val FIL_COLOR_ARCFIT = 0xFFFF5252.toInt()   // rojo suave

    private val lastByKey = HashMap<String, Long>()

    private fun allow(key: String, minIntervalMs: Long): Boolean {
        val now = SystemClock.elapsedRealtime()
        val last = lastByKey[key] ?: 0L
        if (now - last < minIntervalMs) return false
        lastByKey[key] = now
        return true
    }

    /**
     * Dump legacy/compat:
     * detector usado y, opcionalmente, un FIL superpuesto.
     */
    fun dumpRimUsed3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        if (usedPack == null) return

        val key = "RIM_USED_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val overlayBmp = edgeU8ToBitmap(usedPack.edges, usedPack.w, usedPack.h)
            .copy(Bitmap.Config.ARGB_8888, true)

        drawRimOverlayOnEdgeBitmap(
            edgeBmp = overlayBmp,
            res = usedPack.result,
            filPtsGlobal800 = filPtsGlobal800,
            filLabel = if (filPtsGlobal800.isNullOrEmpty()) null else "FIL USED",
            filColor = if (filPtsGlobal800.isNullOrEmpty()) null else FIL_COLOR_DETECTOR,
            drawGuides = true
        )

        saveBitmapToPictures(
            context = context,
            bmp = overlayBmp,
            baseName = "RIM_${debugTag}_USED_OVERLAY"
        )
    }

    /**
     * NUEVO:
     * Dump del detector por separado.
     * Acepta opcionalmente un FIL "detector" para mantener compatibilidad
     * con la llamada actual del pipeline.
     */
    fun dumpRimDetectorUsed3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsDetectorGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        if (usedPack == null) return

        val key = "RIM_DETECTOR_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val overlayBmp = edgeU8ToBitmap(usedPack.edges, usedPack.w, usedPack.h)
            .copy(Bitmap.Config.ARGB_8888, true)

        drawRimOverlayOnEdgeBitmap(
            edgeBmp = overlayBmp,
            res = usedPack.result,
            filPtsGlobal800 = filPtsDetectorGlobal800,
            filLabel = if (filPtsDetectorGlobal800.isNullOrEmpty()) null else FIL_LABEL_DETECTOR,
            filColor = if (filPtsDetectorGlobal800.isNullOrEmpty()) null else FIL_COLOR_DETECTOR,
            drawGuides = true
        )

        saveBitmapToPictures(
            context = context,
            bmp = overlayBmp,
            baseName = "RIM_${debugTag}_DETECTOR_ONLY"
        )
    }

    /**
     * NUEVO:
     * Dump del ArcFit por separado.
     * Base = edge map ROI + FIL final del ArcFit.
     */
    fun dumpRimArcFitUsed3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsArcFitGlobal800: List<PointF>?,
        minIntervalMs: Long = 1200L
    ) {
        if (usedPack == null || filPtsArcFitGlobal800.isNullOrEmpty()) return

        val key = "RIM_ARCFIT_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val overlayBmp = edgeU8ToBitmap(usedPack.edges, usedPack.w, usedPack.h)
            .copy(Bitmap.Config.ARGB_8888, true)

        drawArcFitOnlyOnEdgeBitmap3250(
            edgeBmp = overlayBmp,
            roiPx = usedPack.result.roiPx,
            filPtsGlobal800 = filPtsArcFitGlobal800,
            filLabel = FIL_LABEL_ARCFIT,
            filColor = FIL_COLOR_ARCFIT
        )

        saveBitmapToPictures(
            context = context,
            bmp = overlayBmp,
            baseName = "RIM_${debugTag}_ARCFIT_ONLY"
        )
    }

    /**
     * NUEVO:
     * Emite ambos dumps, separados.
     */
    fun dumpRimDetectorAndArcFit3250(
        context: Context,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsDetectorGlobal800: List<PointF>?,
        filPtsArcFitGlobal800: List<PointF>?,
        minIntervalMs: Long = 1200L
    ) {
        dumpRimDetectorUsed3250(
            context = context,
            usedPack = usedPack,
            debugTag = debugTag,
            filPtsDetectorGlobal800 = filPtsDetectorGlobal800,
            minIntervalMs = minIntervalMs
        )

        dumpRimArcFitUsed3250(
            context = context,
            usedPack = usedPack,
            debugTag = debugTag,
            filPtsArcFitGlobal800 = filPtsArcFitGlobal800,
            minIntervalMs = minIntervalMs
        )
    }

    /**
     * Compatibilidad con tu llamada actual del pipeline.
     * Internamente conserva el comportamiento legacy.
     */
    @Suppress("UNUSED_PARAMETER")
    fun dumpRimRawNormAndUsed3250(
        context: Context,
        rawPack: RimDetectPack3250?,
        normPack: RimDetectPack3250?,
        usedPack: RimDetectPack3250?,
        debugTag: String,
        filPtsGlobal800: List<PointF>? = null,
        minIntervalMs: Long = 1200L
    ) {
        dumpRimUsed3250(
            context = context,
            usedPack = usedPack,
            debugTag = debugTag,
            filPtsGlobal800 = filPtsGlobal800,
            minIntervalMs = minIntervalMs
        )
    }

    /**
     * Base = edge map ROI del pack usado.
     * En este modo dibuja:
     * - bottom final
     * - marcas verticales inner L/R
     * - arcos inner/outer si existen
     * - FIL opcional si se pasa (legacy o detector separado)
     *
     * drawGuides=true agrega probe/walls/seed/top.
     */
    private fun drawRimOverlayOnEdgeBitmap(
        edgeBmp: Bitmap,
        res: RimDetectionResult,
        filPtsGlobal800: List<PointF>? = null,
        filLabel: String? = null,
        filColor: Int? = null,
        drawGuides: Boolean = false
    ) {
        val c = Canvas(edgeBmp)

        val x0 = res.roiPx.left
        val y0 = res.roiPx.top

        fun toLocalX(xGlobal: Float): Float = xGlobal - x0
        fun toLocalY(yGlobal: Float): Float = yGlobal - y0

        fun drawDebugY(
            yGlobalPx: Float,
            label: String,
            paintLine: Paint,
            paintText: Paint
        ) {
            if (!yGlobalPx.isFinite()) return

            val yLocal = toLocalY(yGlobalPx)
            if (!yLocal.isFinite()) return

            c.drawLine(0f, yLocal, edgeBmp.width.toFloat(), yLocal, paintLine)
            c.drawText(
                "$label g=${"%.1f".format(yGlobalPx)} l=${"%.1f".format(yLocal)}",
                12f,
                yLocal - 6f,
                paintText
            )
        }

        val pBottom = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2.0f, edgeBmp.width * 0.0030f)
            color = Color.argb(145, 0, 255, 80)
        }

        val pInnerV = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.3f, edgeBmp.width * 0.0021f)
            color = Color.argb(85, 0, 229, 255)
        }

        val pInnerArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.8f, edgeBmp.width * 0.0027f)
            color = Color.argb(150, 0, 229, 255)
        }

        val pOuterArc = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.5f, edgeBmp.width * 0.0023f)
            color = Color.argb(105, 255, 152, 0)
        }

        val pProbe = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.4f, edgeBmp.width * 0.0021f)
            color = Color.argb(170, 255, 235, 59)
        }

        val pWalls = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.2f, edgeBmp.width * 0.0019f)
            color = Color.argb(150, 121, 85, 72)
        }

        val pSeed = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.6f, edgeBmp.width * 0.0022f)
            color = Color.argb(180, 186, 104, 200)
        }

        val pTop = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2.2f, edgeBmp.width * 0.0032f)
            color = Color.argb(180, 186, 104, 200)
        }

        val pTopMin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.6f, edgeBmp.width * 0.0022f)
            color = Color.argb(210, 255, 0, 255)
        }

        val pTopSearchMin = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.6f, edgeBmp.width * 0.0022f)
            color = Color.argb(210, 0, 255, 255)
        }

        val pExpectedTop = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.8f, edgeBmp.width * 0.0025f)
            color = Color.argb(220, 255, 235, 59)
        }

        val pExpectedBand = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(1.2f, edgeBmp.width * 0.0018f)
            color = Color.argb(170, 255, 255, 255)
        }

        val pWhite = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(13f, edgeBmp.width * 0.020f)
            style = Paint.Style.FILL
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        if (drawGuides) {
            val probeY = toLocalY(res.probeYpx)
            if (probeY.isFinite()) {
                c.drawLine(0f, probeY, edgeBmp.width.toFloat(), probeY, pProbe)
                c.drawText("probe", 12f, probeY - 6f, pWhite)
            }

            val wallsY = toLocalY(res.wallsYpx)
            if (wallsY.isFinite()) {
                c.drawLine(0f, wallsY, edgeBmp.width.toFloat(), wallsY, pWalls)
                c.drawText("walls", 12f, wallsY - 6f, pWhite)
            }

            val seedX = toLocalX(res.seedXpx)
            if (seedX.isFinite()) {
                c.drawLine(seedX, 0f, seedX, edgeBmp.height.toFloat(), pSeed)
                c.drawText("seed", seedX + 6f, 18f, pWhite)
            }

            val topY = toLocalY(res.topYpx)
            if (topY.isFinite()) {
                c.drawLine(0f, topY, edgeBmp.width.toFloat(), topY, pTop)
                c.drawText("topUsed", 12f, topY - 6f, pWhite)
            }

            drawDebugY(res.topMinAllowedYpx, "topMin", pTopMin, pWhite)
            drawDebugY(res.topSearchMinYpx, "topSearchMin", pTopSearchMin, pWhite)
            drawDebugY(res.expectedTopYpx, "expTop", pExpectedTop, pWhite)

            val expTopGlobal = res.expectedTopYpx
            if (expTopGlobal.isFinite() && res.expectedTopTolPx > 0f) {
                val yA = expTopGlobal - res.expectedTopTolPx
                val yB = expTopGlobal + res.expectedTopTolPx
                drawDebugY(yA, "expTop-tol-", pExpectedBand, pWhite)
                drawDebugY(yB, "expTop-tol+", pExpectedBand, pWhite)
            }
        }

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.bottomPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pBottom
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.topPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pTop
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.nasalInnerPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pInnerArc
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.templeInnerPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pInnerArc
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.nasalOuterPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pOuterArc
        )

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = res.templeOuterPolylinePx,
            x0 = x0,
            y0 = y0,
            paint = pOuterArc
        )

        val innerL = toLocalX(res.innerLeftXpx)
        if (innerL.isFinite()) {
            c.drawLine(innerL, 0f, innerL, edgeBmp.height.toFloat(), pInnerV)
        }

        val innerR = toLocalX(res.innerRightXpx)
        if (innerR.isFinite()) {
            c.drawLine(innerR, 0f, innerR, edgeBmp.height.toFloat(), pInnerV)
        }

        if (!filPtsGlobal800.isNullOrEmpty()) {
            val pFil = Paint(Paint.ANTI_ALIAS_FLAG).apply {
                style = Paint.Style.STROKE
                strokeWidth = max(2.1f, edgeBmp.width * 0.0032f)
                color = filColor ?: FIL_COLOR_DETECTOR
            }

            drawPolylineGlobal(
                canvas = c,
                ptsGlobal = filPtsGlobal800,
                x0 = x0,
                y0 = y0,
                paint = pFil
            )

            if (!filLabel.isNullOrBlank()) {
                c.drawText(
                    filLabel,
                    12f,
                    max(18f, pWhite.textSize + 10f),
                    pWhite
                )
            }
        }

        val topCount = res.topPolylinePx?.size ?: 0
        val botCount = res.bottomPolylinePx?.size ?: 0
        val nIn = res.nasalInnerPolylinePx?.size ?: 0
        val tIn = res.templeInnerPolylinePx?.size ?: 0
        val nOut = res.nasalOuterPolylinePx?.size ?: 0
        val tOut = res.templeOuterPolylinePx?.size ?: 0

        val dbgTxt =
            "top=$topCount bot=$botCount nIn=$nIn tIn=$tIn nOut=$nOut tOut=$tOut conf=${"%.3f".format(res.confidence)}"

        c.drawText(
            dbgTxt,
            12f,
            edgeBmp.height - pWhite.textSize - 12f,
            pWhite
        )
    }

    private fun drawPolylineGlobal(
        canvas: Canvas,
        ptsGlobal: List<PointF>?,
        x0: Float,
        y0: Float,
        paint: Paint
    ) {
        if (ptsGlobal.isNullOrEmpty()) return

        val path = Path()
        val p0 = ptsGlobal.first()
        path.moveTo(p0.x - x0, p0.y - y0)

        for (i in 1 until ptsGlobal.size) {
            val p = ptsGlobal[i]
            path.lineTo(p.x - x0, p.y - y0)
        }

        canvas.drawPath(path, paint)
    }

    private fun edgeU8ToBitmap(edges: ByteArray, w: Int, h: Int): Bitmap {
        val bmp = createBitmap(w, h)
        val px = IntArray(w * h)
        val n = min(px.size, edges.size)

        for (i in 0 until n) {
            val v = edges[i].toInt() and 0xFF
            px[i] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
        }

        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    private fun saveBitmapToPictures(
        context: Context,
        bmp: Bitmap,
        baseName: String
    ): Uri? {
        val name = "${baseName}_${System.currentTimeMillis()}.png"

        val resolver = context.contentResolver
        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, name)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, REL_PATH)
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values)
        if (uri == null) {
            Log.w(TAG, "saveBitmapToPictures insert returned null")
            return null
        }

        var os: OutputStream? = null
        var ok = false

        try {
            os = resolver.openOutputStream(uri)
            if (os == null) {
                Log.w(TAG, "openOutputStream null")
            } else {
                ok = bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
        } catch (t: Throwable) {
            Log.e(TAG, "saveBitmapToPictures error", t)
        } finally {
            try {
                os?.close()
            } catch (_: Throwable) {
            }
        }

        if (!ok) {
            try {
                resolver.delete(uri, null, null)
            } catch (_: Throwable) {
            }
            return null
        }

        val done = ContentValues().apply {
            put(MediaStore.Images.Media.IS_PENDING, 0)
        }
        resolver.update(uri, done, null, null)

        Log.d(TAG, "Saved debug png: $name")
        return uri
    }

    fun dumpEdgeFinalWithMask3250(
        context: Context,
        edgeFinalU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        debugTag: String,
        minIntervalMs: Long = 1200L
    ) {
        if (edgeFinalU8 == null || maskU8 == null) return
        if (w <= 0 || h <= 0) return
        if (edgeFinalU8.size < w * h || maskU8.size < w * h) {
            Log.w(
                TAG,
                "dumpEdgeFinalWithMask3250 bad size edge=${edgeFinalU8.size} mask=${maskU8.size} need=${w * h}"
            )
            return
        }

        val key = "EDGE_FINAL_MASK_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val bmp = overlayMaskOnEdge3250(
            edgeU8 = edgeFinalU8,
            maskU8 = maskU8,
            w = w,
            h = h
        )

        saveBitmapToPictures(
            context = context,
            bmp = bmp,
            baseName = "EDGE_${debugTag}_FINAL_MASK_OVERLAY"
        )
    }

    private fun overlayMaskOnEdge3250(
        edgeU8: ByteArray,
        maskU8: ByteArray,
        w: Int,
        h: Int
    ): Bitmap {
        val bmp = createBitmap(w, h)
        val px = IntArray(w * h)

        val alpha = 110f / 255f

        for (i in 0 until w * h) {
            val e = edgeU8[i].toInt() and 0xFF
            val m = maskU8[i].toInt() and 0xFF

            if (m != 0) {
                val r = ((1f - alpha) * e + alpha * 0f).toInt().coerceIn(0, 255)
                val g = ((1f - alpha) * e + alpha * 255f).toInt().coerceIn(0, 255)
                val b = ((1f - alpha) * e + alpha * 80f).toInt().coerceIn(0, 255)
                px[i] = Color.argb(255, r, g, b)
            } else {
                px[i] = Color.argb(255, e, e, e)
            }
        }

        bmp.setPixels(px, 0, w, 0, 0, w, h)
        return bmp
    }

    fun dumpGateOverGray3250(
        context: Context,
        grayRawU8: ByteArray?,
        gateMaskU8: ByteArray?,
        w: Int,
        h: Int,
        debugTag: String,
        minIntervalMs: Long = 1200L
    ) {
        if (grayRawU8 == null || gateMaskU8 == null) return
        if (w <= 0 || h <= 0) return
        if (grayRawU8.size < w * h || gateMaskU8.size < w * h) return

        val key = "EDGE_GATE_OVER_GRAY_$debugTag"
        if (!allow(key, minIntervalMs)) return

        val bmp = createBitmap(w, h)
        val px = IntArray(w * h)

        val alpha = 150f / 255f

        for (i in 0 until w * h) {
            val g = grayRawU8[i].toInt() and 0xFF
            val m = gateMaskU8[i].toInt() and 0xFF

            if (m != 0) {
                val r = ((1f - alpha) * g + alpha * 0f).toInt().coerceIn(0, 255)
                val gg = ((1f - alpha) * g + alpha * 255f).toInt().coerceIn(0, 255)
                val b = ((1f - alpha) * g + alpha * 80f).toInt().coerceIn(0, 255)
                px[i] = Color.argb(255, r, gg, b)
            } else {
                px[i] = Color.argb(255, g, g, g)
            }
        }

        bmp.setPixels(px, 0, w, 0, 0, w, h)

        saveBitmapToPictures(
            context = context,
            bmp = bmp,
            baseName = "EDGE_GATE_OVER_GRAY_${debugTag}"
        )
    }

    private fun drawArcFitOnlyOnEdgeBitmap3250(
        edgeBmp: Bitmap,
        roiPx: RectF,
        filPtsGlobal800: List<PointF>,
        filLabel: String? = null,
        filColor: Int = FIL_COLOR_ARCFIT
    ) {
        if (filPtsGlobal800.isEmpty()) return

        val c = Canvas(edgeBmp)
        val x0 = roiPx.left
        val y0 = roiPx.top

        val pFil = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            style = Paint.Style.STROKE
            strokeWidth = max(2.4f, edgeBmp.width * 0.0036f)
            color = filColor
        }

        val pWhite = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            color = Color.WHITE
            textSize = max(13f, edgeBmp.width * 0.020f)
            style = Paint.Style.FILL
            setShadowLayer(5f, 0f, 0f, Color.BLACK)
        }

        drawPolylineGlobal(
            canvas = c,
            ptsGlobal = filPtsGlobal800,
            x0 = x0,
            y0 = y0,
            paint = pFil
        )

        if (!filLabel.isNullOrBlank()) {
            c.drawText(
                filLabel,
                12f,
                max(18f, pWhite.textSize + 10f),
                pWhite
            )
        }

        c.drawText(
            "pts=${filPtsGlobal800.size}",
            12f,
            edgeBmp.height - pWhite.textSize - 12f,
            pWhite
        )
    }
}
