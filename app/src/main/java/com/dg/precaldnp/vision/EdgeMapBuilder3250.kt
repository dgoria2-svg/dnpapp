@file:Suppress("SameParameterValue", "MemberVisibilityCanBePrivate")

package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.PointF
import android.graphics.Rect
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import org.opencv.android.Utils
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.OutputStream
import java.util.Locale
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.roundToInt

object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    data class RoiEdgePack3250(
        val roiRectGlobal3250: Rect,
        val w3250: Int,
        val h3250: Int,
        val grayU83250: ByteArray,
        val scoreU83250: ByteArray,
        val hScoreU83250: ByteArray,
        val vScoreU83250: ByteArray,
        val cannyU83250: ByteArray,
        val edgeRawU83250: ByteArray,
        val edgePostU83250: ByteArray,
        val dirU83250: ByteArray,
        val thrScore3250: Int,
        val pctlScore3250: Double,
        val nonZeroRaw3250: Int,
        val nonZeroPost3250: Int
    )
    data class Params3250(
        val blurK3250: Int = 3,
        val scharrWeight3250: Double = 1.0,
        val hvSuppressCross3250: Double = 0.60,
        val cannyLowFrac3250: Double = 0.55,
        val cannyHighFrac3250: Double = 1.00,
        val scorePercentile3250: Double = 95.0,
        val scoreMinFrac3250: Double = 0.12,
        val closeRadius3250: Int = 1,
        val dilateIters3250: Int = 1,
        val bridgeGapPx3250: Int = 5,
        val bridgeMinRunPx3250: Int = 2,
        val debugSaveToGallery3250: Boolean = false
    )
    fun buildRoiEdgePackFromBitmap3250(
        ctx: Context?,
        stillBmp: Bitmap,
        roiRectGlobal3250: Rect,
        params3250: Params3250 = Params3250(),
        debugTag3250: String = "ROI",
        filGeometryPtsGlobal3250: List<PointF>? = null
    ): RoiEdgePack3250 {
        val roi = clampRect3250(roiRectGlobal3250, stillBmp.width, stillBmp.height)
        require(roi.width() > 4 && roi.height() > 4) {
            "EDGE3250: ROI invalido ${roi.width()}x${roi.height()}"
        }

        val roiBmp = Bitmap.createBitmap(
            stillBmp,
            roi.left,
            roi.top,
            roi.width(),
            roi.height()
        )

        val rgba = Mat()
        Utils.bitmapToMat(roiBmp, rgba)

        val bgr = Mat()
        when (rgba.channels()) {
            4 -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            3 -> rgba.copyTo(bgr)
            1 -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_GRAY2BGR)
            else -> error("EDGE3250: channels inesperados=${rgba.channels()}")
        }

        val gray = Mat()
        Imgproc.cvtColor(bgr, gray, Imgproc.COLOR_BGR2GRAY)

        val blur = Mat()
        val blurK = ensureOdd3250(max(1, params3250.blurK3250))
        if (blurK > 1) {
            Imgproc.GaussianBlur(gray, blur, Size(blurK.toDouble(), blurK.toDouble()), 0.0)
        } else {
            gray.copyTo(blur)
        }

        val gx16 = Mat()
        val gy16 = Mat()
        Imgproc.Scharr(blur, gx16, CvType.CV_16S, 1, 0, params3250.scharrWeight3250, 0.0)
        Imgproc.Scharr(blur, gy16, CvType.CV_16S, 0, 1, params3250.scharrWeight3250, 0.0)

        val absGx = Mat()
        val absGy = Mat()
        org.opencv.core.Core.convertScaleAbs(gx16, absGx)
        org.opencv.core.Core.convertScaleAbs(gy16, absGy)

        val score = Mat()
        org.opencv.core.Core.max(absGx, absGy, score)

        val absGx32 = Mat()
        val absGy32 = Mat()
        absGx.convertTo(absGx32, CvType.CV_32F)
        absGy.convertTo(absGy32, CvType.CV_32F)

        val hScore32 = Mat()
        val vScore32 = Mat()
        org.opencv.core.Core.addWeighted(
            absGy32,
            1.0,
            absGx32,
            -params3250.hvSuppressCross3250,
            0.0,
            hScore32
        )
        org.opencv.core.Core.addWeighted(
            absGx32,
            1.0,
            absGy32,
            -params3250.hvSuppressCross3250,
            0.0,
            vScore32
        )

        Imgproc.threshold(hScore32, hScore32, 0.0, 0.0, Imgproc.THRESH_TOZERO)
        Imgproc.threshold(vScore32, vScore32, 0.0, 0.0, Imgproc.THRESH_TOZERO)

        val hScore = Mat()
        val vScore = Mat()
        org.opencv.core.Core.convertScaleAbs(hScore32, hScore)
        org.opencv.core.Core.convertScaleAbs(vScore32, vScore)

        val scoreU8 = matU8ToByteArray3250(score)
        val hScoreU8 = matU8ToByteArray3250(hScore)
        val vScoreU8 = matU8ToByteArray3250(vScore)
        val scoreThr = computeScoreThreshold3250(
            scoreU83250 = scoreU8,
            percentile3250 = params3250.scorePercentile3250,
            minFrac3250 = params3250.scoreMinFrac3250
        )

        val scoreBin = Mat(score.rows(), score.cols(), CvType.CV_8UC1)
        Imgproc.threshold(score, scoreBin, scoreThr.toDouble(), 255.0, Imgproc.THRESH_BINARY)

        val cannyHigh = max(10.0, scoreThr * params3250.cannyHighFrac3250)
        val cannyLow = max(5.0, cannyHigh * params3250.cannyLowFrac3250)

        val canny = Mat()
        Imgproc.Canny(blur, canny, cannyLow, cannyHigh, 3, false)

        val edgeRaw = Mat()
        org.opencv.core.Core.max(scoreBin, canny, edgeRaw)

        val edgePost = postProcessEdgeU83250(
            srcU83250 = edgeRaw,
            closeRadius3250 = params3250.closeRadius3250,
            bridgeGapPx3250 = params3250.bridgeGapPx3250,
            bridgeMinRunPx3250 = params3250.bridgeMinRunPx3250,
            dilateIters3250 = params3250.dilateIters3250
        )

        val dirU8 = buildDirMap3250(gx16, gy16)

        val grayU8 = matU8ToByteArray3250(gray)
        val cannyU8 = matU8ToByteArray3250(canny)
        val edgeRawU8 = matU8ToByteArray3250(edgeRaw)
        val edgePostU8 = matU8ToByteArray3250(edgePost)

        val pack = RoiEdgePack3250(
            roiRectGlobal3250 = roi,
            w3250 = roi.width(),
            h3250 = roi.height(),
            grayU83250 = grayU8,
            scoreU83250 = scoreU8,
            hScoreU83250 = hScoreU8,
            vScoreU83250 = vScoreU8,
            cannyU83250 = cannyU8,
            edgeRawU83250 = edgeRawU8,
            edgePostU83250 = edgePostU8,
            dirU83250 = dirU8,
            thrScore3250 = scoreThr,
            pctlScore3250 = params3250.scorePercentile3250,
            nonZeroRaw3250 = countNonZero3250(edgeRawU8),
            nonZeroPost3250 = countNonZero3250(edgePostU8)
        )

        Log.d(
            TAG,
            String.format(
                Locale.US,
                "EDGE3250[%s] roi=(%d,%d %dx%d) thr=%d pctl=%.1f rawNZ=%d postNZ=%d cannyLow=%.1f cannyHigh=%.1f",
                debugTag3250,
                roi.left,
                roi.top,
                roi.width(),
                roi.height(),
                scoreThr,
                params3250.scorePercentile3250,
                pack.nonZeroRaw3250,
                pack.nonZeroPost3250,
                cannyLow,
                cannyHigh
            )
        )

        if (params3250.debugSaveToGallery3250 && ctx != null) {
            debugDumpRoiEdgePack3250(
                ctx = ctx,
                pack3250 = pack,
                debugTag3250 = debugTag3250,
                filGeometryPtsGlobal3250 = filGeometryPtsGlobal3250
            )
        }

        rgba.release()
        bgr.release()
        gray.release()
        blur.release()
        gx16.release()
        gy16.release()
        absGx.release()
        absGy.release()
        score.release()
        scoreBin.release()
        canny.release()
        edgeRaw.release()
        edgePost.release()
        absGx32.release()
        absGy32.release()
        hScore32.release()
        vScore32.release()
        hScore.release()
        vScore.release()

        return pack
    }

    fun debugDumpRoiEdgePack3250(
        ctx: Context,
        pack3250: RoiEdgePack3250,
        debugTag3250: String,
        filGeometryPtsGlobal3250: List<PointF>? = null
    ) {
        val grayBmp = u8ToBitmap3250(pack3250.grayU83250, pack3250.w3250, pack3250.h3250)
        val scoreBmp = u8ToBitmap3250(pack3250.scoreU83250, pack3250.w3250, pack3250.h3250)
        val hScoreBmp = u8ToBitmap3250(pack3250.hScoreU83250, pack3250.w3250, pack3250.h3250)
        val vScoreBmp = u8ToBitmap3250(pack3250.vScoreU83250, pack3250.w3250, pack3250.h3250)
        val cannyBmp = u8ToBitmap3250(pack3250.cannyU83250, pack3250.w3250, pack3250.h3250)
        val rawBmp = u8ToBitmap3250(pack3250.edgeRawU83250, pack3250.w3250, pack3250.h3250)
        val postBmp = u8ToBitmap3250(pack3250.edgePostU83250, pack3250.w3250, pack3250.h3250)

        saveBitmapPng3250(ctx, grayBmp, "EDGE3250_${debugTag3250}_01_gray")
        saveBitmapPng3250(ctx, scoreBmp, "EDGE3250_${debugTag3250}_02_score")
        saveBitmapPng3250(ctx, hScoreBmp, "EDGE3250_${debugTag3250}_03_hscore")
        saveBitmapPng3250(ctx, vScoreBmp, "EDGE3250_${debugTag3250}_04_vscore")
        saveBitmapPng3250(ctx, cannyBmp, "EDGE3250_${debugTag3250}_05_canny")
        saveBitmapPng3250(ctx, rawBmp, "EDGE3250_${debugTag3250}_06_edge_raw")
        saveBitmapPng3250(ctx, postBmp, "EDGE3250_${debugTag3250}_07_detector_input")


        val overlayBmp = postBmp.copy(Bitmap.Config.ARGB_8888, true)
        val ptsRoi = filGeometryPtsGlobal3250
            ?.map {
                PointF(
                    it.x - pack3250.roiRectGlobal3250.left,
                    it.y - pack3250.roiRectGlobal3250.top
                )
            }
            ?.filter { it.x in 0f..(pack3250.w3250 - 1).toFloat() && it.y in 0f..(pack3250.h3250 - 1).toFloat() }


        if (!ptsRoi.isNullOrEmpty()) {
            drawPolylineOnBitmap3250(
                bmp = overlayBmp,
                pts = ptsRoi,
                color = Color.RED,
                strokePx = 2.0f,
                closed = true
            )
        }
        saveBitmapPng3250(ctx, overlayBmp, "EDGE3250_${debugTag3250}_08_detector_input_plus_fil")
    }

    private fun postProcessEdgeU83250(
        srcU83250: Mat,
        closeRadius3250: Int,
        bridgeGapPx3250: Int,
        bridgeMinRunPx3250: Int,
        dilateIters3250: Int
    ): Mat {
        val work = Mat()
        srcU83250.copyTo(work)

        if (closeRadius3250 > 0) {
            val k = 2 * closeRadius3250 + 1
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(k.toDouble(), k.toDouble())
            )
            Imgproc.morphologyEx(work, work, Imgproc.MORPH_CLOSE, kernel)
            kernel.release()
        }

        val tmpBridge = bridgeHorizontalGaps3250(
            srcU83250 = work,
            maxGapPx3250 = bridgeGapPx3250,
            minRunPx3250 = bridgeMinRunPx3250
        )
        work.release()

        if (dilateIters3250 > 0) {
            val kernel = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size(3.0, 3.0)
            )
            Imgproc.dilate(tmpBridge, tmpBridge, kernel, Point(-1.0, -1.0), dilateIters3250)
            kernel.release()
        }

        return tmpBridge
    }

    private fun bridgeHorizontalGaps3250(
        srcU83250: Mat,
        maxGapPx3250: Int,
        minRunPx3250: Int
    ): Mat {
        val w = srcU83250.cols()
        val h = srcU83250.rows()
        val arr = ByteArray(w * h)
        srcU83250.get(0, 0, arr)

        fun isOn(x: Int, y: Int): Boolean {
            return (arr[y * w + x].toInt() and 0xFF) != 0
        }

        fun setOn(x: Int, y: Int) {
            arr[y * w + x] = 0xFF.toByte()
        }

        for (y in 0 until h) {
            var x = 0
            while (x < w) {
                if (!isOn(x, y)) {
                    x++
                    continue
                }

                var run1End = x
                while (run1End + 1 < w && isOn(run1End + 1, y)) run1End++
                val run1Len = run1End - x + 1

                val gapStart = run1End + 1
                var gapEnd = gapStart
                while (gapEnd < w && !isOn(gapEnd, y)) gapEnd++

                if (gapEnd >= w) {
                    x = run1End + 1
                    continue
                }

                val run2Start = gapEnd
                var run2End = run2Start
                while (run2End + 1 < w && isOn(run2End + 1, y)) run2End++
                val run2Len = run2End - run2Start + 1
                val gapLen = run2Start - gapStart

                if (run1Len >= minRunPx3250 && run2Len >= minRunPx3250 && gapLen in 1..maxGapPx3250) {
                    for (xx in gapStart until run2Start) setOn(xx, y)
                }

                x = run2End + 1
            }
        }

        val out = Mat(h, w, CvType.CV_8UC1)
        out.put(0, 0, arr)
        return out
    }

    private fun buildDirMap3250(gx16: Mat, gy16: Mat): ByteArray {
        val w = gx16.cols()
        val h = gx16.rows()

        val gxArr = ShortArray(w * h)
        val gyArr = ShortArray(w * h)
        gx16.get(0, 0, gxArr)
        gy16.get(0, 0, gyArr)

        val out = ByteArray(w * h)

        for (i in out.indices) {
            val gx = gxArr[i].toInt()
            val gy = gyArr[i].toInt()

            if (gx == 0 && gy == 0) {
                out[i] = 255.toByte()
                continue
            }

            val ang = Math.toDegrees(atan2(gy.toDouble(), gx.toDouble()))
            val a = ((ang + 180.0) % 180.0)

            val bin = when {
                a < 22.5 -> 0
                a < 67.5 -> 1
                a < 112.5 -> 2
                a < 157.5 -> 3
                else -> 0
            }
            out[i] = bin.toByte()
        }

        return out
    }

    private fun computeScoreThreshold3250(
        scoreU83250: ByteArray,
        percentile3250: Double,
        minFrac3250: Double
    ): Int {
        val hist = IntArray(256)
        var maxV = 0

        for (b in scoreU83250) {
            val v = b.toInt() and 0xFF
            hist[v]++
            if (v > maxV) maxV = v
        }

        if (maxV <= 0) return 1

        val target = (scoreU83250.size * (percentile3250 / 100.0)).roundToInt()
        var acc = 0
        var pctl = maxV

        for (v in 0..255) {
            acc += hist[v]
            if (acc >= target) {
                pctl = v
                break
            }
        }

        val minThr = max(1, (maxV * minFrac3250).roundToInt())
        return max(pctl, minThr)
    }

    private fun matU8ToByteArray3250(mat: Mat): ByteArray {
        require(mat.type() == CvType.CV_8UC1) {
            "EDGE3250: matU8ToByteArray espera CV_8UC1, llegó type=${mat.type()}"
        }
        val arr = ByteArray(mat.rows() * mat.cols())
        mat.get(0, 0, arr)
        return arr
    }

    private fun u8ToBitmap3250(u8: ByteArray, w: Int, h: Int): Bitmap {
        require(u8.size == w * h) {
            "EDGE3250: u8ToBitmap size mismatch expected=${w * h} actual=${u8.size}"
        }

        val pixels = IntArray(w * h)
        for (i in u8.indices) {
            val v = u8[i].toInt() and 0xFF
            pixels[i] = Color.argb(255, v, v, v)
        }

        return Bitmap.createBitmap(pixels, w, h, Bitmap.Config.ARGB_8888)
    }

    private fun countNonZero3250(arr: ByteArray): Int {
        var c = 0
        for (b in arr) {
            if ((b.toInt() and 0xFF) != 0) c++
        }
        return c
    }

    private fun clampRect3250(r: Rect, maxW: Int, maxH: Int): Rect {
        val l = r.left.coerceIn(0, maxW - 1)
        val t = r.top.coerceIn(0, maxH - 1)
        val rr = r.right.coerceIn(l + 1, maxW)
        val bb = r.bottom.coerceIn(t + 1, maxH)
        return Rect(l, t, rr, bb)
    }

    private fun ensureOdd3250(v: Int): Int = if (v % 2 == 0) v + 1 else v

    private fun drawPolylineOnBitmap3250(
        bmp: Bitmap,
        pts: List<PointF>,
        color: Int,
        strokePx: Float,
        closed: Boolean
    ) {
        if (pts.size < 2) return

        val canvas = Canvas(bmp)
        val p = Paint(Paint.ANTI_ALIAS_FLAG).apply {
            this.color = color
            style = Paint.Style.STROKE
            strokeWidth = strokePx
        }

        for (i in 0 until pts.lastIndex) {
            val a = pts[i]
            val b = pts[i + 1]
            canvas.drawLine(a.x, a.y, b.x, b.y, p)
        }

        if (closed) {
            val a = pts.last()
            val b = pts.first()
            canvas.drawLine(a.x, a.y, b.x, b.y, p)
        }
    }

    private fun saveBitmapPng3250(
        ctx: Context,
        bmp: Bitmap,
        baseName3250: String
    ): Uri? {
        val resolver = ctx.contentResolver
        val fileName = "${baseName3250}_${System.currentTimeMillis()}.png"

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/PrecalDNP")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: return null

        var os: OutputStream? = null
        return try {
            os = resolver.openOutputStream(uri)
                ?: throw IllegalStateException("EDGE3250: openOutputStream devolvió null para $fileName")

            bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            os.flush()

            val done = ContentValues().apply {
                put(MediaStore.Images.Media.IS_PENDING, 0)
            }
            resolver.update(uri, done, null, null)

            uri
        } catch (t: Throwable) {
            Log.e(TAG, "EDGE3250: saveBitmapPng falló $baseName3250", t)
            null
        } finally {
            try {
                os?.close()
            } catch (_: Throwable) {
            }
        }
    }
}
