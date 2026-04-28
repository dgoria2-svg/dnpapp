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
import java.util.Locale
import kotlin.math.PI
import kotlin.math.atan2
import kotlin.math.max
import kotlin.math.roundToInt

object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    class RoiEdgePack3250(
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
        val cannyHighFrac3250: Double = 1.00,
        val scorePercentile3250: Double = 99.0,
        val scoreMinFrac3250: Double = 0.20,
        val closeRadius3250: Int = 1,
        val dilateIters3250: Int = 1,
        val bridgeGapPx3250: Int = 5,
        val bridgeMinRunPx3250: Int = 2,
        val cannyIgnoreZeros3250: Boolean = true,
        val cannyHighPercentile3250: Double = 99.0,
        val cannyLowFrac3250: Double = 0.50,
        val scoreMixWeight3250: Double = 0.85,
        val cannyMixWeight3250: Double = 0.15,
        val debugSaveToGallery3250: Boolean = false
    )

    fun buildRoiEdgePackFromBitmap3250(
        ctx: Context?,
        stillBmp: Bitmap,
        roiRectGlobal3250: Rect,
        params3250: Params3250 = Params3250(),
        debugTag3250: String = "ROI",
        filGeometryPtsGlobal3250: List<PointF>? = null,
        workMaskFullU83250: ByteArray? = null,
        rimBodyMaskFullU83250: ByteArray? = null,
        innerBoundaryMaskFullU83250: ByteArray? = null,
        flattenMaskFullU83250: ByteArray? = null,
        vetoMaskFullU83250: ByteArray? = null
    ): RoiEdgePack3250 {
        val roi = clampRect3250(roiRectGlobal3250, stillBmp.width, stillBmp.height)
        require(roi.width() > 4 && roi.height() > 4) {
            "EDGE3250: ROI invalido ${roi.width()}x${roi.height()}"
        }

        val workMaskRoiU83250 = cropFullMaskToRoi3250(
            full = workMaskFullU83250,
            fullW = stillBmp.width,
            fullH = stillBmp.height,
            roi = roi
        )

        val rimBodyMaskRoiU83250 = cropFullMaskToRoi3250(
            full = rimBodyMaskFullU83250,
            fullW = stillBmp.width,
            fullH = stillBmp.height,
            roi = roi
        )

        val innerBoundaryMaskRoiU83250 = cropFullMaskToRoi3250(
            full = innerBoundaryMaskFullU83250,
            fullW = stillBmp.width,
            fullH = stillBmp.height,
            roi = roi
        )

        val flattenMaskRoiU83250 = cropFullMaskToRoi3250(
            full = flattenMaskFullU83250,
            fullW = stillBmp.width,
            fullH = stillBmp.height,
            roi = roi
        )
        val vetoMaskRoiU83250 = cropFullMaskToRoi3250(
            full = vetoMaskFullU83250,
            fullW = stillBmp.width,
            fullH = stillBmp.height,
            roi = roi
        )
        Log.d(
            TAG,
            "EDGE_MASKS[$debugTag3250] " +
                    "work=${countNonZero3250(workMaskRoiU83250 ?: ByteArray(0))} " +
                    "rim=${countNonZero3250(rimBodyMaskRoiU83250 ?: ByteArray(0))} " +
                    "inner=${countNonZero3250(innerBoundaryMaskRoiU83250 ?: ByteArray(0))} " +
                    "flatten=${countNonZero3250(flattenMaskRoiU83250 ?: ByteArray(0))} " +
                    "veto=${countNonZero3250(vetoMaskRoiU83250 ?: ByteArray(0))}"
        )
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

        val blurSoft = Mat()
        val blurStrong = Mat()

        val blurKSoft = ensureOdd3250(max(1, params3250.blurK3250))
        val blurKStrong = ensureOdd3250(max(blurKSoft + 6, 11))

        if (blurKSoft > 1) {
            Imgproc.GaussianBlur(
                gray,
                blurSoft,
                Size(blurKSoft.toDouble(), blurKSoft.toDouble()),
                0.0
            )
        } else {
            gray.copyTo(blurSoft)
        }

        Imgproc.GaussianBlur(
            gray,
            blurStrong,
            Size(blurKStrong.toDouble(), blurKStrong.toDouble()),
            0.0
        )

        val graySrcU83250 = matU8ToByteArray3250(gray)
        val blurSoftU83250 = matU8ToByteArray3250(blurSoft)
        val blurStrongU83250 = matU8ToByteArray3250(blurStrong)

        val guidedGrayU83250 = buildGuidedGray3250(
            graySrcU83250 = graySrcU83250,
            blurSoftU83250 = blurSoftU83250,
            blurStrongU83250 = blurStrongU83250,
            workMaskU83250 = workMaskRoiU83250,
            rimBodyMaskU83250 = rimBodyMaskRoiU83250,
            innerBoundaryMaskU83250 = innerBoundaryMaskRoiU83250,
            flattenMaskU83250 = flattenMaskRoiU83250
        )

        val blur = Mat(gray.rows(), gray.cols(), CvType.CV_8UC1)
        blur.put(0, 0, guidedGrayU83250)

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

        val vetoRoiU83250 = buildVetoRoi3250(
            flattenMaskU83250 = flattenMaskRoiU83250,
            rimBodyMaskU83250 = rimBodyMaskRoiU83250,
            innerBoundaryMaskU83250 = innerBoundaryMaskRoiU83250,
            vetoMaskRoiU83250 = vetoMaskRoiU83250
        )

        val scoreU8 = applyVeto3250(matU8ToByteArray3250(score), vetoRoiU83250, zeroValue3250 = 0)
        val hScoreU8 = applyVeto3250(matU8ToByteArray3250(hScore), vetoRoiU83250, zeroValue3250 = 0)
        val vScoreU8 = applyVeto3250(matU8ToByteArray3250(vScore), vetoRoiU83250, zeroValue3250 = 0)

        score.put(0, 0, scoreU8)
        hScore.put(0, 0, hScoreU8)
        vScore.put(0, 0, vScoreU8)

        val scoreThr = computeScoreThreshold3250(
            scoreU83250 = scoreU8,
            percentile3250 = params3250.scorePercentile3250,
            minFrac3250 = params3250.scoreMinFrac3250
        )

        val scoreBin = Mat(score.rows(), score.cols(), CvType.CV_8UC1)
        Imgproc.threshold(score, scoreBin, scoreThr.toDouble(), 255.0, Imgproc.THRESH_BINARY)

        val cannyHighBase = computePercentileU83250(
            valuesU83250 = scoreU8,
            percentile3250 = params3250.cannyHighPercentile3250,
            ignoreZeros3250 = params3250.cannyIgnoreZeros3250
        ).coerceAtLeast(10.0)

        val cannyHigh = (cannyHighBase * params3250.cannyHighFrac3250).coerceAtLeast(10.0)
        val cannyLow = (cannyHigh * params3250.cannyLowFrac3250).coerceAtLeast(5.0)

        val canny = Mat()
        Imgproc.Canny(blur, canny, cannyLow, cannyHigh, 3, false)

        val edgeRaw = Mat()
        org.opencv.core.Core.addWeighted(
            scoreBin,
            params3250.scoreMixWeight3250,
            canny,
            params3250.cannyMixWeight3250,
            0.0,
            edgeRaw
        )

        val edgePost = postProcessEdgeU83250(
            srcU83250 = edgeRaw,
            closeRadius3250 = params3250.closeRadius3250,
            bridgeGapPx3250 = params3250.bridgeGapPx3250,
            bridgeMinRunPx3250 = params3250.bridgeMinRunPx3250,
            dilateIters3250 = params3250.dilateIters3250
        )

        val dirU8 = buildDirMap3250(gx16, gy16)

        val cannyU8 = applyVeto3250(matU8ToByteArray3250(canny), vetoRoiU83250, zeroValue3250 = 0)
        val edgeRawU8 = applyVeto3250(matU8ToByteArray3250(edgeRaw), vetoRoiU83250, zeroValue3250 = 0)
        val edgePostU8 = applyVeto3250(matU8ToByteArray3250(edgePost), vetoRoiU83250, zeroValue3250 = 0)
        val dirU8Clean = applyVeto3250(dirU8, vetoRoiU83250, zeroValue3250 = 255)

        val pack = RoiEdgePack3250(
            roiRectGlobal3250 = roi,
            w3250 = roi.width(),
            h3250 = roi.height(),
            grayU83250 = graySrcU83250,
            scoreU83250 = scoreU8,
            hScoreU83250 = hScoreU8,
            vScoreU83250 = vScoreU8,
            cannyU83250 = cannyU8,
            edgeRawU83250 = edgeRawU8,
            edgePostU83250 = edgePostU8,
            dirU83250 = dirU8Clean,
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
                pack.thrScore3250,
                pack.pctlScore3250,
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

        listOf(
            rgba,
            bgr,
            gray,
            blur,
            gx16,
            gy16,
            absGx,
            absGy,
            score,
            scoreBin,
            canny,
            edgeRaw,
            edgePost,
            absGx32,
            absGy32,
            hScore32,
            vScore32,
            hScore,
            vScore,
            blurSoft,
            blurStrong
        ).forEach { it.release() }

        return pack
    }

    fun debugDumpRoiEdgePack3250(
        ctx: Context,
        pack3250: RoiEdgePack3250,
        debugTag3250: String,
        filGeometryPtsGlobal3250: List<PointF>? = null
    ) {
        val labels = listOf(
            "gray",
            "score",
            "hscore",
            "vscore",
            "canny",
            "edge_raw",
            "detector_input"
        )

        val data = listOf(
            pack3250.grayU83250,
            pack3250.scoreU83250,
            pack3250.hScoreU83250,
            pack3250.vScoreU83250,
            pack3250.cannyU83250,
            pack3250.edgeRawU83250,
            pack3250.edgePostU83250
        )

        var postBmp: Bitmap? = null

        data.forEachIndexed { i, bytes ->
            val bmp = u8ToBitmap3250(bytes, pack3250.w3250, pack3250.h3250)
            if (i == 6) postBmp = bmp
            saveBitmapPng3250(
                ctx = ctx,
                bmp = bmp,
                baseName3250 = "EDGE3250_${debugTag3250}_${(i + 1).toString().padStart(2, '0')}_${labels[i]}"
            )
        }

        postBmp?.let { bmp ->
            val overlayBmp = bmp.copy(Bitmap.Config.ARGB_8888, true)

            val ptsRoi = filGeometryPtsGlobal3250
                ?.map {
                    PointF(
                        it.x - pack3250.roiRectGlobal3250.left,
                        it.y - pack3250.roiRectGlobal3250.top
                    )
                }
                ?.filter {
                    it.x in 0f..(pack3250.w3250 - 1).toFloat() &&
                            it.y in 0f..(pack3250.h3250 - 1).toFloat()
                }

            if (!ptsRoi.isNullOrEmpty()) {
                drawPolylineOnBitmap3250(
                    bmp = overlayBmp,
                    pts = ptsRoi,
                    color = Color.RED,
                    strokePx = 2.0f,
                    closed = true
                )
            }

            saveBitmapPng3250(
                ctx = ctx,
                bmp = overlayBmp,
                baseName3250 = "EDGE3250_${debugTag3250}_08_detector_input_plus_fil"
            )
        }
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

        for (y in 0 until h) {
            val rowOff = y * w
            var x = 0

            while (x < w) {
                if ((arr[rowOff + x].toInt() and 0xFF) == 0) {
                    x++
                    continue
                }

                var run1End = x
                while (run1End + 1 < w && (arr[rowOff + run1End + 1].toInt() and 0xFF) != 0) {
                    run1End++
                }
                val run1Len = run1End - x + 1

                val gapStart = run1End + 1
                var gapEnd = gapStart
                while (gapEnd < w && (arr[rowOff + gapEnd].toInt() and 0xFF) == 0) {
                    gapEnd++
                }

                if (gapEnd >= w) {
                    x = run1End + 1
                    continue
                }

                val run2Start = gapEnd
                var run2End = run2Start
                while (run2End + 1 < w && (arr[rowOff + run2End + 1].toInt() and 0xFF) != 0) {
                    run2End++
                }
                val run2Len = run2End - run2Start + 1
                val gapLen = run2Start - gapStart

                if (run1Len >= minRunPx3250 && run2Len >= minRunPx3250 && gapLen in 1..maxGapPx3250) {
                    for (xx in gapStart until run2Start) {
                        arr[rowOff + xx] = 0xFF.toByte()
                    }
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

        val r225 = 22.5 * PI / 180.0
        val r675 = 67.5 * PI / 180.0
        val r1125 = 112.5 * PI / 180.0
        val r1575 = 157.5 * PI / 180.0

        for (i in out.indices) {
            val gx = gxArr[i].toInt()
            val gy = gyArr[i].toInt()

            if (gx == 0 && gy == 0) {
                out[i] = 255.toByte()
                continue
            }

            var ang = atan2(gy.toDouble(), gx.toDouble())
            if (ang < 0.0) ang += PI

            out[i] = when {
                ang < r225 -> 0
                ang < r675 -> 1
                ang < r1125 -> 2
                ang < r1575 -> 3
                else -> 0
            }.toByte()
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

    private fun computePercentileU83250(
        valuesU83250: ByteArray,
        percentile3250: Double,
        ignoreZeros3250: Boolean
    ): Double {
        val hist = IntArray(256)
        var total = 0

        for (b in valuesU83250) {
            val v = b.toInt() and 0xFF
            if (ignoreZeros3250 && v == 0) continue
            hist[v]++
            total++
        }

        if (total <= 0) return 0.0

        val p = percentile3250.coerceIn(0.0, 100.0)
        val target = kotlin.math.ceil(total * (p / 100.0)).toInt().coerceAtLeast(1)

        var acc = 0
        for (v in 0..255) {
            acc += hist[v]
            if (acc >= target) return v.toDouble()
        }

        return 255.0
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

    private fun ensureOdd3250(v: Int): Int {
        return if (v % 2 == 0) v + 1 else v
    }

    private fun cropFullMaskToRoi3250(
        full: ByteArray?,
        fullW: Int,
        fullH: Int,
        roi: Rect
    ): ByteArray? {
        if (full == null) return null
        if (full.size < fullW * fullH) return null

        val w = roi.width()
        val h = roi.height()
        if (w <= 0 || h <= 0) return null

        val out = ByteArray(w * h)
        var dst = 0

        for (y in 0 until h) {
            val srcOff = (roi.top + y) * fullW + roi.left
            full.copyInto(out, dst, srcOff, srcOff + w)
            dst += w
        }


        return out
    }
    private fun buildGuidedGray3250(
        graySrcU83250: ByteArray,
        blurSoftU83250: ByteArray,
        blurStrongU83250: ByteArray,
        workMaskU83250: ByteArray?,
        rimBodyMaskU83250: ByteArray?,
        innerBoundaryMaskU83250: ByteArray?,
        flattenMaskU83250: ByteArray?
    ): ByteArray {
        val n = graySrcU83250.size
        val out = ByteArray(n)

        val hasWork = workMaskU83250 != null
        val hasRim = rimBodyMaskU83250 != null
        val hasInner = innerBoundaryMaskU83250 != null
        val hasFlatten = flattenMaskU83250 != null

        for (i in 0 until n) {
            val gray = graySrcU83250[i].toInt() and 0xFF
            val soft = blurSoftU83250[i].toInt() and 0xFF
            val strong = blurStrongU83250[i].toInt() and 0xFF

            val inInner =
                hasInner && ((innerBoundaryMaskU83250[i].toInt() and 0xFF) != 0)

            val inRim =
                hasRim && ((rimBodyMaskU83250[i].toInt() and 0xFF) != 0)

            val inWork =
                hasWork && ((workMaskU83250[i].toInt() and 0xFF) != 0)

            val flattenW =
                if (hasFlatten) {
                    ((flattenMaskU83250[i].toInt() and 0xFF) / 255f).coerceIn(0f, 1f)
                } else {
                    0f
                }

            val v = when {
                inInner -> {
                    ((gray * 0.88f) + (soft * 0.12f)).roundToInt()
                }

                inRim -> {
                    ((gray * 0.72f) + (soft * 0.28f)).roundToInt()
                }

                inWork -> {
                    val base =
                        (soft * 0.35f) + (strong * 0.65f)

                    val mixed =
                        (base * (1f - flattenW)) + (strong * flattenW)

                    mixed.roundToInt()
                }

                else -> {
                    ((gray * 0.12f) + (strong * 0.88f)).roundToInt()
                }
            }

            out[i] = v.coerceIn(0, 255).toByte()
        }

        return out
    }
    private fun buildVetoRoi3250(
        flattenMaskU83250: ByteArray?,
        rimBodyMaskU83250: ByteArray?,
        innerBoundaryMaskU83250: ByteArray?,
        vetoMaskRoiU83250: ByteArray?
    ): ByteArray? {
        val n = flattenMaskU83250?.size
            ?: vetoMaskRoiU83250?.size
            ?: return null

        val out = ByteArray(n)

        for (i in 0 until n) {
            val inFlatten =
                flattenMaskU83250 != null &&
                        flattenMaskU83250.size > i &&
                        ((flattenMaskU83250[i].toInt() and 0xFF) != 0)

            val inRim =
                rimBodyMaskU83250 != null &&
                        rimBodyMaskU83250.size > i &&
                        ((rimBodyMaskU83250[i].toInt() and 0xFF) != 0)

            val inInner =
                innerBoundaryMaskU83250 != null &&
                        innerBoundaryMaskU83250.size > i &&
                        ((innerBoundaryMaskU83250[i].toInt() and 0xFF) != 0)

            val inDetectorVeto =
                vetoMaskRoiU83250 != null &&
                        vetoMaskRoiU83250.size > i &&
                        ((vetoMaskRoiU83250[i].toInt() and 0xFF) != 0)

            if (
                (inFlatten && !inRim && !inInner) ||
                (inDetectorVeto && !inRim && !inInner)
            ) {
                out[i] = 0xFF.toByte()
            }
        }

        return out
    }

    private fun applyVeto3250(
        srcU83250: ByteArray,
        vetoU83250: ByteArray?,
        zeroValue3250: Int
    ): ByteArray {
        if (vetoU83250 == null || vetoU83250.size < srcU83250.size) return srcU83250

        val out = srcU83250.copyOf()
        val z = zeroValue3250.coerceIn(0, 255).toByte()

        for (i in out.indices) {
            if ((vetoU83250[i].toInt() and 0xFF) != 0) {
                out[i] = z
            }
        }


        return out
    }
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

        return try {
            resolver.openOutputStream(uri)?.use { os ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }
            resolver.update(
                uri,
                ContentValues().apply {
                    put(MediaStore.Images.Media.IS_PENDING, 0)
                },
                null,
                null
            )
            uri
        } catch (t: Throwable) {
            Log.e(TAG, "EDGE3250: saveBitmapPng falló $baseName3250", t)
            null
        }

    }
}

