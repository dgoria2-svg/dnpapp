package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

data class FrontMaskResult3250(
    val ok: Boolean,
    val workMaskU8: ByteArray,
    val rimBodyMaskU8: ByteArray,
    val innerBoundaryMaskU8: ByteArray,
    val lensInteriorMaskU8: ByteArray,
    val flattenMaskU8: ByteArray,
    val w: Int,
    val h: Int,
    val roi: Rect
) {
    override fun equals(other: Any?): Boolean {
        if (this === other) return true
        if (javaClass != other?.javaClass) return false

        other as FrontMaskResult3250

        if (ok != other.ok) return false
        if (w != other.w) return false
        if (h != other.h) return false
        if (!workMaskU8.contentEquals(other.workMaskU8)) return false
        if (!rimBodyMaskU8.contentEquals(other.rimBodyMaskU8)) return false
        if (!innerBoundaryMaskU8.contentEquals(other.innerBoundaryMaskU8)) return false
        if (!lensInteriorMaskU8.contentEquals(other.lensInteriorMaskU8)) return false
        if (!flattenMaskU8.contentEquals(other.flattenMaskU8)) return false
        if (roi != other.roi) return false

        return true
    }

    override fun hashCode(): Int {
        var result = ok.hashCode()
        result = 31 * result + w
        result = 31 * result + h
        result = 31 * result + workMaskU8.contentHashCode()
        result = 31 * result + rimBodyMaskU8.contentHashCode()
        result = 31 * result + innerBoundaryMaskU8.contentHashCode()
        result = 31 * result + lensInteriorMaskU8.contentHashCode()
        result = 31 * result + flattenMaskU8.contentHashCode()
        result = 31 * result + roi.hashCode()
        return result
    }
}

object GlassesFrontMasker3250 {

    fun build(
        src: Bitmap,
        pLeft: PointF?,
        pRight: PointF?,
        browBottomY: Float? = null,
        growPx: Int = 4
    ): FrontMaskResult3250 {

        val w = src.width
        val h = src.height
        val empty = ByteArray(w * h)

        if (pLeft == null || pRight == null) {
            return FrontMaskResult3250(
                ok = false,
                workMaskU8 = empty,
                rimBodyMaskU8 = empty,
                innerBoundaryMaskU8 = empty,
                lensInteriorMaskU8 = empty,
                flattenMaskU8 = empty,
                w = w,
                h = h,
                roi = Rect(0, 0, w, h)
            )
        }

        val pd = abs(pRight.x - pLeft.x).coerceAtLeast(60f)
        val minPx = min(pLeft.x, pRight.x)
        val maxPx = max(pLeft.x, pRight.x)
        val minPy = min(pLeft.y, pRight.y)
        val maxPy = max(pLeft.y, pRight.y)

        val x0 = floor((minPx - 1.05f * pd).toDouble()).toInt().coerceIn(0, w - 2)
        val x1 = ceil((maxPx + 1.05f * pd).toDouble()).toInt().coerceIn(x0 + 2, w)
        val y0Base = browBottomY ?: (minPy - 0.80f * pd)
        val y0 = floor((y0Base - 0.20f * pd).toDouble()).toInt().coerceIn(0, h - 2)
        val y1 = ceil((maxPy + 0.95f * pd).toDouble()).toInt().coerceIn(y0 + 2, h)

        val roiRect = Rect(x0, y0, x1 - x0, y1 - y0)

        val rgba = Mat()
        Utils.bitmapToMat(src, rgba)

        val bgr = Mat()
        when (rgba.channels()) {
            4 -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
            3 -> rgba.copyTo(bgr)
            else -> Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_GRAY2BGR)
        }

        val roiBgr = bgr.submat(roiRect)
        val roiW = roiRect.width
        val roiH = roiRect.height

        val gcMask = Mat(roiH, roiW, CvType.CV_8U, Scalar(Imgproc.GC_PR_BGD.toDouble()))
        val seed = Mat.zeros(roiH, roiW, CvType.CV_8U)

        val l = Point((pLeft.x - x0).toDouble(), (pLeft.y - y0).toDouble())
        val r = Point((pRight.x - x0).toDouble(), (pRight.y - y0).toDouble())
        val cy = (l.y + r.y) * 0.5

        val lensRx = max(24, (0.46f * pd).roundToInt())
        val lensRy = max(18, (0.38f * pd).roundToInt())
        val topBarH = max(5, (0.12f * pd).roundToInt())
        val topBarY = (cy - 0.92 * lensRy).roundToInt()

        Imgproc.ellipse(
            gcMask,
            l,
            Size(lensRx.toDouble(), lensRy.toDouble()),
            0.0,
            0.0,
            360.0,
            Scalar(Imgproc.GC_PR_FGD.toDouble()),
            -1
        )
        Imgproc.ellipse(
            gcMask,
            r,
            Size(lensRx.toDouble(), lensRy.toDouble()),
            0.0,
            0.0,
            360.0,
            Scalar(Imgproc.GC_PR_FGD.toDouble()),
            -1
        )
        Imgproc.ellipse(
            seed,
            l,
            Size(lensRx.toDouble(), lensRy.toDouble()),
            0.0,
            0.0,
            360.0,
            Scalar(255.0),
            -1
        )
        Imgproc.ellipse(
            seed,
            r,
            Size(lensRx.toDouble(), lensRy.toDouble()),
            0.0,
            0.0,
            360.0,
            Scalar(255.0),
            -1
        )

        fillRect(
            gcMask,
            (l.x - lensRx).roundToInt(),
            topBarY,
            (r.x + lensRx).roundToInt(),
            topBarY + topBarH,
            Imgproc.GC_PR_FGD
        )
        fillRect(
            seed,
            (l.x - lensRx).roundToInt(),
            topBarY,
            (r.x + lensRx).roundToInt(),
            topBarY + topBarH,
            255
        )

        val borderX = max(4, (0.05f * roiW).roundToInt())
        val borderY = max(4, (0.05f * roiH).roundToInt())

        fillRect(gcMask, 0, 0, roiW - 1, borderY, Imgproc.GC_BGD)
        fillRect(gcMask, 0, roiH - 1 - borderY, roiW - 1, roiH - 1, Imgproc.GC_BGD)
        fillRect(gcMask, 0, 0, borderX, roiH - 1, Imgproc.GC_BGD)
        fillRect(gcMask, roiW - 1 - borderX, 0, roiW - 1, roiH - 1, Imgproc.GC_BGD)

        val bgdModel = Mat()
        val fgdModel = Mat()
        val rectInit = Rect(1, 1, max(2, roiW - 2), max(2, roiH - 2))

        Imgproc.grabCut(
            roiBgr,
            gcMask,
            rectInit,
            bgdModel,
            fgdModel,
            5,
            Imgproc.GC_INIT_WITH_MASK
        )

        val fg = Mat()
        val pfg = Mat()
        val bin = Mat()

        Core.compare(gcMask, Scalar(Imgproc.GC_FGD.toDouble()), fg, Core.CMP_EQ)
        Core.compare(gcMask, Scalar(Imgproc.GC_PR_FGD.toDouble()), pfg, Core.CMP_EQ)
        Core.bitwise_or(fg, pfg, bin)

        val closeK = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(
                odd(max(5, (0.08f * pd).roundToInt())).toDouble(),
                odd(max(5, (0.08f * pd).roundToInt())).toDouble()
            )
        )
        val openK = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(
                odd(max(3, (0.03f * pd).roundToInt())).toDouble(),
                odd(max(3, (0.03f * pd).roundToInt())).toDouble()
            )
        )
        val seedDilK = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(
                odd(max(7, (0.10f * pd).roundToInt())).toDouble(),
                odd(max(7, (0.10f * pd).roundToInt())).toDouble()
            )
        )

        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_CLOSE, closeK)
        Imgproc.morphologyEx(bin, bin, Imgproc.MORPH_OPEN, openK)

        val seedDil = Mat()
        Imgproc.dilate(seed, seedDil, seedDilK)

        val kept = keepSeededComponents(bin, seedDil)
        val filled = fillExternalContours(kept, 0.002 * roiW * roiH)

        Imgproc.morphologyEx(filled, filled, Imgproc.MORPH_CLOSE, closeK)

        if (growPx > 0) {
            val growK = Imgproc.getStructuringElement(
                Imgproc.MORPH_ELLIPSE,
                Size((growPx * 2 + 1).toDouble(), (growPx * 2 + 1).toDouble())
            )
            Imgproc.dilate(filled, filled, growK)
            growK.release()
        }

        val workRoi = filled.clone()

        val rimThicknessPx = max(3, (0.045f * pd).roundToInt())
        val lensInsetPx = max(rimThicknessPx + 2, (0.09f * pd).roundToInt())
        val innerBandPx = max(2, (0.025f * pd).roundToInt())

        val rimK = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size((rimThicknessPx * 2 + 1).toDouble(), (rimThicknessPx * 2 + 1).toDouble())
        )
        val lensK = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size((lensInsetPx * 2 + 1).toDouble(), (lensInsetPx * 2 + 1).toDouble())
        )
        val innerBandK = Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size((innerBandPx * 2 + 1).toDouble(), (innerBandPx * 2 + 1).toDouble())
        )

        val erodedForRim = Mat()
        Imgproc.erode(workRoi, erodedForRim, rimK)

        val erodedForLens = Mat()
        Imgproc.erode(workRoi, erodedForLens, lensK)

        val rimBodyRoi = Mat()
        Core.subtract(workRoi, erodedForLens, rimBodyRoi)

        val innerBandBase = Mat()
        Imgproc.dilate(erodedForLens, innerBandBase, innerBandK)

        val innerBoundaryRoi = Mat()
        Core.subtract(innerBandBase, erodedForLens, innerBoundaryRoi)

        val lensInteriorRoi = erodedForLens.clone()

        val flattenRoi = buildFlattenWeightFromLensInterior3250(lensInteriorRoi)

        val fullWork = Mat.zeros(h, w, CvType.CV_8U)
        val fullRimBody = Mat.zeros(h, w, CvType.CV_8U)
        val fullInnerBoundary = Mat.zeros(h, w, CvType.CV_8U)
        val fullLensInterior = Mat.zeros(h, w, CvType.CV_8U)
        val fullFlatten = Mat.zeros(h, w, CvType.CV_8U)

        workRoi.copyTo(fullWork.submat(roiRect))
        rimBodyRoi.copyTo(fullRimBody.submat(roiRect))
        innerBoundaryRoi.copyTo(fullInnerBoundary.submat(roiRect))
        lensInteriorRoi.copyTo(fullLensInterior.submat(roiRect))
        flattenRoi.copyTo(fullFlatten.submat(roiRect))

        val workOut = ByteArray(w * h)
        val rimBodyOut = ByteArray(w * h)
        val innerBoundaryOut = ByteArray(w * h)
        val lensInteriorOut = ByteArray(w * h)
        val flattenOut = ByteArray(w * h)

        fullWork.get(0, 0, workOut)
        fullRimBody.get(0, 0, rimBodyOut)
        fullInnerBoundary.get(0, 0, innerBoundaryOut)
        fullLensInterior.get(0, 0, lensInteriorOut)
        fullFlatten.get(0, 0, flattenOut)

        fullWork.release()
        fullRimBody.release()
        fullInnerBoundary.release()
        fullLensInterior.release()
        fullFlatten.release()
        flattenRoi.release()
        lensInteriorRoi.release()
        innerBoundaryRoi.release()
        innerBandBase.release()
        rimBodyRoi.release()
        erodedForLens.release()
        erodedForRim.release()
        rimK.release()
        lensK.release()
        innerBandK.release()
        workRoi.release()
        filled.release()
        kept.release()
        seedDil.release()
        seed.release()
        closeK.release()
        openK.release()
        seedDilK.release()
        bin.release()
        fg.release()
        pfg.release()
        bgdModel.release()
        fgdModel.release()
        gcMask.release()
        roiBgr.release()
        bgr.release()
        rgba.release()

        return FrontMaskResult3250(
            ok = true,
            workMaskU8 = workOut,
            rimBodyMaskU8 = rimBodyOut,
            innerBoundaryMaskU8 = innerBoundaryOut,
            lensInteriorMaskU8 = lensInteriorOut,
            flattenMaskU8 = flattenOut,
            w = w,
            h = h,
            roi = roiRect
        )
    }

    private fun buildFlattenWeightFromLensInterior3250(
        lensInteriorRoi: Mat
    ): Mat {
        val dist = Mat()
        Imgproc.distanceTransform(lensInteriorRoi, dist, Imgproc.DIST_L2, 3)

        val mm = Core.minMaxLoc(dist)
        val maxDist = mm.maxVal.toFloat().coerceAtLeast(1f)

        val startFrac = 0.10f
        val fullFrac = 0.50f

        val startDist = maxDist * startFrac
        val fullDist = maxDist * fullFrac

        val w = lensInteriorRoi.cols()
        val h = lensInteriorRoi.rows()

        val dist32 = FloatArray(w * h)
        dist.get(0, 0, dist32)

        val lensU8 = ByteArray(w * h)
        lensInteriorRoi.get(0, 0, lensU8)

        val outU8 = ByteArray(w * h)

        for (i in dist32.indices) {
            val inside = (lensU8[i].toInt() and 0xFF) != 0
            if (!inside) {
                outU8[i] = 0
                continue
            }

            val d = dist32[i]
            val t = when {
                d <= startDist -> 0f
                d >= fullDist -> 1f
                else -> (d - startDist) / (fullDist - startDist)
            }

            val smooth = t * t * (3f - 2f * t)
            outU8[i] = (smooth * 255f).roundToInt().coerceIn(0, 255).toByte()
        }

        val out = Mat(h, w, CvType.CV_8U)
        out.put(0, 0, outU8)
        dist.release()
        return out
    }

    private fun keepSeededComponents(bin: Mat, seedDil: Mat): Mat {
        val labels = Mat()
        val stats = Mat()
        val centroids = Mat()

        Imgproc.connectedComponentsWithStats(
            bin,
            labels,
            stats,
            centroids,
            8,
            CvType.CV_32S
        )

        var out = Mat.zeros(bin.size(), CvType.CV_8U)

        for (label in 1 until stats.rows()) {
            val comp = Mat()
            val overlap = Mat()

            Core.compare(labels, Scalar(label.toDouble()), comp, Core.CMP_EQ)
            Core.bitwise_and(comp, seedDil, overlap)

            if (Core.countNonZero(overlap) > 0) {
                Core.bitwise_or(out, comp, out)
            }

            comp.release()
            overlap.release()
        }

        if (Core.countNonZero(out) == 0) {
            out.release()
            out = bin.clone()
        }

        labels.release()
        stats.release()
        centroids.release()

        return out
    }

    private fun fillExternalContours(src: Mat, minArea: Double): Mat {
        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        val out = Mat.zeros(src.size(), CvType.CV_8U)

        Imgproc.findContours(
            src.clone(),
            contours,
            hierarchy,
            Imgproc.RETR_EXTERNAL,
            Imgproc.CHAIN_APPROX_SIMPLE
        )

        for (c in contours) {
            if (Imgproc.contourArea(c) >= minArea) {
                Imgproc.drawContours(out, listOf(c), -1, Scalar(255.0), -1)
            }
            c.release()
        }

        hierarchy.release()
        return out
    }

    private fun fillRect(mask: Mat, x0: Int, y0: Int, x1: Int, y1: Int, value: Int) {
        val l = min(x0, x1).coerceIn(0, mask.cols() - 1)
        val r = max(x0, x1).coerceIn(0, mask.cols() - 1)
        val t = min(y0, y1).coerceIn(0, mask.rows() - 1)
        val b = max(y0, y1).coerceIn(0, mask.rows() - 1)

        if (r < l || b < t) return

        Imgproc.rectangle(
            mask,
            Point(l.toDouble(), t.toDouble()),
            Point(r.toDouble(), b.toDouble()),
            Scalar(value.toDouble()),
            -1
        )
    }

    private fun odd(v: Int): Int {
        return if ((v and 1) == 1) v else v + 1
    }
}
