@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.net.Uri
import android.provider.MediaStore
import android.util.Log
import androidx.core.graphics.createBitmap
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.OutputStream
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object EdgeMapBuilder3250 {

    private const val TAG = "EdgeMapBuilder3250"

    private const val EDGE_THR_FRAC = 0.16f
    private const val EDGE_THR_MIN = 22
    private const val P95_CAP_K = 1.05f

    private const val TAN22_NUM = 41
    private const val TAN22_DEN = 100
    private const val TAN67_NUM = 241
    private const val TAN67_DEN = 100

    private const val DEBUG_MAX_DIM_PX = 1600

    private const val RIBBON_INNER_ERODE_PX = 8
    private const val RIBBON_OUTER_DILATE_PX = 12
    private const val RIBBON_THIN_COVERAGE_FRAC = 0.08f
    private const val RIBBON_CLOSE_RADIUS_PX = 2

    private const val BLUR_GRAY_K = 5
    private const val BLUR_SAT_K = 5
    private const val BLUR_GATE_K = 15
    private const val BLUR_TENSOR_K = 11

    private const val CANNY_GRAY_LOW = 36.0
    private const val CANNY_GRAY_HIGH = 96.0

    private const val CANNY_SAT_LOW = 28.0
    private const val CANNY_SAT_HIGH = 84.0

    private const val SCORE_W_GRAY_NORMAL = 0.34f
    private const val SCORE_W_SAT_NORMAL = 0.16f
    private const val SCORE_W_GRAY_CANNY = 0.14f
    private const val SCORE_W_SAT_CANNY = 0.08f
    private const val SCORE_W_COHERENCY = 0.14f
    private const val SCORE_W_GATE_NORMAL = 0.10f
    private const val SCORE_W_BOTTOM_BONUS = 0.04f

    private const val PENALTY_W_GRAY_TANG = 0.10f
    private const val PENALTY_W_SAT_TANG = 0.06f

    private const val ALIGN_FLOOR = 0.12f
    private const val GATE_CONF_FLOOR = 0.25f
    private const val COH_FLOOR = 0.35f

    private const val BOTTOM_SEED_FRAC_START = 0.56f
    private const val BOTTOM_THR_RELAX_MAX = 20

    private const val BRIDGE_MAX_GAP_PX = 2
    private const val FINAL_CLOSE_RADIUS_PX = 1
    private const val FINAL_DILATE_RADIUS_PX = 0

    private const val MASK_DILATE_RADIUS_PX = 0

    data class EdgeStats(
        val maxScore: Float,
        val p95Score: Float,
        val thr: Float,
        val nnz: Int,
        val density: Float,
        val bandPx: Float,
        val samples: Int,
        val topKillY: Int,
        val maskedN: Int,
        val maskedFrac: Float,
        val maskDilateR: Int
    )

    class EdgeMapOut3250(
        val w: Int,
        val h: Int,
        val edgesU8: ByteArray,
        val dirU8: ByteArray,
        val stats: EdgeStats
    )

    private data class EdgeCoreBuild3250(
        val out: EdgeMapOut3250,
        val gateMaskU8: ByteArray?,
        val ribbonMaskU8: ByteArray?,
        val scoreU8: ByteArray?,
        val projNormalU8: ByteArray?,
        val tangentialU8: ByteArray?,
        val coherencyU8: ByteArray?,
        val gateNormalU8: ByteArray?
    )

    fun buildFullFrameEdgePackFromBitmap3250(
        stillBmp: Bitmap,
        borderKillPx: Int,
        topKillY: Int = 0,
        maskFull: ByteArray? = null,
        regionMaskFull3250: ByteArray? = null,
        forceRegion3250: Boolean = false,
        debugTag: String = "FULL",
        statsOut: ((EdgeStats) -> Unit)? = null,
        ctx: Context? = null,
        debugSaveToGallery3250: Boolean = false,
        debugAlsoSaveGray3250: Boolean = false
    ): EdgeMapOut3250 {
        val w = stillBmp.width
        val h = stillBmp.height

        if (w <= 0 || h <= 0) {
            val st = EdgeStats(
                maxScore = 0f,
                p95Score = 0f,
                thr = 0f,
                nnz = 0,
                density = 0f,
                bandPx = 0f,
                samples = 0,
                topKillY = topKillY,
                maskedN = 0,
                maskedFrac = 0f,
                maskDilateR = 0
            )
            return EdgeMapOut3250(0, 0, ByteArray(0), ByteArray(0), st)
        }

        val grayRaw = bitmapToGrayU83250(stillBmp)

        val build = buildFullFrameEdgePackCoreFromBitmap3250(
            stillBmp = stillBmp,
            borderKillPx = borderKillPx,
            topKillY = topKillY,
            maskFull = maskFull,
            regionMaskFull3250 = regionMaskFull3250,
            forceRegion3250 = forceRegion3250,
            debugTag = debugTag,
            statsOut = statsOut
        )

        if (debugSaveToGallery3250 && ctx != null) {
            try {
                debugDumpEdgeTriplet3250(
                    ctx = ctx,
                    w = w,
                    h = h,
                    edgesFullU8 = build.out.edgesU8,
                    maskFullU8 = maskFull,
                    debugTag = debugTag,
                    grayRawU8 = if (debugAlsoSaveGray3250) grayRaw else null,
                    gateMaskU8 = build.gateMaskU8,
                    ribbonMaskU8 = build.ribbonMaskU8,
                    scoreU8 = build.scoreU8,
                    projNormalU8 = build.projNormalU8,
                    tangentialU8 = build.tangentialU8,
                    coherencyU8 = build.coherencyU8,
                    gateNormalU8 = build.gateNormalU8
                )
            } catch (t: Throwable) {
                Log.e(TAG, "EDGE dump save failed", t)
            }
        }

        return build.out
    }

    private fun buildFullFrameEdgePackCoreFromBitmap3250(
        stillBmp: Bitmap,
        borderKillPx: Int,
        topKillY: Int,
        maskFull: ByteArray?,
        regionMaskFull3250: ByteArray?,
        forceRegion3250: Boolean,
        debugTag: String,
        statsOut: ((EdgeStats) -> Unit)?
    ): EdgeCoreBuild3250 {
        val w = stillBmp.width
        val h = stillBmp.height
        val n = w * h

        if (maskFull != null && maskFull.size != n) {
            Log.w(TAG, "EDGE3250[$debugTag] bad maskFull size=${maskFull.size} expected=$n")

            val st = EdgeStats(
                maxScore = 0f,
                p95Score = 0f,
                thr = 0f,
                nnz = 0,
                density = 0f,
                bandPx = 0f,
                samples = 0,
                topKillY = topKillY.coerceIn(0, h),
                maskedN = 0,
                maskedFrac = 0f,
                maskDilateR = 0
            )

            return EdgeCoreBuild3250(
                out = EdgeMapOut3250(w, h, ByteArray(n), ByteArray(n), st),
                gateMaskU8 = null,
                ribbonMaskU8 = null,
                scoreU8 = null,
                projNormalU8 = null,
                tangentialU8 = null,
                coherencyU8 = null,
                gateNormalU8 = null
            )
        }

        val gateMaskU8 = when {
            regionMaskFull3250 == null -> null
            regionMaskFull3250.size != n -> {
                Log.w(
                    TAG,
                    "EDGE3250[$debugTag] bad regionMask size=${regionMaskFull3250.size} expected=$n -> ignored"
                )
                null
            }
            else -> regionMaskFull3250.copyOf()
        }

        val ribbonMaskU8 = buildRibbonMaskFromGate3250(
            gateMaskU8 = gateMaskU8,
            w = w,
            h = h,
            debugTag = debugTag
        )

        val border = borderKillPx.coerceIn(0, min(w, h) / 3)
        val yKill = topKillY.coerceIn(0, h)

        val searchMaskU8 = buildSearchMaskU83250(
            w = w,
            h = h,
            borderKillPx = border,
            topKillY = yKill,
            maskFull = maskFull,
            ribbonMaskU8 = ribbonMaskU8,
            forceRegion3250 = forceRegion3250
        )

        var maskedN = 0
        if (maskFull != null) {
            for (i in 0 until n) {
                if ((maskFull[i].toInt() and 0xFF) != 0) {
                    maskedN++
                }
            }
        }
        val maskedFrac = maskedN.toFloat() / n.toFloat().coerceAtLeast(1f)

        val bandBounds = computeRowBoundsFromMask3250(ribbonMaskU8, w, h)
        val bandYMin = bandBounds.first
        val bandYMax = bandBounds.second

        val rgba = Mat()
        val rgb = Mat()
        val gray = Mat()
        val grayEq = Mat()
        val grayBlur = Mat()

        val hsv = Mat()
        val sat = Mat()
        val satBlur = Mat()

        val gateMat = Mat()
        val gateBlur = Mat()

        val gxGray = Mat()
        val gyGray = Mat()
        val gxSat = Mat()
        val gySat = Mat()
        val gxGate = Mat()
        val gyGate = Mat()

        val absGxGray = Mat()
        val absGyGray = Mat()
        val absGxSat = Mat()
        val absGySat = Mat()
        val absGxGate = Mat()
        val absGyGate = Mat()

        val magGrayU8 = Mat()
        val magSatU8 = Mat()
        val magGateU8 = Mat()

        val cannyGray = Mat()
        val cannySat = Mat()

        val jxx = Mat()
        val jyy = Mat()
        val jxy = Mat()

        val jxxBlur = Mat()
        val jyyBlur = Mat()
        val jxyBlur = Mat()

        try {
            Utils.bitmapToMat(stillBmp, rgba)
            Imgproc.cvtColor(rgba, rgb, Imgproc.COLOR_RGBA2RGB)
            Imgproc.cvtColor(rgba, gray, Imgproc.COLOR_RGBA2GRAY)

            Imgproc.equalizeHist(gray, grayEq)
            Imgproc.GaussianBlur(
                grayEq,
                grayBlur,
                Size(BLUR_GRAY_K.toDouble(), BLUR_GRAY_K.toDouble()),
                0.0
            )

            Imgproc.cvtColor(rgb, hsv, Imgproc.COLOR_RGB2HSV)
            Core.extractChannel(hsv, sat, 1)
            Imgproc.GaussianBlur(
                sat,
                satBlur,
                Size(BLUR_SAT_K.toDouble(), BLUR_SAT_K.toDouble()),
                0.0
            )

            if (gateMaskU8 != null && gateMaskU8.size == n) {
                gateMat.put(0, 0, gateMaskU8)
            } else {
                val full = ByteArray(n) { 0xFF.toByte() }
                gateMat.put(0, 0, full)
            }
            Imgproc.GaussianBlur(
                gateMat,
                gateBlur,
                Size(BLUR_GATE_K.toDouble(), BLUR_GATE_K.toDouble()),
                0.0
            )

            Imgproc.Scharr(grayBlur, gxGray, CvType.CV_32F, 1, 0)
            Imgproc.Scharr(grayBlur, gyGray, CvType.CV_32F, 0, 1)

            Imgproc.Scharr(satBlur, gxSat, CvType.CV_32F, 1, 0)
            Imgproc.Scharr(satBlur, gySat, CvType.CV_32F, 0, 1)

            Imgproc.Scharr(gateBlur, gxGate, CvType.CV_32F, 1, 0)
            Imgproc.Scharr(gateBlur, gyGate, CvType.CV_32F, 0, 1)

            Core.convertScaleAbs(gxGray, absGxGray)
            Core.convertScaleAbs(gyGray, absGyGray)
            Core.addWeighted(absGxGray, 0.5, absGyGray, 0.5, 0.0, magGrayU8)

            Core.convertScaleAbs(gxSat, absGxSat)
            Core.convertScaleAbs(gySat, absGySat)
            Core.addWeighted(absGxSat, 0.5, absGySat, 0.5, 0.0, magSatU8)

            Core.convertScaleAbs(gxGate, absGxGate)
            Core.convertScaleAbs(gyGate, absGyGate)
            Core.addWeighted(absGxGate, 0.5, absGyGate, 0.5, 0.0, magGateU8)

            Imgproc.Canny(grayBlur, cannyGray, CANNY_GRAY_LOW, CANNY_GRAY_HIGH)
            Imgproc.Canny(satBlur, cannySat, CANNY_SAT_LOW, CANNY_SAT_HIGH)

            Core.multiply(gxGray, gxGray, jxx)
            Core.multiply(gyGray, gyGray, jyy)
            Core.multiply(gxGray, gyGray, jxy)

            Imgproc.GaussianBlur(
                jxx,
                jxxBlur,
                Size(BLUR_TENSOR_K.toDouble(), BLUR_TENSOR_K.toDouble()),
                0.0
            )
            Imgproc.GaussianBlur(
                jyy,
                jyyBlur,
                Size(BLUR_TENSOR_K.toDouble(), BLUR_TENSOR_K.toDouble()),
                0.0
            )
            Imgproc.GaussianBlur(
                jxy,
                jxyBlur,
                Size(BLUR_TENSOR_K.toDouble(), BLUR_TENSOR_K.toDouble()),
                0.0
            )

            val scoreU8 = ByteArray(n)
            val dirU8 = ByteArray(n)
            val projNormalU8 = ByteArray(n)
            val tangentialU8 = ByteArray(n)
            val coherencyU8 = ByteArray(n)
            val gateNormalU8 = ByteArray(n)

            val rowMagGray = ByteArray(w)
            val rowMagSat = ByteArray(w)
            val rowMagGate = ByteArray(w)
            val rowCannyGray = ByteArray(w)
            val rowCannySat = ByteArray(w)

            val rowGxGray = FloatArray(w)
            val rowGyGray = FloatArray(w)
            val rowGxSat = FloatArray(w)
            val rowGySat = FloatArray(w)
            val rowGxGate = FloatArray(w)
            val rowGyGate = FloatArray(w)

            val rowJxx = FloatArray(w)
            val rowJyy = FloatArray(w)
            val rowJxy = FloatArray(w)

            var validCount = 0
            var maxScore = 0

            for (y in 0 until h) {
                magGrayU8.get(y, 0, rowMagGray)
                magSatU8.get(y, 0, rowMagSat)
                magGateU8.get(y, 0, rowMagGate)
                cannyGray.get(y, 0, rowCannyGray)
                cannySat.get(y, 0, rowCannySat)

                gxGray.get(y, 0, rowGxGray)
                gyGray.get(y, 0, rowGyGray)
                gxSat.get(y, 0, rowGxSat)
                gySat.get(y, 0, rowGySat)
                gxGate.get(y, 0, rowGxGate)
                gyGate.get(y, 0, rowGyGate)

                jxxBlur.get(y, 0, rowJxx)
                jyyBlur.get(y, 0, rowJyy)
                jxyBlur.get(y, 0, rowJxy)

                val off = y * w
                val bottomFrac = computeBottomFrac3250(y, bandYMin, bandYMax)

                for (x in 0 until w) {
                    val i = off + x

                    if ((searchMaskU8[i].toInt() and 0xFF) == 0) {
                        continue
                    }

                    val gMag = rowMagGray[x].toInt() and 0xFF
                    val sMag = rowMagSat[x].toInt() and 0xFF
                    val gateMag = rowMagGate[x].toInt() and 0xFF

                    val gxg = rowGxGray[x]
                    val gyg = rowGyGray[x]
                    val gxs = rowGxSat[x]
                    val gys = rowGySat[x]
                    val gxMask = rowGxGate[x]
                    val gyMask = rowGyGate[x]

                    val gateNorm = sqrt(gxMask * gxMask + gyMask * gyMask)
                    val hasGateNormal = gateNorm > 1e-3f

                    val nx = if (hasGateNormal) gxMask / gateNorm else 1f
                    val ny = if (hasGateNormal) gyMask / gateNorm else 0f
                    val tx = -ny
                    val ty = nx

                    val projGray = abs(gxg * nx + gyg * ny)
                    val tanGray = abs(gxg * tx + gyg * ty)
                    val projSat = abs(gxs * nx + gys * ny)
                    val tanSat = abs(gxs * tx + gys * ty)

                    val alignGray = max(
                        ALIGN_FLOOR,
                        projGray / (projGray + tanGray + 1e-3f)
                    )
                    val alignSat = max(
                        ALIGN_FLOOR,
                        projSat / (projSat + tanSat + 1e-3f)
                    )

                    val projMix = 0.5f * (
                            (gMag / 255f) * alignGray +
                                    (sMag / 255f) * alignSat
                            )
                    val tangMix = 0.5f * (
                            (gMag / 255f) * (1f - alignGray) +
                                    (sMag / 255f) * (1f - alignSat)
                            )

                    val j11 = rowJxx[x]
                    val j22 = rowJyy[x]
                    val j12 = rowJxy[x]
                    val cohDen = j11 + j22 + 1e-3f
                    val cohNum = sqrt(
                        (j11 - j22) * (j11 - j22) + 4f * j12 * j12
                    )
                    val coherency = (cohNum / cohDen).coerceIn(0f, 1f)

                    val gateConf = max(
                        GATE_CONF_FLOOR,
                        (gateMag / 96f).coerceIn(0f, 1f)
                    )
                    val cohConf = max(COH_FLOOR, coherency)

                    val cGray = if ((rowCannyGray[x].toInt() and 0xFF) != 0) 1f else 0f
                    val cSat = if ((rowCannySat[x].toInt() and 0xFF) != 0) 1f else 0f

                    var scoreF = 255f * (
                            SCORE_W_GRAY_NORMAL * (gMag / 255f) * alignGray +
                                    SCORE_W_SAT_NORMAL * (sMag / 255f) * alignSat +
                                    SCORE_W_GRAY_CANNY * cGray * alignGray +
                                    SCORE_W_SAT_CANNY * cSat * alignSat +
                                    SCORE_W_COHERENCY * cohConf +
                                    SCORE_W_GATE_NORMAL * gateConf +
                                    SCORE_W_BOTTOM_BONUS * bottomFrac -
                                    PENALTY_W_GRAY_TANG * (gMag / 255f) * (1f - alignGray) -
                                    PENALTY_W_SAT_TANG * (sMag / 255f) * (1f - alignSat)
                            )

                    scoreF *= (0.55f + 0.45f * gateConf)
                    scoreF *= (0.60f + 0.40f * cohConf)

                    val score = scoreF.roundToInt().coerceIn(0, 255)
                    if (score <= 0) {
                        continue
                    }

                    scoreU8[i] = score.toByte()
                    dirU8[i] = quantizeDir3250(
                        gx = gxg.toInt(),
                        gy = gyg.toInt(),
                        ax = abs(gxg).toInt(),
                        ay = abs(gyg).toInt()
                    ).toByte()

                    projNormalU8[i] = (projMix * 255f).roundToInt().coerceIn(0, 255).toByte()
                    tangentialU8[i] = (tangMix * 255f).roundToInt().coerceIn(0, 255).toByte()
                    coherencyU8[i] = (coherency * 255f).roundToInt().coerceIn(0, 255).toByte()
                    gateNormalU8[i] = (gateConf * 255f).roundToInt().coerceIn(0, 255).toByte()

                    validCount++
                    if (score > maxScore) {
                        maxScore = score
                    }
                }
            }

            if (validCount < 64 || maxScore <= 0) {
                Log.w(TAG, "EDGE3250[$debugTag] no score signal valid=$validCount max=$maxScore")

                val st = EdgeStats(
                    maxScore = maxScore.toFloat(),
                    p95Score = 0f,
                    thr = 0f,
                    nnz = 0,
                    density = 0f,
                    bandPx = countNonZeroU83250(ribbonMaskU8).toFloat(),
                    samples = validCount,
                    topKillY = yKill,
                    maskedN = maskedN,
                    maskedFrac = maskedFrac,
                    maskDilateR = MASK_DILATE_RADIUS_PX
                )

                statsOut?.invoke(st)

                return EdgeCoreBuild3250(
                    out = EdgeMapOut3250(w, h, ByteArray(n), dirU8, st),
                    gateMaskU8 = gateMaskU8,
                    ribbonMaskU8 = ribbonMaskU8,
                    scoreU8 = scoreU8,
                    projNormalU8 = projNormalU8,
                    tangentialU8 = tangentialU8,
                    coherencyU8 = coherencyU8,
                    gateNormalU8 = gateNormalU8
                )
            }

            val hist = IntArray(256)
            for (i in 0 until n) {
                if ((searchMaskU8[i].toInt() and 0xFF) == 0) {
                    continue
                }
                val s = scoreU8[i].toInt() and 0xFF
                if (s > 0) {
                    hist[s]++
                }
            }

            val p95 = percentileFromU8Hist3250(hist, 0.95f)
            val thrHigh = computeThrHighU8_3250(maxScore, p95)
            val thrLow = max(14, (thrHigh * 0.68f).toInt())

            val classMap = ByteArray(n)
            var candCount = 0
            var strongCount = 0

            for (y in 1 until (h - 1)) {
                val off = y * w
                val bottomFrac = computeBottomFrac3250(y, bandYMin, bandYMax)
                val relax = if (bottomFrac <= BOTTOM_SEED_FRAC_START) {
                    0
                } else {
                    val t = ((bottomFrac - BOTTOM_SEED_FRAC_START) / (1f - BOTTOM_SEED_FRAC_START))
                        .coerceIn(0f, 1f)
                    (BOTTOM_THR_RELAX_MAX * t * t).roundToInt()
                }

                for (x in 1 until (w - 1)) {
                    val i = off + x

                    if ((searchMaskU8[i].toInt() and 0xFF) == 0) {
                        continue
                    }

                    val s = scoreU8[i].toInt() and 0xFF
                    if (s < thrLow) {
                        continue
                    }

                    val d = dirU8[i].toInt() and 0xFF
                    val neighbors = nmsNeighbors3250(i, x, y, w, d)
                    val i1 = neighbors.first
                    val i2 = neighbors.second

                    val s1 = if ((searchMaskU8[i1].toInt() and 0xFF) != 0) {
                        scoreU8[i1].toInt() and 0xFF
                    } else {
                        0
                    }

                    val s2 = if ((searchMaskU8[i2].toInt() and 0xFF) != 0) {
                        scoreU8[i2].toInt() and 0xFF
                    } else {
                        0
                    }

                    if (s < s1 || s < s2) {
                        continue
                    }

                    val thrStrongLocal = max(thrLow, thrHigh - relax)
                    if (s >= thrStrongLocal) {
                        classMap[i] = 2
                        strongCount++
                    } else {
                        classMap[i] = 1
                    }
                    candCount++
                }
            }

            val out = ByteArray(n)

            if (strongCount == 0 || candCount == 0) {
                Log.w(
                    TAG,
                    "EDGE3250[$debugTag] no strong/candidates strong=$strongCount cand=$candCount thrH=$thrHigh thrL=$thrLow"
                )

                val st = EdgeStats(
                    maxScore = maxScore.toFloat(),
                    p95Score = p95.toFloat(),
                    thr = thrHigh.toFloat(),
                    nnz = 0,
                    density = 0f,
                    bandPx = countNonZeroU83250(ribbonMaskU8).toFloat(),
                    samples = validCount,
                    topKillY = yKill,
                    maskedN = maskedN,
                    maskedFrac = maskedFrac,
                    maskDilateR = MASK_DILATE_RADIUS_PX
                )

                statsOut?.invoke(st)

                return EdgeCoreBuild3250(
                    out = EdgeMapOut3250(w, h, out, dirU8, st),
                    gateMaskU8 = gateMaskU8,
                    ribbonMaskU8 = ribbonMaskU8,
                    scoreU8 = scoreU8,
                    projNormalU8 = projNormalU8,
                    tangentialU8 = tangentialU8,
                    coherencyU8 = coherencyU8,
                    gateNormalU8 = gateNormalU8
                )
            }

            val q = IntArray(max(candCount, 1))
            var qh = 0
            var qt = 0

            for (i in 0 until n) {
                if (classMap[i].toInt() == 2) {
                    out[i] = 0xFF.toByte()
                    q[qt++] = i
                }
            }

            while (qh < qt) {
                val i = q[qh++]
                val x = i % w
                val y = i / w

                for (dy in -1..1) {
                    for (dx in -1..1) {
                        if (dx == 0 && dy == 0) {
                            continue
                        }

                        val xx = x + dx
                        val yy = y + dy

                        if (xx <= 0 || xx >= w - 1) {
                            continue
                        }
                        if (yy <= 0 || yy >= h - 1) {
                            continue
                        }

                        val j = yy * w + xx
                        if ((searchMaskU8[j].toInt() and 0xFF) == 0) {
                            continue
                        }

                        if (classMap[j].toInt() == 1) {
                            classMap[j] = 2
                            out[j] = 0xFF.toByte()
                            q[qt++] = j
                        }
                    }
                }
            }

            val outBridged = bridgeHorizontalGapsU83250(
                edges = out,
                dir = dirU8,
                w = w,
                h = h,
                maxGapPx = BRIDGE_MAX_GAP_PX
            )

            val outFinal = morphPostEdgeU83250(
                srcU8 = outBridged,
                keepMaskU8 = searchMaskU8,
                w = w,
                h = h,
                closeRadiusPx = FINAL_CLOSE_RADIUS_PX,
                dilateRadiusPx = FINAL_DILATE_RADIUS_PX
            )

            val dirFinal = dirU8.copyOf()
            for (i in 0 until n) {
                val eNew = outFinal[i].toInt() and 0xFF
                if (eNew == 0) {
                    continue
                }
                val eOld = out[i].toInt() and 0xFF
                if (eOld == 0) {
                    dirFinal[i] = 0xFF.toByte()
                }
            }

            var nnzFinal = 0
            for (b in outFinal) {
                if ((b.toInt() and 0xFF) != 0) {
                    nnzFinal++
                }
            }

            val dens = nnzFinal.toFloat() / n.toFloat().coerceAtLeast(1f)
            val bandPx = countNonZeroU83250(ribbonMaskU8).toFloat()

            Log.d(
                TAG,
                "EDGE3250[$debugTag] max=$maxScore p95=$p95 thrH=$thrHigh thrL=$thrLow " +
                        "strong=$strongCount cand=$candCount nnz=$nnzFinal/$n dens=${fmt4(dens)} " +
                        "bandPx=${bandPx.toInt()} forceRegion=$forceRegion3250 " +
                        "maskedN=$maskedN frac=${fmt4(maskedFrac)} yBand=[$bandYMin,$bandYMax]"
            )

            val st = EdgeStats(
                maxScore = maxScore.toFloat(),
                p95Score = p95.toFloat(),
                thr = thrHigh.toFloat(),
                nnz = nnzFinal,
                density = dens,
                bandPx = bandPx,
                samples = validCount,
                topKillY = yKill,
                maskedN = maskedN,
                maskedFrac = maskedFrac,
                maskDilateR = MASK_DILATE_RADIUS_PX
            )

            statsOut?.invoke(st)

            return EdgeCoreBuild3250(
                out = EdgeMapOut3250(
                    w = w,
                    h = h,
                    edgesU8 = outFinal,
                    dirU8 = dirFinal,
                    stats = st
                ),
                gateMaskU8 = gateMaskU8,
                ribbonMaskU8 = ribbonMaskU8,
                scoreU8 = scoreU8,
                projNormalU8 = projNormalU8,
                tangentialU8 = tangentialU8,
                coherencyU8 = coherencyU8,
                gateNormalU8 = gateNormalU8
            )
        } finally {
            releaseQuiet3250(
                rgba, rgb, gray, grayEq, grayBlur,
                hsv, sat, satBlur,
                gateMat, gateBlur,
                gxGray, gyGray, gxSat, gySat, gxGate, gyGate,
                absGxGray, absGyGray, absGxSat, absGySat, absGxGate, absGyGate,
                magGrayU8, magSatU8, magGateU8,
                cannyGray, cannySat,
                jxx, jyy, jxy, jxxBlur, jyyBlur, jxyBlur
            )
        }
    }

    private fun buildRibbonMaskFromGate3250(
        gateMaskU8: ByteArray?,
        w: Int,
        h: Int,
        debugTag: String
    ): ByteArray? {
        if (gateMaskU8 == null || gateMaskU8.size != w * h) {
            return null
        }

        val nz = countNonZeroU83250(gateMaskU8)
        if (nz <= 0) {
            return null
        }

        val cov = nz.toFloat() / (w * h).toFloat().coerceAtLeast(1f)

        val gateMat = byteMaskToMatU83250(gateMaskU8, w, h)
        val workA = Mat()
        val workB = Mat()
        val ribbon = Mat()

        try {
            if (cov <= RIBBON_THIN_COVERAGE_FRAC) {
                val kDil = ellipseKernel3250(RIBBON_OUTER_DILATE_PX)
                try {
                    Imgproc.dilate(gateMat, workA, kDil)
                } finally {
                    kDil.release()
                }

                val kClose = ellipseKernel3250(RIBBON_CLOSE_RADIUS_PX)
                try {
                    Imgproc.morphologyEx(workA, workB, Imgproc.MORPH_CLOSE, kClose)
                } finally {
                    kClose.release()
                }

                return matU8ToByteArray3250(workB, w, h)
            }

            val kDil = ellipseKernel3250(RIBBON_OUTER_DILATE_PX)
            val kEro = ellipseKernel3250(RIBBON_INNER_ERODE_PX)

            try {
                Imgproc.dilate(gateMat, workA, kDil)
                Imgproc.erode(gateMat, workB, kEro)
            } finally {
                kDil.release()
                kEro.release()
            }

            val notInner = Mat()
            try {
                Core.bitwise_not(workB, notInner)
                Core.bitwise_and(workA, notInner, ribbon)

                val kClose = ellipseKernel3250(RIBBON_CLOSE_RADIUS_PX)
                try {
                    Imgproc.morphologyEx(ribbon, ribbon, Imgproc.MORPH_CLOSE, kClose)
                } finally {
                    kClose.release()
                }

                val out = matU8ToByteArray3250(ribbon, w, h)
                val outNz = countNonZeroU83250(out)

                Log.d(
                    TAG,
                    "EDGE3250[$debugTag] ribbon covGate=${fmt4(cov)} nzGate=$nz nzRibbon=$outNz"
                )

                return out
            } finally {
                releaseQuiet3250(notInner)
            }
        } finally {
            releaseQuiet3250(gateMat, workA, workB, ribbon)
        }
    }

    private fun buildSearchMaskU83250(
        w: Int,
        h: Int,
        borderKillPx: Int,
        topKillY: Int,
        maskFull: ByteArray?,
        ribbonMaskU8: ByteArray?,
        forceRegion3250: Boolean
    ): ByteArray {
        val n = w * h
        val out = ByteArray(n)

        if (ribbonMaskU8 != null && ribbonMaskU8.size == n) {
            System.arraycopy(ribbonMaskU8, 0, out, 0, n)
        } else {
            for (i in 0 until n) {
                out[i] = 0xFF.toByte()
            }
        }

        val b = borderKillPx.coerceIn(0, min(w, h) / 3)
        val yKill = topKillY.coerceIn(0, h)

        for (y in 0 until h) {
            val off = y * w
            val killRow = y < yKill || y < b || y >= h - b

            for (x in 0 until w) {
                val i = off + x

                if (killRow || x < b || x >= w - b) {
                    out[i] = 0
                    continue
                }

                if (maskFull != null && (maskFull[i].toInt() and 0xFF) != 0) {
                    out[i] = 0
                    continue
                }

                if (forceRegion3250 && ribbonMaskU8 != null && (ribbonMaskU8[i].toInt() and 0xFF) == 0) {
                    out[i] = 0
                }
            }
        }

        return out
    }

    private fun morphPostEdgeU83250(
        srcU8: ByteArray,
        keepMaskU8: ByteArray,
        w: Int,
        h: Int,
        closeRadiusPx: Int,
        dilateRadiusPx: Int
    ): ByteArray {
        val src = byteMaskToMatU83250(srcU8, w, h)
        val keep = byteMaskToMatU83250(keepMaskU8, w, h)
        val tmp = Mat()
        val dst = Mat()

        try {
            src.copyTo(tmp)

            if (closeRadiusPx > 0) {
                val kClose = ellipseKernel3250(closeRadiusPx)
                try {
                    Imgproc.morphologyEx(tmp, tmp, Imgproc.MORPH_CLOSE, kClose)
                } finally {
                    kClose.release()
                }
            }

            if (dilateRadiusPx > 0) {
                val kDil = ellipseKernel3250(dilateRadiusPx)
                try {
                    Imgproc.dilate(tmp, tmp, kDil)
                } finally {
                    kDil.release()
                }
            }

            Core.bitwise_and(tmp, keep, dst)
            return matU8ToByteArray3250(dst, w, h)
        } finally {
            releaseQuiet3250(src, keep, tmp, dst)
        }
    }

    private fun saveU8ToGalleryAsPng3250(
        ctx: Context,
        u8img: ByteArray,
        w: Int,
        h: Int,
        displayName: String,
        invert: Boolean
    ): Uri? {
        if (u8img.size != w * h || w <= 0 || h <= 0) {
            return null
        }

        val scale = computeDebugScale3250(w, h)
        val outW = (w * scale).toInt().coerceAtLeast(1)
        val outH = (h * scale).toInt().coerceAtLeast(1)

        val bmp = createBitmap(outW, outH)
        val row = IntArray(outW)

        for (yy in 0 until outH) {
            val ySrc = ((yy.toFloat() / outH) * h).toInt().coerceIn(0, h - 1)
            val offSrc = ySrc * w

            for (xx in 0 until outW) {
                val xSrc = ((xx.toFloat() / outW) * w).toInt().coerceIn(0, w - 1)
                var v = u8img[offSrc + xSrc].toInt() and 0xFF
                if (invert) {
                    v = 255 - v
                }
                row[xx] = (0xFF shl 24) or (v shl 16) or (v shl 8) or v
            }

            bmp.setPixels(row, 0, outW, 0, yy, outW, 1)
        }

        val values = ContentValues().apply {
            put(MediaStore.Images.Media.DISPLAY_NAME, displayName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/png")
            put(MediaStore.Images.Media.RELATIVE_PATH, "Pictures/debug3250")
            put(MediaStore.Images.Media.IS_PENDING, 1)
        }

        val resolver = ctx.contentResolver
        val uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values) ?: run {
            try {
                bmp.recycle()
            } catch (_: Throwable) {
            }
            return null
        }

        try {
            resolver.openOutputStream(uri)?.use { os: OutputStream ->
                bmp.compress(Bitmap.CompressFormat.PNG, 100, os)
            }

            values.clear()
            values.put(MediaStore.Images.Media.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return uri
        } catch (t: Throwable) {
            Log.e(TAG, "saveU8ToGalleryAsPng3250 failed uri=$uri", t)

            try {
                resolver.delete(uri, null, null)
            } catch (_: Throwable) {
            }

            return null
        } finally {
            try {
                bmp.recycle()
            } catch (_: Throwable) {
            }
        }
    }

    private fun debugDumpEdgeTriplet3250(
        ctx: Context,
        w: Int,
        h: Int,
        edgesFullU8: ByteArray,
        maskFullU8: ByteArray?,
        debugTag: String,
        grayRawU8: ByteArray? = null,
        gateMaskU8: ByteArray? = null,
        ribbonMaskU8: ByteArray? = null,
        scoreU8: ByteArray? = null,
        projNormalU8: ByteArray? = null,
        tangentialU8: ByteArray? = null,
        coherencyU8: ByteArray? = null,
        gateNormalU8: ByteArray? = null
    ) {
        if (edgesFullU8.size != w * h) {
            return
        }

        val ts = System.currentTimeMillis()

        if (grayRawU8 != null && grayRawU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = grayRawU8,
                w = w,
                h = h,
                displayName = "EDGE_GRAY_RAW_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (gateMaskU8 != null && gateMaskU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = gateMaskU8,
                w = w,
                h = h,
                displayName = "EDGE_GATE_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (ribbonMaskU8 != null && ribbonMaskU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = ribbonMaskU8,
                w = w,
                h = h,
                displayName = "EDGE_RIBBON_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (scoreU8 != null && scoreU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = scoreU8,
                w = w,
                h = h,
                displayName = "EDGE_SCORE_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (projNormalU8 != null && projNormalU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = projNormalU8,
                w = w,
                h = h,
                displayName = "EDGE_PROJ_NORMAL_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (tangentialU8 != null && tangentialU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = tangentialU8,
                w = w,
                h = h,
                displayName = "EDGE_TANGENTIAL_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (coherencyU8 != null && coherencyU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = coherencyU8,
                w = w,
                h = h,
                displayName = "EDGE_COHERENCY_${debugTag}_${ts}.png",
                invert = false
            )
        }

        if (gateNormalU8 != null && gateNormalU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = gateNormalU8,
                w = w,
                h = h,
                displayName = "EDGE_GATE_NORMAL_${debugTag}_${ts}.png",
                invert = false
            )
        }

        saveU8ToGalleryAsPng3250(
            ctx = ctx,
            u8img = edgesFullU8,
            w = w,
            h = h,
            displayName = "EDGE_FULL_${debugTag}_${ts}.png",
            invert = false
        )

        if (maskFullU8 != null && maskFullU8.size == w * h) {
            saveU8ToGalleryAsPng3250(
                ctx = ctx,
                u8img = maskFullU8,
                w = w,
                h = h,
                displayName = "EDGE_MASK_${debugTag}_${ts}.png",
                invert = false
            )
        }
    }

    private fun bitmapToGrayU83250(stillBmp: Bitmap): ByteArray {
        val w = stillBmp.width
        val h = stillBmp.height
        val gray = ByteArray(w * h)
        val row = IntArray(w)

        for (y in 0 until h) {
            stillBmp.getPixels(row, 0, w, 0, y, w, 1)
            val off = y * w

            for (x in 0 until w) {
                val c = row[x]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val yv = (77 * r + 150 * g + 29 * b + 128) shr 8
                gray[off + x] = yv.toByte()
            }
        }

        return gray
    }

    private fun byteMaskToMatU83250(src: ByteArray, w: Int, h: Int): Mat {
        val mat = Mat(h, w, CvType.CV_8UC1)
        mat.put(0, 0, src)
        return mat
    }

    private fun matU8ToByteArray3250(src: Mat, w: Int, h: Int): ByteArray {
        val out = ByteArray(w * h)
        src.get(0, 0, out)
        return out
    }

    private fun ellipseKernel3250(radiusPx: Int): Mat {
        val r = radiusPx.coerceAtLeast(1)
        val k = 2 * r + 1
        return Imgproc.getStructuringElement(
            Imgproc.MORPH_ELLIPSE,
            Size(k.toDouble(), k.toDouble())
        )
    }

    private fun countNonZeroU83250(src: ByteArray?): Int {
        if (src == null) {
            return 0
        }

        var n = 0
        for (b in src) {
            if ((b.toInt() and 0xFF) != 0) {
                n++
            }
        }
        return n
    }

    private fun computeRowBoundsFromMask3250(
        src: ByteArray?,
        w: Int,
        h: Int
    ): Pair<Int, Int> {
        if (src == null || src.size != w * h) {
            return 0 to (h - 1).coerceAtLeast(0)
        }

        var yMin = h
        var yMax = -1

        for (y in 0 until h) {
            val off = y * w
            var hit = false
            for (x in 0 until w) {
                if ((src[off + x].toInt() and 0xFF) != 0) {
                    hit = true
                    break
                }
            }
            if (hit) {
                if (y < yMin) {
                    yMin = y
                }
                if (y > yMax) {
                    yMax = y
                }
            }
        }

        if (yMax < yMin) {
            return 0 to (h - 1).coerceAtLeast(0)
        }
        return yMin to yMax
    }

    private fun computeBottomFrac3250(
        y: Int,
        yMin: Int,
        yMax: Int
    ): Float {
        val den = (yMax - yMin).coerceAtLeast(1)
        return ((y - yMin).toFloat() / den.toFloat()).coerceIn(0f, 1f)
    }

    private fun fmt4(v: Float): String {
        return String.format(Locale.US, "%.4f", v)
    }

    private fun percentileFromU8Hist3250(hist: IntArray, q: Float): Int {
        var total = 0
        for (c in hist) {
            total += c
        }

        if (total <= 0) {
            return 0
        }

        val target = (total * q).toInt().coerceIn(0, total - 1)
        var acc = 0

        for (i in hist.indices) {
            acc += hist[i]
            if (acc > target) {
                return i
            }
        }

        return hist.lastIndex
    }

    private fun computeThrHighU8_3250(maxScore: Int, p95: Int): Int {
        val thrBase = max(EDGE_THR_MIN, (EDGE_THR_FRAC * maxScore.toFloat()).toInt())
        val thrCap = max(EDGE_THR_MIN, (P95_CAP_K * p95.toFloat()).toInt())
        return min(thrBase, thrCap).coerceAtLeast(EDGE_THR_MIN)
    }

    private fun quantizeDir3250(gx: Int, gy: Int, ax: Int, ay: Int): Int {
        if (ax == 0 && ay == 0) {
            return 0
        }

        if (ay * TAN22_DEN <= ax * TAN22_NUM) {
            return 0
        }

        if (ay * TAN67_DEN >= ax * TAN67_NUM) {
            return 2
        }

        return if ((gx xor gy) >= 0) {
            1
        } else {
            3
        }
    }

    private fun nmsNeighbors3250(
        i: Int,
        x: Int,
        y: Int,
        w: Int,
        dir: Int
    ): Pair<Int, Int> {
        return when (dir) {
            0 -> (i - 1) to (i + 1)
            2 -> (i - w) to (i + w)
            1 -> (i - w - 1) to (i + w + 1)
            else -> (i - w + 1) to (i + w - 1)
        }
    }

    private fun isHorizontalEdgeSeed3250(
        edges: ByteArray,
        dir: ByteArray,
        idx: Int
    ): Boolean {
        if ((edges[idx].toInt() and 0xFF) == 0) {
            return false
        }
        return (dir[idx].toInt() and 0xFF) == 2 || (dir[idx].toInt() and 0xFF) == 0xFF
    }

    private fun bridgeHorizontalGapsU83250(
        edges: ByteArray,
        dir: ByteArray,
        w: Int,
        h: Int,
        maxGapPx: Int = 5
    ): ByteArray {
        if (edges.size != w * h || dir.size != w * h || w <= 2 || h <= 2) {
            return edges
        }

        if (maxGapPx <= 0) {
            return edges
        }

        val out = edges.copyOf()

        for (y in 1 until h - 1) {
            var x = 1

            while (x < w - 1) {
                val i0 = y * w + x

                if (!isHorizontalEdgeSeed3250(edges, dir, i0)) {
                    x++
                    continue
                }

                var bridged = false
                val xMax = min(w - 2, x + maxGapPx + 1)
                var xr = x + 2

                while (xr <= xMax) {
                    val ir = y * w + xr

                    if (isHorizontalEdgeSeed3250(edges, dir, ir)) {
                        for (xx in (x + 1) until xr) {
                            out[y * w + xx] = 0xFF.toByte()
                        }
                        x = xr
                        bridged = true
                        break
                    }

                    xr++
                }

                if (!bridged) {
                    x++
                }
            }
        }

        return out
    }

    private fun computeDebugScale3250(w: Int, h: Int): Float {
        val maxDim = max(w, h).toFloat()
        return if (maxDim <= DEBUG_MAX_DIM_PX) {
            1f
        } else {
            (DEBUG_MAX_DIM_PX / maxDim).coerceIn(0.10f, 1f)
        }
    }

    private fun releaseQuiet3250(vararg mats: Mat) {
        for (m in mats) {
            try {
                m.release()
            } catch (_: Throwable) {
            }
        }
    }
}