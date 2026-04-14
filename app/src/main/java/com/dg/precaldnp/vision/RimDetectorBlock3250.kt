@file:Suppress("SameParameterValue", "UNNECESSARY_NOT_NULL_ASSERTION")

package com.dg.precaldnp.vision

import android.graphics.PointF
import android.graphics.RectF
import android.util.Log
import java.util.Locale
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt


/**
 * RimDetectorBlock3250 (EDGE-IN) — CENTER→OUT (INNER FIRST)
 *
 * ✅ Consume edgeMap binario ROI-local (0/255)
 * ✅ dirU8 opcional (0..3 o 255=don’t care)
 * ✅ maskU8 = máscara de ojo/ceja: mask!=0 => VETO (NO RIM)
 *
 * Regla oficial 3250:
 * - Hay UNA sola referencia de búsqueda: la línea pupilar / bridgeRow => probeYLocal.
 * - INNER y BOTTOM se buscan siempre referidos a esa misma línea oficial.
 * - La máscara NO mueve la referencia geométrica.
 * - Si un pixel cae dentro de máscara => NO VALE; se ignora/salta y se sigue buscando.
 * - Bottom arc debe ser continuo (no saltos grandes).
 * - Bottom barre CENTER→OUT dentro del dominio leftXrightX.
 */
object RimDetectorBlock3250 {

    private const val TAG = "RimDetectorBlock3250"
    private const val DBG = true

    private const val MIN_ROI_W = 120
    private const val MIN_ROI_H = 90

    private const val BAND_HALF_PX = 14
    private const val TOP_SEARCH_PAD = 10

    private const val OK_CONF_MIN = 0.55f

    private const val SCALE_MIN = 0.60f
    private const val SCALE_MAX = 1.35f
    private const val SCALE_STEP = 0.05f

    // Si te sigue agarrando OUTER, bajá W_RATIO_MAX a ~1.12
    private const val W_RATIO_MIN = 0.90f
    private const val W_RATIO_MAX = 1.25f

    // Para evitar “saltos” grandes en el arco bottom
    private const val CONT_JUMP_PX = 7

    // Para evitar “hits” espurios cerca del centro (reflejos / ruido interior)
    private const val LR_MIN_DIST_PX = 10

    private fun d(msg: String) {
        if (DBG) Log.d(TAG, msg)
    }

    // API
    // ==========================================================
    fun detectRim(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray? = null,
        w: Int,
        h: Int,
        hScoreU8: ByteArray? = null,
        vScoreU8: ByteArray? = null,
        roiGlobal: RectF,
        midlineXpx: Float,
        browBottomYpx: Float?,
        filHboxMm: Double,
        filVboxMm: Double?,
        filOverInnerMmPerSide3250: Double,
        profile3250: RimProfile3250,
        bridgeRowYpxGlobal: Float?,
        pupilGlobal: PointF?,
        filPtsMm3250: List<PointF>? = null,
        debugTag: String,
        pxPerMmGuessFace: Float? = null
    ): RimDetectPack3250? {

        var stage = "ENTER"
        fun fail(code: String, msg: String): RimDetectPack3250? {
            Log.w(TAG, "RIM[$debugTag] FAIL@$stage code=$code :: $msg")
            return null
        }
        stage = "SANITY"

        val effectiveOverF = effectiveFilOverPerSide3250(
            profile3250 = profile3250,
            filOverInnerMmPerSide = filOverInnerMmPerSide3250
        ).toFloat()

        val hboxInnerMmF = (filHboxMm.toFloat() - 2f * effectiveOverF)
            .takeIf { it.isFinite() && it > 1e-6f }
            ?: return fail(
                "HBOX",
                "HBOX inner invalid fil=$filHboxMm over=$effectiveOverF profile=$profile3250"
            )

        val vboxInnerMmF = ((filVboxMm ?: 0.0).toFloat() - 2f * effectiveOverF)
            .takeIf { it.isFinite() && it > 1e-6f }
            ?: 0f

        fun logSanityHeader3250() {
            d(
                "RIM[$debugTag] profile=$profile3250 " +
                        "filH=${f13250(filHboxMm.toFloat())} " +
                        "filV=${f13250((filVboxMm ?: Double.NaN).toFloat())} " +
                        "overIn=${f23250(filOverInnerMmPerSide3250.toFloat())} " +
                        "overEff=${f23250(effectiveOverF)} " +
                        "hboxInner=${f23250(hboxInnerMmF)} " +
                        "vboxInner=${f23250(vboxInnerMmF)}"
            )
        }

        if (w <= 0 || h <= 0) {
            return fail("WH", "w/h invalid $w x $h")
        }

        if (w < MIN_ROI_W || h < MIN_ROI_H) {
            return fail("ROI_SMALL", "ROI too small ${w}x${h}")
        }

        if (edgesU8.size != w * h) {
            return fail("EDGE_SIZE", "edgesU8.size=${edgesU8.size} != w*h=${w * h}")
        }

        if (dirU8 != null && dirU8.size != w * h) {
            return fail("DIR_SIZE", "dirU8.size=${dirU8.size} != w*h=${w * h}")
        }

        if (maskU8 != null && maskU8.size != w * h) {
            return fail("MASK_SIZE", "maskU8.size=${maskU8.size} != w*h=${w * h}")
        }

        if (!midlineXpx.isFinite()) {
            return fail("MIDLINE", "midlineXpx invalid=$midlineXpx")
        }

        logSanityHeader3250()


        stage = "EDGE_DENSITY"

        var nnz = 0
        for (i in edgesU8.indices) {
            if ((edgesU8[i].toInt() and 0xFF) != 0) nnz++
        }

        val dens = nnz.toFloat() / (w * h).toFloat().coerceAtLeast(1f)
        d("EDGE[$debugTag] nnz=$nnz/${w * h} dens=${"%.4f".format(Locale.US, dens)}")

        if (nnz < 50) {
            return fail("EDGE_SPARSE", "edgeMap too sparse (nnz<50)")
        }

        val roiLeftG = roiGlobal.left
        val roiTopG = roiGlobal.top

        stage = "PROBE"

        val probeYLocalRaw = run {
            val y0 = bridgeRowYpxGlobal ?: pupilGlobal?.y ?: roiGlobal.centerY()
            (y0 - roiTopG).roundToInt()
        }.coerceIn(0, h - 1)

        val browLocal = browBottomYpx
            ?.let { (it - roiTopG).roundToInt() }
            ?.coerceIn(0, h - 1)

        val topMinAllowedY0 =
            if (browLocal == null) {
                0
            } else {
                val pad = clampInt3250((h * 0.02f).roundToInt(), 6, 18)
                (browLocal + pad).coerceIn(0, h - 1)
            }

        val topMinAllowedY =
            min(
                topMinAllowedY0,
                (probeYLocalRaw - 14).coerceIn(0, h - 1)
            ).coerceIn(0, h - 1)

        val probeYLocal = run {
            val lo = (topMinAllowedY + 2).coerceIn(0, h - 1)
            val hi = (h - 3).coerceIn(0, h - 1)
            clampInRange3250(probeYLocalRaw, lo, hi)
        }

        d(
            "RIM[$debugTag] probeRaw=$probeYLocalRaw probe=$probeYLocal " +
                    "topMin0=$topMinAllowedY0 topMin=$topMinAllowedY browLocal=${browLocal ?: -1}"
        )
        // ---- DBG: máscara (bbox + frac) ----
        if (DBG) {
            if (maskU8 != null && maskU8.size == w * h) {
                var minX = w
                var minY = h
                var maxX = -1
                var maxY = -1
                var n = 0
                for (i in 0 until w * h) {
                    if ((maskU8[i].toInt() and 0xFF) != 0) {
                        n++
                        val x = i % w
                        val y = i / w
                        if (x < minX) minX = x
                        if (y < minY) minY = y
                        if (x > maxX) maxX = x
                        if (y > maxY) maxY = y
                    }
                }
                val frac = n.toFloat() / (w.toFloat() * h.toFloat()).coerceAtLeast(1f)
                if (n > 0) {
                    Log.d(
                        TAG,
                        "MASK[$debugTag] n=$n frac=${f33250(frac)} bbox=($minX,$minY)-($maxX,$maxY)"
                    )
                } else {
                    Log.d(TAG, "MASK[$debugTag] n=0 frac=${f33250(frac)} bbox=none")
                }
            } else {
                Log.d(
                    TAG,
                    "MASK[$debugTag] noneOrBadSize size=${maskU8?.size ?: -1} expected=${w * h}"
                )
            }
        }

        // ---- MIDLINE + SIDE ----
        stage = "MIDLINE"
        val midLocalX = (midlineXpx - roiLeftG)
        val sideSign = when {
            pupilGlobal != null -> if (pupilGlobal.x < midlineXpx) -1f else +1f   // OD=-1, OI=+1
            else -> if (roiGlobal.centerX() < midlineXpx) -1f else +1f
        }
        val nasalAtLeft = sideSign > 0f // OI => nasal a la izquierda (hacia midline)

// ---- pxGuessBase ----
        stage = "PX_GUESS"
        val pxGuessBase = pxPerMmGuessFace ?: run {
            val approx = (w.toFloat() / hboxInnerMmF) * 0.82f
            approx.coerceIn(3.5f, 8.0f)
        }
        if (!pxGuessBase.isFinite() || pxGuessBase <= 0f) {
            return fail("PX_GUESS", "pxGuessBase invalid=$pxGuessBase")
        }
// Regla 3250:
// La geometría del aro sólo existe si hay inner bilateral (A+B).
// Top y Bottom sólo se buscan si existe inner.
// Un estimado puede completar una geometría existente,
// pero nunca crear geometría desde ausencia de observación.
// ==========================================================
// Search over scales
// ==========================================================
        stage = "SCALES"
        val scales = buildScales3250(SCALE_MIN, SCALE_MAX, SCALE_STEP)
        if (scales.isEmpty()) return fail("SCALES", "empty scales")

        var bestConf = -1f
        var bestLeft = -1
        var bestRight = -1
        var bestTop = -1
        var bestBottom = -1
        var bestRefY = -1
        var bestSeedX = -1
        var bestScale = 1.0f
        var bestBottomPoly: List<Pair<Int, Int>> = emptyList()
        var bestTopPoly: List<Pair<Int, Int>> = emptyList()
        var bestTopObservedY: Int? = null
        var bestExpHGuessPx = Float.NaN
        var bestRatioPenalty = 1f

        var bestPartialConf = -1f
        var bestPartialLeft = -1
        var bestPartialRight = -1
        var bestPartialTop = -1
        var bestPartialBottom = -1
        var bestPartialRefY = -1
        var bestPartialSeedX = -1
        var bestPartialScale = Float.NaN
        var bestPartialTopPoly: List<Pair<Int, Int>> = emptyList()
        var bestPartialBottomPoly: List<Pair<Int, Int>> = emptyList()
        var bestPartialNasalInnerPoly: List<Pair<Int, Int>> = emptyList()
        var bestPartialTempleInnerPoly: List<Pair<Int, Int>> = emptyList()
        var bestPartialTopObservedY: Int? = null
        var bestPartialExpHGuessPx = Float.NaN
        var bestPartialRatioPenalty = 1f

        for (scale in scales) {
            val pxGuessThis = (pxGuessBase * scale).coerceIn(3.5f, 8.0f)
            val expWpx = (hboxInnerMmF * pxGuessThis).coerceAtLeast(80f)
            val gapPx = max(18f, 0.10f * expWpx)

            val seamXLocal = midLocalX.coerceIn(0f, (w - 1).toFloat())
            val seedExpected = seamXLocal + sideSign * (gapPx + 0.50f * expWpx)

            val seedXLocal = seedExpected
                .roundToInt()
                .coerceIn(0, w - 1)

            val expHpxGuess =
                if (vboxInnerMmF > 1e-3f) (vboxInnerMmF * pxGuessThis) else (0.72f * expWpx)

            val maxDist = (0.85f * expWpx).roundToInt().coerceAtLeast(60).coerceAtMost(w - 1)
            val distStart = max(LR_MIN_DIST_PX, (0.10f * expWpx).roundToInt())

            if (DBG) {
                Log.d(
                    TAG,
                    "RIMDBG[$debugTag] profile=$profile3250 S=${f23250(scale)} " +
                            "seed=($seedXLocal,$probeYLocal) yRef=$probeYLocal " +
                            "seedExpX=${f13250(seedExpected)} seamX=${f13250(seamXLocal)} side=${
                                f13250(
                                    sideSign
                                )
                            } " +
                            "expW=${f13250(expWpx)} gap=${f13250(gapPx)} expHGuess=${
                                f13250(
                                    expHpxGuess
                                )
                            } pxG=${
                                f23250(
                                    pxGuessThis
                                )
                            } " +
                            "distStart=$distStart maxDist=$maxDist"
                )
            }

            val lr = findInnerWallsAtY3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                vScoreU8 = vScoreU8,
                w = w,
                h = h,
                seedX = seedXLocal,
                yRef = probeYLocal,
                bandHalf = BAND_HALF_PX,
                minDist = distStart,
                maxDist = maxDist,
                profile3250 = profile3250
            )

            if (lr == null) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f23250(scale)} LR_FAIL yRef=$probeYLocal " +
                                "seedX=$seedXLocal probe=$probeYLocal"
                    )
                }
                continue
            }

            val a = lr.first
            val b = lr.second
            val innerW = (b - a)

            if (DBG) {
                val y0h = (probeYLocal - BAND_HALF_PX).coerceIn(0, h - 1)
                val y1h = (probeYLocal + BAND_HALF_PX).coerceIn(0, h - 1)

                var hitsL = 0
                for (yy in y0h..y1h) {
                    val idx = yy * w + a
                    if ((edgesU8[idx].toInt() and 0xFF) != 0 && !isMaskedPx3250(maskU8, idx)) {
                        val ddir = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                        if (isVertEdgeDir3250(ddir)) hitsL++
                    }
                }
                var hitsR = 0
                for (yy in y0h..y1h) {
                    val idx = yy * w + b
                    if ((edgesU8[idx].toInt() and 0xFF) != 0 && !isMaskedPx3250(maskU8, idx)) {
                        val ddir = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
                        if (isVertEdgeDir3250(ddir)) hitsR++
                    }
                }

                val ratioWdbg = innerW.toFloat() / expWpx
                Log.d(
                    TAG,
                    "RIMDBG[$debugTag] S=${f23250(scale)} LR_OK a=$a b=$b innerW=$innerW " +
                            "ratioW=${f33250(ratioWdbg)} yRef=$probeYLocal " +
                            "dL=${abs(a - seedXLocal)} hitsL=$hitsL dR=${abs(b - seedXLocal)} hitsR=$hitsR"
                )
            }

            if (innerW < 60) {
                if (DBG) Log.d(
                    TAG,
                    "RIMDBG[$debugTag] S=${f23250(scale)} DROP innerW<60 innerW=$innerW"
                )
                continue
            }
            val obsInnerWpx = innerW.toFloat()
            val usedInnerWpx = min(obsInnerWpx, expWpx)

            // ratioW: NO cortar candidates -> penalidad
            val ratioW = usedInnerWpx / expWpx
            val ratioPenalty = if (ratioW !in W_RATIO_MIN..W_RATIO_MAX) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f23250(scale)} RATIO_PEN ratioW=${f33250(ratioW)} " +
                                "expW=${f13250(expWpx)} innerW=$innerW"
                    )
                }
                (1f - 1.1f * abs(1f - ratioW)).coerceIn(0.15f, 0.95f)
            } else {
                1f
            }
            val pxPerMmX =
                (usedInnerWpx / hboxInnerMmF).takeIf { it.isFinite() && it > 0f } ?: run {
                    if (DBG) {
                        Log.d(
                            TAG,
                            "RIMDBG[$debugTag] S=${this.f23250(scale)} DROP pxPerMmX invalid " +
                                    "obsW=$obsInnerWpx expW=$expWpx hboxInner=$hboxInnerMmF"
                        )
                    }
                    continue
                }
            if (DBG) {
                Log.d(
                    TAG,
                    "RIMSCALE[$debugTag] S=${f23250(scale)} " +
                            "obsW=${f23250(obsInnerWpx)} expW=${f23250(expWpx)} " +
                            "usedW=${f23250(usedInnerWpx)} pxUsed=${f33250(pxPerMmX)}"
                )
            }
            val filExpectedY = expectedTopBottomFromFilAtSeed3250(
                filPtsMm = filPtsMm3250,
                probeYLocal = probeYLocal,
                pxPerMmX = pxPerMmX
            )

            val expectedTopRawY =
                filExpectedY?.topY
                    ?: if (vboxInnerMmF > 1e-3f) {
                        (probeYLocal - 0.5f * vboxInnerMmF * pxPerMmX).roundToInt()
                    } else {
                        null
                    }

            val expectedBottomY = filExpectedY?.bottomY
                ?.coerceIn(topMinAllowedY, h - 1)
                ?: if (vboxInnerMmF > 1e-3f) {
                    (probeYLocal + 0.5f * vboxInnerMmF * pxPerMmX)
                        .roundToInt()
                        .coerceIn(topMinAllowedY, h - 1)
                } else {
                    null
                }

            val minOuterGapBottomPx = when (profile3250) {
                RimProfile3250.FULL_RIM ->
                    max(3, (0.60f * pxPerMmX).roundToInt())

                RimProfile3250.RANURADO ->
                    max(2, (0.35f * pxPerMmX).roundToInt())

                RimProfile3250.PERFORADO ->
                    0
            }

            val maxOuterGapBottomPx = when (profile3250) {
                RimProfile3250.FULL_RIM ->
                    max(
                        minOuterGapBottomPx + 2,
                        (4.50f * pxPerMmX).roundToInt()
                    )

                RimProfile3250.RANURADO ->
                    max(
                        minOuterGapBottomPx + 2,
                        (2.50f * pxPerMmX).roundToInt()
                    )

                RimProfile3250.PERFORADO ->
                    0
            }

            val bottomTolPx = when (profile3250) {
                RimProfile3250.FULL_RIM ->
                    max(12, (0.80f * pxPerMmX).roundToInt())

                RimProfile3250.RANURADO ->
                    max(10, (1.00f * pxPerMmX).roundToInt())

                RimProfile3250.PERFORADO ->
                    max(14, (1.25f * pxPerMmX).roundToInt())
            }

            val bottomSeedY =
                probeYLocal.coerceIn(0, h - 1)

            val bottomYMax = when (profile3250) {
                RimProfile3250.FULL_RIM ->
                    (probeYLocal + max(96, (1.05f * expWpx).roundToInt()))
                        .coerceIn(bottomSeedY + 12, h - 1)

                RimProfile3250.RANURADO ->
                    (probeYLocal + max(112, (1.15f * expWpx).roundToInt()))
                        .coerceIn(bottomSeedY + 12, h - 1)

                RimProfile3250.PERFORADO ->
                    (probeYLocal + max(128, (1.25f * expWpx).roundToInt()))
                        .coerceIn(bottomSeedY + 12, h - 1)
            }

            val bottomExpectedTolPx = when (profile3250) {
                RimProfile3250.FULL_RIM ->
                    max(bottomTolPx, (0.45f * expWpx).roundToInt())

                RimProfile3250.RANURADO ->
                    max(bottomTolPx, (0.55f * expWpx).roundToInt())

                RimProfile3250.PERFORADO ->
                    max(bottomTolPx, (0.65f * expWpx).roundToInt())
            }

            val bottomPick = pickBottomInsideOut(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                hScoreU8 = hScoreU8,
                w = w,
                h = h,
                leftX = a,
                rightX = b,
                ySeed = bottomSeedY,
                yMax = bottomYMax,
                topMinAllowedY = topMinAllowedY,
                stepX = 4,
                minHits = 28,
                minCoverage = 0.50f,
                contJumpPx = CONT_JUMP_PX,
                profile3250 = profile3250,
                minOuterGapPx = minOuterGapBottomPx,
                maxOuterGapPx = maxOuterGapBottomPx,
                expectedBottomY = expectedBottomY,
                expectedTolPx = bottomExpectedTolPx
            )

            if (bottomPick == null) {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f23250(scale)} BOTTOM_FAIL a=$a b=$b " +
                                "ySeed=$bottomSeedY yMax=$bottomYMax topMin=$topMinAllowedY " +
                                "expBot=${expectedBottomY ?: -1} tol=$bottomExpectedTolPx " +
                                "gap=[$minOuterGapBottomPx..$maxOuterGapBottomPx] " +
                                "profile=$profile3250"
                    )
                }
            } else {
                if (DBG) {
                    Log.d(
                        TAG,
                        "RIMDBG[$debugTag] S=${f23250(scale)} BOTTOM_OK " +
                                "yMed=${bottomPick.yMed} yMax=${bottomPick.yMax} yBot=${bottomPick.yBottomCoherent} " +
                                "hits=${bottomPick.poly.size} cov=${f33250(bottomPick.coverage)} " +
                                "cont=${f33250(bottomPick.continuity)}"
                    )
                }
            }

            val topTolPx = max(14, (1.20f * pxPerMmX).roundToInt())

            val yCapTop = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)

            val expectedTopForSearch =
                when {
                    bottomPick != null && vboxInnerMmF > 1e-3f -> {
                        (bottomPick.yMax - vboxInnerMmF * pxPerMmX).roundToInt()
                    }

                    expectedTopRawY != null -> {
                        expectedTopRawY
                    }

                    else -> {
                        probeYLocal - max(24, (0.20f * expWpx).roundToInt())
                    }
                }

            val expectedTopY =
                expectedTopForSearch.coerceIn(0, yCapTop)

            val topSearchUpPx = 40

            val topSearchMinY =
                (expectedTopY - topSearchUpPx)
                    .coerceIn(0, yCapTop)

            if (DBG) {
                Log.d(
                    TAG,
                    "TOPWIN[$debugTag] a=$a b=$b probe=$probeYLocal topMin=$topMinAllowedY " +
                            "topSearchMin=$topSearchMinY expTop=$expectedTopY tol=$topTolPx"
                )
            }

            val topPick = pickTopInsideOut3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                leftX = a,
                rightX = b,
                ySeed = probeYLocal,
                yMin = topSearchMinY,
                stepX = 4,
                minHits = 20,
                minCoverage = 0.40f,
                contJumpPx = CONT_JUMP_PX,
                expectedTopY = expectedTopY,
                expectedTolPx = topTolPx
            )

            if (DBG) {
                Log.d(
                    TAG,
                    "RIMTOP[$debugTag] S=${f23250(scale)} " +
                            "expTop=$expectedTopY " +
                            "obsTop=${topPick?.yMed ?: -1} " +
                            "cov=${f33250(topPick?.coverage ?: 0f)} " +
                            "cont=${f33250(topPick?.continuity ?: 0f)} " +
                            "conf=${f33250(topPick?.confidence ?: 0f)}"
                )
            }

            val hasTopObs =
                topPick != null &&
                        topPick.coverage > 0f &&
                        topPick.continuity > 0f &&
                        topPick.confidence > 0f

            val hasBottomObs =
                bottomPick != null

            val cand0 =
                when {
                    hasTopObs && hasBottomObs -> {
                        buildNormalCandidate3250(
                            debugTag = debugTag,
                            scale = scale,
                            ratioPenalty = ratioPenalty,
                            topSearchMinY = topSearchMinY,
                            topExpectedY = expectedTopY,
                            probeYLocal = probeYLocal,
                            vboxInnerMmF = vboxInnerMmF,
                            pxPerMmX = pxPerMmX,
                            bottomPick = bottomPick!!,
                            topPick = topPick,
                            h = h
                        )
                    }

                    hasTopObs || hasBottomObs -> {
                        val expHpxGuessUsed =
                            if (expWpx > 1e-3f) {
                                expHpxGuess * (usedInnerWpx / expWpx)
                            } else {
                                expHpxGuess
                            }

                        buildPartialCandidate3250(
                            debugTag = debugTag,
                            scale = scale,
                            ratioPenalty = ratioPenalty,
                            probeYLocal = probeYLocal,
                            topSearchMinY = topSearchMinY,
                            topExpectedY = expectedTopY,
                            topPick = topPick,
                            bottomPick = bottomPick,
                            profile3250 = profile3250,
                            expHpxGuess = expHpxGuessUsed,
                            expWpx = expWpx,
                            innerW = usedInnerWpx.roundToInt(),
                            h = h
                        )
                    }

                    else -> {
                        null
                    }
                }

            val candBase = cand0?.copy(
                left = a,
                right = b,
                refY = probeYLocal,
                seedX = seedXLocal
            ) ?: continue

            val cand = if (candBase.isPartial) {
                val nasalX = if (nasalAtLeft) a else b
                val templeX = if (nasalAtLeft) b else a

                candBase.copy(
                    nasalInnerPoly = buildVerticalGuidePoly3250(
                        x = nasalX,
                        yTop = candBase.top,
                        yBottom = candBase.bottom,
                        h = h
                    ),
                    templeInnerPoly = buildVerticalGuidePoly3250(
                        x = templeX,
                        yTop = candBase.top,
                        yBottom = candBase.bottom,
                        h = h
                    )
                )
            } else {
                candBase
            }

            if (cand.isPartial) {
                if (cand.conf > bestPartialConf) {
                    bestPartialConf = cand.conf
                    bestPartialLeft = cand.left
                    bestPartialRight = cand.right
                    bestPartialTop = cand.top
                    bestPartialBottom = cand.bottom
                    bestPartialRefY = cand.refY
                    bestPartialSeedX = cand.seedX
                    bestPartialScale = cand.scale
                    bestPartialBottomPoly = cand.bottomPoly
                    bestPartialTopPoly = cand.topPoly
                    bestPartialNasalInnerPoly = cand.nasalInnerPoly
                    bestPartialTempleInnerPoly = cand.templeInnerPoly
                    bestPartialTopObservedY = cand.topObservedY
                    bestPartialExpHGuessPx = cand.expHGuessPx
                    bestPartialRatioPenalty = cand.ratioPenalty
                }
            } else {
                if (cand.conf > bestConf) {
                    bestConf = cand.conf
                    bestLeft = cand.left
                    bestRight = cand.right
                    bestTop = cand.top
                    bestBottom = cand.bottom
                    bestScale = cand.scale
                    bestBottomPoly = cand.bottomPoly
                    bestTopPoly = cand.topPoly
                    bestRefY = cand.refY
                    bestSeedX = cand.seedX
                    bestTopObservedY = cand.topObservedY
                    bestExpHGuessPx = cand.expHGuessPx
                    bestRatioPenalty = cand.ratioPenalty
                }
            }
        }

        stage = "BEST_FINAL"

        val winnerIsPartial: Boolean
        val winnerConfRaw: Float
        val winnerLeft: Int
        val winnerRight: Int
        val winnerTop: Int
        val winnerBottom: Int
        val winnerRefY: Int
        val winnerSeedX: Int
        val winnerScale: Float
        val winnerTopPoly: List<Pair<Int, Int>>
        val winnerBottomPoly: List<Pair<Int, Int>>
        val winnerNasalInnerPoly: List<Pair<Int, Int>>
        val winnerTempleInnerPoly: List<Pair<Int, Int>>
        val winnerTopObservedY: Int?
        val winnerExpHGuessPx: Float
        val winnerRatioPenalty: Float

        if (bestConf >= 0f) {
            if (bestConf < OK_CONF_MIN) {
                return fail("LOW_CONF", "bestConf=${f33250(bestConf)} < $OK_CONF_MIN")
            }
            if (bestLeft < 0 || bestRight < 0 || bestBottom < 0 || bestTop < 0) {
                return fail("BEST_INV", "best coords invalid")
            }
            if (bestRefY < 0 || bestSeedX < 0) {
                return fail(
                    "BEST_INV2",
                    "bestRefY/bestSeedX invalid refY=$bestRefY seed=$bestSeedX"
                )
            }

            winnerIsPartial = false
            winnerConfRaw = bestConf
            winnerLeft = bestLeft
            winnerRight = bestRight
            winnerTop = bestTop
            winnerBottom = bestBottom
            winnerRefY = bestRefY
            winnerSeedX = bestSeedX
            winnerScale = bestScale
            winnerTopPoly = bestTopPoly
            winnerBottomPoly = bestBottomPoly
            winnerNasalInnerPoly = emptyList()
            winnerTempleInnerPoly = emptyList()
            winnerTopObservedY = bestTopObservedY
            winnerExpHGuessPx = bestExpHGuessPx
            winnerRatioPenalty = bestRatioPenalty

            d(
                "RIM[$debugTag] BEST_RAW profile=$profile3250 conf=${f33250(winnerConfRaw)} " +
                        "L/R=$winnerLeft/$winnerRight T/B=$winnerTop/$winnerBottom " +
                        "scale=${f23250(winnerScale)} refY=$winnerRefY seedX=$winnerSeedX partial=false"
            )
        } else if (
            bestPartialConf >= 0f &&
            bestPartialLeft >= 0 &&
            bestPartialRight >= 0 &&
            bestPartialTop >= 0 &&
            bestPartialBottom > bestPartialTop &&
            bestPartialRefY >= 0 &&
            bestPartialSeedX >= 0
        ) {
            winnerIsPartial = true
            winnerConfRaw = bestPartialConf
            winnerLeft = bestPartialLeft
            winnerRight = bestPartialRight
            winnerTop = bestPartialTop
            winnerBottom = bestPartialBottom
            winnerRefY = bestPartialRefY
            winnerSeedX = bestPartialSeedX
            winnerScale = bestPartialScale
            winnerTopPoly = bestPartialTopPoly
            winnerBottomPoly = bestPartialBottomPoly
            winnerNasalInnerPoly = bestPartialNasalInnerPoly
            winnerTempleInnerPoly = bestPartialTempleInnerPoly
            winnerTopObservedY = bestPartialTopObservedY
            winnerExpHGuessPx = bestPartialExpHGuessPx
            winnerRatioPenalty = bestPartialRatioPenalty

            d(
                "RIM[$debugTag] BEST_RAW profile=$profile3250 conf=${f33250(winnerConfRaw)} " +
                        "L/R=$winnerLeft/$winnerRight T/B=$winnerTop/$winnerBottom " +
                        "topPts=${winnerTopPoly.size} bottomPts=${winnerBottomPoly.size} " +
                        "scale=${f23250(winnerScale)} refY=$winnerRefY seedX=$winnerSeedX partial=true"
            )
        } else {
            return fail("NO_CAND", "no candidate survived")
        }
        val innerWpx = (winnerRight - winnerLeft).toFloat().coerceAtLeast(1f)
        val pxPerMmXFinal = (innerWpx / hboxInnerMmF).takeIf { it.isFinite() && it > 0f }
            ?: return fail("PXMM_PARTIAL", "pxPerMmX invalid")

        val nasalG = (if (nasalAtLeft) winnerLeft.toFloat() else winnerRight.toFloat()) + roiLeftG
        val templeG = (if (nasalAtLeft) winnerRight.toFloat() else winnerLeft.toFloat()) + roiLeftG

        val innerL = min(nasalG, templeG)
        val innerR = max(nasalG, templeG)

        val refYGlobal = winnerRefY.toFloat() + roiTopG
        val seedXGlobal = winnerSeedX.toFloat() + roiLeftG

        // ----------------------------------------------------------
        // arcos candidatos laterales
        // ----------------------------------------------------------
        val minOuterGapPx = max(3, (0.60f * pxPerMmXFinal).roundToInt())
        val maxOuterGapPx = max(minOuterGapPx + 2, (4.50f * pxPerMmXFinal).roundToInt())

        val sideArcLocal = buildSideArcCandidates3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            vScoreU8 = vScoreU8,
            w = w,
            h = h,
            seedXInside = winnerSeedX,
            yRef = winnerRefY,
            topY = winnerTop,
            bottomY = winnerBottom,
            minOuterGapPx = minOuterGapPx,
            maxOuterGapPx = maxOuterGapPx,
            profile3250 = profile3250
        )

        val innerLeftValidated = keepPolylineNearXRef3250(
        pts = sideArcLocal.innerLeft,
        xRef = winnerLeft.toFloat(),
        sideSign = -1
        )

        val innerRightValidated = keepPolylineNearXRef3250(
        pts = sideArcLocal.innerRight,
        xRef = winnerRight.toFloat(),
        sideSign = +1
        )

        val sideArcLocalValidated = sideArcLocal.copy(
            innerLeft = innerLeftValidated,
            innerRight = innerRightValidated
        )

        val innerLeftLocal = sideArcLocalValidated.innerLeft
        val innerRightLocal = sideArcLocalValidated.innerRight

        val outerLeftPolyG = localPolylineToGlobal3250(sideArcLocalValidated.outerLeft, roiLeftG, roiTopG)
        val outerRightPolyG = localPolylineToGlobal3250(sideArcLocalValidated.outerRight, roiLeftG, roiTopG)

        val winnerNasalInnerLocal = winnerNasalInnerPoly.map { (x, y) ->
            PointF(x.toFloat(), y.toFloat())
        }
        val winnerTempleInnerLocal = winnerTempleInnerPoly.map { (x, y) ->
            PointF(x.toFloat(), y.toFloat())
        }

        val nasalInnerLocal =
            if (winnerIsPartial && winnerNasalInnerLocal.isNotEmpty()) {
                winnerNasalInnerLocal
            } else {
                if (nasalAtLeft) innerLeftLocal else innerRightLocal
            }

        val templeInnerLocal =
            if (winnerIsPartial && winnerTempleInnerLocal.isNotEmpty()) {
                winnerTempleInnerLocal
            } else {
                if (nasalAtLeft) innerRightLocal else innerLeftLocal
            }

        val nasalInnerPolyG = localPolylineToGlobal3250(nasalInnerLocal, roiLeftG, roiTopG)
        val templeInnerPolyG = localPolylineToGlobal3250(templeInnerLocal, roiLeftG, roiTopG)
        val nasalOuterPolyG = if (nasalAtLeft) outerLeftPolyG else outerRightPolyG
        val templeOuterPolyG = if (nasalAtLeft) outerRightPolyG else outerLeftPolyG

        val innerArcLocal = buildOfficialInnerArc3250(
            bottomLocal = winnerBottomPoly.map { (x, y) -> PointF(x.toFloat(), y.toFloat()) },
            topLocal = winnerTopPoly.map { (x, y) -> PointF(x.toFloat(), y.toFloat()) },
            nasalLocal = nasalInnerLocal,
            templeLocal = templeInnerLocal,
            maxJoinDistPx = 14f,
            maxStepDistPx = 12f,
            minRunPts = 6
        )

        val innerArcGlobal =
            if (innerArcLocal.isNotEmpty()) localPolylineToGlobal3250(innerArcLocal, roiLeftG, roiTopG)
            else null
        val topPolyGlobal =
            if (winnerTopPoly.isNotEmpty()) {
                pairPolylineToGlobal3250(
                    pts = winnerTopPoly,
                    roiLeft = roiLeftG,
                    roiTop = roiTopG
                )
            } else {
                null
            }
        val bottomPolyGlobal =
            if (winnerBottomPoly.isNotEmpty()) {
                winnerBottomPoly
                    .map { (x, y) -> PointF(x.toFloat() + roiLeftG, y.toFloat() + roiTopG) }
                    .sortedBy { it.x }
            } else {
                null
            }
        val hasInnerLateralsForGate =
            if (!winnerIsPartial) {
                val winnerInnerW = winnerRight - winnerLeft
                winnerRight > winnerLeft && winnerInnerW >= 60
            } else {
                !nasalInnerPolyG.isNullOrEmpty() && !templeInnerPolyG.isNullOrEmpty()
            }

        val gateResult3250 = applyProfileGate3250(
            RimGateInput3250(
                profile3250 = profile3250,
                baseConfidence = winnerConfRaw,
                isPartial = winnerIsPartial,
                hasInnerLaterals = hasInnerLateralsForGate,
                hasOuterLaterals = !nasalOuterPolyG.isNullOrEmpty() && !templeOuterPolyG.isNullOrEmpty(),
                topObservedY = winnerTopObservedY,
                topUsedY = winnerTop,
                bottomUsedY = winnerBottom,
                expHGuessPx = winnerExpHGuessPx,
                ratioPenalty = winnerRatioPenalty
            )
        )

        if (!gateResult3250.accepted) {
            return fail("PROFILE_GATE", gateResult3250.reason3250)
        }

        val res = RimDetectionResult(
            ok = true,
            confidence = gateResult3250.confidenceOut,
            roiPx = RectF(roiLeftG, roiTopG, roiLeftG + w.toFloat(), roiTopG + h.toFloat()),
            probeYpx = probeYLocal.toFloat() + roiTopG,
            topYpx = winnerTop.toFloat() + roiTopG,
            bottomYpx = winnerBottom.toFloat() + roiTopG,
            innerLeftXpx = innerL,
            innerRightXpx = innerR,
            nasalInnerPolylinePx = nasalInnerPolyG,
            templeInnerPolylinePx = templeInnerPolyG,
            nasalOuterPolylinePx = nasalOuterPolyG,
            templeOuterPolylinePx = templeOuterPolyG,
            nasalInnerXpx = nasalG,
            templeInnerXpx = templeG,
            innerWidthPx = (innerR - innerL),
            heightPx = (winnerBottom - winnerTop).toFloat(),
            wallsYpx = refYGlobal,
            seedXpx = seedXGlobal,
            topPolylinePx = topPolyGlobal,
            bottomPolylinePx = bottomPolyGlobal,
            innerArcPolylinePx = innerArcGlobal,
            outerLeftXpx = null,
            outerRightXpx = null,
            rimThicknessPx = null
        )

           return RimDetectPack3250(
            result = res,
            edges = edgesU8,
            w = w,
            h = h
        )
    }

    /**
     * Mirror fallback
     *
     * Refleja el resultado geométrico de un ojo al otro lado usando la midline global.
     * NO recalcula detección, NO recalcula px/mm, NO reinterpreta filOver/profile:
     * solo espeja la geometría ya resuelta.
     */
    fun mirrorFromOtherEye3250(
        primary: RimDetectionResult,
        targetEdgesU8: ByteArray,
        w: Int,
        h: Int,
        targetRoiGlobal: RectF,
        midlineXpx: Float,
        debugTag: String,
        srcTag: String,
        confScale: Float = 0.60f
    ): RimDetectPack3250? {

        fun fail(msg: String): RimDetectPack3250? {
            Log.w(TAG, "RIM[$debugTag] MIRROR FAIL: $msg")
            return null
        }

        if (w <= 0 || h <= 0) return fail("w/h invalid $w x $h")
        if (targetEdgesU8.size != w * h) {
            return fail("edgesU8.size=${targetEdgesU8.size} != w*h=${w * h}")
        }
        if (!midlineXpx.isFinite()) return fail("midline invalid=$midlineXpx")

        val roiLeft = targetRoiGlobal.left
        val roiTop = targetRoiGlobal.top
        val roiRight = roiLeft + (w - 1).toFloat()
        val roiBottom = roiTop + (h - 1).toFloat()

        fun mirrorX(x: Float): Float = 2f * midlineXpx - x
        fun clampX(x: Float): Float = x.coerceIn(roiLeft + 1f, roiRight - 1f)
        fun clampY(y: Float): Float = y.coerceIn(roiTop + 1f, roiBottom - 1f)

        fun mirrorPolyY(poly: List<PointF>?): List<PointF>? =
            poly
                ?.map { p -> PointF(clampX(mirrorX(p.x)), clampY(p.y)) }
                ?.sortedBy { it.y }

        fun mirrorPolyX(poly: List<PointF>?): List<PointF>? =
            poly
                ?.map { p -> PointF(clampX(mirrorX(p.x)), clampY(p.y)) }
                ?.sortedBy { it.x }

        // ---------- mirror SEMANTIC endpoints ----------
        var nasal = clampX(mirrorX(primary.nasalInnerXpx))
        var temple = clampX(mirrorX(primary.templeInnerXpx))

        // Re-clasificación por seguridad: nasal debe quedar más cerca de la midline que temple.
        if (abs(temple - midlineXpx) < abs(nasal - midlineXpx)) {
            val tmp = nasal
            nasal = temple
            temple = tmp
        }

        // innerLeft/Right = SOLO min/max
        val innerL = min(nasal, temple)
        val innerR = max(nasal, temple)

        val innerW = (innerR - innerL)
        if (!innerW.isFinite() || innerW < 60f) {
            return fail("inner width too small after mirror: $innerW")
        }

        val probeY = clampY(primary.probeYpx)
        val topY = clampY(primary.topYpx)
        val bottomY = clampY(primary.bottomYpx)
        val heightPx = (bottomY - topY).coerceAtLeast(1f)

        // bottom/top poly: mirror + clamp + orden por X
        val bottomPoly = mirrorPolyX(primary.bottomPolylinePx)
        val topPoly = mirrorPolyX(primary.topPolylinePx)

// mirror de arcos candidatos laterales
        val nasalInnerPoly = mirrorPolyY(primary.nasalInnerPolylinePx)
        val templeInnerPoly = mirrorPolyY(primary.templeInnerPolylinePx)
        val nasalOuterPoly = mirrorPolyY(primary.nasalOuterPolylinePx)
        val templeOuterPoly = mirrorPolyY(primary.templeOuterPolylinePx)

        val innerArcPolylinePx =
            buildOfficialInnerArc3250(
                bottomLocal = bottomPoly ?: emptyList(),
                topLocal = topPoly ?: emptyList(),
                nasalLocal = nasalInnerPoly ?: emptyList(),
                templeLocal = templeInnerPoly ?: emptyList(),
                maxJoinDistPx = 14f,
                maxStepDistPx = 12f,
                minRunPts = 6
            ).takeIf { it.isNotEmpty() }
        // anchors
        val wallsY = clampY(primary.wallsYpx)
        val seedX = clampX(mirrorX(primary.seedXpx))

        val conf = (primary.confidence * confScale).coerceIn(0f, 0.99f)

        Log.w(
            TAG,
            "RIM[$debugTag] MIRROR_FROM_$srcTag conf=${f33250(conf)} " +
                    "innerL/R=${f13250(innerL)}/${f13250(innerR)} W=${f13250(innerW)} " +
                    "wallsY=${f13250(wallsY)} seedX=${f13250(seedX)} midX=${f13250(midlineXpx)} " +
                    "nIn=${nasalInnerPoly?.size ?: 0} tIn=${templeInnerPoly?.size ?: 0} " +
                    "nOut=${nasalOuterPoly?.size ?: 0} tOut=${templeOuterPoly?.size ?: 0}"
        )

        val res = RimDetectionResult(
            ok = true,
            confidence = conf,
            roiPx = targetRoiGlobal,

            probeYpx = probeY,
            topYpx = topY,
            bottomYpx = bottomY,

            innerLeftXpx = innerL,
            innerRightXpx = innerR,

            nasalInnerXpx = nasal,
            templeInnerXpx = temple,

            heightPx = heightPx,
            innerWidthPx = innerW,

            topPolylinePx = topPoly,
            bottomPolylinePx = bottomPoly,

            wallsYpx = wallsY,
            seedXpx = seedX,

            outerLeftXpx = null,
            outerRightXpx = null,
            nasalOuterXpx = null,
            templeOuterXpx = null,
            rimThicknessPx = null,
            outerWidthPx = null,

            innerArcPolylinePx = innerArcPolylinePx,
            nasalInnerPolylinePx = nasalInnerPoly,
            templeInnerPolylinePx = templeInnerPoly,
            nasalOuterPolylinePx = nasalOuterPoly,
            templeOuterPolylinePx = templeOuterPoly
        )

        return RimDetectPack3250(
            result = res,
            edges = targetEdgesU8,
            w = w,
            h = h
        )
    }

    /**
     * Bottom: contrato congruente con el detector
     * - MISMA referencia vertical oficial: ySeed
     * - DOMINIO horizontal del aro: leftXrightX
     * - BARRIDO CENTER→OUT
     * - MÁSCARA = veto local (se ignora/salta)
     * - Continuidad guiada por prevY desde el ancla
     */
    /**
     * Bottom: contrato congruente con el detector
     * - MISMA referencia vertical oficial: ySeed
     * - DOMINIO horizontal del aro: leftX..rightX
     * - BARRIDO CENTER→OUT
     * - MÁSCARA = veto local (se ignora/salta)
     * - Continuidad guiada por prevY desde el ancla
     *
     * Ajuste aplicado:
     * - seed más permisiva
     * - bottom acepta edge no perfectamente horizontal
     * - soporte horizontal ahora da bonus, no veto absoluto
     * - menor castigo por expectedY
     * - fallback extra con ventana más amplia antes de declarar miss
     */

    private fun pickBottomInsideOut(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        hScoreU8: ByteArray? = null,
        w: Int,
        h: Int,
        leftX: Int,
        rightX: Int,
        ySeed: Int,
        yMax: Int,
        topMinAllowedY: Int,
        stepX: Int,
        minHits: Int,
        minCoverage: Float,
        contJumpPx: Int,
        profile3250: RimProfile3250,
        minOuterGapPx: Int = 0,
        maxOuterGapPx: Int = 0,
        expectedBottomY: Int? = null,
        expectedTolPx: Int = 42
    ): ArcPick? {

        val xa = (leftX + 8).coerceIn(0, w - 1)
        val xb = (rightX - 8).coerceIn(0, w - 1)
        if (xb - xa < 50) return null

        val yLo = topMinAllowedY.coerceIn(0, h - 1)
        val yHi = yMax.coerceIn(0, h - 1)
        if (yHi - yLo < 10) return null

        val xMid = ((leftX + rightX) / 2).coerceIn(xa, xb)
        val ySeed0 = ySeed.coerceIn(yLo, yHi)

        fun findSeedAtX(x: Int): Int {
            val x0 = (x - stepX * 2).coerceIn(xa, xb)
            val x1 = (x + stepX * 2).coerceIn(xa, xb)

            val y0Soft = expectedBottomY?.let { (it - expectedTolPx).coerceIn(yLo, yHi) } ?: yLo
            val y1Soft = expectedBottomY?.let { (it + expectedTolPx).coerceIn(yLo, yHi) } ?: yHi

            var y = findBestBottomEdgeInWindow3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                hScoreU8 = hScoreU8,
                w = w,
                a = x0,
                b = x1,
                y0 = max(ySeed0, y0Soft),
                y1 = y1Soft,
                expectedY = expectedBottomY ?: ySeed0,
                profile3250 = profile3250,
                minOuterGapPx = minOuterGapPx,
                maxOuterGapPx = maxOuterGapPx
            )

            if (y < 0) {
                y = findBestBottomEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    hScoreU8 = hScoreU8,
                    w = w,
                    a = x0,
                    b = x1,
                    y0 = ySeed0,
                    y1 = yHi,
                    expectedY = expectedBottomY ?: ySeed0,
                    profile3250 = profile3250,
                    minOuterGapPx = minOuterGapPx,
                    maxOuterGapPx = maxOuterGapPx
                )
            }

            if (y < 0) {
                val y0Wide = (ySeed0 - contJumpPx * 3).coerceIn(yLo, yHi)
                val y1Wide = (ySeed0 + contJumpPx * 5).coerceIn(yLo, yHi)

                y = findBestBottomEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    hScoreU8 = hScoreU8,
                    w = w,
                    a = x0,
                    b = x1,
                    y0 = y0Wide,
                    y1 = y1Wide,
                    expectedY = expectedBottomY ?: ySeed0,
                    profile3250 = profile3250,
                    minOuterGapPx = minOuterGapPx,
                    maxOuterGapPx = maxOuterGapPx
                )
            }

            return y
        }
        data class SeedBottom(val x: Int, val y: Int)

        var seedBottom: SeedBottom? = null

        run {
            val y0 = findSeedAtX(xMid)
            if (y0 >= 0) {
                seedBottom = SeedBottom(xMid, y0)
                return@run
            }

            var d = stepX
            while (xMid - d >= xa || xMid + d <= xb) {
                val xl = xMid - d
                if (xl in xa..xb) {
                    val yl = findSeedAtX(xl)
                    if (yl >= 0) {
                        seedBottom = SeedBottom(xl, yl)
                        return@run
                    }
                }

                val xr = xMid + d
                if (xr in xa..xb) {
                    val yr = findSeedAtX(xr)
                    if (yr >= 0) {
                        seedBottom = SeedBottom(xr, yr)
                        return@run
                    }
                }

                d += stepX
            }
        }

        val seedBottomX = seedBottom?.x ?: return null
        val seedBottomY = seedBottom.y

        fun traceHalf(startX: Int, stepSign: Int): List<Pair<Int, Int>> {
            val out = ArrayList<Pair<Int, Int>>()
            var prevY = seedBottomY
            var x = startX
            var missCount = 0

            while (x in xa..xb) {
                val winPad = (contJumpPx + missCount * 2).coerceAtMost(contJumpPx * 3)
                val yWin0 = (prevY - winPad).coerceIn(yLo, yHi)
                val yWin1 = (prevY + winPad).coerceIn(yLo, yHi)

                var bestY = findBestBottomEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    hScoreU8 = hScoreU8,
                    w = w,
                    a = xa,
                    b = xb,
                    y0 = yWin0,
                    y1 = yWin1,
                    expectedY = prevY,
                    profile3250 = profile3250,
                    minOuterGapPx = minOuterGapPx,
                    maxOuterGapPx = maxOuterGapPx
                )

                if (bestY < 0 && expectedBottomY != null) {
                    val ySoft0 = (expectedBottomY - expectedTolPx).coerceIn(yLo, yHi)
                    val ySoft1 = (expectedBottomY + expectedTolPx).coerceIn(yLo, yHi)
                    bestY = findBestBottomEdgeInWindow3250(
                        edgesU8 = edgesU8,
                        dirU8 = dirU8,
                        maskU8 = maskU8,
                        hScoreU8 = hScoreU8,
                        w = w,
                        a = xa,
                        b = xb,
                        y0 = ySoft0,
                        y1 = ySoft1,
                        expectedY = expectedBottomY,
                        profile3250 = profile3250,
                        minOuterGapPx = minOuterGapPx,
                        maxOuterGapPx = maxOuterGapPx
                    )
                }

                if (bestY < 0) {
                    val yWide0 = (prevY - winPad * 2).coerceIn(yLo, yHi)
                    val yWide1 = (prevY + winPad * 2).coerceIn(yLo, yHi)
                    bestY = findBestBottomEdgeInWindow3250(
                        edgesU8 = edgesU8,
                        dirU8 = dirU8,
                        maskU8 = maskU8,
                        hScoreU8 = hScoreU8,
                        w = w,
                        a = xa,
                        b = xb,
                        y0 = yWide0,
                        y1 = yWide1,
                        expectedY = prevY,
                        profile3250 = profile3250,
                        minOuterGapPx = minOuterGapPx,
                        maxOuterGapPx = maxOuterGapPx
                    )
                }

                if (bestY >= 0) {
                    out.add(x to bestY)
                    prevY = bestY
                    missCount = 0
                } else {
                    out.add(x to -1)
                    missCount++
                }

                x += stepSign * stepX
            }

            return out
        }

        val leftHalf = traceHalf(seedBottomX, -1).drop(1)
        val rightHalf = traceHalf(seedBottomX, +1)

        val polyRaw = ArrayList<Pair<Int, Int>>(leftHalf.size + rightHalf.size)
        polyRaw.addAll(leftHalf.asReversed())
        polyRaw.addAll(rightHalf)

        val polyFilled = fillSmallBottomGaps3250(
            poly = polyRaw,
            stepX = stepX,
            maxGapCols = 2,
            maxInterpJumpPx = contJumpPx * 2
        )

        val polySmooth = smoothBottomPolyline3250(polyFilled)

        val poly = keepBestBottomRunByXStrict3250(
            poly = polySmooth,
            maxJumpX = stepX * 2,
            maxJumpY = contJumpPx * 2,
            minRunPoints = minHits,
            maxTotalSecondaryDriftPx = 120
        )

        if (poly.size < minHits) return null

        val xSpan = (poly.last().first - poly.first().first).coerceAtLeast(0)
        val expectedSamples = (xSpan / stepX + 1).coerceAtLeast(1)
        val coverage = poly.size.toFloat() / expectedSamples.toFloat()

        var contSamples = 0
        var contHits = 0
        for (i in 1 until poly.size) {
            contSamples++
            if (abs(poly[i].second - poly[i - 1].second) <= contJumpPx * 2) {
                contHits++
            }
        }
        val continuity = if (contSamples > 0) {
            contHits.toFloat() / contSamples.toFloat()
        } else {
            0f
        }

        if (coverage < minCoverage) return null

// 🔥 NUEVO
        val ys = poly.map { it.second }.sorted()
        val yMed = ys[ys.size / 2]

        val yMax = poly.maxOf { it.second }

        val tolY = 3
        val nearBottom = poly
            .sortedBy { it.first }
            .filter { it.second >= yMax - tolY }

        val yBottomCoherent = if (nearBottom.isNotEmpty()) {
            nearBottom.map { it.second }.average().toFloat().toInt()
        } else {
            yMax
        }

        return ArcPick(
            yMed = yMed,
            yMax = yMax,
            yBottomCoherent = yBottomCoherent,
            poly = poly,
            coverage = coverage,
            continuity = continuity
        )
    }
        private fun fillSmallBottomGaps3250(
        poly: List<Pair<Int, Int>>,
        stepX: Int,
        maxGapCols: Int,
        maxInterpJumpPx: Int
    ): List<Pair<Int, Int>> {
        if (poly.isEmpty()) return emptyList()

        val out = poly.toMutableList()
        var i = 0

        while (i < out.size) {
            if (out[i].second >= 0) {
                i++
                continue
            }

            val start = i
            while (i < out.size && out[i].second < 0) i++
            val end = i - 1

            val gapLen = end - start + 1
            val leftIdx = start - 1
            val rightIdx = i

            if (
                gapLen <= maxGapCols &&
                leftIdx >= 0 &&
                rightIdx < out.size &&
                out[leftIdx].second >= 0 &&
                out[rightIdx].second >= 0
            ) {
                val yL = out[leftIdx].second
                val yR = out[rightIdx].second
                val xL = out[leftIdx].first
                val xR = out[rightIdx].first

                val dxCols = ((xR - xL) / stepX).coerceAtLeast(1)
                val dy = abs(yR - yL)

                if (dy <= maxInterpJumpPx * dxCols) {
                    for (k in 0 until gapLen) {
                        val a = (k + 1).toFloat() / (gapLen + 1).toFloat()
                        val yFill = ((1f - a) * yL + a * yR).roundToInt()
                        out[start + k] = out[start + k].first to yFill
                    }
                }
            }
        }

        return out.filter { it.second >= 0 }
    }

    private fun smoothBottomPolyline3250(
        poly: List<Pair<Int, Int>>
    ): List<Pair<Int, Int>> {
        if (poly.size < 3) return poly

        val n = poly.size
        val med = IntArray(n)

        for (i in 0 until n) {
            val ys = ArrayList<Int>(5)
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                ys.add(poly[j].second)
            }
            ys.sort()
            med[i] = ys[ys.size / 2]
        }

        val out = ArrayList<Pair<Int, Int>>(n)
        for (i in 0 until n) {
            var sum = 0
            var cnt = 0
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                sum += med[j]
                cnt++
            }
            out.add(poly[i].first to (sum / cnt))
        }

        return out
    }

    private fun buildScales3250(minS: Float, maxS: Float, step: Float): FloatArray {
        if (!minS.isFinite() || !maxS.isFinite() || !step.isFinite()) return floatArrayOf()
        if (step <= 0f) return floatArrayOf()
        val out = ArrayList<Float>(32)
        var s = minS
        while (s <= maxS + 1e-6f) {
            out += s
            s += step
        }
        return out.toFloatArray()
    }

    private fun clampInt3250(v: Int, lo: Int, hi: Int): Int = max(lo, min(hi, v))
    private fun clampInRange3250(v: Int, a: Int, b: Int): Int = v.coerceIn(min(a, b), max(a, b))

    /**
     * Busca inner L/R en una banda vertical alrededor de yRef (la referencia oficial),
     * desde seedX hacia afuera, saltando máscara.
     *
     * - mantenemos el mismo contrato Pair<Int,Int>
     * - LR exige evidencia vertical (dirU8)
     * - la máscara veta puntos/soportes, pero NO mueve la referencia
     */
    private fun buildSideArcCandidates3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        vScoreU8: ByteArray? = null,
        w: Int,
        h: Int,
        seedXInside: Int,
        yRef: Int,
        topY: Int,
        bottomY: Int,
        minOuterGapPx: Int,
        maxOuterGapPx: Int,
        profile3250: RimProfile3250,
        stepY: Int = 3
    ): SideArcCandidatesLocal3250 {

        val y0Inner = (topY + 6).coerceIn(0, h - 1)
        val y1Inner = (bottomY - 10).coerceIn(0, h - 1)

        if (y1Inner <= y0Inner) {
            return SideArcCandidatesLocal3250(
                innerLeft = emptyList(),
                innerRight = emptyList(),
                outerLeft = emptyList(),
                outerRight = emptyList()
            )
        }

        val seedRow = buildInnerSeedRow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            vScoreU8 = vScoreU8,
            w = w,
            h = h,
            seedXInside = seedXInside,
            yRef = yRef.coerceIn(y0Inner, y1Inner),
            minDist = 4,
            maxDist = ((w * 0.42f).roundToInt()).coerceAtLeast(24),
            bandHalf = 3,
            profile3250 = profile3250
        )
            ?: return SideArcCandidatesLocal3250(
            innerLeft = emptyList(),
            innerRight = emptyList(),
            outerLeft = emptyList(),
            outerRight = emptyList()
        )

        val innerLeftRaw = traceInnerPolylineFromSeedRow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedRow.leftX,
            seedY = seedRow.y,
            centerInsideSeedX = seedXInside,
            sideSign = -1,
            y0 = y0Inner,
            y1 = y1Inner,
            stepY = stepY,
            profile3250 = profile3250
        )

        val innerRightRaw = traceInnerPolylineFromSeedRow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            seedX = seedRow.rightX,
            seedY = seedRow.y,
            centerInsideSeedX = seedXInside,
            sideSign = +1,
            y0 = y0Inner,
            y1 = y1Inner,
            stepY = stepY,
            profile3250 = profile3250
        )

        val maxJumpY = max(stepY * 2, 6).toFloat()

        val innerLeftSmoothed = smoothSidePolyline3250(
            fillSmallSideGaps3250(
                pts = innerLeftRaw,
                stepY = stepY,
                maxGapRows = 1,          // antes 2
                maxInterpJumpPx = 10f    // antes 18f
            )
        )

        val innerRightSmoothed = smoothSidePolyline3250(
            fillSmallSideGaps3250(
                pts = innerRightRaw,
                stepY = stepY,
                maxGapRows = 1,
                maxInterpJumpPx = 10f
            )
        )

        val innerLeft = keepBestContinuousRunByYStrict3250(
            pts = innerLeftSmoothed,
            maxJumpX = 9f,
            maxJumpY = maxJumpY,
            minRunPoints = 10,
            maxTotalSecondaryDriftPx = 22f
        )

        val innerRight = keepBestContinuousRunByYStrict3250(
            pts = innerRightSmoothed,
            maxJumpX = 9f,
            maxJumpY = maxJumpY,
            minRunPoints = 10,
            maxTotalSecondaryDriftPx = 22f
        )

        val outerLeftRaw = if (profile3250 == RimProfile3250.PERFORADO || innerLeft.isEmpty()) {
            emptyList()
        } else {
            traceOuterFromInnerGuide3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                innerGuide = innerLeft,
                y0 = y0Inner,
                y1 = y1Inner,
                stepY = stepY,
                outwardSign = -1,
                minOuterGapPx = minOuterGapPx,
                maxOuterGapPx = maxOuterGapPx,
                profile3250 = profile3250
            )
        }

        val outerRightRaw = if (profile3250 == RimProfile3250.PERFORADO || innerRight.isEmpty()) {
            emptyList()
        } else {
            traceOuterFromInnerGuide3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                innerGuide = innerRight,
                y0 = y0Inner,
                y1 = y1Inner,
                stepY = stepY,
                outwardSign = +1,
                minOuterGapPx = minOuterGapPx,
                maxOuterGapPx = maxOuterGapPx,
                profile3250 = profile3250
            )
        }

        val outerLeft = keepBestContinuousRunByYStrict3250(
            pts = smoothSidePolyline3250(
                fillSmallSideGaps3250(
                    pts = outerLeftRaw,
                    stepY = stepY,
                    maxGapRows = 1,
                    maxInterpJumpPx = 12f
                )
            ),
            maxJumpX = 12f,
            maxJumpY = maxJumpY,
            minRunPoints = if (profile3250 == RimProfile3250.FULL_RIM) 6 else 4,
            maxTotalSecondaryDriftPx = 28f
        )

        val outerRight = keepBestContinuousRunByYStrict3250(
            pts = smoothSidePolyline3250(
                fillSmallSideGaps3250(
                    pts = outerRightRaw,
                    stepY = stepY,
                    maxGapRows = 1,
                    maxInterpJumpPx = 12f
                )
            ),
            maxJumpX = 12f,
            maxJumpY = maxJumpY,
            minRunPoints = if (profile3250 == RimProfile3250.FULL_RIM) 6 else 4,
            maxTotalSecondaryDriftPx = 28f
        )

        return SideArcCandidatesLocal3250(
            innerLeft = innerLeft,
            innerRight = innerRight,
            outerLeft = outerLeft,
            outerRight = outerRight
        )
    }

    private fun keepBestContinuousRunByYStrict3250(
        pts: List<PointF>,
        maxJumpX: Float,
        maxJumpY: Float,
        minRunPoints: Int,
        maxTotalSecondaryDriftPx: Float
    ): List<PointF> {
        if (pts.isEmpty()) return emptyList()
        if (pts.size == 1) return if (minRunPoints <= 1) pts else emptyList()

        val sorted = pts.sortedWith(compareBy<PointF> { it.y }.thenBy { it.x })

        data class Run(val pts: MutableList<PointF> = mutableListOf()) {
            fun ySpan(): Float =
                if (pts.size >= 2) pts.last().y - pts.first().y else 0f

            fun xSpan(): Float {
                if (pts.isEmpty()) return 0f
                var minX = Float.POSITIVE_INFINITY
                var maxX = Float.NEGATIVE_INFINITY
                for (p in pts) {
                    if (p.x < minX) minX = p.x
                    if (p.x > maxX) maxX = p.x
                }
                return maxX - minX
            }

            // serrucho lateral acumulado mientras avanzo sobre Y
            fun totalAbsDx(): Float {
                if (pts.size < 2) return 0f
                var acc = 0f
                for (i in 1 until pts.size) {
                    acc += abs(pts[i].x - pts[i - 1].x)
                }
                return acc
            }

            fun score(): Float {
                return pts.size * 100f +
                        ySpan() * 2f -
                        totalAbsDx() * 1.5f -
                        xSpan() * 0.5f
            }
        }

        val runs = ArrayList<Run>()
        var current = Run()
        current.pts.add(sorted.first())

        for (i in 1 until sorted.size) {
            val prev = sorted[i - 1]
            val now = sorted[i]

            val dy = now.y - prev.y
            val dx = abs(now.x - prev.x)

            val contiguous =
                dy in 0f..maxJumpY &&
                        dx <= maxJumpX

            if (contiguous) {
                current.pts.add(now)
            } else {
                runs.add(current)
                current = Run()
                current.pts.add(now)
            }
        }
        runs.add(current)

        val filtered = runs.filter { run ->
            run.pts.size >= minRunPoints &&
                    run.ySpan() > 0f &&
                    run.totalAbsDx() <= maxTotalSecondaryDriftPx
        }

        val best = filtered.maxByOrNull { it.score() } ?: return emptyList()
        return best.pts
    }
    private fun buildInnerSeedRow3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        vScoreU8: ByteArray? = null,
        w: Int,
        h: Int,
        seedXInside: Int,
        yRef: Int,
        minDist: Int,
        maxDist: Int,
        bandHalf: Int,
        profile3250: RimProfile3250
    ): InnerSeedRow3250? {
        val ySeed = yRef.coerceIn(0, h - 1)

        val pair = findInnerWallsAtY3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            vScoreU8 = vScoreU8,
            w = w,
            h = h,
            seedX = seedXInside,
            yRef = ySeed,
            bandHalf = bandHalf,
            minDist = minDist,
            maxDist = maxDist,
            profile3250 = profile3250
        ) ?: return null

        return InnerSeedRow3250(
            y = ySeed,
            leftX = pair.first,
            rightX = pair.second
        )
    }
    private fun traceInnerPolylineFromSeedRow3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        seedX: Int,
        seedY: Int,
        centerInsideSeedX: Int,
        sideSign: Int,
        y0: Int,
        y1: Int,
        stepY: Int,
        profile3250: RimProfile3250
    ): List<PointF> {

        val up = traceInnerHalf3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            startX = seedX,
            centerInsideSeedX = centerInsideSeedX,
            sideSign = sideSign,
            yStart = seedY - stepY,
            yEndInclusive = y0,
            yStep = -stepY,
            profile3250 = profile3250
        )


        val down = traceInnerHalf3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            startX = seedX,
            centerInsideSeedX = centerInsideSeedX,
            sideSign = sideSign,
            yStart = seedY + stepY,
            yEndInclusive = y1,
            yStep = +stepY,
            profile3250 = profile3250
        )

        val out = ArrayList<PointF>(up.size + 1 + down.size)
        out.addAll(up.reversed())
        out.add(PointF(seedX.toFloat(), seedY.toFloat()))
        out.addAll(down)
        return out.sortedBy { it.y }
    }

    private fun traceInnerHalf3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        vScoreU8: ByteArray? = null,
        w: Int,
        h: Int,
        startX: Int,
        centerInsideSeedX: Int,
        sideSign: Int,
        yStart: Int,
        yEndInclusive: Int,
        yStep: Int,
        profile3250: RimProfile3250
    ): List<PointF> {

        val out = ArrayList<PointF>()
        var prevX = startX.coerceIn(0, w - 1)
        var prevDx = 0
        var missCount = 0

        var y = yStart
        while (if (yStep > 0) y <= yEndInclusive else y >= yEndInclusive) {

            val expectedX = (prevX + prevDx).coerceIn(0, w - 1)
            val radius = when (missCount) {
                0 -> 10
                1 -> 14
                else -> 18
            }

            val xFound = searchNearestInnerAroundPrevX3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                vScoreU8 = vScoreU8,
                w = w,
                h = h,
                y = y,
                prevX = expectedX,
                centerInsideSeedX = centerInsideSeedX,
                sideSign = sideSign,
                searchRadiusPx = radius,
                profile3250 = profile3250
            )

            if (xFound != null) {
                val dxNow = xFound - prevX

                if (abs(dxNow) > 12) {
                    missCount++
                    if (missCount > 1) break
                } else {
                    out.add(PointF(xFound.toFloat(), y.toFloat()))
                    prevDx = dxNow
                    prevX = xFound
                    missCount = 0
                }
            } else {
                missCount++
                if (missCount > 1) break
            }

            y += yStep
        }

        return out
    }
    private fun traceOuterFromInnerGuide3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        innerGuide: List<PointF>,
        y0: Int,
        y1: Int,
        stepY: Int,
        outwardSign: Int,
        minOuterGapPx: Int,
        maxOuterGapPx: Int,
        profile3250: RimProfile3250
    ): List<PointF> {
        if (innerGuide.isEmpty()) return emptyList()
        if (profile3250 != RimProfile3250.FULL_RIM) return emptyList()

        val raw = ArrayList<PointF>()
        var prevGap: Int? = null

        var y = y1
        while (y >= y0) {
            val innerPt = innerGuide.minByOrNull { abs(it.y - y.toFloat()) }
            val innerX = innerPt?.x?.roundToInt()

            if (innerX != null) {
                val xOuter = searchOuterFromInner3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    w = w,
                    h = h,
                    y = y,
                    innerX = innerX,
                    outwardSign = outwardSign,
                    prevGapPx = prevGap,
                    minOuterGapPx = minOuterGapPx,
                    maxOuterGapPx = maxOuterGapPx,
                    profile3250 = profile3250
                )

                if (xOuter != null) {
                    raw.add(PointF(xOuter.toFloat(), y.toFloat()))
                    prevGap = abs(xOuter - innerX)
                }
            }

            y -= stepY
        }

        return raw.sortedBy { it.y }
    }
    private fun fillSmallSideGaps3250(
        pts: List<PointF>,
        stepY: Int,
        maxGapRows: Int,
        maxInterpJumpPx: Float
    ): List<PointF> {
        if (pts.isEmpty()) return emptyList()
        if (pts.size == 1) return pts

        val sorted = pts.sortedBy { it.y }
        val out = ArrayList<PointF>()
        out.add(sorted.first())

        for (i in 1 until sorted.size) {
            val a = sorted[i - 1]
            val b = sorted[i]

            val dy = (b.y - a.y).roundToInt()
            val rowsMissing = dy / stepY - 1

            if (
                rowsMissing in 1..maxGapRows &&
                abs(b.x - a.x) <= maxInterpJumpPx
            ) {
                for (k in 1..rowsMissing) {
                    val t = k.toFloat() / (rowsMissing + 1).toFloat()
                    val x = (1f - t) * a.x + t * b.x
                    val y = a.y + stepY * k
                    out.add(PointF(x, y))
                }
            }

            out.add(b)
        }

        return out
    }

    private fun smoothSidePolyline3250(
        pts: List<PointF>
    ): List<PointF> {
        if (pts.size < 3) return pts

        val sorted = pts.sortedBy { it.y }
        val n = sorted.size
        val medX = FloatArray(n)

        for (i in 0 until n) {
            val xs = ArrayList<Float>(5)
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                xs.add(sorted[j].x)
            }
            xs.sort()
            medX[i] = xs[xs.size / 2]
        }

        val out = ArrayList<PointF>(n)
        for (i in 0 until n) {
            var sum = 0f
            var cnt = 0
            for (k in -2..2) {
                val j = (i + k).coerceIn(0, n - 1)
                sum += medX[j]
                cnt++
            }
            out.add(PointF(sum / cnt.toFloat(), sorted[i].y))
        }

        return out
    }
    private fun localPolylineToGlobal3250(
        pts: List<PointF>,
        roiLeft: Float,
        roiTop: Float
    ): List<PointF>? {
        if (pts.isEmpty()) return null
        return pts.map { p -> PointF(p.x + roiLeft, p.y + roiTop) }
    }

    private fun dist3250(a: PointF, b: PointF): Float {
        val dx = a.x - b.x
        val dy = a.y - b.y
        return kotlin.math.sqrt(dx * dx + dy * dy)
    }

    private fun takeContiguousRunFromBottom3250(
        sidePts: List<PointF>,
        bottomAnchor: PointF,
        maxJoinDistPx: Float = 14f,
        maxStepDistPx: Float = 12f,
        minRunPts: Int = 6
    ): List<PointF> {
        if (sidePts.size < minRunPts) return emptyList()

        // bottom -> top
        val pts = sidePts.sortedByDescending { it.y }

        val startIdx = pts.indices.minByOrNull { i -> dist3250(pts[i], bottomAnchor) } ?: return emptyList()
        if (dist3250(pts[startIdx], bottomAnchor) > maxJoinDistPx) return emptyList()

        val out = ArrayList<PointF>()
        var prev = pts[startIdx]
        out.add(prev)

        for (i in (startIdx + 1) until pts.size) {
            val p = pts[i]

            val dyUp = prev.y - p.y
            if (dyUp <= 0f) break
            if (dyUp > maxStepDistPx) break
            if (dist3250(prev, p) > maxStepDistPx) break

            out.add(p)
            prev = p
        }

        return if (out.size >= minRunPts) out.reversed() else emptyList() // top -> bottom
    }

    private fun nearestIdxOnPolyline3250(poly: List<PointF>, p: PointF): Int {
        return poly.indices.minByOrNull { i -> dist3250(poly[i], p) } ?: 0
    }

    private fun buildOfficialInnerArc3250(
        bottomLocal: List<PointF>,
        topLocal: List<PointF>,
        nasalLocal: List<PointF>,
        templeLocal: List<PointF>,
        maxJoinDistPx: Float = 14f,
        maxStepDistPx: Float = 12f,
        minRunPts: Int = 6
    ): List<PointF> {
        if (bottomLocal.size < 3) return emptyList()
        if (topLocal.size < 3) return emptyList()
        if (nasalLocal.size < minRunPts) return emptyList()
        if (templeLocal.size < minRunPts) return emptyList()

        val bottom = bottomLocal.sortedBy { it.x }
        val top = topLocal.sortedBy { it.x }

        val leftBottom = bottom.first()
        val rightBottom = bottom.last()
        val leftTop = top.first()
        val rightTop = top.last()

        val nasalBottomRef =
            if (dist3250(nasalLocal.maxBy { it.y }, leftBottom) <= dist3250(nasalLocal.maxBy { it.y }, rightBottom)) {
                leftBottom
            } else {
                rightBottom
            }

        val templeBottomRef =
            if (dist3250(templeLocal.maxBy { it.y }, leftBottom) <= dist3250(templeLocal.maxBy { it.y }, rightBottom)) {
                leftBottom
            } else {
                rightBottom
            }

        val nasalTopRef =
            if (dist3250(nasalLocal.minBy { it.y }, leftTop) <= dist3250(nasalLocal.minBy { it.y }, rightTop)) {
                leftTop
            } else {
                rightTop
            }

        val templeTopRef =
            if (dist3250(templeLocal.minBy { it.y }, leftTop) <= dist3250(templeLocal.minBy { it.y }, rightTop)) {
                leftTop
            } else {
                rightTop
            }

        val nasalRun = takeContiguousRunFromBottom3250(
            sidePts = nasalLocal,
            bottomAnchor = nasalBottomRef,
            maxJoinDistPx = maxJoinDistPx,
            maxStepDistPx = maxStepDistPx,
            minRunPts = minRunPts
        )

        val templeRun = takeContiguousRunFromBottom3250(
            sidePts = templeLocal,
            bottomAnchor = templeBottomRef,
            maxJoinDistPx = maxJoinDistPx,
            maxStepDistPx = maxStepDistPx,
            minRunPts = minRunPts
        )

        if (nasalRun.isEmpty() || templeRun.isEmpty()) return emptyList()

        val nasalTopPt = nasalRun.first()
        val templeTopPt = templeRun.first()
        val nasalBottomPt = nasalRun.last()
        val templeBottomPt = templeRun.last()

        if (
            dist3250(nasalTopPt, nasalTopRef) > maxJoinDistPx ||
            dist3250(templeTopPt, templeTopRef) > maxJoinDistPx
        ) {
            return emptyList()
        }

        val iTop0 = nearestIdxOnPolyline3250(top, nasalTopPt)
        val iTop1 = nearestIdxOnPolyline3250(top, templeTopPt)

        val topSegment =
            if (iTop0 <= iTop1) {
                top.subList(iTop0, iTop1 + 1)
            } else {
                top.subList(iTop1, iTop0 + 1).reversed()
            }

        val iBot0 = nearestIdxOnPolyline3250(bottom, nasalBottomPt)
        val iBot1 = nearestIdxOnPolyline3250(bottom, templeBottomPt)

        val bottomSegment =
            if (iBot0 <= iBot1) {
                bottom.subList(iBot0, iBot1 + 1)
            } else {
                bottom.subList(iBot1, iBot0 + 1).reversed()
            }

        val out = ArrayList<PointF>(
            nasalRun.size + topSegment.size + templeRun.size + bottomSegment.size
        )

        out.addAll(nasalRun)
        out.addAll(topSegment.drop(1))
        out.addAll(templeRun.asReversed().drop(1))
        out.addAll(bottomSegment.drop(1))

        return out
    }
    private fun keepPolylineNearXRef3250(
        pts: List<PointF>,
        xRef: Float,
        sideSign: Int,
        tolTowardCenterPx: Float = 8f,
        tolOutwardPx: Float = 26f,
        minPts: Int = 6
    ): List<PointF> {
        if (pts.isEmpty()) return emptyList()

        val out = pts.filter { p ->
            val dx = p.x - xRef
            if (sideSign < 0) {
                dx <= tolTowardCenterPx && dx >= -tolOutwardPx
            } else {
                dx >= -tolTowardCenterPx && dx <= tolOutwardPx
            }
        }

        return if (out.size >= minPts) out else emptyList()
    }
    private fun norm013250(v: Float, lo: Float, hi: Float): Float {
        if (hi <= lo) return 0f
        return ((v - lo) / (hi - lo)).coerceIn(0f, 1f)
    }

    private fun keepBestBottomRunByXStrict3250(
        poly: List<Pair<Int, Int>>,
        maxJumpX: Int,
        maxJumpY: Int,
        minRunPoints: Int,
        maxTotalSecondaryDriftPx: Int
    ): List<Pair<Int, Int>> {
        if (poly.isEmpty()) return emptyList()
        if (poly.size == 1) return if (minRunPoints <= 1) poly else emptyList()

        val sorted = poly
            .filter { it.second >= 0 }
            .sortedWith(compareBy<Pair<Int, Int>> { it.first }.thenBy { it.second })

        if (sorted.isEmpty()) return emptyList()
        if (sorted.size == 1) return if (minRunPoints <= 1) sorted else emptyList()

        data class Run(val pts: MutableList<Pair<Int, Int>> = mutableListOf()) {
            fun xSpan(): Int =
                if (pts.size >= 2) pts.last().first - pts.first().first else 0

            fun ySpan(): Int {
                if (pts.isEmpty()) return 0
                var minY = Int.MAX_VALUE
                var maxY = Int.MIN_VALUE
                for ((_, y) in pts) {
                    if (y < minY) minY = y
                    if (y > maxY) maxY = y
                }
                return maxY - minY
            }

            fun totalAbsDy(): Int {
                if (pts.size < 2) return 0
                var acc = 0
                for (i in 1 until pts.size) {
                    acc += abs(pts[i].second - pts[i - 1].second)
                }
                return acc
            }

            fun score(): Float {
                return pts.size * 100f +
                        xSpan() * 2f -
                        totalAbsDy() * 1.5f -
                        ySpan() * 0.5f
            }
        }

        val runs = ArrayList<Run>()
        var current = Run()
        current.pts.add(sorted.first())

        for (i in 1 until sorted.size) {
            val prev = sorted[i - 1]
            val now = sorted[i]

            val dx = now.first - prev.first
            val dy = abs(now.second - prev.second)

            val contiguous =
                dx in 0..maxJumpX &&
                        dy <= maxJumpY

            if (contiguous) {
                current.pts.add(now)
            } else {
                runs.add(current)
                current = Run()
                current.pts.add(now)
            }
        }
        runs.add(current)

        val filtered = runs.filter { run ->
            run.pts.size >= minRunPoints &&
                    run.xSpan() > 0 &&
                    run.totalAbsDy() <= maxTotalSecondaryDriftPx
        }

        val best = filtered.maxByOrNull { it.score() } ?: return emptyList()
        return best.pts
    }
    fun detectRimAutoProfile3250(
        edgesU8: ByteArray,
        dirU8: ByteArray? = null,
        maskU8: ByteArray? = null,
        w: Int,
        h: Int,
        hScoreU8: ByteArray? = null,
        vScoreU8: ByteArray? = null,
        roiGlobal: RectF,
        midlineXpx: Float,
        browBottomYpx: Float?,
        filHboxMm: Double,
        filVboxMm: Double?,
        filOverInnerMmPerSide3250: Double,
        bridgeRowYpxGlobal: Float?,
        pupilGlobal: PointF?,
        debugTag: String,
        filPtsMm3250: List<PointF>? = null,
        pxPerMmGuessFace: Float? = null

    ): RimDetectProfilePick3250? {

        val profiles = listOf(
            RimProfile3250.FULL_RIM,
            RimProfile3250.RANURADO,
            RimProfile3250.PERFORADO
        )

        var best: RimDetectProfilePick3250? = null

        for (profile in profiles) {
            val pack = detectRim(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                hScoreU8 = hScoreU8,
                vScoreU8 = vScoreU8,
                roiGlobal = roiGlobal,
                midlineXpx = midlineXpx,
                browBottomYpx = browBottomYpx,
                filHboxMm = filHboxMm,
                filVboxMm = filVboxMm,
                filOverInnerMmPerSide3250 = filOverInnerMmPerSide3250,
                profile3250 = profile,
                bridgeRowYpxGlobal = bridgeRowYpxGlobal,
                pupilGlobal = pupilGlobal,
                filPtsMm3250 = filPtsMm3250,
                debugTag = "$debugTag/$profile",
                pxPerMmGuessFace = pxPerMmGuessFace
            ) ?: continue

            val score = pack.result.confidence

            if (best == null || score > best.score3250) {
                best = RimDetectProfilePick3250(
                    profile3250 = profile,
                    pack3250 = pack,
                    score3250 = score
                )
            }
        }

        return best
    }

    private fun pickTopInsideOut3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        hScoreU8: ByteArray? = null,
        w: Int,
        h: Int,
        leftX: Int,
        rightX: Int,
        ySeed: Int,
        yMin: Int,
        stepX: Int,
        minHits: Int,
        minCoverage: Float,
        contJumpPx: Int,
        expectedTopY: Int?,
        expectedTolPx: Int
    ): TopPick3250? {

        val xa = (leftX + 8).coerceIn(0, w - 1)
        val xb = (rightX - 8).coerceIn(0, w - 1)
        if (xb - xa < 50) return null

        val yLo = yMin.coerceIn(0, h - 1)
        val yHi = ySeed.coerceIn(0, h - 1)
        if (yHi - yLo < 10) return null

        val xMid = ((leftX + rightX) / 2).coerceIn(xa, xb)
        val ySeed0 = ySeed.coerceIn(yLo, yHi)
        val tol = expectedTolPx.coerceAtLeast(12)

        fun findSeedAtX(x: Int): Int {
            val x0 = (x - stepX * 2).coerceIn(xa, xb)
            val x1 = (x + stepX * 2).coerceIn(xa, xb)

            val y0Soft = expectedTopY?.let { (it - tol).coerceIn(yLo, yHi) } ?: yLo
            val y1Soft = expectedTopY?.let { (it + tol).coerceIn(yLo, yHi) } ?: yHi

            var y = findBestTopEdgeInWindow3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                hScoreU8 = hScoreU8,
                w = w,
                a = x0,
                b = x1,
                y0 = y0Soft,
                y1 = min(y1Soft, ySeed0),
                expectedY = expectedTopY ?: y0Soft
            )

            if (y < 0) {
                y = findBestTopEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    hScoreU8 = hScoreU8,
                    w = w,
                    a = x0,
                    b = x1,
                    y0 = yLo,
                    y1 = ySeed0,
                    expectedY = expectedTopY ?: yLo
                )
            }

            if (y < 0) {
                val yCenter = expectedTopY ?: ySeed0
                val y0Wide = (yCenter - contJumpPx * 5).coerceIn(yLo, yHi)
                val y1Wide = (yCenter + contJumpPx * 3).coerceIn(yLo, yHi)

                y = findBestTopEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    hScoreU8 = hScoreU8,
                    w = w,
                    a = x0,
                    b = x1,
                    y0 = y0Wide,
                    y1 = y1Wide,
                    expectedY = expectedTopY ?: yCenter
                )
            }

            return y
        }
        data class SeedTop(val x: Int, val y: Int)

        var seedTop: SeedTop? = null

        run {
            val y0 = findSeedAtX(xMid)
            if (y0 >= 0) {
                seedTop = SeedTop(xMid, y0)
                return@run
            }

            var d = stepX
            while (xMid - d >= xa || xMid + d <= xb) {
                val xl = xMid - d
                if (xl in xa..xb) {
                    val yl = findSeedAtX(xl)
                    if (yl >= 0) {
                        seedTop = SeedTop(xl, yl)
                        return@run
                    }
                }

                val xr = xMid + d
                if (xr in xa..xb) {
                    val yr = findSeedAtX(xr)
                    if (yr >= 0) {
                        seedTop = SeedTop(xr, yr)
                        return@run
                    }
                }

                d += stepX
            }
        }

        val seedTopX = seedTop?.x ?: return null
        val seedTopY = seedTop.y

        fun traceHalf(startX: Int, stepSign: Int): List<Pair<Int, Int>> {
            val out = ArrayList<Pair<Int, Int>>()
            var prevY = seedTopY
            var x = startX
            var missCount = 0

            while (x in xa..xb) {
                val winPad = (contJumpPx + missCount * 2).coerceAtMost(contJumpPx * 3)
                val yWin0 = (prevY - winPad).coerceIn(yLo, yHi)
                val yWin1 = (prevY + winPad).coerceIn(yLo, yHi)

                var bestY = findBestTopEdgeInWindow3250(
                    edgesU8 = edgesU8,
                    dirU8 = dirU8,
                    maskU8 = maskU8,
                    hScoreU8 = hScoreU8,
                    w = w,
                    a = xa,
                    b = xb,
                    y0 = yWin0,
                    y1 = yWin1,
                    expectedY = prevY
                )

                if (bestY < 0 && expectedTopY != null) {
                    val ySoft0 = (expectedTopY - tol).coerceIn(yLo, yHi)
                    val ySoft1 = (expectedTopY + tol).coerceIn(yLo, yHi)

                    bestY = findBestTopEdgeInWindow3250(
                        edgesU8 = edgesU8,
                        dirU8 = dirU8,
                        maskU8 = maskU8,
                        hScoreU8 = hScoreU8,
                        w = w,
                        a = xa,
                        b = xb,
                        y0 = ySoft0,
                        y1 = ySoft1,
                        expectedY = expectedTopY
                    )
                }

                if (bestY < 0) {
                    val yWide0 = (prevY - winPad * 2).coerceIn(yLo, yHi)
                    val yWide1 = (prevY + winPad * 2).coerceIn(yLo, yHi)

                    bestY = findBestTopEdgeInWindow3250(
                        edgesU8 = edgesU8,
                        dirU8 = dirU8,
                        maskU8 = maskU8,
                        hScoreU8 = hScoreU8,
                        w = w,
                        a = xa,
                        b = xb,
                        y0 = yWide0,
                        y1 = yWide1,
                        expectedY = prevY
                    )
                }

                if (bestY >= 0) {
                    out.add(x to bestY)
                    prevY = bestY
                    missCount = 0
                } else {
                    out.add(x to -1)
                    missCount++
                }

                x += stepSign * stepX
            }

            return out
        }

        val leftHalf = traceHalf(seedTopX, -1).drop(1)
        val rightHalf = traceHalf(seedTopX, +1)

        val polyRaw = ArrayList<Pair<Int, Int>>(leftHalf.size + rightHalf.size)
        polyRaw.addAll(leftHalf.asReversed())
        polyRaw.addAll(rightHalf)

        val polyFilled = fillSmallBottomGaps3250(
            poly = polyRaw,
            stepX = stepX,
            maxGapCols = 2,
            maxInterpJumpPx = contJumpPx * 2
        )

        val polySmooth = smoothBottomPolyline3250(polyFilled)

        val poly = keepBestBottomRunByXStrict3250(
            poly = polySmooth,
            maxJumpX = stepX * 2,
            maxJumpY = contJumpPx * 2,
            minRunPoints = minHits,
            maxTotalSecondaryDriftPx = 120
        )

        if (poly.size < minHits) return null

        val xSpan = (poly.last().first - poly.first().first).coerceAtLeast(0)
        val expectedSamples = (xSpan / stepX + 1).coerceAtLeast(1)
        val coverage = poly.size.toFloat() / expectedSamples.toFloat()

        var contSamples = 0
        var contHits = 0
        for (i in 1 until poly.size) {
            contSamples++
            if (abs(poly[i].second - poly[i - 1].second) <= contJumpPx * 2) {
                contHits++
            }
        }

        val continuity = if (contSamples > 0) {
            contHits.toFloat() / contSamples.toFloat()
        } else {
            0f
        }

        if (coverage < minCoverage) return null

        val ys = poly.map { it.second }.sorted()
        val yMed = ys[ys.size / 2]

        val yMin = poly.minOf { it.second }

        val tolY = 3
        val nearTop = poly
            .sortedBy { it.first }
            .filter { it.second <= yMin + tolY }

        val yTopCoherent = if (nearTop.isNotEmpty()) {
            nearTop.map { it.second }.average().toFloat().toInt()
        } else {
            yMin
        }
        val confCoverage =
            norm013250(coverage, 0.25f, 0.80f)

        val confContinuity =
            norm013250(continuity, 0.35f, 0.90f)

        val confExpected =
            if (expectedTopY != null) {
                val dy = abs(yTopCoherent - expectedTopY).toFloat()
                (1f - (dy / expectedTolPx.toFloat().coerceAtLeast(1f))).coerceIn(0f, 1f)
            } else {
                0.5f
            }

        val confidence =
            (
                    0.35f * confCoverage +
                            0.30f * confContinuity +
                            0.35f * confExpected
                    ).coerceIn(0f, 1f)

        return TopPick3250(
            yMed = yMed,
            yMin = yMin,
            yTopCoherent = yTopCoherent,
            poly = poly,
            coverage = coverage,
            continuity = continuity,
            confidence = confidence
        )
    }
    private fun pairPolylineToGlobal3250(
        pts: List<Pair<Int, Int>>,
        roiLeft: Float,
        roiTop: Float
    ): List<PointF>? {
        if (pts.isEmpty()) return null

        return pts.map { (x, y) ->
            PointF(
                x.toFloat() + roiLeft,
                y.toFloat() + roiTop
            )
        }
    }
    private fun buildNormalCandidate3250(
        debugTag: String,
        scale: Float,
        ratioPenalty: Float,
        probeYLocal: Int,
        vboxInnerMmF: Float,
        pxPerMmX: Float,
        bottomPick: ArcPick,
        topPick: TopPick3250?,
        topSearchMinY: Int,
        topExpectedY: Int?,
        h: Int
    ): ScaleCandidate3250?{

        val bottomY = bottomPick.yMax

        val yCap = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)
        val expHpx = if (vboxInnerMmF > 1e-3f) {
            vboxInnerMmF * pxPerMmX
        } else {
            Float.NaN
        }

        val topEstimatedY = when {
            topExpectedY != null -> {
                topExpectedY.coerceIn(topSearchMinY, yCap)
            }

            expHpx.isFinite() && expHpx > 60f -> {
                (bottomY - expHpx).roundToInt().coerceIn(topSearchMinY, yCap)
            }

            else -> {
                topSearchMinY.coerceAtMost(yCap)
            }
        }

        val topObservedY =
            when {
                topPick == null -> null
                topPick.yTopCoherent > 0 -> topPick.yTopCoherent
                topPick.yMed > 0 -> topPick.yMed
                else -> null
            }

        val topConfidence =
            if (topObservedY != null && topPick != null) {
                topPick.confidence.coerceIn(0f, 1f)
            } else {
                0f
            }

        val topPoly =
            if (topObservedY != null && topPick != null) {
                topPick.poly
            } else {
                emptyList()
            }

        val topUsedY =
            (topObservedY ?: topEstimatedY)
                .coerceIn(topSearchMinY, yCap)

        val innerH = (bottomY - topUsedY).coerceAtLeast(1)
        val minH0 = max(70, (h * 0.18f).roundToInt())
        val maxH0 = (h * 0.92f).roundToInt()
        if (innerH !in minH0..maxH0) {
            return null
        }

        val confCoverage =
            norm013250(bottomPick.coverage, 0.55f, 0.95f)

        val confContinuity =
            norm013250(bottomPick.continuity, 0.65f, 0.98f)

        val confSamples =
            norm013250(bottomPick.poly.size.toFloat(), 28f, 60f)

        val confTop =
            norm013250(topConfidence, 0.20f, 0.80f)

        var conf =
            (
                    0.40f * confCoverage +
                            0.25f * confContinuity +
                            0.10f * confSamples +
                            0.25f * confTop
                    ).coerceIn(0f, 1f)

        conf *=
            (0.75f + 0.25f * ratioPenalty).coerceIn(0.75f, 1.00f)

        conf =
            conf.coerceIn(0f, 1f)

        if (DBG) {
            Log.d(
                TAG,
                "RIMCONF[$debugTag] S=${f23250(scale)} " +
                        "topObs=${topObservedY ?: -1} " +
                        "topEst=$topEstimatedY " +
                        "topUsed=$topUsedY " +
                        "topConf=${f33250(topConfidence)} " +
                        "topPts=${topPoly.size} " +
                        "cov=${f33250(bottomPick.coverage)} " +
                        "cont=${f33250(bottomPick.continuity)} " +
                        "n=${bottomPick.poly.size} " +
                        "ratioPen=${f33250(ratioPenalty)} " +
                        "conf=${f33250(conf)}"
            )

        }

        return ScaleCandidate3250(
            conf = conf,
            left = -1,
            right = -1,
            top = topUsedY,
            bottom = bottomY,
            refY = probeYLocal,
            seedX = -1,
            scale = scale,
            bottomPoly = bottomPick.poly,
            topPoly = topPoly,
            topObservedY = topObservedY,
            topConfidence = topConfidence,
            ratioPenalty = ratioPenalty,
            expHGuessPx = if (expHpx.isFinite() && expHpx > 0f) expHpx else innerH.toFloat(),
            isPartial = false
        )
    }

    private fun buildPartialCandidate3250(
        debugTag: String,
        scale: Float,
        ratioPenalty: Float,
        probeYLocal: Int,
        topPick: TopPick3250?,
        bottomPick: ArcPick?,
        profile3250: RimProfile3250,
        expHpxGuess: Float,
        expWpx: Float,
        innerW: Int,
        topSearchMinY: Int,
        topExpectedY: Int?,
        h: Int
    ): ScaleCandidate3250?
    {

        val yCapPartial = (probeYLocal - TOP_SEARCH_PAD).coerceIn(0, h - 1)
        val topEstimatedY = when {
            topExpectedY != null -> {
                topExpectedY.coerceIn(topSearchMinY, yCapPartial)
            }

            else -> {
                topSearchMinY.coerceAtMost(yCapPartial)
            }
        }
        val topObservedY =
            when {
                topPick == null -> null
                topPick.yTopCoherent > 0 -> topPick.yTopCoherent
                topPick.yMed > 0 -> topPick.yMed
                else -> null
            }

        val topConfidence =
            if (topObservedY != null && topPick != null) {
                topPick.confidence.coerceIn(0f, 1f)
            } else {
                0f
            }

        val topPoly =
            if (topObservedY != null && topPick != null) {
                topPick.poly
            } else {
                emptyList()
            }

        val topUsedY =
            (topObservedY ?: topEstimatedY)
                .coerceIn(topSearchMinY, yCapPartial)

        val minBottomGapPx = when (profile3250) {
            RimProfile3250.FULL_RIM ->
                max(10, (0.08f * expWpx).roundToInt())

            RimProfile3250.RANURADO ->
                max(8, (0.06f * expWpx).roundToInt())

            RimProfile3250.PERFORADO ->
                max(6, (0.05f * expWpx).roundToInt())
        }

        val bottomObservedY =
            when {
                bottomPick == null -> null
                bottomPick.yBottomCoherent > 0 -> bottomPick.yBottomCoherent
                bottomPick.yMed > 0 -> bottomPick.yMed
                else -> null
            }

        val bottomEstimatedY =
            if (expHpxGuess.isFinite() && expHpxGuess > 60f) {
                (topUsedY + expHpxGuess.roundToInt())
                    .coerceIn(topUsedY + minBottomGapPx, h - 1)
            } else {
                (probeYLocal + (0.70f * expWpx).roundToInt())
                    .coerceIn(topUsedY + minBottomGapPx, h - 1)
            }

        val bottomUsedY =
            (bottomObservedY ?: bottomEstimatedY)
                .coerceIn(topUsedY + minBottomGapPx, h - 1)

        if (bottomUsedY <= topUsedY) {
            return null
        }

        fun clamp01(v: Float): Float = v.coerceIn(0f, 1f)

        val confTop =
            norm013250(topConfidence, 0.20f, 0.80f)

        val confWidthRatio =
            clamp01(((innerW.toFloat() / expWpx) - 0.70f) / 0.60f)

        val confInnerW =
            clamp01((innerW - 60f) / 180f)

        val confBottom =
            if (bottomObservedY != null) 1f else 0f

        val partialConf =
            (
                    0.25f * ratioPenalty +
                            0.20f * confWidthRatio +
                            0.15f * confInnerW +
                            0.15f * confTop +
                            0.25f * confBottom
                    ).coerceIn(0f, 0.90f)

        if (DBG) {
            Log.d(
                TAG,
                "RIMDBG[$debugTag] S=${f23250(scale)} PARTIAL_OK " +
                        "topObs=${topObservedY ?: -1} " +
                        "topEst=${topEstimatedY} " +
                        "topUsed=${topUsedY} " +
                        "botObs=${bottomObservedY} ?: -1} " +
                        "botEst=${bottomEstimatedY} " +
                        "botUsed=${bottomUsedY} " +
                        "conf=${f33250(partialConf)}"
            )
        }
        return ScaleCandidate3250(
        conf = partialConf,
        left = -1,
        right = -1,
        top = topUsedY,
        bottom = bottomUsedY,
        refY = probeYLocal,
        seedX = -1,
        scale = scale,
        bottomPoly = bottomPick?.poly ?: emptyList(),
        topPoly = topPoly,
        topObservedY = topObservedY,
        topConfidence = topConfidence,
        ratioPenalty = ratioPenalty,
        expHGuessPx = expHpxGuess,
        isPartial = true
    )
    }
    private fun buildVerticalGuidePoly3250(
        x: Int,
        yTop: Int,
        yBottom: Int,
        h: Int,
        marginTop: Int = 6,
        marginBottom: Int = 6,
        step: Int = 4
    ): List<Pair<Int, Int>> {
        val y0 = (yTop + marginTop).coerceIn(0, h - 1)
        val y1 = (yBottom - marginBottom).coerceIn(0, h - 1)
        if (y1 <= y0) return emptyList()

        val out = ArrayList<Pair<Int, Int>>()
        var y = y0
        while (y <= y1) {
            out.add(x to y)
            y += step
        }
        if (out.isEmpty() || out.last().second != y1) {
            out.add(x to y1)
        }
        return out
    }
    private data class FilExpectedY3250(
        val topY: Int,
        val bottomY: Int
    )

    private fun expectedTopBottomFromFilAtSeed3250(
        filPtsMm: List<PointF>?,
        probeYLocal: Int,
        pxPerMmX: Float
    ): FilExpectedY3250? {
        if (filPtsMm == null || filPtsMm.size < 601) return null
        if (!pxPerMmX.isFinite() || pxPerMmX <= 0f) return null

        var minX = Float.POSITIVE_INFINITY
        var minY = Float.POSITIVE_INFINITY
        var maxX = Float.NEGATIVE_INFINITY
        var maxY = Float.NEGATIVE_INFINITY

        for (p in filPtsMm) {
            if (p.x < minX) minX = p.x
            if (p.y < minY) minY = p.y
            if (p.x > maxX) maxX = p.x
            if (p.y > maxY) maxY = p.y
        }

        val cyMm = (minY + maxY) * 0.5f

        val r201 = filPtsMm[200]
        val r601 = filPtsMm[600]

        val topY = (probeYLocal - (r201.y - cyMm) * pxPerMmX).roundToInt()
        val bottomY = (probeYLocal - (r601.y - cyMm) * pxPerMmX).roundToInt()

        return FilExpectedY3250(
            topY = topY,
            bottomY = bottomY
        )
    }
    private fun f13250(x: Float): String = String.format(Locale.US, "%.1f", x)
    private fun f23250(x: Float): String = String.format(Locale.US, "%.2f", x)
    private fun f33250(x: Float): String = String.format(Locale.US, "%.3f", x)
}

