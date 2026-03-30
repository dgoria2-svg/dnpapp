@file:Suppress("SameParameterValue")

package com.dg.precaldnp.vision

import android.graphics.Bitmap
import android.graphics.PointF
import android.graphics.RectF
import kotlin.math.abs
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sqrt

object FilBoundaryProbe3250 {

    enum class Side3250 { OD, OI }
    enum class Sector3250 { BOTTOM, NASAL, TEMPORAL, TOP, LATERAL }

    data class ProbeInput3250(
        val stillBmp: Bitmap,
        val filPtsGlobal: List<PointF>,
        val pupilGlobal: PointF,
        val midlineXpx: Float,
        val side3250: Side3250,
        val eyeBrowMaskFullU8: ByteArray? = null,   // mask != 0 => veto
        val edgeScoreFullU8: ByteArray? = null,     // opcional: EDGE_SCORE_* o similar
        val edgeBinFullU8: ByteArray? = null,       // opcional: EDGE_FULL_*
        val roiGlobal: RectF? = null,               // opcional: para acotar validación
        val insideTolPx: Float = 2.0f,
        val outsideTolPx: Float = 10.0f,
        val stepPx: Float = 1.0f,
        val sampleHalfPx: Float = 1.5f,
        val continuityMaxJumpPx: Float = 5.0f,
        val minRunLen3250: Int = 8,
        val topPenalty3250: Float = 0.35f,
        val bottomBonus3250: Float = 0.35f,
        val lateralBonus3250: Float = 0.12f,
        val nasalBonus3250: Float = 0.16f,
        val temporalBonus3250: Float = 0.10f,
        val maskPenalty3250: Float = 1.25f,
        val crossMidPenalty3250: Float = 1.25f,
        val roiOutPenalty3250: Float = 0.70f,
        val minAcceptScore3250: Float = 0.18f
    )

    data class ProbeSample3250(
        val idx: Int,
        val basePt: PointF,
        val hitPt: PointF?,
        val tangent: PointF,
        val outwardNormal: PointF,
        val offsetPx: Float,
        val score: Float,
        val valid: Boolean,
        val sector3250: Sector3250,
        val crossedMidline: Boolean,
        val hitMasked: Boolean,
        val outOfRoi: Boolean,
        val edgeScore01: Float,
        val edgeBin01: Float,
        val contrast01: Float,
        val grad01: Float,
        val sectorWeight: Float
    )

    data class ProbeResult3250(
        val samples: List<ProbeSample3250>,
        val supportedPts: List<PointF>,
        val supportedIdx: IntArray,
        val supportFrac: Float,
        val supportBottomFrac: Float,
        val supportNasalFrac: Float,
        val supportTemporalFrac: Float,
        val supportTopFrac: Float,
        val meanScore: Float,
        val meanOffsetPx: Float,
        val smoothedPts: List<PointF>
    )

    fun probeBoundary3250(inp: ProbeInput3250): ProbeResult3250 {
        val w = inp.stillBmp.width
        val h = inp.stillBmp.height
        val n = w * h

        require(w > 0 && h > 0) { "Bitmap vacío" }
        require(inp.filPtsGlobal.size >= 8) { "filPtsGlobal insuficiente" }

        if (inp.eyeBrowMaskFullU8 != null) {
            require(inp.eyeBrowMaskFullU8.size == n) { "eyeBrowMaskFullU8 size inválido" }
        }
        if (inp.edgeScoreFullU8 != null) {
            require(inp.edgeScoreFullU8.size == n) { "edgeScoreFullU8 size inválido" }
        }
        if (inp.edgeBinFullU8 != null) {
            require(inp.edgeBinFullU8.size == n) { "edgeBinFullU8 size inválido" }
        }

        val gray = bitmapToGrayU83250(inp.stillBmp)
        val m = inp.filPtsGlobal.size
        val centroid = centroid3250(inp.filPtsGlobal)

        val samples = ArrayList<ProbeSample3250>(m)

        for (i in 0 until m) {
            val prev = inp.filPtsGlobal[(i - 1 + m) % m]
            val curr = inp.filPtsGlobal[i]
            val next = inp.filPtsGlobal[(i + 1) % m]

            val tangent = normalize3250(
                PointF(
                    next.x - prev.x,
                    next.y - prev.y
                )
            )

            val nA = PointF(-tangent.y, tangent.x)
            val nB = PointF(tangent.y, -tangent.x)

            val outward = chooseOutwardNormal3250(
                curr = curr,
                nA = nA,
                nB = nB,
                pupil = inp.pupilGlobal,
                centroid = centroid
            )

            val sector = classifySector3250(
                pt = curr,
                pupil = inp.pupilGlobal,
                midlineX = inp.midlineXpx,
                side = inp.side3250,
                outward = outward
            )

            var bestScore = Float.NEGATIVE_INFINITY
            var bestOffset = 0f
            var bestPt: PointF? = null
            var bestCrossMid = false
            var bestMasked = false
            var bestOutOfRoi = false
            var bestEdgeScore01 = 0f
            var bestEdgeBin01 = 0f
            var bestContrast01 = 0f
            var bestGrad01 = 0f
            var bestSectorWeight = sectorWeight3250(
                sector = sector,
                bottomBonus = inp.bottomBonus3250,
                lateralBonus = inp.lateralBonus3250,
                nasalBonus = inp.nasalBonus3250,
                temporalBonus = inp.temporalBonus3250,
                topPenalty = inp.topPenalty3250
            )

            var d = -inp.insideTolPx
            while (d <= inp.outsideTolPx + 1e-3f) {
                val x = curr.x + outward.x * d
                val y = curr.y + outward.y * d

                val crossMid = crossesMidline3250(x, inp.midlineXpx, inp.side3250)
                val masked = isMasked3250(inp.eyeBrowMaskFullU8, w, h, x, y)
                val outOfRoi = inp.roiGlobal?.contains(x, y)?.not() ?: false

                val edgeScore01 = readU8Norm3250(inp.edgeScoreFullU8, w, h, x, y)
                val edgeBin01 = readU8Norm3250(inp.edgeBinFullU8, w, h, x, y)

                val contrast01 = normalContrast3250(
                    gray = gray,
                    w = w,
                    h = h,
                    x = x,
                    y = y,
                    nx = outward.x,
                    ny = outward.y,
                    halfPx = inp.sampleHalfPx
                )

                val grad01 = normalGradient3250(
                    gray = gray,
                    w = w,
                    h = h,
                    x = x,
                    y = y,
                    nx = outward.x,
                    ny = outward.y
                )

                var score = 0f
                score += 0.36f * edgeScore01
                score += 0.14f * edgeBin01
                score += 0.30f * contrast01
                score += 0.20f * grad01
                score += bestSectorWeight

                if (masked) {
                    score -= inp.maskPenalty3250
                }
                if (crossMid) {
                    score -= inp.crossMidPenalty3250
                }
                if (outOfRoi) {
                    score -= inp.roiOutPenalty3250
                }

                if (score > bestScore) {
                    bestScore = score
                    bestOffset = d
                    bestPt = PointF(x, y)
                    bestCrossMid = crossMid
                    bestMasked = masked
                    bestOutOfRoi = outOfRoi
                    bestEdgeScore01 = edgeScore01
                    bestEdgeBin01 = edgeBin01
                    bestContrast01 = contrast01
                    bestGrad01 = grad01
                }

                d += inp.stepPx
            }

            val valid = bestPt != null &&
                    bestScore >= inp.minAcceptScore3250 &&
                    !bestCrossMid &&
                    !bestMasked &&
                    !bestOutOfRoi

            samples += ProbeSample3250(
                idx = i,
                basePt = curr,
                hitPt = if (valid) bestPt else null,
                tangent = tangent,
                outwardNormal = outward,
                offsetPx = bestOffset,
                score = bestScore,
                valid = valid,
                sector3250 = sector,
                crossedMidline = bestCrossMid,
                hitMasked = bestMasked,
                outOfRoi = bestOutOfRoi,
                edgeScore01 = bestEdgeScore01,
                edgeBin01 = bestEdgeBin01,
                contrast01 = bestContrast01,
                grad01 = bestGrad01,
                sectorWeight = bestSectorWeight
            )
        }

        val continuityFiltered = continuityFilter3250(
            samples = samples,
            maxJumpPx = inp.continuityMaxJumpPx,
            minRunLen = inp.minRunLen3250
        )

        val supportedPts = ArrayList<PointF>()
        val supportedIdx = ArrayList<Int>()

        for (s in continuityFiltered) {
            if (s.valid && s.hitPt != null) {
                supportedPts += s.hitPt
                supportedIdx += s.idx
            }
        }

        val smoothedPts = smoothSupported3250(continuityFiltered)

        val supportFrac = supportedPts.size.toFloat() / samples.size.toFloat().coerceAtLeast(1f)
        val supportBottomFrac = sectorSupportFrac3250(continuityFiltered, Sector3250.BOTTOM)
        val supportNasalFrac = sectorSupportFrac3250(continuityFiltered, Sector3250.NASAL)
        val supportTemporalFrac = sectorSupportFrac3250(continuityFiltered, Sector3250.TEMPORAL)
        val supportTopFrac = sectorSupportFrac3250(continuityFiltered, Sector3250.TOP)

        var sumScore = 0f
        var sumOffset = 0f
        var cnt = 0
        for (s in continuityFiltered) {
            if (s.valid) {
                sumScore += s.score
                sumOffset += s.offsetPx
                cnt++
            }
        }

        return ProbeResult3250(
            samples = continuityFiltered,
            supportedPts = supportedPts,
            supportedIdx = supportedIdx.toIntArray(),
            supportFrac = supportFrac,
            supportBottomFrac = supportBottomFrac,
            supportNasalFrac = supportNasalFrac,
            supportTemporalFrac = supportTemporalFrac,
            supportTopFrac = supportTopFrac,
            meanScore = if (cnt > 0) sumScore / cnt else 0f,
            meanOffsetPx = if (cnt > 0) sumOffset / cnt else 0f,
            smoothedPts = smoothedPts
        )
    }

    private fun continuityFilter3250(
        samples: List<ProbeSample3250>,
        maxJumpPx: Float,
        minRunLen: Int
    ): List<ProbeSample3250> {
        if (samples.isEmpty()) {
            return samples
        }

        val out = samples.toMutableList()
        val n = out.size

        fun dist(a: PointF?, b: PointF?): Float {
            if (a == null || b == null) return Float.POSITIVE_INFINITY
            val dx = a.x - b.x
            val dy = a.y - b.y
            return sqrt(dx * dx + dy * dy)
        }

        val keep = BooleanArray(n) { false }
        var runStart = -1

        for (i in 0 until n) {
            val curr = out[i]
            val prev = out[(i - 1 + n) % n]

            val continues = curr.valid &&
                    curr.hitPt != null &&
                    prev.valid &&
                    prev.hitPt != null &&
                    dist(curr.hitPt, prev.hitPt) <= maxJumpPx

            if (curr.valid && runStart < 0) {
                runStart = i
            }

            if (!continues && runStart >= 0) {
                val runEnd = i - 1
                val len = ringRunLength3250(runStart, runEnd, n)
                if (len >= minRunLen) {
                    markRun3250(keep, runStart, runEnd)
                }
                if (curr.valid) {
                    runStart = i
                } else {
                    runStart = -1
                }
            }
        }

        if (runStart >= 0) {
            val last = n - 1
            val len = ringRunLength3250(runStart, last, n)
            if (len >= minRunLen) {
                markRun3250(keep, runStart, last)
            }
        }

        for (i in 0 until n) {
            if (!keep[i]) {
                val s = out[i]
                out[i] = s.copy(valid = false, hitPt = null)
            }
        }

        return out
    }

    private fun smoothSupported3250(samples: List<ProbeSample3250>): List<PointF> {
        val out = ArrayList<PointF>()
        val n = samples.size
        if (n == 0) return out

        for (i in 0 until n) {
            val a = samples[(i - 1 + n) % n]
            val b = samples[i]
            val c = samples[(i + 1) % n]

            if (!(b.valid && b.hitPt != null)) {
                continue
            }

            val pts = ArrayList<PointF>(3)
            if (a.valid && a.hitPt != null) pts += a.hitPt
            pts += b.hitPt
            if (c.valid && c.hitPt != null) pts += c.hitPt

            var sx = 0f
            var sy = 0f
            for (p in pts) {
                sx += p.x
                sy += p.y
            }
            out += PointF(sx / pts.size, sy / pts.size)
        }

        return out
    }

    private fun sectorSupportFrac3250(
        samples: List<ProbeSample3250>,
        sector: Sector3250
    ): Float {
        var total = 0
        var ok = 0
        for (s in samples) {
            if (s.sector3250 == sector) {
                total++
                if (s.valid) ok++
            }
        }
        return if (total > 0) ok.toFloat() / total.toFloat() else 0f
    }

    private fun sectorWeight3250(
        sector: Sector3250,
        bottomBonus: Float,
        lateralBonus: Float,
        nasalBonus: Float,
        temporalBonus: Float,
        topPenalty: Float
    ): Float {
        return when (sector) {
            Sector3250.BOTTOM -> bottomBonus
            Sector3250.NASAL -> nasalBonus
            Sector3250.TEMPORAL -> temporalBonus
            Sector3250.LATERAL -> lateralBonus
            Sector3250.TOP -> -topPenalty
        }
    }

    private fun classifySector3250(
        pt: PointF,
        pupil: PointF,
        midlineX: Float,
        side: Side3250,
        outward: PointF
    ): Sector3250 {
        if (outward.y > 0.55f) return Sector3250.BOTTOM
        if (outward.y < -0.55f) return Sector3250.TOP

        val towardMidline = when (side) {
            Side3250.OD -> pt.x > pupil.x || pt.x > midlineX
            Side3250.OI -> pt.x < pupil.x || pt.x < midlineX
        }

        return if (towardMidline) {
            Sector3250.NASAL
        } else {
            Sector3250.TEMPORAL
        }
    }

    private fun chooseOutwardNormal3250(
        curr: PointF,
        nA: PointF,
        nB: PointF,
        pupil: PointF,
        centroid: PointF
    ): PointF {
        val toPupilX = curr.x - pupil.x
        val toPupilY = curr.y - pupil.y
        val sA = nA.x * toPupilX + nA.y * toPupilY
        val sB = nB.x * toPupilX + nB.y * toPupilY

        if (abs(sA - sB) > 1e-3f) {
            return if (sA >= sB) nA else nB
        }

        val toCentX = curr.x - centroid.x
        val toCentY = curr.y - centroid.y
        val cA = nA.x * toCentX + nA.y * toCentY
        val cB = nB.x * toCentX + nB.y * toCentY
        return if (cA >= cB) nA else nB
    }

    private fun crossesMidline3250(
        x: Float,
        midlineX: Float,
        side: Side3250
    ): Boolean {
        return when (side) {
            Side3250.OD -> x > midlineX
            Side3250.OI -> x < midlineX
        }
    }

    private fun isMasked3250(
        mask: ByteArray?,
        w: Int,
        h: Int,
        x: Float,
        y: Float
    ): Boolean {
        if (mask == null) return false
        val xi = x.roundToInt()
        val yi = y.roundToInt()
        if (xi !in 0 until w || yi !in 0 until h) return true
        val v = mask[yi * w + xi].toInt() and 0xFF
        return v != 0
    }

    private fun normalContrast3250(
        gray: ByteArray,
        w: Int,
        h: Int,
        x: Float,
        y: Float,
        nx: Float,
        ny: Float,
        halfPx: Float
    ): Float {
        val in1 = bilinearGray3250(gray, w, h, x - nx * halfPx, y - ny * halfPx)
        val in2 = bilinearGray3250(gray, w, h, x - nx * (halfPx + 1.5f), y - ny * (halfPx + 1.5f))
        val out1 = bilinearGray3250(gray, w, h, x + nx * halfPx, y + ny * halfPx)
        val out2 = bilinearGray3250(gray, w, h, x + nx * (halfPx + 1.5f), y + ny * (halfPx + 1.5f))

        val inside = 0.5f * (in1 + in2)
        val outside = 0.5f * (out1 + out2)
        return (abs(outside - inside) / 255f).coerceIn(0f, 1f)
    }

    private fun normalGradient3250(
        gray: ByteArray,
        w: Int,
        h: Int,
        x: Float,
        y: Float,
        nx: Float,
        ny: Float
    ): Float {
        val a = bilinearGray3250(gray, w, h, x - nx, y - ny)
        val b = bilinearGray3250(gray, w, h, x + nx, y + ny)
        return (abs(b - a) / 255f).coerceIn(0f, 1f)
    }

    private fun bitmapToGrayU83250(bmp: Bitmap): ByteArray {
        val w = bmp.width
        val h = bmp.height
        val out = ByteArray(w * h)
        val row = IntArray(w)

        for (y in 0 until h) {
            bmp.getPixels(row, 0, w, 0, y, w, 1)
            val off = y * w
            for (x in 0 until w) {
                val c = row[x]
                val r = (c shr 16) and 0xFF
                val g = (c shr 8) and 0xFF
                val b = c and 0xFF
                val yy = (77 * r + 150 * g + 29 * b + 128) shr 8
                out[off + x] = yy.toByte()
            }
        }
        return out
    }

    private fun bilinearGray3250(
        gray: ByteArray,
        w: Int,
        h: Int,
        x: Float,
        y: Float
    ): Float {
        val xx = x.coerceIn(0f, (w - 1).toFloat())
        val yy = y.coerceIn(0f, (h - 1).toFloat())

        val x0 = xx.toInt()
        val y0 = yy.toInt()
        val x1 = min(x0 + 1, w - 1)
        val y1 = min(y0 + 1, h - 1)

        val fx = xx - x0
        val fy = yy - y0

        val p00 = u8(gray[y0 * w + x0]).toFloat()
        val p10 = u8(gray[y0 * w + x1]).toFloat()
        val p01 = u8(gray[y1 * w + x0]).toFloat()
        val p11 = u8(gray[y1 * w + x1]).toFloat()

        val a = p00 * (1f - fx) + p10 * fx
        val b = p01 * (1f - fx) + p11 * fx
        return a * (1f - fy) + b * fy
    }

    private fun readU8Norm3250(
        src: ByteArray?,
        w: Int,
        h: Int,
        x: Float,
        y: Float
    ): Float {
        if (src == null) return 0f
        val xi = x.roundToInt()
        val yi = y.roundToInt()
        if (xi !in 0 until w || yi !in 0 until h) return 0f
        return (u8(src[yi * w + xi]) / 255f).coerceIn(0f, 1f)
    }

    private fun centroid3250(pts: List<PointF>): PointF {
        var sx = 0f
        var sy = 0f
        for (p in pts) {
            sx += p.x
            sy += p.y
        }
        val n = pts.size.coerceAtLeast(1)
        return PointF(sx / n, sy / n)
    }

    private fun normalize3250(p: PointF): PointF {
        val d = sqrt(p.x * p.x + p.y * p.y)
        if (d <= 1e-6f) return PointF(1f, 0f)
        return PointF(p.x / d, p.y / d)
    }

    private fun ringRunLength3250(start: Int, end: Int, n: Int): Int {
        return if (end >= start) {
            end - start + 1
        } else {
            (n - start) + (end + 1)
        }
    }

    private fun markRun3250(keep: BooleanArray, start: Int, end: Int) {
        val n = keep.size
        if (end >= start) {
            for (i in start..end) keep[i] = true
        } else {
            for (i in start until n) keep[i] = true
            for (i in 0..end) keep[i] = true
        }
    }
    fun probeFromPolyline3250(
        stillBmp: Bitmap,
        poly: List<PointF>,
        pupil: PointF,
        midlineX: Float,
        mask: ByteArray?,
        edge: ByteArray,
        w: Int,
        h: Int
    ): ProbeQuickResult3250 {

        if (poly.size < 8) {
            return ProbeQuickResult3250(0f, 0f)
        }

        val n = poly.size
        var ok = 0
        var okBottom = 0
        var totalBottom = 0

        for (i in 0 until n) {

            val prev = poly[(i - 1 + n) % n]
            val curr = poly[i]
            val next = poly[(i + 1) % n]

            // tangente
            val tx = next.x - prev.x
            val ty = next.y - prev.y
            val norm = kotlin.math.sqrt(tx * tx + ty * ty).coerceAtLeast(1e-6f)

            val nx = -ty / norm
            val ny = tx / norm

            // clasificación simple bottom/top
            val isBottom = ny > 0.4f
            if (isBottom) totalBottom++

            // sample afuera del FIL
            val x = (curr.x + nx * 4f).toInt()
            val y = (curr.y + ny * 4f).toInt()

            if (x !in 0 until w || y !in 0 until h) continue

            val idx = y * w + x

            // máscara veto
            if (mask != null && (mask[idx].toInt() and 0xFF) != 0) continue

            // midline veto (simple)
            if ((curr.x < midlineX && x > midlineX) ||
                (curr.x > midlineX && x < midlineX)
            ) continue

            val e = edge[idx].toInt() and 0xFF

            if (e > 0) {
                ok++
                if (isBottom) okBottom++
            }
        }

        val support = ok.toFloat() / n.toFloat()
        val supportBottom = if (totalBottom > 0) {
            okBottom.toFloat() / totalBottom.toFloat()
        } else 0f

        return ProbeQuickResult3250(
            supportFrac = support,
            supportBottomFrac = supportBottom
        )
    }

    data class ProbeQuickResult3250(
        val supportFrac: Float,
        val supportBottomFrac: Float
    )

    private fun u8(b: Byte): Int = b.toInt() and 0xFF
}