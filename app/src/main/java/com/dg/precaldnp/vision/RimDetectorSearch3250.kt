package com.dg.precaldnp.vision

import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

    internal fun verticalSupportAt3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        x: Int,
        y: Int,
        halfY: Int = 2
    ): Int {
        if (x !in 0 until w || y !in 0 until h) return 0

        var s = 0
        val y0 = (y - halfY).coerceIn(0, h - 1)
        val y1 = (y + halfY).coerceIn(0, h - 1)

        for (yy in y0..y1) {
            val idx = yy * w + x
            if (isMaskedPx3250(maskU8, idx)) continue
            if ((edgesU8[idx].toInt() and 0xFF) == 0) continue

            val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
            if (isVertEdgeDir3250(d)) s++
        }

        return s
    }
    internal fun horizontalSupportAt3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        w: Int,
        h: Int,
        x: Int,
        y: Int,
        halfX: Int = 2,
        halfY: Int = 1
    ): Int {
        if (x !in 0 until w || y !in 0 until h) return 0

        val x0 = (x - halfX).coerceIn(0, w - 1)
        val x1 = (x + halfX).coerceIn(0, w - 1)
        val y0 = (y - halfY).coerceIn(0, h - 1)
        val y1 = (y + halfY).coerceIn(0, h - 1)

        var s = 0

        for (yy in y0..y1) {
            val base = yy * w
            for (xx in x0..x1) {
                val idx = base + xx
                if (isMaskedPx3250(maskU8, idx)) continue
                if ((edgesU8[idx].toInt() and 0xFF) == 0) continue

                val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)

                s += if (yy == y) 2 else 1

                if (dirU8 == null || isHorzEdgeDir3250(d)) {
                    s += if (yy == y) 2 else 1
                }
            }
        }

        return s
    }
internal fun findInnerWallsAtY3250(
    edgesU8: ByteArray,
    dirU8: ByteArray? = null,
    maskU8: ByteArray?,
    vScoreU8: ByteArray? = null,
    w: Int,
    h: Int,
    seedX: Int,
    yRef: Int,
    bandHalf: Int,
    minDist: Int,
    maxDist: Int,
    profile3250: RimProfile3250
): Pair<Int, Int>? {
    val ys = ArrayList<Int>(2 * bandHalf + 1)
    for (dy in -bandHalf..bandHalf) {
        ys.add((yRef + dy).coerceIn(0, h - 1))
    }

    val leftHits = ArrayList<Int>()
    val rightHits = ArrayList<Int>()

    val innerMinStep = minDist.coerceAtLeast(1)

    val minSupportInner = when (profile3250) {
        RimProfile3250.PERFORADO -> 1
        RimProfile3250.FULL_RIM,
        RimProfile3250.RANURADO -> 2
    }

    val firstHitWindowPx = when (profile3250) {
        RimProfile3250.FULL_RIM -> 6
        RimProfile3250.RANURADO -> 5
        RimProfile3250.PERFORADO -> 4
    }

    fun scanInner(dx: Int, y: Int): Int? {
        val xStart = seedX.coerceIn(0, w - 1)

        var firstHitStep = -1
        var bestX = -1
        var bestScore = Int.MIN_VALUE

        for (step in innerMinStep..maxDist) {
            val x = xStart + dx * step
            if (x !in 0 until w) break

            val idx = y * w + x

            if (isMaskedPx3250(maskU8, idx)) {
                if (firstHitStep >= 0 && step - firstHitStep > firstHitWindowPx) break
                continue
            }

            if ((edgesU8[idx].toInt() and 0xFF) == 0) {
                if (firstHitStep >= 0 && step - firstHitStep > firstHitWindowPx) break
                continue
            }

            val support = verticalSupportAt3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                x = x,
                y = y,
                halfY = 2
            )
            if (support < minSupportInner) {
                if (firstHitStep >= 0 && step - firstHitStep > firstHitWindowPx) break
                continue
            }

            val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
            val dirGood = isVertEdgeDir3250(d)
            if (!dirGood) {
                if (firstHitStep >= 0 && step - firstHitStep > firstHitWindowPx) break
                continue
            }

            if (firstHitStep < 0) firstHitStep = step

            val vScore = if (vScoreU8 != null && vScoreU8.size == w * h) {
                vScoreU8[idx].toInt() and 0xFF
            } else {
                0
            }

            val dist = abs(x - xStart)

            val score =
                support * 100 +
                        vScore * 2 -
                        dist * 4

            if (score > bestScore) {
                bestScore = score
                bestX = x
            }

            if (step - firstHitStep >= firstHitWindowPx) break
        }

        return bestX.takeIf { it >= 0 }
    }

    for (y in ys) {
        scanInner(dx = -1, y = y)?.let { leftHits.add(it) }
        scanInner(dx = +1, y = y)?.let { rightHits.add(it) }
    }

    if (leftHits.isEmpty() || rightHits.isEmpty()) return null

    val left = medianInt3250(leftHits)
    val right = medianInt3250(rightHits)

    if (left >= right) return null
    if (abs(right - left) < 40) return null

    return left to right
}
internal fun searchNearestInnerAroundPrevX3250(
    edgesU8: ByteArray,
    dirU8: ByteArray? = null,
    maskU8: ByteArray? = null,
    vScoreU8: ByteArray? = null,
    w: Int,
    h: Int,
    y: Int,
    prevX: Int,
    centerInsideSeedX: Int,
    sideSign: Int,
    searchRadiusPx: Int,
    profile3250: RimProfile3250
): Int? {
    fun onCorrectSide(x: Int): Boolean {
        return if (sideSign < 0) x < centerInsideSeedX else x > centerInsideSeedX
    }

    fun isValidInnerAt(x: Int): Boolean {
        if (x !in 0 until w) return false
        if (!onCorrectSide(x)) return false

        val idx = y * w + x
        if (isMaskedPx3250(maskU8, idx)) return false
        if ((edgesU8[idx].toInt() and 0xFF) == 0) return false

        val support = verticalSupportAt3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            x = x,
            y = y,
            halfY = 2
        )

        val minSupport = when (profile3250) {
            RimProfile3250.PERFORADO -> 1
            RimProfile3250.FULL_RIM,
            RimProfile3250.RANURADO -> 2
        }

        if (support < minSupport) return false

        if (dirU8 != null) {
            val d = dirU8[idx].toInt() and 0xFF
            if (!isVertEdgeDir3250(d)) return false
        }

        return true
    }

    for (d in 0..searchRadiusPx) {
        if (d == 0) {
            if (isValidInnerAt(prevX)) return prevX
        } else {
            val x1 = prevX - d
            val x2 = prevX + d

            if (sideSign < 0) {
                if (isValidInnerAt(x1)) return x1
                if (isValidInnerAt(x2)) return x2
            } else {
                if (isValidInnerAt(x2)) return x2
                if (isValidInnerAt(x1)) return x1
            }
        }
    }

    return null
}
internal fun searchOuterFromInner3250(
    edgesU8: ByteArray,
    dirU8: ByteArray?,
    maskU8: ByteArray?,
    w: Int,
    h: Int,
    y: Int,
    innerX: Int,
    outwardSign: Int,
    prevGapPx: Int?,
    minOuterGapPx: Int,
    maxOuterGapPx: Int,
    profile3250: RimProfile3250,
    minSupport: Int = 2
): Int? {
    if (profile3250 != RimProfile3250.FULL_RIM) return null

    fun isValidOuterAt(x: Int): Boolean {
        if (x !in 0 until w) return false

        val idx = y * w + x
        if (isMaskedPx3250(maskU8, idx)) return false
        if ((edgesU8[idx].toInt() and 0xFF) == 0) return false

        if (dirU8 != null) {
            val d = dirU8[idx].toInt() and 0xFF
            if (!isVertEdgeDir3250(d)) return false
        }

        val support = verticalSupportAt3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            w = w,
            h = h,
            x = x,
            y = y,
            halfY = 2
        )

        return support >= minSupport
    }

    for (gap in minOuterGapPx..maxOuterGapPx) {
        val x = innerX + outwardSign * gap
        if (x !in 0 until w) break
        if (isValidOuterAt(x)) return x
    }

    return null
}

    internal fun collectBottomCandidatesInWindow3250(
        edgesU8: ByteArray,
        dirU8: ByteArray?,
        maskU8: ByteArray?,
        hScoreU8: ByteArray?,
        w: Int,
        x: Int,
        y0: Int,
        y1: Int,
        expectedY: Int
    ): List<BottomCand3250> {
        val h = edgesU8.size / w
        if (h <= 0) return emptyList()

        val lo = min(y0, y1).coerceIn(0, h - 1)
        val hi = max(y0, y1).coerceIn(0, h - 1)

        val raw = ArrayList<BottomCand3250>()

        for (y in lo..hi) {
            val idx = y * w + x
            val e = edgesU8[idx].toInt() and 0xFF
            if (e == 0) continue
            if (isMasked3250(maskU8, idx)) continue

            val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
            val dirScore = when {
                dirU8 == null -> 100
                isHorzEdgeDir3250(d) -> 100
                else -> 45
            }

            val support = horizontalSupportAt3250(
                edgesU8 = edgesU8,
                dirU8 = dirU8,
                maskU8 = maskU8,
                w = w,
                h = h,
                x = x,
                y = y,
                halfX = 2,
                halfY = 1
            )
            if (support <= 0) continue

            val hScore = if (hScoreU8 != null && hScoreU8.size == w * h) {
                hScoreU8[idx].toInt() and 0xFF
            } else {
                0
            }

            val score =
                support * 100 +
                        dirScore +
                        hScore * 2 -
                        abs(y - expectedY) * 8
            raw.add(
                BottomCand3250(
                    y = y,
                    score = score,
                    support = support,
                    dirScore = dirScore
                )
            )
        }

        if (raw.isEmpty()) return emptyList()

        val grouped = ArrayList<BottomCand3250>()
        var i = 0
        while (i < raw.size) {
            var best = raw[i]
            var j = i + 1
            while (j < raw.size && raw[j].y - raw[j - 1].y <= 2) {
                if (raw[j].score > best.score) best = raw[j]
                j++
            }
            grouped.add(best)
            i = j
        }

        return grouped.sortedBy { it.y }
    }
    internal fun chooseSingleBottomInnerCandidate3250(
        candidates: List<BottomCand3250>,
        expectedY: Int,
        profile3250: RimProfile3250
    ): BottomCand3250? {
        if (candidates.isEmpty()) return null

        var best: BottomCand3250? = null
        var bestScore = Int.MIN_VALUE

        for (c in candidates) {
            val belowPenalty = when (profile3250) {
                RimProfile3250.FULL_RIM -> max(0, c.y - expectedY) * 16
                RimProfile3250.RANURADO -> max(0, c.y - expectedY) * 10
                RimProfile3250.PERFORADO -> max(0, c.y - expectedY) * 6
            }

            val dyPenalty = abs(c.y - expectedY) * 12
            val dirBonus = if (c.dirScore >= 100) 40 else 0
            val supportBonus = c.support * 12

            val finalScore =
                c.score +
                        dirBonus +
                        supportBonus -
                        dyPenalty -
                        belowPenalty

            if (finalScore > bestScore) {
                bestScore = finalScore
                best = c
            }
        }

        return best
    }
internal fun findBestBottomEdgeInWindow3250(
    edgesU8: ByteArray,
    dirU8: ByteArray?,
    maskU8: ByteArray?,
    hScoreU8: ByteArray?,
    w: Int,
    a: Int,
    b: Int,
    y0: Int,
    y1: Int,
    expectedY: Int,
    profile3250: RimProfile3250,
    minOuterGapPx: Int = 0,
    maxOuterGapPx: Int = 0
): Int {
    if (w <= 0) return -1

    val h = edgesU8.size / w
    if (h <= 0) return -1

    val lo = min(y0, y1).coerceIn(0, h - 1)
    val hi = max(y0, y1).coerceIn(0, h - 1)
    val xl = min(a, b).coerceIn(0, w - 1)
    val xr = max(a, b).coerceIn(0, w - 1)

    data class Hit(val x: Int, val y: Int, val score: Int)

    val hits = ArrayList<Hit>()

    for (xx in xl..xr) {
        val candidates = collectBottomCandidatesInWindow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            hScoreU8 = hScoreU8,
            w = w,
            x = xx,
            y0 = lo,
            y1 = hi,
            expectedY = expectedY
        )
        if (candidates.isEmpty()) continue

        var bestYLocal = -1
        var bestScoreLocal = Int.MIN_VALUE

        if (
            profile3250 == RimProfile3250.FULL_RIM &&
            minOuterGapPx > 0 &&
            maxOuterGapPx >= minOuterGapPx
        ) {
            val gapTarget = ((minOuterGapPx + maxOuterGapPx) * 0.5f)

            for (i in candidates.indices) {
                val inner = candidates[i]
                for (j in i + 1 until candidates.size) {
                    val outer = candidates[j]
                    val gap = outer.y - inner.y
                    if (gap !in minOuterGapPx..maxOuterGapPx) continue

                    val expectedPenalty =
                        abs(inner.y - expectedY) * 14 +
                                max(0, inner.y - expectedY) * 18

                    val gapPenalty =
                        abs(gap.toFloat() - gapTarget).roundToInt() * 4

                    val pairScore =
                        inner.score +
                                (outer.score / 2) +
                                inner.support * 16 +
                                outer.support * 8 +
                                (if (inner.dirScore >= 100) 50 else 0) +
                                (if (outer.dirScore >= 100) 20 else 0) -
                                expectedPenalty -
                                gapPenalty

                    if (pairScore > bestScoreLocal) {
                        bestScoreLocal = pairScore
                        bestYLocal = inner.y
                    }
                }
            }
        }

        if (bestYLocal < 0) {
            val bestSingle = chooseSingleBottomInnerCandidate3250(
                candidates = candidates,
                expectedY = expectedY,
                profile3250 = profile3250
            )

            if (bestSingle != null) {
                bestYLocal = bestSingle.y
                bestScoreLocal =
                    bestSingle.score +
                            (if (bestSingle.dirScore >= 100) 40 else 0) +
                            bestSingle.support * 12
            }
        }

        if (bestYLocal >= 0) {
            hits.add(Hit(xx, bestYLocal, bestScoreLocal))
        }
    }

    if (hits.isEmpty()) return -1

    data class Group(
        val ys: ArrayList<Int>,
        val xs: ArrayList<Int>,
        var scoreSum: Int
    )

    hits.sortBy { it.y }

    val groups = ArrayList<Group>()
    var current = Group(
        ys = arrayListOf(hits[0].y),
        xs = arrayListOf(hits[0].x),
        scoreSum = hits[0].score
    )

    for (i in 1 until hits.size) {
        val hit = hits[i]
        val prevY = current.ys.last()

        if (abs(hit.y - prevY) <= 2) {
            current.ys.add(hit.y)
            current.xs.add(hit.x)
            current.scoreSum += hit.score
        } else {
            groups.add(current)
            current = Group(
                ys = arrayListOf(hit.y),
                xs = arrayListOf(hit.x),
                scoreSum = hit.score
            )
        }
    }
    groups.add(current)

    var bestY = -1
    var bestGroupScore = Int.MIN_VALUE

    for (g in groups) {
        val xs = g.xs.sorted()

        var run = 1
        var bestRun = 1
        for (i in 1 until xs.size) {
            run = if (xs[i] == xs[i - 1] + 1) run + 1 else 1
            if (run > bestRun) bestRun = run
        }

        val yMed = medianInt3250(g.ys)
        val density = (g.xs.size * 100) / max(1, xr - xl + 1)

        val groupScore =
            bestRun * 10000 +
                    g.xs.size * 2000 +
                    density * 50 +
                    g.scoreSum -
                    abs(yMed - expectedY) * 8

        if (groupScore > bestGroupScore) {
            bestGroupScore = groupScore
            bestY = yMed
        }
    }

    return bestY
}
internal fun collectTopCandidatesInWindow3250(
    edgesU8: ByteArray,
    dirU8: ByteArray?,
    maskU8: ByteArray?,
    hScoreU8: ByteArray? = null,
    w: Int,
    x: Int,
    y0: Int,
    y1: Int,
    expectedY: Int
): List<TopCand3250> {
    val h = edgesU8.size / w
    if (h <= 0) return emptyList()

    val lo = min(y0, y1).coerceIn(0, h - 1)
    val hi = max(y0, y1).coerceIn(0, h - 1)

    val raw = ArrayList<TopCand3250>()

    for (y in lo..hi) {
        val idx = y * w + x
        val e = edgesU8[idx].toInt() and 0xFF
        if (e == 0) continue
        if (isMasked3250(maskU8, idx)) continue

        val d = if (dirU8 == null) 255 else (dirU8[idx].toInt() and 0xFF)
        val dirScore = when {
            dirU8 == null -> 100
            isHorzEdgeDir3250(d) -> 100
            else -> 45
        }

        val support = horizontalSupportAt3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = null,
            w = w,
            h = h,
            x = x,
            y = y,
            halfX = 2,
            halfY = 1
        )
        if (support <= 0) continue

        val hScore = if (hScoreU8 != null && hScoreU8.size == w * h) {
            hScoreU8[idx].toInt() and 0xFF
        } else {
            0
        }

        val score =
            support * 100 +
                    dirScore +
                    hScore * 2 -
                    abs(y - expectedY) * 8

        raw.add(
            TopCand3250(
                y = y,
                score = score,
                support = support,
                dirScore = dirScore
            )
        )
    }

    if (raw.isEmpty()) return emptyList()

    val grouped = ArrayList<TopCand3250>()
    var i = 0
    while (i < raw.size) {
        var best = raw[i]
        var j = i + 1
        while (j < raw.size && raw[j].y - raw[j - 1].y <= 2) {
            if (raw[j].score > best.score) best = raw[j]
            j++
        }
        grouped.add(best)
        i = j
    }

    return grouped.sortedBy { it.y }
}
internal fun chooseSingleTopInnerCandidate3250(
    candidates: List<TopCand3250>,
    expectedY: Int
): TopCand3250? {
    if (candidates.isEmpty()) return null

    var best: TopCand3250? = null
    var bestScore = Int.MIN_VALUE

    for (c in candidates) {
        val belowPenalty = max(0, c.y - expectedY) * 16
        val dyPenalty = abs(c.y - expectedY) * 12
        val dirBonus = if (c.dirScore >= 100) 40 else 0
        val supportBonus = c.support * 12

        val finalScore =
            c.score +
                    dirBonus +
                    supportBonus -
                    dyPenalty -
                    belowPenalty

        if (finalScore > bestScore) {
            bestScore = finalScore
            best = c
        }
    }

    return best
}
internal fun findBestTopEdgeInWindow3250(
    edgesU8: ByteArray,
    dirU8: ByteArray?,
    maskU8: ByteArray?,
    hScoreU8: ByteArray? = null,
    w: Int,
    a: Int,
    b: Int,
    y0: Int,
    y1: Int,
    expectedY: Int
): Int {
    if (w <= 0) return -1

    val h = edgesU8.size / w
    if (h <= 0) return -1

    val lo = min(y0, y1).coerceIn(0, h - 1)
    val hi = max(y0, y1).coerceIn(0, h - 1)
    val xl = min(a, b).coerceIn(0, w - 1)
    val xr = max(a, b).coerceIn(0, w - 1)

    data class Hit(val x: Int, val y: Int, val score: Int)

    val hits = ArrayList<Hit>()

    for (xx in xl..xr) {
        val candidates = collectTopCandidatesInWindow3250(
            edgesU8 = edgesU8,
            dirU8 = dirU8,
            maskU8 = maskU8,
            hScoreU8 = hScoreU8,
            w = w,
            x = xx,
            y0 = lo,
            y1 = hi,
            expectedY = expectedY
        )
        if (candidates.isEmpty()) continue

        val best = chooseSingleTopInnerCandidate3250(
            candidates = candidates,
            expectedY = expectedY
        ) ?: continue

        val finalScore =
            best.score +
                    (if (best.dirScore >= 100) 40 else 0) +
                    best.support * 12

        hits.add(Hit(xx, best.y, finalScore))
    }

    if (hits.isEmpty()) return -1

    data class Group(
        val ys: ArrayList<Int>,
        val xs: ArrayList<Int>,
        var scoreSum: Int
    )

    hits.sortBy { it.y }

    val groups = ArrayList<Group>()
    var current = Group(
        ys = arrayListOf(hits[0].y),
        xs = arrayListOf(hits[0].x),
        scoreSum = hits[0].score
    )

    for (i in 1 until hits.size) {
        val hit = hits[i]
        val prevY = current.ys.last()

        if (abs(hit.y - prevY) <= 2) {
            current.ys.add(hit.y)
            current.xs.add(hit.x)
            current.scoreSum += hit.score
        } else {
            groups.add(current)
            current = Group(
                ys = arrayListOf(hit.y),
                xs = arrayListOf(hit.x),
                scoreSum = hit.score
            )
        }
    }
    groups.add(current)

    var bestY = -1
    var bestGroupScore = Int.MIN_VALUE

    for (g in groups) {
        val xs = g.xs.sorted()

        var run = 1
        var bestRun = 1
        for (i in 1 until xs.size) {
            run = if (xs[i] == xs[i - 1] + 1) run + 1 else 1
            if (run > bestRun) bestRun = run
        }

        val yMed = medianInt3250(g.ys)
        val density = (g.xs.size * 100) / max(1, xr - xl + 1)

        val groupScore =
            bestRun * 10000 +
                    g.xs.size * 2000 +
                    density * 50 +
                    g.scoreSum -
                    abs(yMed - expectedY) * 8

        if (groupScore > bestGroupScore) {
            bestGroupScore = groupScore
            bestY = yMed
        }
    }

    return bestY
}