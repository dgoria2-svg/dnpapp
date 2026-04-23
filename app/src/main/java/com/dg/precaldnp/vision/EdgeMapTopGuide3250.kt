package com.dg.precaldnp.vision

import android.graphics.PointF
import kotlin.math.abs
import kotlin.math.roundToInt

object EdgeMapTopGuide3250 {

    fun buildTopGuideYAtX3250(
        poly: List<PointF>
    ): (Int) -> Int? {
        if (poly.isEmpty()) return { null }

        val pts = poly.sortedBy { it.x }

        return guide@{ x: Int ->
            var best: PointF? = null
            var bestDx = Int.MAX_VALUE

            for (p in pts) {
                val dx = abs(p.x.roundToInt() - x)
                if (dx < bestDx) {
                    bestDx = dx
                    best = p
                }
            }

            best?.y?.roundToInt()
        }
    }

    fun minY3250(poly: List<PointF>): Int? {
        if (poly.isEmpty()) return null
        var minY = Int.MAX_VALUE
        for (p in poly) {
            val y = p.y.roundToInt()
            if (y < minY) minY = y
        }
        return minY
    }

    fun maxY3250(poly: List<PointF>): Int? {
        if (poly.isEmpty()) return null
        var maxY = Int.MIN_VALUE
        for (p in poly) {
            val y = p.y.roundToInt()
            if (y > maxY) maxY = y
        }
        return maxY
    }

    fun meanY3250(poly: List<PointF>): Int? {
        if (poly.isEmpty()) return null
        var sum = 0f
        for (p in poly) sum += p.y
        return (sum / poly.size.toFloat()).roundToInt()
    }
}