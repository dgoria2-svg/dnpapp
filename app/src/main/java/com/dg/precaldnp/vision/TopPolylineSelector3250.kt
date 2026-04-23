package com.dg.precaldnp.vision

object TopPolylineSelector3250 {

    fun pickTopPolyline3250(
        polys: List<EdgeMapPolylineDebug3250.Polyline3250>,
        minPoints: Int = 20
    ): EdgeMapPolylineDebug3250.Polyline3250? {

        var best: EdgeMapPolylineDebug3250.Polyline3250? = null
        var bestScore = Float.NEGATIVE_INFINITY

        for (p in polys) {
            if (p.pointCount < minPoints) continue

            val pts = p.pointsLocal
            if (pts.isEmpty()) continue

            var minX = Float.POSITIVE_INFINITY
            var maxX = Float.NEGATIVE_INFINITY
            var minY = Float.POSITIVE_INFINITY
            var maxY = Float.NEGATIVE_INFINITY

            for (pt in pts) {
                if (pt.x < minX) minX = pt.x
                if (pt.x > maxX) maxX = pt.x
                if (pt.y < minY) minY = pt.y
                if (pt.y > maxY) maxY = pt.y
            }

            val width = maxX - minX
            val height = maxY - minY

            if (width <= 0f) continue

            val flatness = width / (height + 1f)
            val density = p.pointCount.toFloat() / (width + 1f)
            val score = width * 1.0f + flatness * 50f + density * 10f

            if (score > bestScore) {
                bestScore = score
                best = p
            }
        }

        return best
    }
}