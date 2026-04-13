package com.dg.precaldnp.vision

import android.graphics.PointF
import java.util.Locale
import kotlin.math.max
import kotlin.math.min

internal fun isMasked3250(maskU8: ByteArray?, idx: Int): Boolean =
    maskU8 != null && ((maskU8[idx].toInt() and 0xFF) != 0)

internal fun isMaskedPx3250(maskU8: ByteArray?, idx: Int): Boolean =
    isMasked3250(maskU8, idx)

internal fun isVertEdgeDir3250(d: Int): Boolean {
    val dd = d and 0xFF
    if (dd == 255) return true
    if (dd !in 0..3) return true
    return dd == 0 || dd == 1 || dd == 3
}

internal fun isHorzEdgeDir3250(d: Int): Boolean {
    val dd = d and 0xFF
    if (dd == 255) return true
    if (dd !in 0..3) return true
    return dd == 2 || dd == 1 || dd == 3
}

internal fun buildScales3250(minS: Float, maxS: Float, step: Float): FloatArray {
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

internal fun clampInt3250(v: Int, lo: Int, hi: Int): Int =
    max(lo, min(hi, v))

internal fun clampInRange3250(v: Int, a: Int, b: Int): Int =
    v.coerceIn(min(a, b), max(a, b))

internal fun norm013250(v: Float, lo: Float, hi: Float): Float {
    if (hi <= lo) return 0f
    return ((v - lo) / (hi - lo)).coerceIn(0f, 1f)
}

internal fun medianInt3250(v: List<Int>): Int {
    if (v.isEmpty()) return 0
    val s = v.sorted()
    val m = s.size / 2
    return if (s.size % 2 == 1) s[m] else ((s[m - 1] + s[m]) / 2)
}

internal fun pairPolylineToGlobal3250(
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

internal data class RimSearchPolicy3250(
    val requireOuter: Boolean,
    val requireTop: Boolean,
    val innerMandatory: Boolean,
    val validateOuterAgainstInner: Boolean
)

internal fun buildRimSearchPolicy3250(
    profile3250: RimProfile3250
): RimSearchPolicy3250 =
    when (profile3250) {
        RimProfile3250.FULL_RIM -> RimSearchPolicy3250(
            requireOuter = true,
            requireTop = false,
            innerMandatory = true,
            validateOuterAgainstInner = true
        )
        RimProfile3250.RANURADO -> RimSearchPolicy3250(
            requireOuter = false,
            requireTop = true,
            innerMandatory = true,
            validateOuterAgainstInner = false
        )
        RimProfile3250.PERFORADO -> RimSearchPolicy3250(
            requireOuter = false,
            requireTop = false,
            innerMandatory = true,
            validateOuterAgainstInner = false
        )
    }

internal fun f13250(x: Float): String =
    String.format(Locale.US, "%.1f", x)

internal fun f23250(x: Float): String =
    String.format(Locale.US, "%.2f", x)

internal fun f33250(x: Float): String =
    String.format(Locale.US, "%.3f", x)