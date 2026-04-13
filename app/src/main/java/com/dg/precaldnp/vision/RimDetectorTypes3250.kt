package com.dg.precaldnp.vision

import android.graphics.PointF

internal data class ArcPick(
    val yMed: Int,
    val yMax: Int,
    val yBottomCoherent: Int,
    val poly: List<Pair<Int, Int>>,
    val coverage: Float,
    val continuity: Float
)

internal data class TopCand3250(
    val y: Int,
    val score: Int,
    val support: Int,
    val dirScore: Int
)

internal data class InnerSeedRow3250(
    val y: Int,
    val leftX: Int,
    val rightX: Int
)

internal data class SideArcCandidatesLocal3250(
    val innerLeft: List<PointF>,
    val innerRight: List<PointF>,
    val outerLeft: List<PointF>,
    val outerRight: List<PointF>
)

internal data class ScaleCandidate3250(
    val conf: Float,
    val left: Int,
    val right: Int,
    val top: Int,
    val bottom: Int,
    val refY: Int,
    val seedX: Int,
    val scale: Float,
    val bottomPoly: List<Pair<Int, Int>>,
    val topPoly: List<Pair<Int, Int>>,
    val nasalInnerPoly: List<Pair<Int, Int>> = emptyList(),
    val templeInnerPoly: List<Pair<Int, Int>> = emptyList(),
    val topObservedY: Int?,
    val topConfidence: Float,
    val topMinAllowedY: Int = -1,
    val topSearchMinY: Int = -1,
    val expectedTopY: Int = -1,
    val expectedTopTolPx: Int = 0,

    val isPartial: Boolean
)

internal data class ResolvedTop3250(
    val usedY: Int,
    val estimatedY: Int,
    val observedY: Int?,
    val poly: List<Pair<Int, Int>>,
    val confidence: Float
)

data class RimDetectProfilePick3250(
    val profile3250: RimProfile3250,
    val pack3250: RimDetectPack3250,
    val score3250: Float
)

internal data class TopPick3250(
    val yMed: Int,
    val yMin: Int,
    val yTopCoherent: Int,
    val poly: List<Pair<Int, Int>>,
    val coverage: Float,
    val continuity: Float,
    val confidence: Float
)

internal data class BottomCand3250(
    val y: Int,
    val score: Int,
    val support: Int,
    val dirScore: Int
)