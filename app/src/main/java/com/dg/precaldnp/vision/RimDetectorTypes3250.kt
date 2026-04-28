package com.dg.precaldnp.vision

import android.graphics.PointF

internal data class ArcPick(
    val yMed: Int,
    val yMax: Int,
    val yBottomCoherent: Int,
    val poly: List<Pair<Int, Int>>,
    val coverage: Float,
    val continuity: Float,
    val hasOuter: Boolean = false,
    val outerYMed: Int? = null
)

internal data class TopCand3250(
    val y: Int,
    val score: Int,
    val support: Int,
    val dirScore: Int,
    val source3250: RimWinnerSource3250
)

internal data class TopPick3250(
    val yMed: Int,
    val yMin: Int,
    val yTopCoherent: Int,
    val poly: List<Pair<Int, Int>>,
    val coverage: Float,
    val continuity: Float,
    val confidence: Float,
    val source3250: RimWinnerSource3250,
    val hasOuter3250: Boolean
)

internal data class InnerSeedRow3250(
    val y: Int,
    val leftX: Int,
    val rightX: Int
)

data class SideArcCandidatesLocal3250(
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
    val expHGuessPx: Float,
    val ratioPenalty: Float,
    val expectedTopTolPx: Int = 0,

    val isPartial: Boolean
)

internal data class BottomEdgePick3250(
    val innerY: Int,
    val hasOuter: Boolean,
    val outerY: Int?
)
internal data class ResolvedTop3250(
    val usedY: Int,
    val estimatedY: Int,
    val observedY: Int?,
    val poly: List<Pair<Int, Int>>,
    val confidence: Float
)
internal data class TopEdgePick3250(
    val y: Int,
    val source3250: RimWinnerSource3250
)

data class RimDetectProfilePick3250(
    val profile3250: RimProfile3250,
    val pack3250: RimDetectPack3250,
    val score3250: Float
)

internal data class BottomCand3250(
    val y: Int,
    val score: Int,
    val support: Int,
    val dirScore: Int
)
internal data class RimGateInput3250(
    val profile3250: RimProfile3250,
    val baseConfidence: Float,
    val isPartial: Boolean,
    val hasInnerLaterals: Boolean,
    val hasBottomInner: Boolean,
    val hasBottomOuter: Boolean,
    val hasTopOuter: Boolean,
    val topObservedY: Int?,
    val bottomObservedY: Int?,
    val topUsedY: Int,
    val bottomUsedY: Int,
    val expHGuessPx: Float,
    val ratioPenalty: Float,
    val winnerSource3250: RimWinnerSource3250
)
internal data class RimGateResult3250(
    val accepted: Boolean,
    val confidenceOut: Float,
    val reason3250: String
)
