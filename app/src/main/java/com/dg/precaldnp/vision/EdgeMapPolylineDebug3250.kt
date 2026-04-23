package com.dg.precaldnp.vision

import android.graphics.PointF
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfPoint
import org.opencv.core.Point
import org.opencv.core.Rect
import org.opencv.imgproc.Imgproc

object EdgeMapPolylineDebug3250 {
    data class Polyline3250(
        val pointsLocal: List<PointF>,
        val pointsGlobal: List<PointF>,
        val areaPx: Double,
        val perimeterPx: Double,
        val bboxLocal: Rect,
        val pointCount: Int
    )

    data class Result3250(
        val polylines: List<Polyline3250>,
        val width: Int,
        val height: Int
    )

    fun extractAll3250(
        edgeU8: ByteArray,
        w: Int,
        h: Int,
        roiLeft: Int = 0,
        roiTop: Int = 0,
        minAreaPx: Double = 1.0,
        minPerimeterPx: Double = 8.0,
        minPoints: Int = 8,
        approxEpsilonPx: Double = 0.0
    ): Result3250 {
        require(w > 0 && h > 0)
        require(edgeU8.size >= w * h)

        val src = Mat(h, w, CvType.CV_8UC1)
        src.put(0, 0, edgeU8)

        val bin = Mat()
        Imgproc.threshold(src, bin, 0.0, 255.0, Imgproc.THRESH_BINARY)

        val contours = ArrayList<MatOfPoint>()
        val hierarchy = Mat()
        Imgproc.findContours(
            bin.clone(),
            contours,
            hierarchy,
            Imgproc.RETR_LIST,
            Imgproc.CHAIN_APPROX_NONE
        )

        val out = ArrayList<Polyline3250>(contours.size)

        for (c in contours) {
            if (c.empty()) continue

            val area = kotlin.math.abs(Imgproc.contourArea(c))
            val perimeter = Imgproc.arcLength(org.opencv.core.MatOfPoint2f(*c.toArray()), true)
            val pts0 = c.toArray().toList()

            if (area < minAreaPx) continue
            if (perimeter < minPerimeterPx) continue
            if (pts0.size < minPoints) continue

            val pts = if (approxEpsilonPx > 0.0) {
                val c2f = org.opencv.core.MatOfPoint2f(*c.toArray())
                val approx2f = org.opencv.core.MatOfPoint2f()
                Imgproc.approxPolyDP(c2f, approx2f, approxEpsilonPx, true)
                approx2f.toArray().toList()
            } else {
                pts0
            }

            if (pts.size < minPoints) continue

            val local = ArrayList<PointF>(pts.size)
            val global = ArrayList<PointF>(pts.size)

            for (p in pts) {
                val xl = p.x.toFloat()
                val yl = p.y.toFloat()
                local.add(PointF(xl, yl))
                global.add(PointF(xl + roiLeft.toFloat(), yl + roiTop.toFloat()))
            }

            val bbox = Imgproc.boundingRect(MatOfPoint(*pts.map { Point(it.x, it.y) }.toTypedArray()))

            out.add(
                Polyline3250(
                    pointsLocal = local,
                    pointsGlobal = global,
                    areaPx = area,
                    perimeterPx = perimeter,
                    bboxLocal = bbox,
                    pointCount = pts.size
                )
            )
        }

        val sorted = out.sortedWith(
            compareByDescending<Polyline3250> { it.perimeterPx }
                .thenByDescending { it.areaPx }
                .thenByDescending { it.pointCount }
        )

        return Result3250(
            polylines = sorted,
            width = w,
            height = h
        )
    }

    fun extractAllFromMat3250(
        edgeMatU8: Mat,
        roiLeft: Int = 0,
        roiTop: Int = 0,
        minAreaPx: Double = 1.0,
        minPerimeterPx: Double = 8.0,
        minPoints: Int = 8,
        approxEpsilonPx: Double = 0.0
    ): Result3250 {
        require(edgeMatU8.type() == CvType.CV_8UC1)

        val w = edgeMatU8.cols()
        val h = edgeMatU8.rows()
        val buf = ByteArray(w * h)
        edgeMatU8.get(0, 0, buf)

        return extractAll3250(
            edgeU8 = buf,
            w = w,
            h = h,
            roiLeft = roiLeft,
            roiTop = roiTop,
            minAreaPx = minAreaPx,
            minPerimeterPx = minPerimeterPx,
            minPoints = minPoints,
            approxEpsilonPx = approxEpsilonPx
        )
    }
}