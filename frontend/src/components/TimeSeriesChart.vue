<script setup>
import { ref, onMounted, onUnmounted, watch, nextTick } from 'vue'
import * as d3 from 'd3'

const props = defineProps({
  series: { type: Array, required: true },    // [{time, value, index, outlier, global_mask}]
  segments: { type: Array, default: () => [] }, // [{start, end, score, length}]
  pointName: { type: String, default: '' },
  method: { type: String, default: '' },
})

const chartRef = ref(null)
const tooltipRef = ref(null)
let resizeObserver = null

// Layout constants
const margin = { top: 20, right: 60, bottom: 100, left: 70 }
const contextMargin = { top: 0, right: 60, bottom: 20, left: 70 }
const contextHeight = 60
const gap = 30

function render() {
  if (!chartRef.value || !props.series.length) return

  const container = chartRef.value
  d3.select(container).selectAll('*').remove()

  const totalWidth = container.clientWidth
  const totalHeight = 460
  const mainHeight = totalHeight - margin.top - margin.bottom - contextHeight - gap

  const svg = d3.select(container)
    .append('svg')
    .attr('width', totalWidth)
    .attr('height', totalHeight)

  // Clip path for main chart
  svg.append('defs').append('clipPath')
    .attr('id', 'clip-main')
    .append('rect')
    .attr('width', totalWidth - margin.left - margin.right)
    .attr('height', mainHeight)

  const data = props.series
  const xDomain = [0, data.length - 1]
  const values = data.map(d => d.value).filter(v => v != null)
  const yMin = d3.min(values)
  const yMax = d3.max(values)
  const yPad = (yMax - yMin) * 0.05 || 1

  // ---- Scales ----
  const xScale = d3.scaleLinear()
    .domain(xDomain)
    .range([0, totalWidth - margin.left - margin.right])

  const yScale = d3.scaleLinear()
    .domain([yMin - yPad, yMax + yPad])
    .range([mainHeight, 0])

  const contextXScale = d3.scaleLinear()
    .domain(xDomain)
    .range([0, totalWidth - contextMargin.left - contextMargin.right])

  const contextYScale = d3.scaleLinear()
    .domain([yMin - yPad, yMax + yPad])
    .range([contextHeight, 0])

  // ---- Main chart group ----
  const main = svg.append('g')
    .attr('transform', `translate(${margin.left},${margin.top})`)

  // Anomaly segment highlights (behind line)
  const segmentGroup = main.append('g').attr('clip-path', 'url(#clip-main)')
  function drawSegments(xS) {
    segmentGroup.selectAll('.segment-rect').remove()
    for (const seg of props.segments) {
      segmentGroup.append('rect')
        .attr('class', 'segment-rect')
        .attr('x', xS(seg.start))
        .attr('width', Math.max(1, xS(seg.end) - xS(seg.start)))
        .attr('y', 0)
        .attr('height', mainHeight)
        .attr('fill', '#F8766D')
        .attr('opacity', 0.15 + Math.min(0.35, (seg.score || 0) * 0.05))
    }
  }
  drawSegments(xScale)

  // Line generator
  const line = d3.line()
    .defined(d => d.value != null)
    .x(d => xScale(d.index))
    .y(d => yScale(d.value))

  // Main line
  const linePath = main.append('g')
    .attr('clip-path', 'url(#clip-main)')
    .append('path')
    .datum(data)
    .attr('fill', 'none')
    .attr('stroke', '#409EFF')
    .attr('stroke-width', 1.2)
    .attr('d', line)

  // Outlier points
  const outlierGroup = main.append('g').attr('clip-path', 'url(#clip-main)')
  function drawOutliers(xS) {
    outlierGroup.selectAll('.outlier-dot').remove()
    const outliers = data.filter(d => d.outlier === 1 && d.value != null)
    outlierGroup.selectAll('.outlier-dot')
      .data(outliers)
      .enter().append('circle')
      .attr('class', 'outlier-dot')
      .attr('cx', d => xS(d.index))
      .attr('cy', d => yScale(d.value))
      .attr('r', 2.5)
      .attr('fill', '#F8766D')
  }
  drawOutliers(xScale)

  // Axes
  const xAxis = main.append('g')
    .attr('transform', `translate(0,${mainHeight})`)
    .call(d3.axisBottom(xScale).ticks(8).tickFormat(i => {
      const d = data[Math.round(i)]
      return d?.time ? d.time.split(' ')[0] : ''
    }))

  xAxis.selectAll('text')
    .attr('transform', 'rotate(-30)')
    .style('text-anchor', 'end')
    .style('font-size', '11px')

  const yAxis = main.append('g')
    .call(d3.axisLeft(yScale).ticks(6))

  // Y-axis label
  main.append('text')
    .attr('transform', 'rotate(-90)')
    .attr('y', -55)
    .attr('x', -mainHeight / 2)
    .attr('text-anchor', 'middle')
    .style('font-size', '12px')
    .style('fill', '#666')
    .text(props.pointName)

  // ---- Tooltip ----
  const tooltipLine = main.append('line')
    .attr('stroke', '#999')
    .attr('stroke-dasharray', '3,3')
    .attr('y1', 0)
    .attr('y2', mainHeight)
    .style('display', 'none')

  const tooltipDot = main.append('circle')
    .attr('r', 4)
    .attr('fill', '#409EFF')
    .attr('stroke', '#fff')
    .attr('stroke-width', 2)
    .style('display', 'none')

  const tooltip = d3.select(tooltipRef.value)

  // Hover overlay
  main.append('rect')
    .attr('width', totalWidth - margin.left - margin.right)
    .attr('height', mainHeight)
    .attr('fill', 'transparent')
    .on('mousemove', function(event) {
      const [mx] = d3.pointer(event)
      const idx = Math.round(currentXScale.invert(mx))
      const d = data[Math.max(0, Math.min(data.length - 1, idx))]
      if (!d || d.value == null) return

      tooltipLine.style('display', null)
        .attr('x1', currentXScale(d.index))
        .attr('x2', currentXScale(d.index))

      tooltipDot.style('display', null)
        .attr('cx', currentXScale(d.index))
        .attr('cy', yScale(d.value))

      tooltip.style('display', 'block')
        .style('left', `${event.offsetX + 15}px`)
        .style('top', `${event.offsetY - 10}px`)
        .html(`
          <div><b>${d.time || 'idx: ' + d.index}</b></div>
          <div>Value: ${d.value.toFixed(4)}</div>
          ${d.outlier ? '<div style="color:#F8766D">Outlier</div>' : ''}
        `)
    })
    .on('mouseleave', function() {
      tooltipLine.style('display', 'none')
      tooltipDot.style('display', 'none')
      tooltip.style('display', 'none')
    })

  // ---- Context chart (overview) ----
  const contextTop = margin.top + mainHeight + gap
  const context = svg.append('g')
    .attr('transform', `translate(${contextMargin.left},${contextTop})`)

  // Downsample for context
  const step = Math.max(1, Math.floor(data.length / 1000))
  const contextData = data.filter((_, i) => i % step === 0)

  const contextLine = d3.line()
    .defined(d => d.value != null)
    .x(d => contextXScale(d.index))
    .y(d => contextYScale(d.value))

  // Context segment highlights
  for (const seg of props.segments) {
    context.append('rect')
      .attr('x', contextXScale(seg.start))
      .attr('width', Math.max(1, contextXScale(seg.end) - contextXScale(seg.start)))
      .attr('y', 0)
      .attr('height', contextHeight)
      .attr('fill', '#F8766D')
      .attr('opacity', 0.2)
  }

  context.append('path')
    .datum(contextData)
    .attr('fill', 'none')
    .attr('stroke', '#409EFF')
    .attr('stroke-width', 0.8)
    .attr('d', contextLine)

  context.append('g')
    .attr('transform', `translate(0,${contextHeight})`)
    .call(d3.axisBottom(contextXScale).ticks(6).tickFormat(i => {
      const d = data[Math.round(i)]
      return d?.time ? d.time.split(' ')[0] : ''
    }))
    .selectAll('text')
    .style('font-size', '10px')

  // ---- Brush on context for zoom ----
  let currentXScale = xScale.copy()

  const brush = d3.brushX()
    .extent([[0, 0], [totalWidth - contextMargin.left - contextMargin.right, contextHeight]])
    .on('brush end', function(event) {
      if (!event.selection) {
        // Reset to full view
        currentXScale = xScale.copy()
      } else {
        const [x0, x1] = event.selection.map(contextXScale.invert)
        currentXScale = xScale.copy().domain([x0, x1])
      }

      // Update main chart
      const newLine = d3.line()
        .defined(d => d.value != null)
        .x(d => currentXScale(d.index))
        .y(d => yScale(d.value))

      linePath.attr('d', newLine)

      xAxis.call(d3.axisBottom(currentXScale).ticks(8).tickFormat(i => {
        const d = data[Math.round(i)]
        return d?.time ? d.time.split(' ')[0] : ''
      }))
      xAxis.selectAll('text')
        .attr('transform', 'rotate(-30)')
        .style('text-anchor', 'end')
        .style('font-size', '11px')

      drawSegments(currentXScale)
      drawOutliers(currentXScale)
    })

  context.append('g')
    .attr('class', 'brush')
    .call(brush)
}

onMounted(() => {
  nextTick(render)
  resizeObserver = new ResizeObserver(() => render())
  if (chartRef.value) resizeObserver.observe(chartRef.value)
})

onUnmounted(() => {
  if (resizeObserver) resizeObserver.disconnect()
})

watch(() => [props.series, props.segments], () => nextTick(render), { deep: true })
</script>

<template>
  <div style="position: relative">
    <div ref="chartRef" style="width: 100%; min-height: 460px"></div>
    <div
      ref="tooltipRef"
      style="display: none; position: absolute; background: rgba(255,255,255,0.95); border: 1px solid #ddd; border-radius: 4px; padding: 6px 10px; font-size: 12px; pointer-events: none; box-shadow: 0 2px 8px rgba(0,0,0,0.1); z-index: 10"
    ></div>
  </div>
</template>
