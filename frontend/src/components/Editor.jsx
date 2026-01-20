/**
 * Editor Component
 * 
 * Full-featured sticker editor with:
 * - Manual mask editing (erase/restore brushes)
 * - Re-background testing
 * - Border customization
 * - Re-analyze capability
 * - Undo/Redo support
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage } from 'react-konva'
import useImage from 'use-image'
import { HexColorPicker } from 'react-colorful'
import {
    getImageUrl,
    generateBorder,
    downloadImage,
    saveMask,
    generateRebackground,
    getBackgroundPresets,
    reanalyzeJob,
    getJob
} from '../api'

// View modes for the canvas
const VIEW_MODES = {
    FINAL: 'final',   // Show final sticker
    MASK: 'mask',     // Show mask for editing
    EDGE: 'edge'      // Show edge overlay
}

// Brush tools
const TOOLS = {
    NONE: 'none',
    ERASE: 'erase',     // Remove background (paint black on mask)
    RESTORE: 'restore'  // Restore sticker (paint white on mask)
}

function Editor({ job, onBack, onJobUpdate }) {
    // =========================================================================
    // STATE
    // =========================================================================

    // View
    const [viewMode, setViewMode] = useState(VIEW_MODES.FINAL)
    const [stageSize, setStageSize] = useState({ width: 800, height: 600 })

    // Tools
    const [activeTool, setActiveTool] = useState(TOOLS.NONE)
    const [brushSize, setBrushSize] = useState(20)
    const [isDrawing, setIsDrawing] = useState(false)

    // Mask editing
    const [maskCanvas, setMaskCanvas] = useState(null)
    const [maskHistory, setMaskHistory] = useState([])
    const [historyIndex, setHistoryIndex] = useState(-1)
    const [isSaving, setIsSaving] = useState(false)
    const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)

    // Border
    const [borderEnabled, setBorderEnabled] = useState(false)
    const [borderThickness, setBorderThickness] = useState(10)
    const [borderColor, setBorderColor] = useState('#ffffff')
    const [hexInput, setHexInput] = useState('#ffffff')
    const [showColorPicker, setShowColorPicker] = useState(false)
    const [isGeneratingBorder, setIsGeneratingBorder] = useState(false)

    // Re-background
    const [backgroundPresets, setBackgroundPresets] = useState([])
    const [activeBackground, setActiveBackground] = useState('transparent')
    const [previewUrl, setPreviewUrl] = useState(null)

    // Re-analyze
    const [isReanalyzing, setIsReanalyzing] = useState(false)

    // Refs
    const containerRef = useRef(null)
    const stageRef = useRef(null)
    const borderTimeoutRef = useRef(null)

    // =========================================================================
    // IMAGE LOADING
    // =========================================================================

    const transparentUrl = getImageUrl(job.paths?.transparent)
    const maskUrl = getImageUrl(job.paths?.edited_mask || job.paths?.mask)
    const borderUrl = getImageUrl(job.paths?.with_border)
    const originalUrl = getImageUrl(job.paths?.original)

    const [transparentImage] = useImage(transparentUrl, 'anonymous')
    const [maskImage] = useImage(maskUrl, 'anonymous')
    const [borderImage] = useImage(borderUrl, 'anonymous')
    const [previewImage] = useImage(previewUrl, 'anonymous')

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    // Load background presets
    useEffect(() => {
        getBackgroundPresets()
            .then(data => setBackgroundPresets(data.presets || []))
            .catch(console.error)
    }, [])

    // Initialize mask canvas when mask image loads
    useEffect(() => {
        if (maskImage && !maskCanvas) {
            const canvas = document.createElement('canvas')
            canvas.width = maskImage.width
            canvas.height = maskImage.height
            const ctx = canvas.getContext('2d')
            ctx.drawImage(maskImage, 0, 0)
            setMaskCanvas(canvas)

            // Initialize history
            setMaskHistory([canvas.toDataURL('image/png')])
            setHistoryIndex(0)
        }
    }, [maskImage, maskCanvas])

    // Resize stage to fit container
    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect()
                setStageSize({
                    width: Math.max(rect.width - 20, 400),
                    height: Math.max(rect.height - 20, 300)
                })
            }
        }
        updateSize()
        window.addEventListener('resize', updateSize)
        return () => window.removeEventListener('resize', updateSize)
    }, [])

    // =========================================================================
    // COMPUTED VALUES
    // =========================================================================

    const getImageDimensions = useCallback(() => {
        const img = transparentImage || maskImage
        if (!img) return { x: 0, y: 0, width: 0, height: 0, scale: 1 }

        const padding = 60
        const maxWidth = stageSize.width - padding * 2
        const maxHeight = stageSize.height - padding * 2

        const imgScale = Math.min(maxWidth / img.width, maxHeight / img.height, 1)
        const width = img.width * imgScale
        const height = img.height * imgScale
        const x = (stageSize.width - width) / 2
        const y = (stageSize.height - height) / 2

        return { x, y, width, height, scale: imgScale }
    }, [transparentImage, maskImage, stageSize])

    const dims = getImageDimensions()
    const canUndo = historyIndex > 0
    const canRedo = historyIndex < maskHistory.length - 1

    // =========================================================================
    // BRUSH DRAWING
    // =========================================================================

    const handleMouseDown = (e) => {
        if (activeTool === TOOLS.NONE || !maskCanvas) return
        setIsDrawing(true)
        draw(e)
    }

    const handleMouseMove = (e) => {
        if (!isDrawing || activeTool === TOOLS.NONE || !maskCanvas) return
        draw(e)
    }

    const handleMouseUp = () => {
        if (isDrawing && maskCanvas) {
            setIsDrawing(false)
            // Save to history
            const newData = maskCanvas.toDataURL('image/png')
            const newHistory = maskHistory.slice(0, historyIndex + 1)
            newHistory.push(newData)
            setMaskHistory(newHistory)
            setHistoryIndex(newHistory.length - 1)
            setHasUnsavedChanges(true)
        }
    }

    const draw = (e) => {
        if (!maskCanvas || !stageRef.current) return

        const stage = stageRef.current
        const pos = stage.getPointerPosition()
        if (!pos) return

        // Convert screen coords to mask coords
        const maskX = (pos.x - dims.x) / dims.scale
        const maskY = (pos.y - dims.y) / dims.scale

        const ctx = maskCanvas.getContext('2d')
        ctx.beginPath()
        ctx.arc(maskX, maskY, brushSize / 2, 0, Math.PI * 2)
        ctx.fillStyle = activeTool === TOOLS.ERASE ? 'black' : 'white'
        ctx.fill()

        // Force re-render by creating new canvas reference
        setMaskCanvas(prev => {
            const newCanvas = document.createElement('canvas')
            newCanvas.width = prev.width
            newCanvas.height = prev.height
            newCanvas.getContext('2d').drawImage(prev, 0, 0)
            return newCanvas
        })
    }

    // =========================================================================
    // UNDO / REDO
    // =========================================================================

    const handleUndo = useCallback(() => {
        if (!canUndo || !maskCanvas) return
        const newIndex = historyIndex - 1
        setHistoryIndex(newIndex)

        const img = new Image()
        img.onload = () => {
            const ctx = maskCanvas.getContext('2d')
            ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
            ctx.drawImage(img, 0, 0)
            setMaskCanvas(c => {
                const n = document.createElement('canvas')
                n.width = c.width; n.height = c.height
                n.getContext('2d').drawImage(c, 0, 0)
                return n
            })
        }
        img.src = maskHistory[newIndex]
    }, [canUndo, historyIndex, maskHistory, maskCanvas])

    const handleRedo = useCallback(() => {
        if (!canRedo || !maskCanvas) return
        const newIndex = historyIndex + 1
        setHistoryIndex(newIndex)

        const img = new Image()
        img.onload = () => {
            const ctx = maskCanvas.getContext('2d')
            ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
            ctx.drawImage(img, 0, 0)
            setMaskCanvas(c => {
                const n = document.createElement('canvas')
                n.width = c.width; n.height = c.height
                n.getContext('2d').drawImage(c, 0, 0)
                return n
            })
        }
        img.src = maskHistory[newIndex]
    }, [canRedo, historyIndex, maskHistory, maskCanvas])

    // =========================================================================
    // SAVE MASK
    // =========================================================================

    const handleSaveMask = async () => {
        if (!maskCanvas) return
        setIsSaving(true)
        try {
            const maskData = maskCanvas.toDataURL('image/png')
            const result = await saveMask(job.id, maskData)
            if (result.success) {
                onJobUpdate({ ...job, paths: result.paths })
                setHasUnsavedChanges(false)
            }
        } catch (error) {
            console.error('Save failed:', error)
            alert('Failed to save mask')
        } finally {
            setIsSaving(false)
        }
    }

    // =========================================================================
    // RE-ANALYZE
    // =========================================================================

    const handleReanalyze = async () => {
        if (!confirm('This will reprocess the image through AI. Any manual edits will be lost. Continue?')) {
            return
        }

        setIsReanalyzing(true)
        try {
            await reanalyzeJob(job.id)
            // Poll for completion
            const pollForComplete = async () => {
                const updatedJob = await getJob(job.id)
                if (updatedJob.status === 'complete') {
                    onJobUpdate(updatedJob)
                    setMaskCanvas(null) // Reset to reload new mask
                    setHasUnsavedChanges(false)
                    setIsReanalyzing(false)
                } else if (updatedJob.status === 'error') {
                    alert('Re-analysis failed: ' + updatedJob.error)
                    setIsReanalyzing(false)
                } else {
                    setTimeout(pollForComplete, 1000)
                }
            }
            pollForComplete()
        } catch (error) {
            console.error('Re-analyze failed:', error)
            alert('Failed to start re-analysis')
            setIsReanalyzing(false)
        }
    }

    // =========================================================================
    // BORDER GENERATION
    // =========================================================================

    const handleGenerateBorder = useCallback(async () => {
        if (!borderEnabled) return
        setIsGeneratingBorder(true)
        try {
            const result = await generateBorder(job.id, borderThickness, borderColor)
            onJobUpdate({ ...job, paths: { ...job.paths, with_border: result.path } })
        } catch (error) {
            console.error('Border generation failed:', error)
        } finally {
            setIsGeneratingBorder(false)
        }
    }, [job, borderThickness, borderColor, borderEnabled, onJobUpdate])

    // Debounced border generation
    useEffect(() => {
        if (!borderEnabled) return
        if (borderTimeoutRef.current) clearTimeout(borderTimeoutRef.current)
        borderTimeoutRef.current = setTimeout(handleGenerateBorder, 300)
        return () => { if (borderTimeoutRef.current) clearTimeout(borderTimeoutRef.current) }
    }, [borderThickness, borderColor, borderEnabled])

    // =========================================================================
    // RE-BACKGROUND
    // =========================================================================

    const handleBackgroundChange = async (preset) => {
        setActiveBackground(preset.id)

        if (preset.id === 'transparent') {
            setPreviewUrl(null)
            return
        }

        try {
            const result = await generateRebackground(job.id, {
                type: preset.type,
                color1: preset.color1,
                color2: preset.color2
            })
            setPreviewUrl(getImageUrl(result.path) + '?t=' + Date.now())
        } catch (error) {
            console.error('Background generation failed:', error)
        }
    }

    // =========================================================================
    // DOWNLOAD
    // =========================================================================

    const handleDownload = async (withBorder = false) => {
        try {
            await downloadImage(job.id, withBorder && borderEnabled)
        } catch (error) {
            console.error('Download failed:', error)
            alert('Download failed')
        }
    }

    // =========================================================================
    // HELPERS
    // =========================================================================

    const handleHexChange = (value) => {
        setHexInput(value)
        if (/^#[0-9A-Fa-f]{6}$/.test(value)) {
            setBorderColor(value)
        }
    }

    const getDisplayImage = () => {
        if (viewMode === VIEW_MODES.MASK) return null
        if (activeBackground !== 'transparent' && previewImage) return previewImage
        if (borderEnabled && borderImage) return borderImage
        return transparentImage
    }

    const presetColors = ['#ffffff', '#000000', '#ef4444', '#f97316', '#eab308', '#22c55e', '#3b82f6', '#8b5cf6']

    // =========================================================================
    // RENDER
    // =========================================================================

    const isLoading = isGeneratingBorder || isSaving || isReanalyzing

    return (
        <div className="editor-panel">
            {/* Canvas Area */}
            <div className="canvas-container" ref={containerRef}>
                <div className="canvas-checker" />

                <Stage
                    ref={stageRef}
                    width={stageSize.width}
                    height={stageSize.height}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    style={{ cursor: activeTool !== TOOLS.NONE ? 'crosshair' : 'default' }}
                >
                    <Layer>
                        {/* Main image */}
                        {viewMode !== VIEW_MODES.MASK && getDisplayImage() && (
                            <KonvaImage
                                image={getDisplayImage()}
                                x={dims.x} y={dims.y}
                                width={dims.width} height={dims.height}
                            />
                        )}

                        {/* Mask view */}
                        {viewMode === VIEW_MODES.MASK && maskCanvas && (
                            <KonvaImage
                                image={maskCanvas}
                                x={dims.x} y={dims.y}
                                width={dims.width} height={dims.height}
                            />
                        )}

                        {/* Edge overlay */}
                        {viewMode === VIEW_MODES.EDGE && maskImage && (
                            <KonvaImage
                                image={maskImage}
                                x={dims.x} y={dims.y}
                                width={dims.width} height={dims.height}
                                opacity={0.4}
                            />
                        )}
                    </Layer>
                </Stage>

                {/* Loading overlay */}
                {isLoading && (
                    <div className="canvas-loading">
                        <div className="spinner" />
                        <span>
                            {isReanalyzing ? 'Re-analyzing...' : isSaving ? 'Saving...' : 'Generating...'}
                        </span>
                    </div>
                )}

                {/* View mode toggle */}
                <div className="view-mode-toggle">
                    <button className={`btn btn-icon ${viewMode === VIEW_MODES.FINAL ? 'active' : ''}`}
                        onClick={() => setViewMode(VIEW_MODES.FINAL)} title="Final View">🖼️</button>
                    <button className={`btn btn-icon ${viewMode === VIEW_MODES.EDGE ? 'active' : ''}`}
                        onClick={() => setViewMode(VIEW_MODES.EDGE)} title="Edge Overlay">✂️</button>
                    <button className={`btn btn-icon ${viewMode === VIEW_MODES.MASK ? 'active' : ''}`}
                        onClick={() => setViewMode(VIEW_MODES.MASK)} title="Mask View">🎭</button>
                </div>
            </div>

            {/* Controls Panel */}
            <div className="controls-panel">
                {/* Header */}
                <div className="editor-header">
                    <button className="btn btn-secondary" onClick={onBack}>← Back</button>
                    <span className="editor-filename" title={job.filename}>{job.filename}</span>
                </div>

                {/* Quality Info */}
                {job.edge_confidence && (
                    <div className={`quality-badge ${job.edge_confidence}`}>
                        <span className="quality-icon">
                            {job.edge_confidence === 'high' ? '✅' : job.edge_confidence === 'medium' ? '✓' : '⚠️'}
                        </span>
                        <span>Edge Quality: {job.edge_confidence}</span>
                    </div>
                )}

                {/* Manual Edit Tools */}
                <div className="control-section">
                    <div className="section-header">
                        <span className="section-title">Manual Edit</span>
                        {hasUnsavedChanges && <span className="unsaved-badge">Unsaved</span>}
                    </div>

                    <div className="tool-buttons">
                        <button className={`btn btn-tool ${activeTool === TOOLS.ERASE ? 'active erase' : ''}`}
                            onClick={() => setActiveTool(activeTool === TOOLS.ERASE ? TOOLS.NONE : TOOLS.ERASE)}>
                            🧹 Erase
                        </button>
                        <button className={`btn btn-tool ${activeTool === TOOLS.RESTORE ? 'active restore' : ''}`}
                            onClick={() => setActiveTool(activeTool === TOOLS.RESTORE ? TOOLS.NONE : TOOLS.RESTORE)}>
                            ✨ Restore
                        </button>
                    </div>

                    {activeTool !== TOOLS.NONE && (
                        <div className="slider-container">
                            <div className="slider-header">
                                <span>Brush Size</span>
                                <span className="slider-value">{brushSize}px</span>
                            </div>
                            <input type="range" className="slider" min="5" max="100" value={brushSize}
                                onChange={(e) => setBrushSize(parseInt(e.target.value))} />
                        </div>
                    )}

                    <div className="button-row">
                        <button className="btn btn-icon" onClick={handleUndo} disabled={!canUndo} title="Undo">↩️</button>
                        <button className="btn btn-icon" onClick={handleRedo} disabled={!canRedo} title="Redo">↪️</button>
                        <button className="btn btn-success flex-1" onClick={handleSaveMask}
                            disabled={isSaving || !hasUnsavedChanges}>
                            💾 {isSaving ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>
                </div>

                {/* Re-analyze */}
                <div className="control-section">
                    <span className="section-title">AI Processing</span>
                    <button className="btn btn-secondary" onClick={handleReanalyze} disabled={isReanalyzing}>
                        🔄 {isReanalyzing ? 'Processing...' : 'Re-Analyze with AI'}
                    </button>
                    <p className="hint-text">Reprocess the image if the initial extraction wasn't perfect</p>
                </div>

                {/* Test Backgrounds */}
                <div className="control-section">
                    <span className="section-title">Test Backgrounds</span>
                    <div className="background-presets">
                        {backgroundPresets.map(preset => (
                            <button key={preset.id}
                                className={`btn btn-preset ${activeBackground === preset.id ? 'active' : ''}`}
                                style={{ backgroundColor: preset.color1 || '#888' }}
                                onClick={() => handleBackgroundChange(preset)}
                                title={preset.name}>
                                {preset.name}
                            </button>
                        ))}
                    </div>
                </div>

                {/* Border */}
                <div className="control-section">
                    <span className="section-title">Border</span>
                    <div className="toggle">
                        <span>Add Border</span>
                        <div className={`toggle-switch ${borderEnabled ? 'active' : ''}`}
                            onClick={() => setBorderEnabled(!borderEnabled)} />
                    </div>

                    {borderEnabled && (
                        <>
                            <div className="slider-container">
                                <div className="slider-header">
                                    <span>Thickness</span>
                                    <span className="slider-value">{borderThickness}px</span>
                                </div>
                                <input type="range" className="slider" min="1" max="50" value={borderThickness}
                                    onChange={(e) => setBorderThickness(parseInt(e.target.value))} />
                            </div>

                            <div className="color-picker-container">
                                <span className="color-label">Color</span>
                                <div className="color-preview">
                                    <div className="color-swatch" style={{ backgroundColor: borderColor }}
                                        onClick={() => setShowColorPicker(!showColorPicker)} />
                                    <input type="text" className="hex-input" value={hexInput}
                                        onChange={(e) => handleHexChange(e.target.value)} placeholder="#FFFFFF" />
                                </div>
                                {showColorPicker && (
                                    <div className="color-picker-dropdown">
                                        <HexColorPicker color={borderColor} onChange={(c) => { setBorderColor(c); setHexInput(c); }} />
                                        <div className="preset-colors">
                                            {presetColors.map(color => (
                                                <div key={color} className={`preset-swatch ${borderColor === color ? 'active' : ''}`}
                                                    style={{ backgroundColor: color }}
                                                    onClick={() => { setBorderColor(color); setHexInput(color); }} />
                                            ))}
                                        </div>
                                    </div>
                                )}
                            </div>
                        </>
                    )}
                </div>

                {/* Export */}
                <div className="control-section export-section">
                    <span className="section-title">Export</span>
                    <button className="btn btn-primary btn-large" onClick={() => handleDownload(true)}>
                        📥 Download PNG
                    </button>
                    {borderEnabled && (
                        <button className="btn btn-secondary" onClick={() => handleDownload(false)}>
                            Download without border
                        </button>
                    )}
                </div>
            </div>
        </div>
    )
}

export default Editor
