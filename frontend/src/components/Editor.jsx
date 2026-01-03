import { useState, useEffect, useRef, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage, Line } from 'react-konva'
import useImage from 'use-image'
import { HexColorPicker } from 'react-colorful'
import { getImageUrl, generateBorder, downloadImage } from '../api'

/**
 * STICKER EDITOR - Precision Sticker Cutter
 * 
 * This is NOT an image editor.
 * It is a precision tool for sticker cutting and border generation.
 * 
 * Three visual layers:
 * 1. Sticker Preview (final composited output)
 * 2. Edge Overlay Mode (shows boundary clearly)
 * 3. Border Preview Layer
 */

// View modes for the editor
const VIEW_MODES = {
    FINAL: 'final',      // Show final sticker only
    EDGE: 'edge',        // Show cut edge overlay
    MASK: 'mask'         // Show mask only
}

function Editor({ job, onBack, onJobUpdate }) {
    // Border state
    const [borderEnabled, setBorderEnabled] = useState(false)
    const [borderThickness, setBorderThickness] = useState(10)
    const [borderColor, setBorderColor] = useState('#ffffff')
    const [hexInput, setHexInput] = useState('#ffffff')
    const [showColorPicker, setShowColorPicker] = useState(false)

    // View state
    const [viewMode, setViewMode] = useState(VIEW_MODES.FINAL)
    const [isGenerating, setIsGenerating] = useState(false)
    const [stageSize, setStageSize] = useState({ width: 800, height: 600 })

    // Refs
    const containerRef = useRef(null)
    const borderTimeoutRef = useRef(null)

    // Image URLs
    const transparentUrl = getImageUrl(job.paths?.transparent)
    const maskUrl = getImageUrl(job.paths?.mask)
    const borderUrl = getImageUrl(job.paths?.with_border)

    // Load images
    const [transparentImage] = useImage(transparentUrl, 'anonymous')
    const [maskImage] = useImage(maskUrl, 'anonymous')
    const [borderImage] = useImage(borderUrl, 'anonymous')

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

    // Calculate image dimensions to fit in stage (pixel-accurate zoom)
    const getImageDimensions = useCallback(() => {
        const img = transparentImage || maskImage
        if (!img) return { x: 0, y: 0, width: 0, height: 0, scale: 1 }

        const padding = 60
        const maxWidth = stageSize.width - padding * 2
        const maxHeight = stageSize.height - padding * 2

        const scale = Math.min(maxWidth / img.width, maxHeight / img.height, 1)
        const width = img.width * scale
        const height = img.height * scale
        const x = (stageSize.width - width) / 2
        const y = (stageSize.height - height) / 2

        return { x, y, width, height, scale }
    }, [transparentImage, maskImage, stageSize])

    // Generate border with debounce (no lag on slider)
    const handleGenerateBorder = useCallback(async () => {
        if (!borderEnabled) return

        setIsGenerating(true)
        try {
            const result = await generateBorder(job.id, borderThickness, borderColor)
            onJobUpdate({
                ...job,
                paths: { ...job.paths, with_border: result.path }
            })
        } catch (error) {
            console.error('Border generation failed:', error)
        } finally {
            setIsGenerating(false)
        }
    }, [job, borderThickness, borderColor, borderEnabled, onJobUpdate])

    // Debounced border generation
    useEffect(() => {
        if (!borderEnabled) return

        // Clear previous timeout
        if (borderTimeoutRef.current) {
            clearTimeout(borderTimeoutRef.current)
        }

        // Debounce 300ms for smooth slider feel
        borderTimeoutRef.current = setTimeout(() => {
            handleGenerateBorder()
        }, 300)

        return () => {
            if (borderTimeoutRef.current) {
                clearTimeout(borderTimeoutRef.current)
            }
        }
    }, [borderThickness, borderColor, borderEnabled])

    // Handle hex input
    const handleHexChange = (value) => {
        setHexInput(value)
        if (/^#[0-9A-Fa-f]{6}$/.test(value)) {
            setBorderColor(value)
        }
    }

    const handleDownload = async (withBorder = false) => {
        try {
            await downloadImage(job.id, withBorder && borderEnabled)
        } catch (error) {
            console.error('Download failed:', error)
        }
    }

    const dims = getImageDimensions()

    // Determine which image to display based on view mode
    const getDisplayImage = () => {
        if (viewMode === VIEW_MODES.MASK) return maskImage
        if (borderEnabled && borderImage && viewMode === VIEW_MODES.FINAL) return borderImage
        return transparentImage
    }

    const displayImage = getDisplayImage()

    // Preset colors for quick selection
    const presetColors = [
        '#ffffff', '#000000', '#ef4444', '#f97316',
        '#eab308', '#22c55e', '#3b82f6', '#8b5cf6'
    ]

    return (
        <div className="editor-panel">
            {/* Canvas Area */}
            <div className="canvas-container" ref={containerRef}>
                {/* Checker pattern for transparency */}
                <div className="canvas-checker" />

                <Stage width={stageSize.width} height={stageSize.height}>
                    <Layer>
                        {displayImage && (
                            <KonvaImage
                                image={displayImage}
                                x={dims.x}
                                y={dims.y}
                                width={dims.width}
                                height={dims.height}
                            />
                        )}

                        {/* Edge overlay when in EDGE mode */}
                        {viewMode === VIEW_MODES.EDGE && maskImage && (
                            <KonvaImage
                                image={maskImage}
                                x={dims.x}
                                y={dims.y}
                                width={dims.width}
                                height={dims.height}
                                opacity={0.5}
                                globalCompositeOperation="source-over"
                            />
                        )}
                    </Layer>
                </Stage>

                {/* Loading overlay */}
                {isGenerating && (
                    <div className="canvas-loading">
                        <div className="spinner" />
                        <span>Generating border...</span>
                    </div>
                )}

                {/* View mode toggle */}
                <div className="view-mode-toggle">
                    <button
                        className={`btn btn-icon ${viewMode === VIEW_MODES.FINAL ? 'active' : ''}`}
                        onClick={() => setViewMode(VIEW_MODES.FINAL)}
                        title="Show Final Only"
                    >
                        🖼️
                    </button>
                    <button
                        className={`btn btn-icon ${viewMode === VIEW_MODES.EDGE ? 'active' : ''}`}
                        onClick={() => setViewMode(VIEW_MODES.EDGE)}
                        title="Show Cut Edge"
                    >
                        ✂️
                    </button>
                    <button
                        className={`btn btn-icon ${viewMode === VIEW_MODES.MASK ? 'active' : ''}`}
                        onClick={() => setViewMode(VIEW_MODES.MASK)}
                        title="Show Mask"
                    >
                        🎭
                    </button>
                </div>
            </div>

            {/* Controls Panel */}
            <div className="controls-panel">
                {/* Back button */}
                <button className="btn btn-secondary" onClick={onBack}>
                    ← Back to Gallery
                </button>

                {/* Validation Status */}
                {job.edge_confidence && (
                    <div className="control-section">
                        <span className="control-section-title">Quality Check</span>
                        <div className={`validation-status ${job.edge_confidence}`}>
                            <span className="validation-icon">
                                {job.edge_confidence === 'high' && '✅'}
                                {job.edge_confidence === 'medium' && '✓'}
                                {job.edge_confidence === 'low' && '⚠️'}
                                {job.edge_confidence === 'failed' && '❌'}
                            </span>
                            <span>Edge: {job.edge_confidence}</span>
                        </div>
                        {job.warnings && job.warnings.length > 0 && (
                            <div className="warnings-list">
                                {job.warnings.map((w, i) => (
                                    <div key={i} className="warning-item">⚠️ {w}</div>
                                ))}
                            </div>
                        )}
                        {job.needs_review && (
                            <div className="needs-review-badge">
                                Manual review recommended
                            </div>
                        )}
                    </div>
                )}

                {/* File info */}
                <div className="control-section">
                    <span className="control-section-title">File</span>
                    <div className="file-info">
                        {job.filename}
                    </div>
                </div>

                {/* Border Controls */}
                <div className="control-section">
                    <span className="control-section-title">Border / Outline</span>

                    {/* Toggle */}
                    <div className="toggle">
                        <span>Enable Border</span>
                        <div
                            className={`toggle-switch ${borderEnabled ? 'active' : ''}`}
                            onClick={() => setBorderEnabled(!borderEnabled)}
                        />
                    </div>

                    {borderEnabled && (
                        <>
                            {/* Thickness - Analog feel slider */}
                            <div className="slider-container">
                                <div className="slider-header">
                                    <span>Thickness</span>
                                    <span className="slider-value">{borderThickness}px</span>
                                </div>
                                <input
                                    type="range"
                                    className="slider"
                                    min="1"
                                    max="50"
                                    value={borderThickness}
                                    onChange={(e) => setBorderThickness(parseInt(e.target.value))}
                                />
                            </div>

                            {/* Color with HEX input */}
                            <div className="color-picker-container">
                                <span className="color-label">Color</span>
                                <div className="color-preview">
                                    <div
                                        className="color-swatch"
                                        style={{ backgroundColor: borderColor }}
                                        onClick={() => setShowColorPicker(!showColorPicker)}
                                    />
                                    <input
                                        type="text"
                                        className="hex-input"
                                        value={hexInput}
                                        onChange={(e) => handleHexChange(e.target.value)}
                                        placeholder="#FFFFFF"
                                    />
                                </div>

                                {showColorPicker && (
                                    <div className="color-picker-dropdown">
                                        <HexColorPicker
                                            color={borderColor}
                                            onChange={(c) => {
                                                setBorderColor(c)
                                                setHexInput(c)
                                            }}
                                        />
                                        {/* Preset colors */}
                                        <div className="preset-colors">
                                            {presetColors.map(color => (
                                                <div
                                                    key={color}
                                                    className={`preset-swatch ${borderColor === color ? 'active' : ''}`}
                                                    style={{ backgroundColor: color }}
                                                    onClick={() => {
                                                        setBorderColor(color)
                                                        setHexInput(color)
                                                    }}
                                                />
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
                    <span className="control-section-title">Export</span>

                    <button
                        className="btn btn-primary btn-large"
                        onClick={() => handleDownload(true)}
                    >
                        📥 Download PNG
                    </button>

                    {borderEnabled && (
                        <button
                            className="btn btn-secondary"
                            onClick={() => handleDownload(false)}
                        >
                            Download without border
                        </button>
                    )}
                </div>
            </div>
        </div>
    )
}

export default Editor
