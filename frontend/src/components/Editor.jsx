/**
 * Editor Component
 * 
 * Professional sticker editor with:
 * - Pixel-accurate mask editing (interpolation + softness)
 * - Multi-layered canvas (Background, Image, Mask, Preview)
 * - High-precision Zoom & Pan (Wheel + Space-drag)
 * - Undo/Redo history
 * - Border and Test backgrounds
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { Stage, Layer, Image as KonvaImage, Circle, Group } from 'react-konva'
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

// View modes
const VIEW_MODES = {
    FINAL: 'final',     // Result on test background
    MASK: 'mask',       // Pure black/white mask
    OVERLAY: 'overlay'  // Image + colored mask overlay
}

// Brush tools
const TOOLS = {
    NONE: 'none',
    ERASE: 'erase',     // Paint black on mask
    RESTORE: 'restore', // Paint white on mask
    PAN: 'pan'          // Drag to move
}

function Editor({ job, onBack, onJobUpdate }) {
    // =========================================================================
    // STATE
    // =========================================================================

    // Canvas View State
    const [viewMode, setViewMode] = useState(VIEW_MODES.FINAL)
    const [stageSize, setStageSize] = useState({ width: 800, height: 600 })
    const [scale, setScale] = useState(1)
    const [position, setPosition] = useState({ x: 0, y: 0 })
    const [isSpacePressed, setIsSpacePressed] = useState(false)

    // Brush Tools State
    const [activeTool, setActiveTool] = useState(TOOLS.NONE)
    const [brushSize, setBrushSize] = useState(30)
    const [brushSoftness, setBrushSoftness] = useState(0) // 0 to 100
    const [isDrawing, setIsDrawing] = useState(false)
    const [lastPos, setLastPos] = useState(null)
    const [cursorPos, setCursorPos] = useState({ x: -100, y: -100 })

    // Mask editing state
    const [maskCanvas, setMaskCanvas] = useState(null)
    const [maskHistory, setMaskHistory] = useState([])
    const [historyIndex, setHistoryIndex] = useState(-1)
    const [isSaving, setIsSaving] = useState(false)
    const [hasUnsavedChanges, setHasUnsavedChanges] = useState(false)
    const [maskOpacity, setMaskOpacity] = useState(0.5)

    // Border and UI state
    const [borderEnabled, setBorderEnabled] = useState(false)
    const [borderThickness, setBorderThickness] = useState(10)
    const [borderColor, setBorderColor] = useState('#ffffff')
    const [showColorPicker, setShowColorPicker] = useState(false)
    const [isGeneratingBorder, setIsGeneratingBorder] = useState(false)
    const [backgroundPresets, setBackgroundPresets] = useState([])
    const [activeBackground, setActiveBackground] = useState('checker')
    const [previewUrl, setPreviewUrl] = useState(null)
    const [isReanalyzing, setIsReanalyzing] = useState(false)

    // Refs
    const containerRef = useRef(null)
    const stageRef = useRef(null)
    const maskLayerRef = useRef(null)
    const borderTimeoutRef = useRef(null)

    // =========================================================================
    // IMAGE LOADING
    // =========================================================================

    const transparentUrl = getImageUrl(job.paths?.transparent)
    const maskUrl = getImageUrl(job.paths?.edited_mask || job.paths?.mask)
    const borderUrl = getImageUrl(job.paths?.with_border)
    const originalUrl = getImageUrl(job.paths?.original)

    const [transparentImg] = useImage(transparentUrl, 'anonymous')
    const [maskImg] = useImage(maskUrl, 'anonymous')
    const [borderImg] = useImage(borderUrl, 'anonymous')
    const [originalImg] = useImage(originalUrl, 'anonymous')
    const [previewImg] = useImage(previewUrl, 'anonymous')

    // =========================================================================
    // INITIALIZATION
    // =========================================================================

    useEffect(() => {
        getBackgroundPresets().then(data => setBackgroundPresets(data.presets || []))
        
        // Keyboard listeners
        const handleKeyDown = (e) => {
            if (e.code === 'Space') setIsSpacePressed(true)
            if (e.key === '[') setBrushSize(prev => Math.max(5, prev - 5))
            if (e.key === ']') setBrushSize(prev => Math.min(200, prev + 5))
            if (e.metaKey || e.ctrlKey) {
                if (e.key === 'z') {
                    if (e.shiftKey) handleRedo(); else handleUndo()
                    e.preventDefault()
                }
            }
        }
        const handleKeyUp = (e) => {
            if (e.code === 'Space') setIsSpacePressed(false)
        }
        window.addEventListener('keydown', handleKeyDown)
        window.addEventListener('keyup', handleKeyUp)
        return () => {
            window.removeEventListener('keydown', handleKeyDown)
            window.removeEventListener('keyup', handleKeyUp)
        }
    }, [])

    // Initialize mask canvas
    useEffect(() => {
        if (maskImg && !maskCanvas) {
            const canvas = document.createElement('canvas')
            canvas.width = maskImg.width
            canvas.height = maskImg.height
            const ctx = canvas.getContext('2d')
            ctx.drawImage(maskImg, 0, 0)
            setMaskCanvas(canvas)
            setMaskHistory([canvas.toDataURL('image/png')])
            setHistoryIndex(0)

            // Initial auto-zoom to fit
            const padding = 40
            const availableW = stageSize.width - padding
            const availableH = stageSize.height - padding
            const fitScale = Math.min(availableW / maskImg.width, availableH / maskImg.height, 1)
            setScale(fitScale)
            setPosition({
                x: (stageSize.width - maskImg.width * fitScale) / 2,
                y: (stageSize.height - maskImg.height * fitScale) / 2
            })
        }
    }, [maskImg, maskCanvas, stageSize])

    // Update stage size
    useEffect(() => {
        const updateSize = () => {
            if (containerRef.current) {
                const rect = containerRef.current.getBoundingClientRect()
                setStageSize({ width: rect.width, height: rect.height })
            }
        }
        updateSize(); window.addEventListener('resize', updateSize)
        return () => window.removeEventListener('resize', updateSize)
    }, [])

    // =========================================================================
    // ZOOM & PAN
    // =========================================================================

    const handleWheel = (e) => {
        e.evt.preventDefault()
        const stage = stageRef.current
        const oldScale = scale
        const pointer = stage.getPointerPosition()

        const mousePointTo = {
            x: (pointer.x - position.x) / oldScale,
            y: (pointer.y - position.y) / oldScale,
        }

        const speed = 1.1
        const newScale = e.evt.deltaY < 0 ? oldScale * speed : oldScale / speed
        const clampedScale = Math.max(0.1, Math.min(newScale, 20))

        setScale(clampedScale)
        setPosition({
            x: pointer.x - mousePointTo.x * clampedScale,
            y: pointer.y - mousePointTo.y * clampedScale,
        })
    }

    const handleStageDrag = (e) => {
        if (isSpacePressed || activeTool === TOOLS.PAN) {
            setPosition(e.target.position())
        }
    }

    // =========================================================================
    // BRUSH DRAWING
    // =========================================================================

    const getLocalPointerPos = useCallback(() => {
        const stage = stageRef.current
        const pos = stage.getPointerPosition()
        if (!pos) return null
        return {
            x: (pos.x - position.x) / scale,
            y: (pos.y - position.y) / scale
        }
    }, [position, scale])

    const handleMouseDown = (e) => {
        if (isSpacePressed || activeTool === TOOLS.NONE || activeTool === TOOLS.PAN) return
        setIsDrawing(true)
        const pos = getLocalPointerPos()
        setLastPos(pos)
        draw(pos, pos)
    }

    const handleMouseMove = (e) => {
        const pos = getLocalPointerPos()
        if (pos) setCursorPos(pos)

        if (!isDrawing) return
        draw(lastPos, pos)
        setLastPos(pos)
    }

    const handleMouseUp = () => {
        if (isDrawing) {
            setIsDrawing(false)
            setLastPos(null)
            saveToHistory()
        }
    }

    const draw = (start, end) => {
        if (!maskCanvas || !start || !end) return
        const ctx = maskCanvas.getContext('2d')
        
        ctx.lineJoin = 'round'
        ctx.lineCap = 'round'
        ctx.lineWidth = brushSize
        ctx.strokeStyle = activeTool === TOOLS.ERASE ? 'black' : 'white'
        
        // Softness implementation using shadow or gradient
        if (brushSoftness > 0) {
            ctx.shadowBlur = (brushSize * brushSoftness) / 100
            ctx.shadowColor = ctx.strokeStyle
        } else {
            ctx.shadowBlur = 0
        }

        ctx.beginPath()
        ctx.moveTo(start.x, start.y)
        ctx.lineTo(end.x, end.y)
        ctx.stroke()

        // Sync Konva
        maskLayerRef.current.batchDraw()
        setHasUnsavedChanges(true)
    }

    // =========================================================================
    // HISTORY & SAVE
    // =========================================================================

    const saveToHistory = () => {
        const newData = maskCanvas.toDataURL('image/png')
        const newHistory = maskHistory.slice(0, historyIndex + 1)
        newHistory.push(newData)
        if (newHistory.length > 20) newHistory.shift() // Limit history
        setMaskHistory(newHistory)
        setHistoryIndex(newHistory.length - 1)
    }

    const handleUndo = () => {
        if (historyIndex <= 0) return
        loadHistory(historyIndex - 1)
    }

    const handleRedo = () => {
        if (historyIndex >= maskHistory.length - 1) return
        loadHistory(historyIndex + 1)
    }

    const loadHistory = (index) => {
        const img = new Image()
        img.onload = () => {
            const ctx = maskCanvas.getContext('2d')
            ctx.clearRect(0, 0, maskCanvas.width, maskCanvas.height)
            ctx.drawImage(img, 0, 0)
            setHistoryIndex(index)
            maskLayerRef.current.batchDraw()
        }
        img.src = maskHistory[index]
    }

    const handleSave = async () => {
        setIsSaving(true)
        try {
            const data = maskCanvas.toDataURL('image/png')
            const res = await saveMask(job.id, data)
            onJobUpdate({ ...job, paths: res.paths })
            setHasUnsavedChanges(false)
        } catch (e) { alert('Save failed') }
        finally { setIsSaving(false) }
    }

    const handleReanalyze = async () => {
        setIsReanalyzing(true)
        try {
            await reanalyzeJob(job.id)
            alert('Reprocessing started. Returning to gallery.')
            onBack() 
        } catch (e) { alert('Reprocessing failed') }
        finally { setIsReanalyzing(false) }
    }

    // =========================================================================
    // BORDER & BG
    // =========================================================================

    useEffect(() => {
        if (!borderEnabled) return
        if (borderTimeoutRef.current) clearTimeout(borderTimeoutRef.current)
        borderTimeoutRef.current = setTimeout(async () => {
            setIsGeneratingBorder(true)
            const res = await generateBorder(job.id, borderThickness, borderColor)
            onJobUpdate({ ...job, paths: { ...job.paths, with_border: res.path } })
            setIsGeneratingBorder(false)
        }, 300)
    }, [borderThickness, borderColor, borderEnabled])

    const handleBackgroundChange = async (preset) => {
        setActiveBackground(preset.id)
        if (preset.id === 'checker') { setPreviewUrl(null); return }
        const res = await generateRebackground(job.id, {
            type: preset.type, color1: preset.color1, color2: preset.color2
        })
        setPreviewUrl(getImageUrl(res.path) + '?t=' + Date.now())
    }

    // =========================================================================
    // RENDER HELPERS
    // =========================================================================

    const getResultImage = () => {
        if (activeBackground !== 'checker' && previewImg) return previewImg
        if (borderEnabled && borderImg) return borderImg
        return transparentImg
    }

    const isPanning = isSpacePressed || activeTool === TOOLS.PAN

    return (
        <div className="editor-panel professional">
            {/* Main Stage */}
            <div className="canvas-container" ref={containerRef}>
                <div className={`canvas-checker ${activeBackground}`} />
                
                <Stage
                    ref={stageRef}
                    width={stageSize.width}
                    height={stageSize.height}
                    scaleX={scale} scaleY={scale}
                    x={position.x} y={position.y}
                    onWheel={handleWheel}
                    draggable={isPanning}
                    onDragMove={handleStageDrag}
                    onMouseDown={handleMouseDown}
                    onMouseMove={handleMouseMove}
                    onMouseUp={handleMouseUp}
                    onMouseLeave={handleMouseUp}
                    style={{ cursor: isPanning ? 'grab' : activeTool !== TOOLS.NONE ? 'none' : 'default' }}
                >
                    {/* Layer 1: Original Image */}
                    <Layer>
                        {originalImg && (
                            <KonvaImage image={originalImg} opacity={viewMode === VIEW_MODES.MASK ? 0 : 1} />
                        )}
                    </Layer>

                    {/* Layer 2: Result Layer (overrides original if in FINAL mode) */}
                    <Layer visible={viewMode === VIEW_MODES.FINAL}>
                        {getResultImage() && <KonvaImage image={getResultImage()} />}
                    </Layer>

                    {/* Layer 3: Mask Layer */}
                    <Layer ref={maskLayerRef} visible={viewMode !== VIEW_MODES.FINAL}>
                        {maskCanvas && (
                            <Group opacity={viewMode === VIEW_MODES.OVERLAY ? maskOpacity : 1}>
                                <KonvaImage 
                                    image={maskCanvas} 
                                    filters={viewMode === VIEW_MODES.OVERLAY ? [] : []} // Could add color filter for overlay
                                />
                            </Group>
                        )}
                    </Layer>

                    {/* Layer 4: UI/Cursor */}
                    <Layer>
                        {!isPanning && activeTool !== TOOLS.NONE && (
                            <Circle 
                                x={cursorPos.x} y={cursorPos.y}
                                radius={brushSize / 2}
                                stroke="white" strokeWidth={2 / scale}
                                dash={[4 / scale, 4 / scale]}
                            />
                        )}
                    </Layer>
                </Stage>

                {/* Status Bar */}
                <div className="stage-status">
                    <span>{Math.round(scale * 100)}%</span>
                    <span>{job.edge_confidence} quality</span>
                </div>

                {/* Floating View Controls */}
                <div className="view-mode-toggle professional">
                    <button className={viewMode === VIEW_MODES.FINAL ? 'active' : ''} onClick={() => setViewMode(VIEW_MODES.FINAL)}>Final View</button>
                    <button className={viewMode === VIEW_MODES.OVERLAY ? 'active' : ''} onClick={() => setViewMode(VIEW_MODES.OVERLAY)}>Overlay</button>
                    <button className={viewMode === VIEW_MODES.MASK ? 'active' : ''} onClick={() => setViewMode(VIEW_MODES.MASK)}>Mask Only</button>
                </div>
            </div>

            {/* Sidebar Controls */}
            <div className="controls-panel">
                <div className="panel-section">
                    <button className="btn btn-back" onClick={onBack}>← All Stickers</button>
                    <h2 className="sticker-name">{job.filename}</h2>
                </div>

                {/* Review Section */}
                {job.needs_review && (
                    <div className="panel-section review-alert">
                        <div className="flex-row">
                            <h3 className="section-title warning">Needs Review</h3>
                            <button 
                                className={`btn-reprocess ${isReanalyzing ? 'loading' : ''}`} 
                                onClick={handleReanalyze}
                                disabled={isReanalyzing}
                                title="Run AI engine again to try and fix issues"
                            >
                                {isReanalyzing ? '...' : '🔄 AI Reprocess'}
                            </button>
                        </div>
                        <ul className="review-list">
                            {(job.review_reasons || []).map((reason, i) => (
                                <li key={i} className="review-item">{reason}</li>
                            ))}
                        </ul>
                    </div>
                )}

                {/* Tool Selection */}
                <div className="panel-section">
                    <h3 className="section-title">Edit Mask</h3>
                    <div className="tool-grid">
                        <button className={`btn-tool ${activeTool === TOOLS.ERASE ? 'active' : ''}`} onClick={() => setActiveTool(TOOLS.ERASE)}>
                            <span className="icon">🧹</span> Erase
                        </button>
                        <button className={`btn-tool ${activeTool === TOOLS.RESTORE ? 'active' : ''}`} onClick={() => setActiveTool(TOOLS.RESTORE)}>
                            <span className="icon">✨</span> Restore
                        </button>
                    </div>

                    <div className="control-group">
                        <label>Brush Size: {brushSize}px</label>
                        <input type="range" min="2" max="200" value={brushSize} onChange={e => setBrushSize(Number(e.target.value))} />
                    </div>

                    <div className="control-group">
                        <label>Softness: {brushSoftness}%</label>
                        <input type="range" min="0" max="100" value={brushSoftness} onChange={e => setBrushSoftness(Number(e.target.value))} />
                    </div>

                    {viewMode === VIEW_MODES.OVERLAY && (
                        <div className="control-group">
                            <label>Mask Opacity: {Math.round(maskOpacity * 100)}%</label>
                            <input type="range" min="0" max="1" step="0.01" value={maskOpacity} onChange={e => setMaskOpacity(Number(e.target.value))} />
                        </div>
                    )}

                    <div className="history-actions">
                        <button className="btn-icon" onClick={handleUndo} disabled={historyIndex <= 0}>↩️ Undo</button>
                        <button className="btn-icon" onClick={handleRedo} disabled={historyIndex >= maskHistory.length - 1}>↪️ Redo</button>
                        <button className={`btn-save ${hasUnsavedChanges ? 'primary' : ''}`} onClick={handleSave} disabled={!hasUnsavedChanges || isSaving}>
                            {isSaving ? 'Saving...' : 'Save Changes'}
                        </button>
                    </div>
                </div>

                {/* Border & Backgrounds */}
                <div className="panel-section">
                    <div className="flex-row">
                        <h3 className="section-title">Border</h3>
                        <div className={`toggle ${borderEnabled ? 'on' : ''}`} onClick={() => setBorderEnabled(!borderEnabled)} />
                    </div>
                    {borderEnabled && (
                        <div className="border-controls">
                            <input type="range" min="1" max="50" value={borderThickness} onChange={e => setBorderThickness(Number(e.target.value))} />
                            <div className="color-row">
                                <div className="color-swatch" style={{ background: borderColor }} onClick={() => setShowColorPicker(!showColorPicker)} />
                                <input type="text" value={borderColor} onChange={e => setBorderColor(e.target.value)} />
                            </div>
                            {showColorPicker && <div className="absolute-picker"><HexColorPicker color={borderColor} onChange={setBorderColor} /></div>}
                        </div>
                    )}
                </div>

                <div className="panel-section">
                    <h3 className="section-title">Background Test</h3>
                    <div className="bg-grid">
                        {backgroundPresets.map(p => (
                            <button key={p.id} className={`bg-swatch ${activeBackground === p.id ? 'active' : ''}`} 
                                style={{ background: p.color1 || '#ccc' }}
                                onClick={() => handleBackgroundChange(p)} />
                        ))}
                    </div>
                </div>

                <div className="panel-section export">
                    <button className="btn-primary btn-large" onClick={() => downloadImage(job.id, borderEnabled)}>
                        Download Sticker
                    </button>
                </div>
            </div>

            <style jsx>{`
                .professional {
                    --bg-dark: #1e1e1e;
                    --panel-bg: #2d2d2d;
                    --accent: #3b82f6;
                    --text: #e5e5e5;
                    --border: #404040;
                    --warning: #f59e0b;
                    --error: #ef4444;
                }
                .editor-panel { display: flex; height: 100vh; background: var(--bg-dark); color: var(--text); overflow: hidden; }
                .canvas-container { flex: 1; position: relative; cursor: crosshair; min-width: 0; }
                .canvas-checker { position: absolute; inset: 0; z-index: 0; }
                .canvas-checker.checker { background-image: conic-gradient(#333 90deg, #444 90deg 180deg, #333 180deg 270deg, #444 270deg); background-size: 40px 40px; }
                .controls-panel { width: 320px; min-width: 320px; background: var(--panel-bg); border-left: 1px solid var(--border); padding: 20px; overflow-y: auto; }
                .panel-section { margin-bottom: 24px; padding-bottom: 24px; border-bottom: 1px solid var(--border); }
                .section-title { font-size: 14px; text-transform: uppercase; color: #888; margin-bottom: 12px; }
                .section-title.warning { color: var(--warning); }
                .tool-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 10px; margin-bottom: 20px; }
                .btn-tool { background: #333; border: 1px solid var(--border); padding: 12px; border-radius: 8px; color: white; cursor: pointer; }
                .btn-tool.active { background: var(--accent); border-color: var(--accent); }
                .control-group { margin-bottom: 15px; }
                .control-group label { display: block; font-size: 13px; margin-bottom: 8px; }
                input[type="range"] { width: 100%; height: 4px; background: #444; border-radius: 2px; appearance: none; }
                .history-actions { display: flex; gap: 8px; margin-top: 20px; }
                .btn-save { flex: 1; padding: 10px; border-radius: 6px; border: none; background: #444; color: white; cursor: pointer; }
                .btn-save.primary { background: var(--accent); }
                .view-mode-toggle { 
                    position: absolute; 
                    bottom: 20px; 
                    left: 50%; 
                    transform: translateX(-50%); 
                    background: rgba(45, 45, 45, 0.9); 
                    padding: 4px; 
                    border-radius: 12px; 
                    display: flex; 
                    gap: 2px; 
                    box-shadow: 0 4px 20px rgba(0,0,0,0.4);
                    backdrop-filter: blur(10px);
                    max-width: 90vw;
                    overflow-x: auto;
                }
                .view-mode-toggle::-webkit-scrollbar { display: none; }
                .view-mode-toggle button { 
                    background: none; 
                    border: none; 
                    color: #aaa; 
                    padding: 8px 16px; 
                    border-radius: 8px; 
                    cursor: pointer; 
                    white-space: nowrap;
                    font-size: 13px;
                    transition: all 0.2s;
                }
                .view-mode-toggle button.active { background: var(--accent); color: white; }
                
                .review-alert { background: rgba(245, 158, 11, 0.05); margin: -20px -20px 24px -20px; padding: 20px; border-bottom: 1px solid rgba(245, 158, 11, 0.2); }
                .review-list { list-style: none; padding: 0; margin: 0; }
                .review-item { font-size: 12px; color: #ccc; padding: 4px 0 4px 20px; position: relative; }
                .review-item::before { content: '•'; position: absolute; left: 5px; color: var(--warning); }
                .btn-reprocess { 
                    background: rgba(245, 158, 11, 0.1); 
                    border: 1px solid var(--warning); 
                    color: var(--warning); 
                    padding: 4px 8px; 
                    border-radius: 4px; 
                    font-size: 11px; 
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .btn-reprocess:hover { background: var(--warning); color: black; }
                .btn-reprocess.loading { opacity: 0.5; cursor: wait; }

                .flex-row { display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px; }
                .bg-grid { display: grid; grid-template-columns: repeat(4, 1fr); gap: 8px; }
                .bg-swatch { aspect-ratio: 1; border-radius: 4px; border: 2px solid transparent; cursor: pointer; }
                .bg-swatch.active { border-color: var(--accent); }
                .stage-status { position: absolute; top: 20px; left: 20px; background: rgba(0,0,0,0.5); padding: 5px 10px; border-radius: 5px; font-size: 12px; display: flex; gap: 10px; }
            `}</style>
        </div>
    )
}

export default Editor
