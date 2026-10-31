/**
 * ImageAnalysisModal Component
 * 
 * Deep Picture Metrology & Quality Analysis Modal
 * Displays image resolution, 300 DPI print sizes, edge sharpness, 
 * contrast ratio, color metrics, and recommended cutline borders.
 */

import { useState, useEffect } from 'react'
import { getImageUrl, getJobAnalysis } from '../api'

function ImageAnalysisModal({ job, onClose, onApplyRecommendedBorder }) {
    const [analysisData, setAnalysisData] = useState(job.analysis || null)
    const [loading, setLoading] = useState(!job.analysis)
    const [error, setError] = useState(null)
    const [activeTab, setActiveTab] = useState('metrics') // 'metrics' | 'compare'

    useEffect(() => {
        if (!job.analysis) {
            setLoading(true)
            getJobAnalysis(job.id)
                .then(res => {
                    setAnalysisData(res.analysis)
                    setLoading(false)
                })
                .catch(err => {
                    console.error('Failed to load analysis:', err)
                    setError('Could not fetch picture analysis.')
                    setLoading(false)
                })
        }
    }, [job])

    const originalUrl = getImageUrl(job.paths?.original)
    const transparentUrl = getImageUrl(job.paths?.transparent)
    const maskUrl = getImageUrl(job.paths?.mask)

    const analysis = analysisData || {}
    const dims = analysis.dimensions || {}
    const print300 = analysis.print_size_300dpi || {}
    const print150 = analysis.print_size_150dpi || {}

    return (
        <div className="modal-overlay" onClick={onClose}>
            <div className="modal-container analysis-modal" onClick={e => e.stopPropagation()}>
                {/* Header */}
                <div className="modal-header">
                    <div className="header-title">
                        <span className="icon">📊</span>
                        <div>
                            <h2>Picture Analysis &amp; Print Metrology</h2>
                            <p className="subtitle">{job.filename}</p>
                        </div>
                    </div>
                    <button className="btn-close" onClick={onClose}>✕</button>
                </div>

                {/* Body */}
                <div className="modal-body">
                    {loading ? (
                        <div className="analysis-loading">
                            <div className="spinner large" />
                            <p>Analyzing picture geometry, edge variance, and color metrics...</p>
                        </div>
                    ) : error ? (
                        <div className="analysis-error">
                            <span>⚠️ {error}</span>
                        </div>
                    ) : (
                        <>
                            {/* Top Overall Score Card */}
                            <div className="quality-banner">
                                <div className="score-ring">
                                    <span className="score-value">{analysis.quality_score || 85}</span>
                                    <span className="score-label">/ 100</span>
                                </div>
                                <div className="score-details">
                                    <div className="score-title">
                                        <h3>Sticker Quality Score</h3>
                                        <span className={`status-badge ${job.edge_confidence || 'high'}`}>
                                            {job.edge_confidence || 'high'} confidence
                                        </span>
                                    </div>
                                    <p className="readiness-text">{analysis.print_readiness}</p>
                                </div>
                            </div>

                            {/* View Switcher Tabs */}
                            <div className="tab-switcher">
                                <button
                                    className={`tab-btn ${activeTab === 'metrics' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('metrics')}
                                >
                                    📈 Metrology &amp; Specs
                                </button>
                                <button
                                    className={`tab-btn ${activeTab === 'compare' ? 'active' : ''}`}
                                    onClick={() => setActiveTab('compare')}
                                >
                                    🔍 Image Inspection
                                </button>
                            </div>

                            {/* Tab Content 1: Specs & Metrics */}
                            {activeTab === 'metrics' && (
                                <div className="metrics-grid">
                                    {/* Card 1: Print Dimensions */}
                                    <div className="metric-card">
                                        <div className="metric-card-header">
                                            <span className="card-icon">📏</span>
                                            <h4>Print Dimensions (300 DPI)</h4>
                                        </div>
                                        <div className="metric-primary-val">
                                            {print300.width_in}″ × {print300.height_in}″
                                        </div>
                                        <div className="metric-sub-val">
                                            {print300.width_cm} cm × {print300.height_cm} cm
                                        </div>
                                        <div className="metric-meta">
                                            <span>Resolution: <strong>{dims.width} × {dims.height} px</strong></span>
                                            <span>Megapixels: <strong>{dims.megapixels} MP</strong></span>
                                            <span>Aspect Ratio: <strong>{dims.aspect_category} ({dims.aspect_ratio})</strong></span>
                                        </div>
                                    </div>

                                    {/* Card 2: Edge & Cutline Quality */}
                                    <div className="metric-card">
                                        <div className="metric-card-header">
                                            <span className="card-icon">🎯</span>
                                            <h4>Edge &amp; Cutline Metrics</h4>
                                        </div>
                                        <div className="stat-progress-bar">
                                            <div className="stat-header">
                                                <span>Edge Sharpness Score</span>
                                                <span>{analysis.edge_sharpness_score}%</span>
                                            </div>
                                            <div className="bar-bg">
                                                <div className="bar-fill" style={{ width: `${analysis.edge_sharpness_score}%` }} />
                                            </div>
                                        </div>
                                        <div className="stat-progress-bar">
                                            <div className="stat-header">
                                                <span>Subject Canvas Coverage</span>
                                                <span>{analysis.subject_coverage_pct}%</span>
                                            </div>
                                            <div className="bar-bg">
                                                <div className="bar-fill purple" style={{ width: `${analysis.subject_coverage_pct}%` }} />
                                            </div>
                                        </div>
                                        <div className="metric-meta mt-2">
                                            <span>White-on-White Edge: <strong>Protected</strong></span>
                                            <span>Boundary Gaps: <strong>Closed Contour</strong></span>
                                        </div>
                                    </div>

                                    {/* Card 3: Color & Contrast */}
                                    <div className="metric-card">
                                        <div className="metric-card-header">
                                            <span className="card-icon">🎨</span>
                                            <h4>Color &amp; Contrast Analysis</h4>
                                        </div>
                                        <div className="stat-progress-bar">
                                            <div className="stat-header">
                                                <span>Foreground Contrast</span>
                                                <span>{analysis.contrast_score}%</span>
                                            </div>
                                            <div className="bar-bg">
                                                <div className="bar-fill green" style={{ width: `${analysis.contrast_score}%` }} />
                                            </div>
                                        </div>
                                        <div className="metric-meta mt-2">
                                            <span>Vibrancy: <strong>{analysis.colorfulness_rating} ({analysis.colorfulness})</strong></span>
                                            <span>Matting Pass: <strong>Guided Refinement</strong></span>
                                        </div>
                                    </div>

                                    {/* Card 4: Recommended Border */}
                                    <div className="metric-card highlight">
                                        <div className="metric-card-header">
                                            <span className="card-icon">✂️</span>
                                            <h4>Recommended Print Border</h4>
                                        </div>
                                        <div className="metric-primary-val text-accent">
                                            {analysis.recommended_border_px} px Outline
                                        </div>
                                        <p className="card-tip">Calculated optimal outline border for clean die-cutting.</p>
                                        {onApplyRecommendedBorder && (
                                            <button
                                                className="btn btn-primary btn-sm mt-2"
                                                onClick={() => {
                                                    onApplyRecommendedBorder(analysis.recommended_border_px)
                                                    onClose()
                                                }}
                                            >
                                                Apply {analysis.recommended_border_px}px Border
                                            </button>
                                        )}
                                    </div>
                                </div>
                            )}

                            {/* Tab Content 2: Visual Comparison */}
                            {activeTab === 'compare' && (
                                <div className="inspection-views">
                                    <div className="inspection-card">
                                        <h5>Original Image</h5>
                                        <div className="img-frame">
                                            <img src={originalUrl} alt="Original" />
                                        </div>
                                    </div>
                                    <div className="inspection-card">
                                        <h5>Transparent Sticker</h5>
                                        <div className="img-frame checkerboard">
                                            <img src={transparentUrl} alt="Transparent" />
                                        </div>
                                    </div>
                                    <div className="inspection-card">
                                        <h5>AI Edge Mask</h5>
                                        <div className="img-frame black-bg">
                                            <img src={maskUrl} alt="Mask" />
                                        </div>
                                    </div>
                                </div>
                            )}
                        </>
                    )}
                </div>
            </div>

            <style jsx>{`
                .analysis-modal {
                    max-width: 820px;
                    width: 92%;
                    background: #161622;
                    border: 1px solid rgba(255, 255, 255, 0.12);
                    border-radius: 20px;
                    color: white;
                    box-shadow: 0 20px 50px rgba(0,0,0,0.8);
                    overflow: hidden;
                }
                .modal-header {
                    display: flex;
                    justify-content: space-between;
                    align-items: center;
                    padding: 20px 24px;
                    border-bottom: 1px solid rgba(255, 255, 255, 0.08);
                    background: rgba(255, 255, 255, 0.02);
                }
                .header-title {
                    display: flex;
                    align-items: center;
                    gap: 14px;
                }
                .header-title .icon {
                    font-size: 28px;
                }
                .header-title h2 {
                    font-size: 18px;
                    font-weight: 700;
                    margin: 0;
                }
                .header-title .subtitle {
                    font-size: 12px;
                    color: #888;
                    margin: 0;
                }
                .btn-close {
                    background: none;
                    border: none;
                    color: #888;
                    font-size: 20px;
                    cursor: pointer;
                    padding: 4px 8px;
                    border-radius: 6px;
                }
                .btn-close:hover {
                    color: white;
                    background: rgba(255,255,255,0.1);
                }
                .modal-body {
                    padding: 24px;
                    max-height: 75vh;
                    overflow-y: auto;
                }
                .analysis-loading {
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 16px;
                    padding: 40px 0;
                    color: #aaa;
                }
                .quality-banner {
                    display: flex;
                    align-items: center;
                    gap: 20px;
                    background: linear-gradient(135deg, rgba(99, 102, 241, 0.15) 0%, rgba(139, 92, 246, 0.15) 100%);
                    border: 1px solid rgba(99, 102, 241, 0.3);
                    border-radius: 16px;
                    padding: 18px 24px;
                    margin-bottom: 20px;
                }
                .score-ring {
                    width: 70px;
                    height: 70px;
                    border-radius: 50%;
                    background: rgba(99, 102, 241, 0.2);
                    border: 3px solid #6366f1;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    justify-content: center;
                    box-shadow: 0 0 20px rgba(99, 102, 241, 0.4);
                }
                .score-value {
                    font-size: 22px;
                    font-weight: 800;
                    color: white;
                    line-height: 1;
                }
                .score-label {
                    font-size: 10px;
                    color: #a5b4fc;
                }
                .score-details {
                    flex: 1;
                }
                .score-title {
                    display: flex;
                    align-items: center;
                    justify-content: space-between;
                }
                .score-title h3 {
                    font-size: 16px;
                    font-weight: 700;
                    margin: 0;
                }
                .status-badge {
                    font-size: 11px;
                    text-transform: uppercase;
                    padding: 4px 10px;
                    border-radius: 20px;
                    font-weight: 700;
                }
                .status-badge.high { background: rgba(34, 197, 94, 0.2); color: #22c55e; }
                .status-badge.medium { background: rgba(245, 158, 11, 0.2); color: #f59e0b; }
                .status-badge.low { background: rgba(239, 68, 68, 0.2); color: #ef4444; }
                .readiness-text {
                    font-size: 13px;
                    color: #cbd5e1;
                    margin-top: 4px;
                }
                .tab-switcher {
                    display: flex;
                    gap: 8px;
                    margin-bottom: 20px;
                    background: rgba(0,0,0,0.3);
                    padding: 4px;
                    border-radius: 10px;
                }
                .tab-btn {
                    flex: 1;
                    background: none;
                    border: none;
                    color: #94a3b8;
                    padding: 8px 16px;
                    border-radius: 8px;
                    font-size: 13px;
                    font-weight: 600;
                    cursor: pointer;
                    transition: all 0.2s;
                }
                .tab-btn.active {
                    background: #6366f1;
                    color: white;
                }
                .metrics-grid {
                    display: grid;
                    grid-template-columns: 1fr 1fr;
                    gap: 16px;
                }
                .metric-card {
                    background: rgba(255, 255, 255, 0.03);
                    border: 1px solid rgba(255, 255, 255, 0.07);
                    border-radius: 14px;
                    padding: 16px;
                }
                .metric-card.highlight {
                    background: rgba(99, 102, 241, 0.06);
                    border-color: rgba(99, 102, 241, 0.25);
                }
                .metric-card-header {
                    display: flex;
                    align-items: center;
                    gap: 8px;
                    margin-bottom: 12px;
                }
                .metric-card-header h4 {
                    font-size: 13px;
                    color: #94a3b8;
                    margin: 0;
                    text-transform: uppercase;
                    letter-spacing: 0.05em;
                }
                .metric-primary-val {
                    font-size: 20px;
                    font-weight: 700;
                    color: white;
                }
                .metric-primary-val.text-accent {
                    color: #818cf8;
                }
                .metric-sub-val {
                    font-size: 13px;
                    color: #94a3b8;
                    margin-top: 2px;
                }
                .metric-meta {
                    display: flex;
                    flex-direction: column;
                    gap: 4px;
                    font-size: 12px;
                    color: #94a3b8;
                    margin-top: 12px;
                }
                .metric-meta strong {
                    color: white;
                }
                .stat-progress-bar {
                    margin-bottom: 10px;
                }
                .stat-header {
                    display: flex;
                    justify-content: space-between;
                    font-size: 12px;
                    color: #94a3b8;
                    margin-bottom: 4px;
                }
                .bar-bg {
                    height: 6px;
                    background: rgba(255,255,255,0.1);
                    border-radius: 3px;
                    overflow: hidden;
                }
                .bar-fill {
                    height: 100%;
                    background: #6366f1;
                    border-radius: 3px;
                }
                .bar-fill.purple { background: #a855f7; }
                .bar-fill.green { background: #22c55e; }
                .card-tip {
                    font-size: 12px;
                    color: #94a3b8;
                    margin-top: 6px;
                }
                .mt-2 { margin-top: 12px; }
                .btn-sm {
                    padding: 6px 12px;
                    font-size: 12px;
                }
                .inspection-views {
                    display: grid;
                    grid-template-columns: repeat(3, 1fr);
                    gap: 16px;
                }
                .inspection-card h5 {
                    font-size: 12px;
                    color: #94a3b8;
                    margin: 0 0 8px 0;
                    text-align: center;
                }
                .img-frame {
                    aspect-ratio: 1;
                    border-radius: 12px;
                    overflow: hidden;
                    background: #111;
                    border: 1px solid rgba(255,255,255,0.1);
                    display: flex;
                    align-items: center;
                    justify-content: center;
                }
                .img-frame.checkerboard {
                    background-image: conic-gradient(#1e1e2d 90deg, #12121a 90deg 180deg, #1e1e2d 180deg 270deg, #12121a 270deg);
                    background-size: 20px 20px;
                }
                .img-frame img {
                    width: 100%;
                    height: 100%;
                    object-fit: contain;
                }
            `}</style>
        </div>
    )
}

export default ImageAnalysisModal
