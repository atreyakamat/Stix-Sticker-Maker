/**
 * Gallery Component
 * 
 * Displays all uploaded stickers in a grid.
 * Shows processing status, edge confidence, and allows batch export.
 */

import { useState } from 'react'
import { getImageUrl, exportBatch } from '../api'

function Gallery({ jobs, onSelect, onUploadMore }) {
    const [isExporting, setIsExporting] = useState(false)
    const [selectedJobIds, setSelectedJobIds] = useState([])

    const completedJobs = jobs.filter(j => j.status === 'complete')
    const processingJobs = jobs.filter(j => j.status === 'processing' || j.status === 'pending')

    // Handle batch export (download selected or all if none selected)
    const handleExport = async () => {
        const idsToExport = selectedJobIds.length > 0 
            ? selectedJobIds 
            : completedJobs.map(j => j.id)

        if (idsToExport.length === 0) return

        setIsExporting(true)
        try {
            await exportBatch(idsToExport, true)
        } catch (error) {
            console.error('Export failed:', error)
            alert('Export failed. Please try again.')
        } finally {
            setIsExporting(false)
        }
    }

    const toggleSelection = (jobId, e) => {
        e.stopPropagation() // Don't trigger the onSelect (editor)
        setSelectedJobIds(prev => 
            prev.includes(jobId) 
                ? prev.filter(id => id !== jobId) 
                : [...prev, jobId]
        )
    }

    const selectAll = () => {
        setSelectedJobIds(completedJobs.map(j => j.id))
    }

    const clearSelection = () => {
        setSelectedJobIds([])
    }

    return (
        <div className="gallery-container">
            {/* Header */}
            <div className="gallery-header">
                <div className="gallery-title">
                    <h2>Your Stickers</h2>
                    <span className="gallery-count">
                        {completedJobs.length} ready
                        {processingJobs.length > 0 && ` • ${processingJobs.length} processing`}
                    </span>
                </div>
                <div className="gallery-actions">
                    {selectedJobIds.length > 0 ? (
                        <button className="btn btn-text" onClick={clearSelection}>
                            Clear Selection ({selectedJobIds.length})
                        </button>
                    ) : (
                        completedJobs.length > 0 && (
                            <button className="btn btn-text" onClick={selectAll}>
                                Select All
                            </button>
                        )
                    )}
                    <button className="btn btn-secondary" onClick={onUploadMore}>
                        + Add More
                    </button>
                </div>
            </div>

            {/* Processing Status Banner */}
            {processingJobs.length > 0 && (
                <div className="processing-banner">
                    <div className="spinner" />
                    <span>
                        Processing {processingJobs.length} image{processingJobs.length > 1 ? 's' : ''}...
                    </span>
                    <span className="processing-hint">
                        Click on completed stickers to edit
                    </span>
                </div>
            )}

            {/* Empty State */}
            {jobs.length === 0 && (
                <div className="empty-state">
                    <div className="empty-state-icon">📭</div>
                    <p>No stickers yet</p>
                    <button className="btn btn-primary" onClick={onUploadMore}>
                        Upload your first sticker
                    </button>
                </div>
            )}

            {/* Gallery Grid */}
            <div className="gallery">
                {jobs.map(job => (
                    <GalleryItem
                        key={job.id}
                        job={job}
                        isSelected={selectedJobIds.includes(job.id)}
                        onClick={() => onSelect(job)}
                        onToggleSelect={(e) => toggleSelection(job.id, e)}
                    />
                ))}
            </div>

            {/* Batch Actions Footer */}
            {completedJobs.length > 0 && (
                <div className="gallery-footer">
                    <button
                        className={`btn btn-primary btn-large ${selectedJobIds.length > 0 ? 'pulse' : ''}`}
                        onClick={handleExport}
                        disabled={isExporting}
                    >
                        {isExporting ? (
                            <>
                                <span className="spinner small" />
                                Exporting...
                            </>
                        ) : (
                            <>
                                📦 {selectedJobIds.length > 0 
                                    ? `Download Selected (${selectedJobIds.length})` 
                                    : `Download All as ZIP (${completedJobs.length})`}
                            </>
                        )}
                    </button>
                </div>
            )}
        </div>
    )
}

/**
 * Individual gallery item
 */
function GalleryItem({ job, isSelected, onClick, onToggleSelect }) {
    const imageUrl = job.status === 'complete'
        ? getImageUrl(job.paths?.transparent)
        : null

    // Determine status indicator
    const getStatusIcon = () => {
        if (job.status === 'complete') {
            switch (job.edge_confidence) {
                case 'high': return '✅'
                case 'medium': return '✓'
                case 'low': return '⚠️'
                case 'failed': return '❌'
                default: return '✓'
            }
        }
        if (job.status === 'processing') return `⏳ ${job.progress}%`
        if (job.status === 'pending') return '⏳ Queued'
        if (job.status === 'error') return '❌ Error'
        return ''
    }

    return (
        <div
            className={`gallery-item ${isSelected ? 'selected' : ''} ${job.needs_review ? 'needs-review' : ''} ${job.status}`}
            onClick={onClick}
            title={job.needs_review ? `Needs Review: \n${(job.review_reasons || []).join('\n')}` : job.filename}
        >
            {/* Selection Checkbox */}
            {job.status === 'complete' && (
                <div 
                    className={`item-selection-overlay ${isSelected ? 'active' : ''}`}
                    onClick={onToggleSelect}
                >
                    <div className="checkbox">
                        {isSelected && '✓'}
                    </div>
                </div>
            )}

            {/* Needs Review Badge */}
            {job.needs_review && <div className="review-badge">Review</div>}

            {/* Image or Loading State */}
            <div className="gallery-item-image">
                {imageUrl ? (
                    <img src={imageUrl} alt={job.filename} loading="lazy" />
                ) : (
                    <div className="gallery-item-placeholder">
                        {job.status === 'error' ? (
                            <span className="error-icon">❌</span>
                        ) : (
                            <div className="spinner" />
                        )}
                    </div>
                )}
            </div>

            {/* Status Bar */}
            <div className={`gallery-item-status ${job.status}`}>
                <span className="status-icon">{getStatusIcon()}</span>
                <span className="status-filename">{job.filename}</span>
            </div>

            {/* Progress Bar (only when processing) */}
            {job.status === 'processing' && (
                <div className="gallery-item-progress">
                    <div
                        className="gallery-item-progress-fill"
                        style={{ width: `${job.progress}%` }}
                    />
                </div>
            )}
        </div>
    )
}

export default Gallery
