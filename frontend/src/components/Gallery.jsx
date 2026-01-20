/**
 * Gallery Component
 * 
 * Displays all uploaded stickers in a grid.
 * Shows processing status, edge confidence, and allows batch export.
 */

import { useState } from 'react'
import { getImageUrl, exportBatch } from '../api'

function Gallery({ jobs, selectedJob, onSelect, onUploadMore }) {
    const [isExporting, setIsExporting] = useState(false)

    const completedJobs = jobs.filter(j => j.status === 'complete')
    const processingJobs = jobs.filter(j => j.status === 'processing' || j.status === 'pending')

    // Handle batch export (download all as ZIP)
    const handleExportAll = async () => {
        if (completedJobs.length === 0) return

        setIsExporting(true)
        try {
            await exportBatch(completedJobs.map(j => j.id), true)
        } catch (error) {
            console.error('Export failed:', error)
            alert('Export failed. Please try again.')
        } finally {
            setIsExporting(false)
        }
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
                        isSelected={selectedJob?.id === job.id}
                        onClick={() => onSelect(job)}
                    />
                ))}
            </div>

            {/* Batch Actions Footer */}
            {completedJobs.length > 0 && (
                <div className="gallery-footer">
                    <button
                        className="btn btn-primary btn-large"
                        onClick={handleExportAll}
                        disabled={isExporting}
                    >
                        {isExporting ? (
                            <>
                                <span className="spinner small" />
                                Exporting...
                            </>
                        ) : (
                            <>
                                📦 Download All as ZIP ({completedJobs.length} stickers)
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
function GalleryItem({ job, isSelected, onClick }) {
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
            title={job.filename}
        >
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
