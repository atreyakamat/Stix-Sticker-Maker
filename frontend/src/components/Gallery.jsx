import { getImageUrl } from '../api'

function Gallery({ jobs, selectedJob, onSelect, onUploadMore }) {
    const completedJobs = jobs.filter(j => j.status === 'complete')

    return (
        <div style={{ flex: 1, display: 'flex', flexDirection: 'column', gap: '1rem' }}>
            {/* Header */}
            <div style={{
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'center',
                padding: '0 0.5rem'
            }}>
                <h2 style={{ fontSize: '1.25rem', fontWeight: 600 }}>
                    Your Stickers
                    <span style={{
                        marginLeft: '0.5rem',
                        fontSize: '0.875rem',
                        color: 'var(--text-secondary)'
                    }}>
                        ({completedJobs.length} ready)
                    </span>
                </h2>
                <button className="btn btn-primary" onClick={onUploadMore}>
                    + Add More
                </button>
            </div>

            {/* Processing Status */}
            {jobs.some(j => j.status === 'processing' || j.status === 'pending') && (
                <div style={{
                    padding: '1rem',
                    background: 'rgba(99, 102, 241, 0.1)',
                    borderRadius: 'var(--radius-lg)',
                    border: '1px solid rgba(99, 102, 241, 0.2)'
                }}>
                    <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem' }}>
                        <div className="spinner"></div>
                        <span>
                            Processing {jobs.filter(j => j.status === 'processing').length} image(s)...
                        </span>
                    </div>
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

            {/* Batch Actions */}
            {completedJobs.length > 1 && (
                <div style={{
                    display: 'flex',
                    justifyContent: 'center',
                    padding: '1rem',
                    borderTop: '1px solid var(--glass-border)'
                }}>
                    <button
                        className="btn btn-secondary btn-large"
                        onClick={() => {
                            // Batch export will be handled elsewhere
                            console.log('Batch export:', completedJobs.map(j => j.id))
                        }}
                    >
                        📦 Export All ({completedJobs.length} stickers)
                    </button>
                </div>
            )}
        </div>
    )
}

function GalleryItem({ job, isSelected, onClick }) {
    const imageUrl = job.status === 'complete'
        ? getImageUrl(job.paths?.transparent)
        : null

    return (
        <div
            className={`gallery-item ${isSelected ? 'selected' : ''} ${job.needs_review ? 'needs-review' : ''}`}
            onClick={onClick}
        >
            {/* Needs Review Badge */}
            {job.needs_review && (
                <div className="review-badge">Review</div>
            )}

            {imageUrl ? (
                <img src={imageUrl} alt={job.filename} />
            ) : (
                <div style={{
                    width: '100%',
                    height: '100%',
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'center'
                }}>
                    {job.status === 'error' ? (
                        <span style={{ fontSize: '2rem' }}>❌</span>
                    ) : (
                        <div className="spinner"></div>
                    )}
                </div>
            )}

            <div className={`gallery-item-status ${job.status}`}>
                {job.status === 'complete' && (
                    <span>
                        {job.edge_confidence === 'high' && '✅'}
                        {job.edge_confidence === 'medium' && '✓'}
                        {job.edge_confidence === 'low' && '⚠️'}
                        {job.edge_confidence === 'failed' && '❌'}
                        {!job.edge_confidence && '✓'}
                    </span>
                )}
                {job.status === 'processing' && <span>⏳ {job.progress}%</span>}
                {job.status === 'pending' && <span>⏳ Waiting...</span>}
                {job.status === 'error' && <span>Error</span>}
                <span style={{
                    flex: 1,
                    textOverflow: 'ellipsis',
                    overflow: 'hidden',
                    whiteSpace: 'nowrap'
                }}>
                    {job.filename}
                </span>
            </div>

            {job.status === 'processing' && (
                <div className="progress-bar" style={{
                    position: 'absolute',
                    bottom: 0,
                    left: 0,
                    right: 0,
                    borderRadius: 0
                }}>
                    <div
                        className="progress-bar-fill"
                        style={{ width: `${job.progress}%` }}
                    />
                </div>
            )}
        </div>
    )
}

export default Gallery
