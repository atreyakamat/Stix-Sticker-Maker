/**
 * UploadZone Component
 * 
 * Drag-and-drop upload area for sticker images.
 * Accepts PNG, JPG, WEBP files. Supports batch uploads.
 */

import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

function UploadZone({ onUpload, isUploading = false }) {
    const [tier, setTier] = useState('balanced')

    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            onUpload(acceptedFiles, tier)
        }
    }, [onUpload, tier])

    const { getRootProps, getInputProps, isDragActive } = useDropzone({
        onDrop,
        accept: {
            'image/png': ['.png'],
            'image/jpeg': ['.jpg', '.jpeg', '.jfif'],
            'image/webp': ['.webp'],
            'image/heic': ['.heic']
        },
        multiple: true,
        disabled: isUploading
    })

    return (
        <div className="upload-container">
            <div
                {...getRootProps()}
                className={`upload-zone ${isDragActive ? 'active' : ''} ${isUploading ? 'uploading' : ''}`}
            >
                <input {...getInputProps()} />

                {/* Icon */}
                <div className="upload-icon">
                    {isUploading ? (
                        <div className="spinner large" />
                    ) : isDragActive ? (
                        '📥'
                    ) : (
                        '✨'
                    )}
                </div>

                {/* Main text */}
                <h2 className="upload-text">
                    {isUploading
                        ? 'Uploading...'
                        : isDragActive
                            ? 'Drop your stickers here...'
                            : 'Drop stickers or click to upload'
                    }
                </h2>

                {/* Hint */}
                <p className="upload-hint">
                    {isUploading
                        ? 'Please wait while we process your images'
                        : 'PNG, JPG, WEBP • Batch upload supported'
                    }
                </p>
            </div>

            {/* Tier Selector */}
            {!isUploading && (
                <div className="tier-selector">
                    <span className="selector-label">Processing Quality:</span>
                    <div className="tier-buttons">
                        <button className={`btn-tier ${tier === 'fast' ? 'active' : ''}`} onClick={() => setTier('fast')}>
                            <span className="icon">⚡</span> Fast
                        </button>
                        <button className={`btn-tier ${tier === 'balanced' ? 'active' : ''}`} onClick={() => setTier('balanced')}>
                            <span className="icon">⚖️</span> Balanced
                        </button>
                        <button className={`btn-tier ${tier === 'quality' ? 'active' : ''}`} onClick={() => setTier('quality')}>
                            <span className="icon">💎</span> High Quality
                        </button>
                    </div>
                </div>
            )}

            {/* Feature pills */}
            <div className="feature-pills">
                <div className="feature-pill">
                    <span className="feature-icon">🎯</span>
                    <div className="feature-content">
                        <span className="feature-title">AI Background Removal</span>
                        <span className="feature-desc">BiRefNet model with edge-first architecture</span>
                    </div>
                </div>
                <div className="feature-pill">
                    <span className="feature-icon">✂️</span>
                    <div className="feature-content">
                        <span className="feature-title">Pixel-Accurate Mask Editor</span>
                        <span className="feature-desc">Erase &amp; restore with brushes, zoom, and softness</span>
                    </div>
                </div>
                <div className="feature-pill">
                    <span className="feature-icon">🎨</span>
                    <div className="feature-content">
                        <span className="feature-title">Professional Quality</span>
                        <span className="feature-desc">Undo/redo, white-on-white support, batch export</span>
                    </div>
                </div>
            </div>
            <style jsx>{`
                .tier-selector {
                    margin-top: 20px;
                    display: flex;
                    flex-direction: column;
                    align-items: center;
                    gap: 10px;
                }
                .selector-label {
                    font-size: 13px;
                    color: #888;
                    text-transform: uppercase;
                }
                .tier-buttons {
                    display: flex;
                    gap: 10px;
                }
                .btn-tier {
                    background: #2d2d2d;
                    border: 1px solid #404040;
                    color: #888;
                    padding: 8px 16px;
                    border-radius: 20px;
                    cursor: pointer;
                    display: flex;
                    align-items: center;
                    gap: 6px;
                    transition: all 0.2s;
                    font-size: 14px;
                }
                .btn-tier.active {
                    background: #3b82f6;
                    border-color: #3b82f6;
                    color: white;
                    box-shadow: 0 0 15px rgba(59, 130, 246, 0.4);
                }
                .btn-tier:hover:not(.active) {
                    border-color: #666;
                    color: #ccc;
                }
            `}</style>
        </div>
    )
}

export default UploadZone
