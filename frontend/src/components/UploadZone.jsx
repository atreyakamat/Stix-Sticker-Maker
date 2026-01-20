/**
 * UploadZone Component
 * 
 * Drag-and-drop upload area for sticker images.
 * Accepts PNG, JPG, WEBP files. Supports batch uploads.
 */

import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function UploadZone({ onUpload, isUploading = false }) {
    const onDrop = useCallback((acceptedFiles) => {
        if (acceptedFiles.length > 0) {
            onUpload(acceptedFiles)
        }
    }, [onUpload])

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

            {/* Feature pills */}
            <div className="feature-pills">
                <div className="feature-pill">
                    <span className="feature-icon">🎯</span>
                    <div className="feature-content">
                        <span className="feature-title">AI Background Removal</span>
                        <span className="feature-desc">BiRefNet model for precise extraction</span>
                    </div>
                </div>
                <div className="feature-pill">
                    <span className="feature-icon">✂️</span>
                    <div className="feature-content">
                        <span className="feature-title">Edge-First Detection</span>
                        <span className="feature-desc">Works on white-on-white stickers</span>
                    </div>
                </div>
                <div className="feature-pill">
                    <span className="feature-icon">🎨</span>
                    <div className="feature-content">
                        <span className="feature-title">Manual Refinement</span>
                        <span className="feature-desc">Brush tools for perfect edges</span>
                    </div>
                </div>
            </div>
        </div>
    )
}

export default UploadZone
