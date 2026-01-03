import { useCallback } from 'react'
import { useDropzone } from 'react-dropzone'

function UploadZone({ onUpload }) {
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
        multiple: true
    })

    return (
        <div
            {...getRootProps()}
            className={`upload-zone ${isDragActive ? 'active' : ''}`}
        >
            <input {...getInputProps()} />

            <div className="upload-icon">
                {isDragActive ? '📥' : '✨'}
            </div>

            <h2 className="upload-text">
                {isDragActive
                    ? 'Drop your stickers here...'
                    : 'Drop stickers or click to upload'
                }
            </h2>

            <p className="upload-hint">
                Supports PNG, JPG, WEBP • Batch upload supported
            </p>

            <div style={{ marginTop: '2rem', display: 'flex', gap: '1rem' }}>
                <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(99, 102, 241, 0.1)',
                    borderRadius: '0.5rem',
                    fontSize: '0.875rem'
                }}>
                    🎯 AI Background Removal
                </div>
                <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(139, 92, 246, 0.1)',
                    borderRadius: '0.5rem',
                    fontSize: '0.875rem'
                }}>
                    ✂️ Edge Detection
                </div>
                <div style={{
                    padding: '0.75rem 1rem',
                    background: 'rgba(168, 85, 247, 0.1)',
                    borderRadius: '0.5rem',
                    fontSize: '0.875rem'
                }}>
                    🎨 Custom Borders
                </div>
            </div>
        </div>
    )
}

export default UploadZone
