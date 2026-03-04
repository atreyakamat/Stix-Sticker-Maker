/**
 * Stix - AI Sticker Maker
 * Main Application Component
 * 
 * Flow: Upload → Gallery → Editor
 */

import { useState, useEffect, useCallback } from 'react'
import './index.css'
import UploadZone from './components/UploadZone'
import Gallery from './components/Gallery'
import Editor from './components/Editor'
import { uploadImages, getAllJobs } from './api'

// View states
const VIEWS = {
  UPLOAD: 'upload',
  GALLERY: 'gallery',
  EDITOR: 'editor'
}

function App() {
  // State
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [view, setView] = useState(VIEWS.UPLOAD)
  const [isUploading, setIsUploading] = useState(false)

  // =========================================================================
  // JOB POLLING
  // Auto-refresh jobs while any are processing
  // =========================================================================
  useEffect(() => {
    const hasProcessingJobs = jobs.some(j =>
      j.status === 'pending' || j.status === 'processing'
    )

    if (!hasProcessingJobs) return

    const interval = setInterval(async () => {
      try {
        const updatedJobs = await getAllJobs()
        setJobs(updatedJobs)
      } catch (error) {
        console.error('Failed to refresh jobs:', error)
      }
    }, 1000)

    return () => clearInterval(interval)
  }, [jobs])

  // =========================================================================
  // HANDLERS
  // =========================================================================

  /**
   * Handle file upload
   * Creates jobs and switches to gallery view
   */
  const handleUpload = useCallback(async (files, tier) => {
    if (files.length === 0) return

    setIsUploading(true)
    try {
      const newJobs = await uploadImages(files, tier)

      if (newJobs.length === 0) {
        alert('No valid images found. Please upload PNG, JPG, or WEBP files.')
        return
      }

      setJobs(prev => [...prev, ...newJobs])
      setView(VIEWS.GALLERY)
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Upload failed. Please check that the backend server is running.')
    } finally {
      setIsUploading(false)
    }
  }, [])

  /**
   * Handle sticker selection from gallery
   * Opens editor if job is complete
   */
  const handleSelectJob = useCallback((job) => {
    setSelectedJob(job)
    if (job.status === 'complete') {
      setView(VIEWS.EDITOR)
    }
  }, [])

  /**
   * Handle back navigation from editor
   */
  const handleBack = useCallback(() => {
    setSelectedJob(null)
    setView(VIEWS.GALLERY)
  }, [])

  /**
   * Handle job updates from editor
   * Updates both the jobs list and selected job
   */
  const handleJobUpdate = useCallback((updatedJob) => {
    setJobs(prev => prev.map(j =>
      j.id === updatedJob.id ? updatedJob : j
    ))
    setSelectedJob(updatedJob)
  }, [])

  // =========================================================================
  // RENDER
  // =========================================================================

  const completedCount = jobs.filter(j => j.status === 'complete').length
  const processingCount = jobs.filter(j => j.status === 'processing' || j.status === 'pending').length

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="logo" onClick={() => jobs.length === 0 ? null : setView(VIEWS.GALLERY)}>
          <span className="logo-icon">✦</span>
          <span>Stix</span>
        </div>

        {/* Navigation - only show when there are jobs */}
        {jobs.length > 0 && (
          <nav className="nav-tabs">
            <button
              className={`nav-tab ${view === VIEWS.UPLOAD ? 'active' : ''}`}
              onClick={() => setView(VIEWS.UPLOAD)}
            >
              + New Upload
            </button>
            <button
              className={`nav-tab ${view === VIEWS.GALLERY || view === VIEWS.EDITOR ? 'active' : ''}`}
              onClick={() => setView(VIEWS.GALLERY)}
            >
              Gallery
              {completedCount > 0 && <span className="badge">{completedCount}</span>}
              {processingCount > 0 && <span className="badge processing">{processingCount}</span>}
            </button>
          </nav>
        )}
      </header>

      {/* Main Content */}
      <main className="main">
        {view === VIEWS.UPLOAD && (
          <UploadZone
            onUpload={handleUpload}
            isUploading={isUploading}
          />
        )}

        {view === VIEWS.GALLERY && (
          <Gallery
            jobs={jobs}
            selectedJob={selectedJob}
            onSelect={handleSelectJob}
            onUploadMore={() => setView(VIEWS.UPLOAD)}
          />
        )}

        {view === VIEWS.EDITOR && selectedJob && (
          <Editor
            job={selectedJob}
            onBack={handleBack}
            onJobUpdate={handleJobUpdate}
          />
        )}
      </main>

      {/* Footer */}
      <footer className="footer">
        <span>Stix v2.0</span>
        <span className="footer-divider">•</span>
        <span>Professional Sticker Studio</span>
      </footer>
    </div>
  )
}

export default App
