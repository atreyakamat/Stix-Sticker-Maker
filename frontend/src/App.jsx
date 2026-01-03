import { useState, useEffect, useCallback } from 'react'
import './index.css'
import UploadZone from './components/UploadZone'
import Gallery from './components/Gallery'
import Editor from './components/Editor'
import { uploadImages, getJob, getAllJobs } from './api'

function App() {
  const [jobs, setJobs] = useState([])
  const [selectedJob, setSelectedJob] = useState(null)
  const [view, setView] = useState('upload') // 'upload', 'gallery', 'editor'

  // Poll for job updates
  useEffect(() => {
    if (jobs.length === 0) return

    const processingJobs = jobs.filter(j =>
      j.status === 'pending' || j.status === 'processing'
    )

    if (processingJobs.length === 0) return

    const interval = setInterval(async () => {
      const updatedJobs = await getAllJobs()
      setJobs(updatedJobs)
    }, 1000)

    return () => clearInterval(interval)
  }, [jobs])

  const handleUpload = useCallback(async (files) => {
    try {
      const newJobs = await uploadImages(files)
      if (newJobs.length === 0) {
        alert('No valid images were found in your selection.')
        return
      }
      setJobs(prev => [...prev, ...newJobs])
      setView('gallery')
    } catch (error) {
      console.error('Upload failed:', error)
      alert('Upload failed. Is your backend server running? Check terminal for errors.')
    }
  }, [])

  const handleSelectJob = useCallback((job) => {
    setSelectedJob(job)
    if (job.status === 'complete') {
      setView('editor')
    }
  }, [])

  const handleBack = useCallback(() => {
    setSelectedJob(null)
    setView('gallery')
  }, [])

  const handleJobUpdate = useCallback((updatedJob) => {
    setJobs(prev => prev.map(j =>
      j.id === updatedJob.id ? updatedJob : j
    ))
    setSelectedJob(updatedJob)
  }, [])

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">✦</span>
          <span>Stix</span>
        </div>
        <nav style={{ display: 'flex', gap: '0.5rem' }}>
          {jobs.length > 0 && (
            <>
              <button
                className={`btn btn-secondary ${view === 'upload' ? 'active' : ''}`}
                onClick={() => setView('upload')}
              >
                + New
              </button>
              <button
                className={`btn btn-secondary ${view === 'gallery' ? 'active' : ''}`}
                onClick={() => setView('gallery')}
              >
                Gallery ({jobs.length})
              </button>
            </>
          )}
        </nav>
      </header>

      <main className="main">
        {view === 'upload' && (
          <UploadZone onUpload={handleUpload} />
        )}

        {view === 'gallery' && (
          <Gallery
            jobs={jobs}
            selectedJob={selectedJob}
            onSelect={handleSelectJob}
            onUploadMore={() => setView('upload')}
          />
        )}

        {view === 'editor' && selectedJob && (
          <Editor
            job={selectedJob}
            onBack={handleBack}
            onJobUpdate={handleJobUpdate}
          />
        )}
      </main>
    </div>
  )
}

export default App
