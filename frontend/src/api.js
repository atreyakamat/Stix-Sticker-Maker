/**
 * API client for Stix backend
 */

const API_BASE = 'http://localhost:8000/api'

export async function uploadImages(files) {
    const formData = new FormData()
    files.forEach(file => {
        formData.append('files', file)
    })

    const response = await fetch(`${API_BASE}/upload`, {
        method: 'POST',
        body: formData,
    })

    if (!response.ok) {
        throw new Error('Upload failed')
    }

    return response.json()
}

export async function getJob(jobId) {
    const response = await fetch(`${API_BASE}/jobs/${jobId}`)

    if (!response.ok) {
        throw new Error('Failed to get job status')
    }

    return response.json()
}

export async function getAllJobs() {
    const response = await fetch(`${API_BASE}/jobs`)

    if (!response.ok) {
        throw new Error('Failed to get jobs')
    }

    return response.json()
}

export async function generateBorder(jobId, thickness, color) {
    const response = await fetch(`${API_BASE}/border`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            job_id: jobId,
            thickness,
            color,
        }),
    })

    if (!response.ok) {
        throw new Error('Failed to generate border')
    }

    return response.json()
}

export async function downloadImage(jobId, withBorder = false) {
    const url = `${API_BASE}/download/${jobId}?with_border=${withBorder}`

    const response = await fetch(url)
    if (!response.ok) {
        throw new Error('Download failed')
    }

    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = downloadUrl
    a.download = `stix_${jobId}.png`
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(downloadUrl)
}

export async function exportBatch(jobIds, includeBorder = true) {
    const response = await fetch(`${API_BASE}/export/batch`, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            job_ids: jobIds,
            include_border: includeBorder,
        }),
    })

    if (!response.ok) {
        throw new Error('Batch export failed')
    }

    const blob = await response.blob()
    const downloadUrl = window.URL.createObjectURL(blob)

    const a = document.createElement('a')
    a.href = downloadUrl
    a.download = 'stix_batch.zip'
    document.body.appendChild(a)
    a.click()
    document.body.removeChild(a)
    window.URL.revokeObjectURL(downloadUrl)
}

export function getImageUrl(path) {
    if (!path) return null
    // Path comes as /output/filename.png from backend
    return `http://localhost:8000${path}`
}
