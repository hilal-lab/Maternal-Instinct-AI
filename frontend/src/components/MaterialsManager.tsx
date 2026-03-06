'use client';

import { useState, useEffect, useRef } from 'react';
import { Upload, Trash2, FileText, RefreshCw, BookOpen } from 'lucide-react';
import { documentsAPI } from '@/lib/api';
import type { Document } from '@/types/api';

export default function MaterialsManager() {
  const [documents, setDocuments] = useState<Document[]>([]);
  const [loading, setLoading] = useState(true);
  const [uploading, setUploading] = useState(false);
  const [dragOver, setDragOver] = useState(false);
  const [uploadError, setUploadError] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  useEffect(() => {
    loadDocuments();
  }, []);

  const loadDocuments = async () => {
    setLoading(true);
    try {
      const data = await documentsAPI.list();
      setDocuments(data);
    } catch (error) {
      console.error('Failed to load documents:', error);
    } finally {
      setLoading(false);
    }
  };

  const handleFiles = async (files: FileList | null) => {
    if (!files || files.length === 0) return;
    setUploadError(null);

    const allowed = ['application/pdf', 'text/plain', 'text/markdown'];
    const file = files[0];

    if (!allowed.includes(file.type) && !file.name.endsWith('.md')) {
      setUploadError('Format tidak didukung. Gunakan PDF, TXT, atau Markdown (.md)');
      return;
    }

    setUploading(true);
    try {
      await documentsAPI.upload(file);
      await loadDocuments();
    } catch (error: any) {
      setUploadError(
        error.response?.data?.detail || 'Gagal mengunggah file. Coba lagi.'
      );
    } finally {
      setUploading(false);
    }
  };

  const handleDelete = async (id: number, filename: string) => {
    if (!confirm(`Hapus "${filename}"?`)) return;
    try {
      await documentsAPI.delete(id);
      await loadDocuments();
    } catch (error) {
      console.error('Failed to delete document:', error);
    }
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setDragOver(false);
    handleFiles(e.dataTransfer.files);
  };

  const formatSize = (bytes: number) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const contentTypeLabel = (ct: string) => {
    if (ct === 'application/pdf') return 'PDF';
    if (ct === 'text/markdown' || ct.includes('markdown')) return 'Markdown';
    if (ct === 'text/plain') return 'TXT';
    return ct;
  };

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Materi Belajar</h2>
          <p className="text-sm text-gray-600 mt-1">
            Upload PDF atau teks untuk diindeks ke RAG knowledge base
          </p>
        </div>
        <button onClick={loadDocuments} className="btn-secondary flex items-center space-x-2">
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Upload Area */}
      <div
        onDragOver={(e) => { e.preventDefault(); setDragOver(true); }}
        onDragLeave={() => setDragOver(false)}
        onDrop={handleDrop}
        onClick={() => fileInputRef.current?.click()}
        className={`border-2 border-dashed rounded-xl p-10 text-center cursor-pointer transition-colors ${
          dragOver
            ? 'border-primary-500 bg-primary-50'
            : 'border-gray-300 hover:border-primary-400 hover:bg-gray-50'
        }`}
      >
        <input
          ref={fileInputRef}
          type="file"
          accept=".pdf,.txt,.md,text/plain,text/markdown,application/pdf"
          className="hidden"
          onChange={(e) => handleFiles(e.target.files)}
        />
        {uploading ? (
          <div className="flex flex-col items-center space-y-2">
            <RefreshCw className="w-10 h-10 text-primary-500 animate-spin" />
            <p className="text-sm text-gray-600">Sedang mengunggah dan mengindeks...</p>
          </div>
        ) : (
          <div className="flex flex-col items-center space-y-2">
            <Upload className="w-10 h-10 text-gray-400" />
            <p className="text-base font-medium text-gray-700">
              Seret file ke sini atau klik untuk memilih
            </p>
            <p className="text-xs text-gray-500">PDF, TXT, Markdown — maks. 10 MB</p>
          </div>
        )}
      </div>

      {uploadError && (
        <div className="p-3 bg-red-50 border border-red-200 rounded-lg text-sm text-red-700">
          {uploadError}
        </div>
      )}

      {/* Document List */}
      <div>
        <h3 className="text-lg font-semibold text-gray-900 mb-3">
          Dokumen Terindeks ({documents.length})
        </h3>

        {loading ? (
          <div className="card text-center text-gray-500">Memuat dokumen...</div>
        ) : documents.length === 0 ? (
          <div className="card text-center">
            <BookOpen className="w-12 h-12 mx-auto mb-2 text-gray-400" />
            <p className="text-gray-500">Belum ada dokumen. Upload file untuk memulai.</p>
          </div>
        ) : (
          <div className="space-y-3">
            {documents.map((doc) => (
              <div key={doc.id} className="card flex items-center justify-between hover:shadow-md transition-shadow">
                <div className="flex items-center space-x-4">
                  <div className="p-2 bg-indigo-100 rounded-lg">
                    <FileText className="w-5 h-5 text-indigo-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{doc.filename}</p>
                    <div className="flex items-center space-x-3 text-xs text-gray-500 mt-0.5">
                      <span className="bg-gray-100 px-2 py-0.5 rounded">
                        {contentTypeLabel(doc.content_type)}
                      </span>
                      <span>{formatSize(doc.file_size)}</span>
                      <span>{doc.chunk_count} chunks</span>
                      <span>{doc.created_at.split('T')[0]}</span>
                    </div>
                  </div>
                </div>
                <button
                  onClick={() => handleDelete(doc.id, doc.filename)}
                  className="p-2 text-red-500 hover:bg-red-50 rounded transition-colors"
                  title="Hapus dokumen"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
