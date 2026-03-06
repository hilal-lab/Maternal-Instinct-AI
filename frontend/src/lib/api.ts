import axios from 'axios';
import type { ChatResponse, ChatMessage, ScheduleTask, Analytics, Document } from '@/types/api';

const API_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';

const api = axios.create({
  baseURL: `${API_URL}/api`,
  headers: {
    'Content-Type': 'application/json',
  },
  timeout: 120000, // 2 minutes for LLM responses
});

// Chat endpoints
export const chatAPI = {
  sendMessage: async (message: string): Promise<ChatResponse> => {
    const { data } = await api.post<ChatResponse>('/chat', { message });
    return data;
  },

  getHistory: async (limit: number = 50): Promise<ChatMessage[]> => {
    const { data } = await api.get<ChatMessage[]>('/chat/history', {
      params: { limit },
    });
    return data;
  },

  clearHistory: async (): Promise<void> => {
    await api.delete('/chat/history');
  },
};

// Schedule endpoints
export const scheduleAPI = {
  getTasks: async (): Promise<ScheduleTask[]> => {
    const { data } = await api.get<ScheduleTask[]>('/schedule');
    return data;
  },

  createTask: async (task: Omit<ScheduleTask, 'id' | 'created_at'>): Promise<ScheduleTask> => {
    const { data } = await api.post<ScheduleTask>('/schedule', task);
    return data;
  },

  updateTask: async (id: number, updates: Partial<ScheduleTask>): Promise<ScheduleTask> => {
    const { data } = await api.put<ScheduleTask>(`/schedule/${id}`, updates);
    return data;
  },

  deleteTask: async (id: number): Promise<void> => {
    await api.delete(`/schedule/${id}`);
  },
};

// Analytics endpoints
export const analyticsAPI = {
  getStats: async (): Promise<Analytics> => {
    const { data } = await api.get<Analytics>('/analytics');
    return data;
  },
};

// Documents endpoints
export const documentsAPI = {
  list: async (): Promise<Document[]> => {
    const { data } = await api.get<Document[]>('/documents');
    return data;
  },

  upload: async (file: File): Promise<Document> => {
    const formData = new FormData();
    formData.append('file', file);
    const { data } = await api.post<Document>('/documents/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' },
    });
    return data;
  },

  delete: async (id: number): Promise<void> => {
    await api.delete(`/documents/${id}`);
  },
};

// Health check
export const healthCheck = async (): Promise<boolean> => {
  try {
    const { data } = await axios.get(`${API_URL}/`);
    return data.status === 'running';
  } catch {
    return false;
  }
};

export default api;
