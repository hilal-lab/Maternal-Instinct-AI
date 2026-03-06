'use client';

import { useState, useEffect } from 'react';
import { MessageSquare, Calendar, BarChart3, Settings } from 'lucide-react';
import ChatInterface from '@/components/ChatInterface';
import ScheduleManager from '@/components/ScheduleManager';
import AnalyticsDashboard from '@/components/AnalyticsDashboard';
import { healthCheck } from '@/lib/api';

type Tab = 'chat' | 'schedule' | 'analytics' | 'settings';

export default function Home() {
  const [activeTab, setActiveTab] = useState<Tab>('chat');
  const [apiStatus, setApiStatus] = useState<'checking' | 'online' | 'offline'>('checking');

  useEffect(() => {
    const checkAPI = async () => {
      const isOnline = await healthCheck();
      setApiStatus(isOnline ? 'online' : 'offline');
    };
    
    checkAPI();
    const interval = setInterval(checkAPI, 30000); // Check every 30s
    
    return () => clearInterval(interval);
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-pink-50 via-purple-50 to-indigo-50">
      {/* Header */}
      <header className="bg-white border-b border-gray-200 shadow-sm">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex justify-between items-center h-16">
            <div className="flex items-center space-x-3">
              <div className="w-10 h-10 bg-gradient-to-br from-primary-500 to-purple-600 rounded-lg flex items-center justify-center">
                <span className="text-white font-bold text-xl">M</span>
              </div>
              <div>
                <h1 className="text-xl font-bold text-gray-900">Maternal Instinct AI</h1>
                <p className="text-xs text-gray-500">Proactive Stress Management</p>
              </div>
            </div>

            {/* API Status */}
            <div className="flex items-center space-x-2">
              <div className={`w-2 h-2 rounded-full ${
                apiStatus === 'online' ? 'bg-green-500' : 
                apiStatus === 'offline' ? 'bg-red-500' : 
                'bg-yellow-500'
              }`} />
              <span className="text-sm text-gray-600">
                {apiStatus === 'online' ? 'Backend Online' : 
                 apiStatus === 'offline' ? 'Backend Offline' : 
                 'Checking...'}
              </span>
            </div>
          </div>
        </div>
      </header>

      {/* Navigation Tabs */}
      <nav className="bg-white border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8">
          <div className="flex space-x-8">
            <TabButton
              icon={<MessageSquare className="w-5 h-5" />}
              label="Chat"
              active={activeTab === 'chat'}
              onClick={() => setActiveTab('chat')}
            />
            <TabButton
              icon={<Calendar className="w-5 h-5" />}
              label="Schedule"
              active={activeTab === 'schedule'}
              onClick={() => setActiveTab('schedule')}
            />
            <TabButton
              icon={<BarChart3 className="w-5 h-5" />}
              label="Analytics"
              active={activeTab === 'analytics'}
              onClick={() => setActiveTab('analytics')}
            />
          </div>
        </div>
      </nav>

      {/* Main Content */}
      <main className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {apiStatus === 'offline' && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <p className="text-red-800 text-sm">
              ⚠️ Backend API tidak terhubung. Pastikan backend server berjalan di{' '}
              <code className="bg-red-100 px-1 py-0.5 rounded">http://localhost:8000</code>
            </p>
          </div>
        )}

        {activeTab === 'chat' && <ChatInterface />}
        {activeTab === 'schedule' && <ScheduleManager />}
        {activeTab === 'analytics' && <AnalyticsDashboard />}
      </main>

      {/* Footer */}
      <footer className="bg-white border-t border-gray-200 mt-12">
        <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-6">
          <p className="text-center text-sm text-gray-500">
            Maternal Instinct AI v2.0 — Powered by{' '}
            <span className="font-semibold text-primary-600">Ollama (llama3.1:8b + bge-m3)</span>
            {' '}— 100% Local & Offline
          </p>
        </div>
      </footer>
    </div>
  );
}

function TabButton({
  icon,
  label,
  active,
  onClick,
}: {
  icon: React.ReactNode;
  label: string;
  active: boolean;
  onClick: () => void;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex items-center space-x-2 py-4 border-b-2 transition-colors ${
        active
          ? 'border-primary-600 text-primary-600'
          : 'border-transparent text-gray-600 hover:text-gray-900 hover:border-gray-300'
      }`}
    >
      {icon}
      <span className="font-medium">{label}</span>
    </button>
  );
}
