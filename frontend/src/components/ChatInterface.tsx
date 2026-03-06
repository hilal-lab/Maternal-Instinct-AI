'use client';

import { useState, useEffect, useRef } from 'react';
import { Send, Loader2, Trash2, Eye, EyeOff } from 'lucide-react';
import { chatAPI } from '@/lib/api';
import type { ChatMessage, ChatResponse } from '@/types/api';
import { format } from 'date-fns';

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showLayers, setShowLayers] = useState(false);
  const [currentLayers, setCurrentLayers] = useState<ChatResponse['layers'] | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    loadHistory();
  }, []);

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  const loadHistory = async () => {
    try {
      const history = await chatAPI.getHistory(50);
      setMessages(history);
    } catch (error) {
      console.error('Failed to load history:', error);
    }
  };

  const handleSend = async () => {
    if (!input.trim() || loading) return;

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    // Add user message immediately
    const tempUserMsg: ChatMessage = {
      role: 'user',
      content: userMessage,
    };
    setMessages(prev => [...prev, tempUserMsg]);

    try {
      const response = await chatAPI.sendMessage(userMessage);
      
      // Add assistant response
      const assistantMsg: ChatMessage = {
        role: 'assistant',
        content: response.response,
        emotion: response.layers.layer1.emotion,
        intent: response.layers.layer1.intent,
        intensity: response.layers.layer1.intensity,
        layer3_status: response.layers.layer3.status,
        layer4_rewritten: response.layers.layer4.is_rewritten,
      };
      
      setMessages(prev => [...prev, assistantMsg]);
      setCurrentLayers(response.layers);
      
    } catch (error: any) {
      console.error('Chat error:', error);
      const errorMsg: ChatMessage = {
        role: 'assistant',
        content: `❌ Error: ${error.response?.data?.detail || error.message || 'Failed to send message'}`,
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setLoading(false);
    }
  };

  const handleClearHistory = async () => {
    if (!confirm('Hapus semua riwayat chat?')) return;
    
    try {
      await chatAPI.clearHistory();
      setMessages([]);
      setCurrentLayers(null);
    } catch (error) {
      console.error('Failed to clear history:', error);
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* Chat Area */}
      <div className="lg:col-span-2">
        <div className="card h-[calc(100vh-280px)] flex flex-col">
          {/* Header */}
          <div className="flex justify-between items-center pb-4 border-b border-gray-200">
            <h2 className="text-lg font-semibold text-gray-900">Chat with AI</h2>
            <div className="flex space-x-2">
              <button
                onClick={() => setShowLayers(!showLayers)}
                className="btn-secondary text-sm flex items-center space-x-1"
              >
                {showLayers ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                <span>{showLayers ? 'Hide' : 'Show'} Layers</span>
              </button>
              <button
                onClick={handleClearHistory}
                className="btn-secondary text-sm flex items-center space-x-1 text-red-600 hover:bg-red-50"
              >
                <Trash2 className="w-4 h-4" />
                <span>Clear</span>
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 py-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-8">
                <p className="text-lg font-medium mb-2">Selamat datang! 👋</p>
                <p className="text-sm">Mulai percakapan dengan mengetik pesan di bawah.</p>
              </div>
            ) : (
              messages.map((msg, idx) => (
                <MessageBubble key={idx} message={msg} />
              ))
            )}
            {loading && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin text-primary-600" />
                  <span className="text-sm text-gray-600">AI sedang berpikir...</span>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="pt-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder="Ketik pesan Anda... (Enter untuk kirim, Shift+Enter untuk baris baru)"
                className="input-field resize-none h-20"
                disabled={loading}
              />
              <button
                onClick={handleSend}
                disabled={loading || !input.trim()}
                className="btn-primary px-6"
              >
                {loading ? (
                  <Loader2 className="w-5 h-5 animate-spin" />
                ) : (
                  <Send className="w-5 h-5" />
                )}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* Layers Info */}
      <div className="lg:col-span-1">
        <div className="card h-[calc(100vh-280px)] overflow-y-auto">
          <h2 className="text-lg font-semibold text-gray-900 mb-4">4-Layer Architecture</h2>
          
          {!showLayers || !currentLayers ? (
            <div className="text-center text-gray-500 mt-8">
              <p className="text-sm">Send a message to see layer details</p>
            </div>
          ) : (
            <div className="space-y-4">
              {/* Layer 1 */}
              <LayerCard
                title="Layer 1: Orchestrator"
                color="blue"
                content={
                  <>
                    <InfoRow label="Intent" value={currentLayers.layer1.intent} />
                    <InfoRow label="Emotion" value={currentLayers.layer1.emotion} />
                    <InfoRow 
                      label="Intensity" 
                      value={`${(currentLayers.layer1.intensity * 100).toFixed(0)}%`} 
                    />
                  </>
                }
              />

              {/* Layer 2 */}
              <LayerCard
                title="Layer 2: Specialist"
                color="green"
                content={
                  <>
                    <InfoRow label="Agent" value={currentLayers.layer2.agent_used} />
                    <InfoRow 
                      label="Response" 
                      value={`${currentLayers.layer2.raw_response.length} chars`} 
                    />
                  </>
                }
              />

              {/* Layer 3 */}
              <LayerCard
                title="Layer 3: Ethics"
                color={currentLayers.layer3.status === 'PASS' ? 'green' : 'yellow'}
                content={
                  <>
                    <InfoRow label="Status" value={currentLayers.layer3.status} />
                    {currentLayers.layer3.note && (
                      <p className="text-xs text-gray-600 mt-2">{currentLayers.layer3.note}</p>
                    )}
                  </>
                }
              />

              {/* Layer 4 */}
              <LayerCard
                title="Layer 4: Guardrail"
                color={currentLayers.layer4.is_rewritten ? 'purple' : 'gray'}
                content={
                  <>
                    <InfoRow 
                      label="Rewritten" 
                      value={currentLayers.layer4.is_rewritten ? 'Yes' : 'No'} 
                    />
                    {currentLayers.layer4.is_rewritten && (
                      <p className="text-xs text-gray-600 mt-2">
                        Maternal tone applied
                      </p>
                    )}
                  </>
                }
              />
            </div>
          )}
        </div>
      </div>
    </div>
  );
}

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg px-4 py-3 ${
        isUser 
          ? 'bg-primary-600 text-white' 
          : 'bg-gray-100 text-gray-900'
      }`}>
        <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        
        {!isUser && message.emotion && (
          <div className="mt-2 pt-2 border-t border-gray-300 flex flex-wrap gap-2 text-xs">
            <span className="bg-white/20 px-2 py-0.5 rounded">
              {message.emotion}
            </span>
            {message.layer4_rewritten && (
              <span className="bg-purple-500/20 px-2 py-0.5 rounded">
                ✨ Rewritten
              </span>
            )}
            {message.layer3_status === 'VIOLATION' && (
              <span className="bg-red-500/20 px-2 py-0.5 rounded">
                ⚠️ Warning
              </span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function LayerCard({ 
  title, 
  color, 
  content 
}: { 
  title: string; 
  color: string; 
  content: React.ReactNode;
}) {
  const colorClasses = {
    blue: 'border-blue-200 bg-blue-50',
    green: 'border-green-200 bg-green-50',
    yellow: 'border-yellow-200 bg-yellow-50',
    purple: 'border-purple-200 bg-purple-50',
    gray: 'border-gray-200 bg-gray-50',
  };

  return (
    <div className={`border-2 rounded-lg p-3 ${colorClasses[color as keyof typeof colorClasses]}`}>
      <h3 className="font-semibold text-sm text-gray-900 mb-2">{title}</h3>
      {content}
    </div>
  );
}

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center text-xs">
      <span className="text-gray-600">{label}:</span>
      <span className="font-medium text-gray-900">{value}</span>
    </div>
  );
}
