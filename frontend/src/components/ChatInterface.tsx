'use client';

import { useState, useEffect, useRef, useCallback } from 'react';
import { Send, Loader2, Trash2, Eye, EyeOff, AlertTriangle, CheckCircle, XCircle } from 'lucide-react';
import ReactMarkdown from 'react-markdown';
import { chatAPI } from '@/lib/api';
import type { ChatMessage, ChatResponse } from '@/types/api';

// ─── Types for WebSocket events ───────────────────────────────────────────────

interface LayerEvent {
  type: 'layer_start' | 'layer_progress' | 'layer_done';
  layer: 1 | 2 | 3 | 4;
  step?: string;
  message: string;
  // layer_done extras
  agent?: string;
  raw_draft?: string;
  status?: string;
  decision?: string;
  is_rewritten?: boolean;
  intent?: string;
  emotion?: string;
  intensity?: number;
}

interface ToolConfirmEvent {
  type: 'tool_confirm';
  confirm_id: string;
  tool: string;
  args: Record<string, unknown>;
  description: string;
  message: string;
}

interface PendingConfirmation {
  confirm_id: string;
  tool: string;
  args: Record<string, unknown>;
  description: string;
}

interface PipelineState {
  activeLayer: 1 | 2 | 3 | 4 | null;
  completedLayers: Set<number>;
  events: LayerEvent[];
  rawDraft: string | null;
  agent: string | null;
  workloadDecision: string | null;
  ethicsStatus: string | null;
  isRewritten: boolean | null;
  intent: string | null;
  emotion: string | null;
  intensity: number | null;
}

const EMPTY_PIPELINE: PipelineState = {
  activeLayer: null,
  completedLayers: new Set(),
  events: [],
  rawDraft: null,
  agent: null,
  workloadDecision: null,
  ethicsStatus: null,
  isRewritten: null,
  intent: null,
  emotion: null,
  intensity: null,
};

const WS_URL =
  (process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000')
    .replace(/^http/, 'ws') + '/api/ws/chat';

// ─── Main component ───────────────────────────────────────────────────────────

export default function ChatInterface() {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState('');
  const [loading, setLoading] = useState(false);
  const [showLayers, setShowLayers] = useState(true);
  const [pipeline, setPipeline] = useState<PipelineState>(EMPTY_PIPELINE);
  const [currentLayers, setCurrentLayers] = useState<ChatResponse['layers'] | null>(null);
  const [wsConnected, setWsConnected] = useState(false);
  const [pendingConfirmation, setPendingConfirmation] = useState<PendingConfirmation | null>(null);

  const messagesEndRef = useRef<HTMLDivElement>(null);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectTimer = useRef<ReturnType<typeof setTimeout> | null>(null);

  // ── WebSocket lifecycle ────────────────────────────────────────────────────

  const connectWS = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => setWsConnected(true);

    ws.onclose = () => {
      setWsConnected(false);
      // Reconnect after 3 s
      reconnectTimer.current = setTimeout(connectWS, 3000);
    };

    ws.onerror = () => ws.close();

    ws.onmessage = (evt) => {
      const event = JSON.parse(evt.data);

      if (event.type === 'response') {
        const result: ChatResponse = event.data;
        const assistantMsg: ChatMessage = {
          role: 'assistant',
          content: result.response,
          emotion: result.layers.layer1.emotion,
          intent: result.layers.layer1.intent,
          intensity: result.layers.layer1.intensity,
          layer3_status: result.layers.layer3.status,
          layer4_rewritten: result.layers.layer4.is_rewritten,
        };
        setMessages(prev => [...prev, assistantMsg]);
        setCurrentLayers(result.layers);
        setLoading(false);
        setPendingConfirmation(null);
        // Mark all layers done
        setPipeline(prev => ({
          ...prev,
          activeLayer: null,
          completedLayers: new Set([1, 2, 3, 4]),
        }));
        return;
      }

      if (event.type === 'tool_confirm') {
        // Pipeline suspended — show inline confirmation card
        const confirmEvt = event as ToolConfirmEvent;
        setPendingConfirmation({
          confirm_id: confirmEvt.confirm_id,
          tool: confirmEvt.tool,
          args: confirmEvt.args,
          description: confirmEvt.description,
        });
        // Keep pipeline visible but stop the "thinking" spinner
        setLoading(false);
        return;
      }

      if (event.type === 'error') {
        setLoading(false);
        setPendingConfirmation(null);
        setMessages(prev => [...prev, {
          role: 'assistant',
          content: `Terjadi kesalahan: ${event.message}`,
        }]);
        return;
      }

      // layer_start | layer_progress | layer_done
      const layerEvt = event as LayerEvent;
      setPipeline(prev => {
        const next = { ...prev, events: [...prev.events, layerEvt] };

        if (layerEvt.type === 'layer_start') {
          next.activeLayer = layerEvt.layer;
        }

        if (layerEvt.type === 'layer_progress') {
          if (layerEvt.intent) next.intent = layerEvt.intent;
          if (layerEvt.emotion) next.emotion = layerEvt.emotion;
          if (layerEvt.intensity != null) next.intensity = layerEvt.intensity;
          if (layerEvt.decision) next.workloadDecision = layerEvt.decision;
        }

        if (layerEvt.type === 'layer_done') {
          next.completedLayers = new Set([...prev.completedLayers, layerEvt.layer]);
          if (layerEvt.raw_draft) next.rawDraft = layerEvt.raw_draft;
          if (layerEvt.agent) next.agent = layerEvt.agent;
          if (layerEvt.status) next.ethicsStatus = layerEvt.status;
          if (layerEvt.is_rewritten != null) next.isRewritten = layerEvt.is_rewritten;
        }

        return next;
      });
    };
  }, []);

  useEffect(() => {
    loadHistory();
    connectWS();
    return () => {
      if (reconnectTimer.current) clearTimeout(reconnectTimer.current);
      wsRef.current?.close();
    };
  }, [connectWS]);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages, pendingConfirmation]);

  // ── Actions ────────────────────────────────────────────────────────────────

  const loadHistory = async () => {
    try {
      const history = await chatAPI.getHistory(50);
      setMessages(history);
    } catch (e) {
      console.error('Failed to load history:', e);
    }
  };

  const handleSend = () => {
    if (!input.trim() || loading || pendingConfirmation) return;
    if (!wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) {
      alert('WebSocket belum terhubung. Coba lagi sebentar.');
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setLoading(true);

    // Reset pipeline state for new message
    setPipeline({ ...EMPTY_PIPELINE, completedLayers: new Set() });
    setCurrentLayers(null);

    setMessages(prev => [...prev, { role: 'user', content: userMessage }]);
    wsRef.current.send(JSON.stringify({ type: 'message', message: userMessage }));
  };

  const handleConfirm = (approved: boolean) => {
    if (!pendingConfirmation || !wsRef.current || wsRef.current.readyState !== WebSocket.OPEN) return;

    // Resume loading state while the pipeline finishes
    setLoading(true);

    wsRef.current.send(JSON.stringify({
      type: 'tool_confirm_response',
      confirm_id: pendingConfirmation.confirm_id,
      approved,
    }));

    setPendingConfirmation(null);
  };

  const handleClearHistory = async () => {
    if (!confirm('Hapus semua riwayat chat?')) return;
    try {
      await chatAPI.clearHistory();
      setMessages([]);
      setCurrentLayers(null);
      setPipeline(EMPTY_PIPELINE);
      setPendingConfirmation(null);
    } catch (e) {
      console.error('Failed to clear history:', e);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  // ── Render ─────────────────────────────────────────────────────────────────

  return (
    <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
      {/* ── Chat Area ─────────────────────────────────────────────────────── */}
      <div className="lg:col-span-2">
        <div className="card h-[calc(100vh-280px)] flex flex-col">
          {/* Header */}
          <div className="flex justify-between items-center pb-4 border-b border-gray-200">
            <div className="flex items-center space-x-2">
              <h2 className="text-lg font-semibold text-gray-900">Chat dengan Ara</h2>
              <span className={`w-2 h-2 rounded-full ${wsConnected ? 'bg-green-500' : 'bg-red-400'}`} title={wsConnected ? 'WebSocket terhubung' : 'Memulihkan koneksi...'} />
            </div>
            <div className="flex space-x-2">
              <button onClick={() => setShowLayers(!showLayers)} className="btn-secondary text-sm flex items-center space-x-1">
                {showLayers ? <EyeOff className="w-4 h-4" /> : <Eye className="w-4 h-4" />}
                <span>{showLayers ? 'Sembunyikan' : 'Tampilkan'} Panel</span>
              </button>
              <button onClick={handleClearHistory} className="btn-secondary text-sm flex items-center space-x-1 text-red-600 hover:bg-red-50">
                <Trash2 className="w-4 h-4" />
                <span>Hapus</span>
              </button>
            </div>
          </div>

          {/* Messages */}
          <div className="flex-1 overflow-y-auto space-y-4 py-4">
            {messages.length === 0 ? (
              <div className="text-center text-gray-500 mt-8">
                <p className="text-lg font-medium mb-2">Selamat datang!</p>
                <p className="text-sm">Mulai percakapan dengan mengetik pesan di bawah.</p>
              </div>
            ) : (
              messages.map((msg, idx) => <MessageBubble key={idx} message={msg} />)
            )}
            {loading && !pendingConfirmation && (
              <div className="flex justify-start">
                <div className="bg-gray-100 rounded-lg px-4 py-3 flex items-center space-x-2">
                  <Loader2 className="w-4 h-4 animate-spin text-primary-600" />
                  <span className="text-sm text-gray-600">Ara sedang berpikir...</span>
                </div>
              </div>
            )}
            {/* Inline tool confirmation card */}
            {pendingConfirmation && (
              <ToolConfirmCard
                confirmation={pendingConfirmation}
                onApprove={() => handleConfirm(true)}
                onDeny={() => handleConfirm(false)}
              />
            )}
            <div ref={messagesEndRef} />
          </div>

          {/* Input */}
          <div className="pt-4 border-t border-gray-200">
            <div className="flex space-x-2">
              <textarea
                value={input}
                onChange={e => setInput(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder={
                  pendingConfirmation
                    ? 'Harap jawab konfirmasi di atas terlebih dahulu...'
                    : 'Ketik pesan... (Enter kirim, Shift+Enter baris baru)'
                }
                className="input-field resize-none h-20"
                disabled={loading || !!pendingConfirmation}
              />
              <button onClick={handleSend} disabled={loading || !input.trim() || !!pendingConfirmation} className="btn-primary px-6">
                {loading && !pendingConfirmation ? <Loader2 className="w-5 h-5 animate-spin" /> : <Send className="w-5 h-5" />}
              </button>
            </div>
          </div>
        </div>
      </div>

      {/* ── Right Panel: Layer Tracker ───────────────────────────────────── */}
      {showLayers && (
        <div className="lg:col-span-1 flex flex-col gap-4">
          {/* Live layer progress */}
          <div className="card overflow-y-auto" style={{ maxHeight: 'calc(50vh - 60px)' }}>
            <h2 className="text-base font-semibold text-gray-900 mb-3">Pipeline Progress</h2>
            <div className="space-y-2">
              {([1, 2, 3, 4] as const).map(n => (
                <LayerStatus
                  key={n}
                  layer={n}
                  isActive={pipeline.activeLayer === n}
                  isDone={pipeline.completedLayers.has(n)}
                  isPending={!pipeline.completedLayers.has(n) && pipeline.activeLayer !== n}
                  events={pipeline.events.filter(e => e.layer === n)}
                />
              ))}
            </div>
          </div>

          {/* Raw draft from Layer 2 */}
          {(loading || pendingConfirmation || pipeline.rawDraft) && (
            <div className="card overflow-y-auto" style={{ maxHeight: 'calc(50vh - 60px)' }}>
              <h2 className="text-base font-semibold text-gray-900 mb-2 flex items-center gap-2">
                Raw Draft
                {(loading || pendingConfirmation) && !pipeline.rawDraft && (
                  <Loader2 className="w-3 h-3 animate-spin text-gray-400" />
                )}
              </h2>
              {pipeline.rawDraft ? (
                <pre className="text-xs text-gray-700 whitespace-pre-wrap font-mono leading-relaxed">
                  {pipeline.rawDraft}
                </pre>
              ) : (
                <p className="text-xs text-gray-400 italic">Menunggu Layer 2...</p>
              )}
            </div>
          )}

          {/* Final layer details (after response arrives) */}
          {currentLayers && !loading && !pendingConfirmation && (
            <div className="card overflow-y-auto" style={{ maxHeight: 'calc(50vh - 60px)' }}>
              <h2 className="text-base font-semibold text-gray-900 mb-3">Layer Details</h2>
              <div className="space-y-3">
                <DetailCard title="Layer 1 — Orchestrator" color="blue">
                  <Row label="Intent" value={currentLayers.layer1.intent} />
                  <Row label="Emosi" value={currentLayers.layer1.emotion} />
                  <Row label="Intensitas" value={`${(currentLayers.layer1.intensity * 100).toFixed(0)}%`} />
                </DetailCard>
                <DetailCard title="Layer 2 — Specialist" color="green">
                  <Row label="Agen" value={currentLayers.layer2.agent_used} />
                  <Row label="Panjang" value={`${currentLayers.layer2.raw_response.length} karakter`} />
                </DetailCard>
                <DetailCard title="Layer 3 — Ethics" color={currentLayers.layer3.status === 'PASS' ? 'green' : 'yellow'}>
                  <Row label="Status" value={currentLayers.layer3.status} />
                  {currentLayers.layer3.note && (
                    <p className="text-xs text-gray-600 mt-1">{currentLayers.layer3.note}</p>
                  )}
                </DetailCard>
                <DetailCard title="Layer 4 — Guardrail" color={currentLayers.layer4.is_rewritten ? 'purple' : 'gray'}>
                  <Row label="Ditulis ulang" value={currentLayers.layer4.is_rewritten ? 'Ya' : 'Tidak'} />
                </DetailCard>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}

// ─── Sub-components ───────────────────────────────────────────────────────────

function MessageBubble({ message }: { message: ChatMessage }) {
  const isUser = message.role === 'user';
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'}`}>
      <div className={`max-w-[80%] rounded-lg px-4 py-3 ${isUser ? 'bg-primary-600 text-white' : 'bg-gray-100 text-gray-900'}`}>
        {isUser ? (
          <p className="text-sm whitespace-pre-wrap">{message.content}</p>
        ) : (
          <div className="text-sm prose prose-sm max-w-none
            prose-p:my-1 prose-p:leading-relaxed
            prose-strong:font-semibold prose-strong:text-gray-900
            prose-ul:my-1 prose-ul:pl-4
            prose-ol:my-1 prose-ol:pl-4
            prose-li:my-0.5
            prose-headings:font-semibold prose-headings:text-gray-900
          ">
            <ReactMarkdown>{message.content}</ReactMarkdown>
          </div>
        )}
        {!isUser && message.emotion && (
          <div className="mt-2 pt-2 border-t border-gray-300 flex flex-wrap gap-2 text-xs">
            <span className="bg-white/40 px-2 py-0.5 rounded">{message.emotion}</span>
            {message.layer4_rewritten && (
              <span className="bg-purple-500/20 px-2 py-0.5 rounded">Ditulis ulang</span>
            )}
            {message.layer3_status === 'VIOLATION' && (
              <span className="bg-red-500/20 px-2 py-0.5 rounded">Peringatan etika</span>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

function ToolConfirmCard({
  confirmation,
  onApprove,
  onDeny,
}: {
  confirmation: PendingConfirmation;
  onApprove: () => void;
  onDeny: () => void;
}) {
  const argsDisplay = Object.entries(confirmation.args)
    .map(([k, v]) => `${k}: ${String(v)}`)
    .join(', ');

  return (
    <div className="flex justify-start">
      <div className="max-w-[85%] rounded-xl border-2 border-amber-300 bg-amber-50 px-4 py-4 shadow-sm">
        {/* Header */}
        <div className="flex items-center gap-2 mb-3">
          <AlertTriangle className="w-5 h-5 text-amber-600 flex-shrink-0" />
          <span className="text-sm font-semibold text-amber-800">Konfirmasi Tindakan Ara</span>
        </div>

        {/* Description */}
        <p className="text-sm text-gray-700 mb-2">{confirmation.description}</p>

        {/* Tool + args details */}
        <div className="bg-white/70 rounded-lg px-3 py-2 mb-4 text-xs text-gray-600 font-mono border border-amber-200">
          <span className="font-semibold text-amber-700">{confirmation.tool}</span>
          {argsDisplay && <span className="ml-1 text-gray-500">({argsDisplay})</span>}
        </div>

        {/* Action buttons */}
        <div className="flex gap-2">
          <button
            onClick={onApprove}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-green-600 hover:bg-green-700 text-white text-sm font-medium transition-colors"
          >
            <CheckCircle className="w-4 h-4" />
            Ya, lanjutkan
          </button>
          <button
            onClick={onDeny}
            className="flex items-center gap-1.5 px-4 py-2 rounded-lg bg-white hover:bg-gray-50 text-gray-700 text-sm font-medium border border-gray-300 transition-colors"
          >
            <XCircle className="w-4 h-4" />
            Batal
          </button>
        </div>
      </div>
    </div>
  );
}

function LayerStatus({
  layer,
  isActive,
  isDone,
  isPending,
  events,
}: {
  layer: 1 | 2 | 3 | 4;
  isActive: boolean;
  isDone: boolean;
  isPending: boolean;
  events: LayerEvent[];
}) {
  const labels: Record<number, string> = {
    1: 'L1 — Orchestrator',
    2: 'L2 — Specialist',
    3: 'L3 — Ethics',
    4: 'L4 — Guardrail',
  };

  const lastEvent = events[events.length - 1];

  let dot = 'bg-gray-300';
  let textColor = 'text-gray-400';
  let border = 'border-gray-200';
  let bg = 'bg-gray-50';

  if (isActive) {
    dot = 'bg-yellow-400 animate-pulse';
    textColor = 'text-yellow-800';
    border = 'border-yellow-300';
    bg = 'bg-yellow-50';
  } else if (isDone) {
    dot = 'bg-green-500';
    textColor = 'text-green-800';
    border = 'border-green-200';
    bg = 'bg-green-50';
  }

  return (
    <div className={`rounded-lg border px-3 py-2 ${border} ${bg}`}>
      <div className="flex items-center gap-2">
        <span className={`w-2 h-2 rounded-full flex-shrink-0 ${dot}`} />
        <span className={`text-xs font-semibold ${textColor}`}>{labels[layer]}</span>
        {isActive && <Loader2 className="w-3 h-3 animate-spin text-yellow-600 ml-auto" />}
        {isDone && <span className="text-green-600 text-xs ml-auto">✓</span>}
        {isPending && <span className="text-gray-400 text-xs ml-auto">—</span>}
      </div>
      {lastEvent && (
        <p className="text-xs text-gray-600 mt-1 pl-4 truncate" title={lastEvent.message}>
          {lastEvent.message}
        </p>
      )}
    </div>
  );
}

function DetailCard({
  title,
  color,
  children,
}: {
  title: string;
  color: string;
  children: React.ReactNode;
}) {
  const colors: Record<string, string> = {
    blue: 'border-blue-200 bg-blue-50',
    green: 'border-green-200 bg-green-50',
    yellow: 'border-yellow-200 bg-yellow-50',
    purple: 'border-purple-200 bg-purple-50',
    gray: 'border-gray-200 bg-gray-50',
  };
  return (
    <div className={`border rounded-lg p-3 ${colors[color] ?? colors.gray}`}>
      <h3 className="font-semibold text-xs text-gray-700 mb-2">{title}</h3>
      {children}
    </div>
  );
}

function Row({ label, value }: { label: string; value: string }) {
  return (
    <div className="flex justify-between items-center text-xs">
      <span className="text-gray-500">{label}:</span>
      <span className="font-medium text-gray-800 text-right max-w-[60%] truncate" title={value}>{value}</span>
    </div>
  );
}
