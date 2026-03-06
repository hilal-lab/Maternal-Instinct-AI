'use client';

import { useState, useEffect } from 'react';
import { TrendingUp, AlertTriangle, RefreshCw, Activity } from 'lucide-react';
import { analyticsAPI } from '@/lib/api';
import type { Analytics } from '@/types/api';

export default function AnalyticsDashboard() {
  const [stats, setStats] = useState<Analytics | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    setLoading(true);
    try {
      const data = await analyticsAPI.getStats();
      setStats(data);
    } catch (error) {
      console.error('Failed to load analytics:', error);
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="card">Loading analytics...</div>;
  }

  if (!stats) {
    return <div className="card text-gray-500">No analytics data available</div>;
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex justify-between items-center">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">Analytics Dashboard</h2>
          <p className="text-sm text-gray-600 mt-1">
            System performance and usage insights
          </p>
        </div>
        <button onClick={loadStats} className="btn-secondary flex items-center space-x-2">
          <RefreshCw className="w-4 h-4" />
          <span>Refresh</span>
        </button>
      </div>

      {/* Stats Grid */}
      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-6">
        <StatCard
          title="Total Chats"
          value={stats.total_chats}
          icon={<Activity className="w-6 h-6" />}
          color="blue"
        />
        <StatCard
          title="Avg Intensity"
          value={`${(stats.avg_intensity * 100).toFixed(0)}%`}
          icon={<TrendingUp className="w-6 h-6" />}
          color="green"
        />
        <StatCard
          title="Ethics Violations"
          value={stats.layer3_violations}
          icon={<AlertTriangle className="w-6 h-6" />}
          color="yellow"
        />
        <StatCard
          title="Maternal Rewrites"
          value={stats.layer4_rewrites}
          icon={<RefreshCw className="w-6 h-6" />}
          color="purple"
        />
      </div>

      {/* Charts */}
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Emotion Distribution */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Emotion Distribution</h3>
          <div className="space-y-3">
            {Object.entries(stats.emotion_distribution).length === 0 ? (
              <p className="text-sm text-gray-500">No emotion data yet</p>
            ) : (
              Object.entries(stats.emotion_distribution)
                .sort(([, a], [, b]) => b - a)
                .map(([emotion, count]) => (
                  <BarChart
                    key={emotion}
                    label={emotion}
                    value={count}
                    max={Math.max(...Object.values(stats.emotion_distribution))}
                    color="blue"
                  />
                ))
            )}
          </div>
        </div>

        {/* Intent Distribution */}
        <div className="card">
          <h3 className="text-lg font-semibold mb-4">Intent Distribution</h3>
          <div className="space-y-3">
            {Object.entries(stats.intent_distribution).length === 0 ? (
              <p className="text-sm text-gray-500">No intent data yet</p>
            ) : (
              Object.entries(stats.intent_distribution)
                .sort(([, a], [, b]) => b - a)
                .map(([intent, count]) => (
                  <BarChart
                    key={intent}
                    label={intent}
                    value={count}
                    max={Math.max(...Object.values(stats.intent_distribution))}
                    color="green"
                  />
                ))
            )}
          </div>
        </div>
      </div>

      {/* Insights */}
      <div className="card">
        <h3 className="text-lg font-semibold mb-4">📊 Insights</h3>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <InsightCard
            title="Layer 3 (Ethics) Activity"
            description={`${stats.layer3_violations} violations detected out of ${stats.total_chats} total chats`}
            percentage={(stats.layer3_violations / stats.total_chats * 100).toFixed(1)}
            type="warning"
          />
          <InsightCard
            title="Layer 4 (Guardrail) Activity"
            description={`${stats.layer4_rewrites} responses rewritten with maternal tone`}
            percentage={(stats.layer4_rewrites / stats.total_chats * 100).toFixed(1)}
            type="success"
          />
        </div>
      </div>
    </div>
  );
}

function StatCard({
  title,
  value,
  icon,
  color,
}: {
  title: string;
  value: number | string;
  icon: React.ReactNode;
  color: 'blue' | 'green' | 'yellow' | 'purple';
}) {
  const colors = {
    blue: 'bg-blue-100 text-blue-600',
    green: 'bg-green-100 text-green-600',
    yellow: 'bg-yellow-100 text-yellow-600',
    purple: 'bg-purple-100 text-purple-600',
  };

  return (
    <div className="card">
      <div className="flex items-center justify-between">
        <div>
          <p className="text-sm text-gray-600 mb-1">{title}</p>
          <p className="text-3xl font-bold text-gray-900">{value}</p>
        </div>
        <div className={`p-3 rounded-lg ${colors[color]}`}>
          {icon}
        </div>
      </div>
    </div>
  );
}

function BarChart({
  label,
  value,
  max,
  color,
}: {
  label: string;
  value: number;
  max: number;
  color: 'blue' | 'green';
}) {
  const percentage = (value / max) * 100;
  const colors = {
    blue: 'bg-blue-500',
    green: 'bg-green-500',
  };

  return (
    <div>
      <div className="flex justify-between text-sm mb-1">
        <span className="text-gray-700">{label}</span>
        <span className="font-medium text-gray-900">{value}</span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
        <div
          className={`h-2 rounded-full ${colors[color]} transition-all duration-300`}
          style={{ width: `${percentage}%` }}
        />
      </div>
    </div>
  );
}

function InsightCard({
  title,
  description,
  percentage,
  type,
}: {
  title: string;
  description: string;
  percentage: string;
  type: 'warning' | 'success';
}) {
  const colors = {
    warning: 'border-yellow-200 bg-yellow-50',
    success: 'border-green-200 bg-green-50',
  };

  return (
    <div className={`border-2 rounded-lg p-4 ${colors[type]}`}>
      <h4 className="font-semibold text-gray-900 mb-1">{title}</h4>
      <p className="text-sm text-gray-700 mb-2">{description}</p>
      <p className="text-2xl font-bold text-gray-900">{percentage}%</p>
    </div>
  );
}
