// API Response Types
export interface ChatMessage {
  id?: number;
  role: 'user' | 'assistant';
  content: string;
  emotion?: string;
  intent?: string;
  intensity?: number;
  layer3_status?: string;
  layer4_rewritten?: boolean;
  created_at?: string;
}

export interface LayerOneData {
  intent: string;
  emotion: string;
  intensity: number;
  flags: Record<string, any>;
}

export interface LayerTwoData {
  agent_used: string;
  raw_response: string;
}

export interface LayerThreeData {
  status: string;
  note: string | null;
  policy_recommendations: string[];
}

export interface LayerFourData {
  is_rewritten: boolean;
  original: string;
  final: string;
}

export interface LayersData {
  layer1: LayerOneData;
  layer2: LayerTwoData;
  layer3: LayerThreeData;
  layer4: LayerFourData;
}

export interface ChatResponse {
  response: string;
  layers: LayersData;
}

export interface ScheduleTask {
  id?: number;
  task: string;
  deadline: string;
  est_hours: number;
  priority: 'LOW' | 'MEDIUM' | 'HIGH' | 'URGENT';
  status: 'PENDING' | 'IN_PROGRESS' | 'COMPLETED';
  created_at?: string;
}

export interface Analytics {
  total_chats: number;
  avg_intensity: number;
  emotion_distribution: Record<string, number>;
  intent_distribution: Record<string, number>;
  layer3_violations: number;
  layer4_rewrites: number;
}
