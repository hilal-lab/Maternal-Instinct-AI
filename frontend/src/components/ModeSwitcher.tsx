'use client';

import { useState } from 'react';
import { MessageCircle, BookOpen, ChevronDown } from 'lucide-react';

interface ModeSwitcherProps {
  currentMode: 'conversation' | 'learning';
  onModeChange: (mode: 'conversation' | 'learning') => void;
}

export default function ModeSwitcher({ currentMode, onModeChange }: ModeSwitcherProps) {
  const [showDropdown, setShowDropdown] = useState(false);

  const modes = [
    {
      id: 'conversation' as const,
      label: 'Conversation Mode',
      icon: MessageCircle,
      description: 'Ngobrol santai dengan Ara',
      color: 'text-blue-600',
      bgColor: 'bg-blue-50',
      borderColor: 'border-blue-200',
    },
    {
      id: 'learning' as const,
      label: 'Learning Mode',
      icon: BookOpen,
      description: 'Belajar materi dari dokumen',
      color: 'text-green-600',
      bgColor: 'bg-green-50',
      borderColor: 'border-green-200',
    },
  ];

  const activeMode = modes.find(m => m.id === currentMode) || modes[0];
  const ActiveIcon = activeMode.icon;

  return (
    <div className="relative">
      <button
        onClick={() => setShowDropdown(!showDropdown)}
        className={`flex items-center space-x-2 px-4 py-2 rounded-lg border-2 transition-all ${activeMode.borderColor} ${activeMode.bgColor}`}
      >
        <ActiveIcon className={`w-5 h-5 ${activeMode.color}`} />
        <span className={`font-medium ${activeMode.color}`}>
          {activeMode.label}
        </span>
        <ChevronDown className={`w-4 h-4 ${activeMode.color} transition-transform ${showDropdown ? 'rotate-180' : ''}`} />
      </button>

      {showDropdown && (
        <>
          <div 
            className="fixed inset-0 z-10" 
            onClick={() => setShowDropdown(false)} 
          />
          <div className="absolute top-full left-0 mt-2 w-72 bg-white rounded-xl shadow-lg border border-gray-200 z-20 overflow-hidden">
            <div className="p-2">
              <p className="text-xs font-semibold text-gray-500 uppercase px-3 py-2">
                Pilih Mode
              </p>
              {modes.map((mode) => {
                const Icon = mode.icon;
                const isActive = mode.id === currentMode;
                
                return (
                  <button
                    key={mode.id}
                    onClick={() => {
                      onModeChange(mode.id);
                      setShowDropdown(false);
                    }}
                    className={`w-full flex items-start space-x-3 p-3 rounded-lg transition-colors ${
                      isActive 
                        ? `${mode.bgColor} ${mode.borderColor} border` 
                        : 'hover:bg-gray-50'
                    }`}
                  >
                    <div className={`p-2 rounded-lg ${isActive ? mode.bgColor : 'bg-gray-100'}`}>
                      <Icon className={`w-5 h-5 ${isActive ? mode.color : 'text-gray-500'}`} />
                    </div>
                    <div className="text-left">
                      <p className={`font-medium ${isActive ? mode.color : 'text-gray-900'}`}>
                        {mode.label}
                      </p>
                      <p className="text-sm text-gray-500">
                        {mode.description}
                      </p>
                    </div>
                    {isActive && (
                      <div className="ml-auto">
                        <div className={`w-2 h-2 rounded-full ${mode.color.replace('text-', 'bg-')}`} />
                      </div>
                    )}
                  </button>
                );
              })}
            </div>
            
            <div className="border-t border-gray-100 p-3 bg-gray-50">
              <p className="text-xs text-gray-500 text-center">
                💡 Ketik &quot;belajar [topik]&quot; untuk otomatis beralih ke Learning Mode
              </p>
            </div>
          </div>
        </>
      )}
    </div>
  );
}
