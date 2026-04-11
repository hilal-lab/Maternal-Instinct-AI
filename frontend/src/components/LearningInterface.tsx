'use client';

import { useState, useEffect, useRef } from 'react';
import { BookOpen, CheckCircle, ArrowRight, RotateCcw, Loader2 } from 'lucide-react';
import { learningAPI } from '@/lib/api';
import ReactMarkdown from 'react-markdown';

interface LessonSection {
  type: string;
  content: string;
  duration_mins: number;
}

interface QuizQuestion {
  id: number;
  type: string;
  question: string;
  options: string[];
  correct_answer?: string;
  explanation?: string;
}

interface LearningState {
  status: 'idle' | 'loading' | 'lesson' | 'quiz' | 'result' | 'error';
  topic: string;
  level: string;
  sessionId?: number;
  sections?: LessonSection[];
  currentSection?: number;
  questions?: QuizQuestion[];
  answers: Record<number, string>;
  quizResult?: any;
  message?: string;
}

const INITIAL_STATE: LearningState = {
  status: 'idle',
  topic: '',
  level: 'pemula',
  answers: {},
};

export default function LearningInterface() {
  const [state, setState] = useState<LearningState>(INITIAL_STATE);
  const [topicInput, setTopicInput] = useState('');
  const [level, setLevel] = useState('pemula');
  const [loading, setLoading] = useState(false);
  const contentRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (contentRef.current) {
      contentRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [state.currentSection, state.status]);

  const startLesson = async () => {
    if (!topicInput.trim()) return;
    
    setLoading(true);
    setState({
      ...INITIAL_STATE,
      status: 'loading',
      topic: topicInput.trim(),
      level,
    });

    try {
      const result = await learningAPI.startLesson(topicInput.trim(), level);
      
      if (result.type === 'lesson') {
        setState({
          status: 'lesson',
          topic: topicInput.trim(),
          level,
          sessionId: result.session.id,
          sections: result.sections,
          currentSection: 0,
          answers: {},
          message: result.message,
        });
      } else if (result.type === 'quiz') {
        setState({
          status: 'quiz',
          topic: topicInput.trim(),
          level,
          sessionId: result.session.id,
          questions: result.questions,
          answers: {},
          message: result.message,
        });
      }
    } catch (error) {
      console.error('Failed to start lesson:', error);
      setState({
        ...INITIAL_STATE,
        status: 'error',
        message: 'Gagal memulai pembelajaran. Coba lagi.',
      });
    } finally {
      setLoading(false);
    }
  };

  const continueLesson = () => {
    if (!state.sessionId || !state.sections) return;
    
    const nextSection = (state.currentSection || 0) + 1;
    
    if (nextSection >= state.sections.length) {
      // Lesson complete, start quiz
      startQuiz();
    } else {
      setState(prev => ({
        ...prev,
        currentSection: nextSection,
      }));
    }
  };

  const startQuiz = async () => {
    if (!state.sessionId) return;
    
    setLoading(true);
    try {
      const result = await learningAPI.continueLesson(state.sessionId);
      
      if (result.type === 'quiz' || result.questions) {
        setState(prev => ({
          ...prev,
          status: 'quiz',
          questions: result.questions || result.data?.questions,
          answers: {},
        }));
      } else {
        // No quiz available, show completion
        setState(prev => ({
          ...prev,
          status: 'result',
          message: 'Belajar selesai! 📚',
          quizResult: {
            score: 100,
            recommendations: ['Materi sudah selesai dipelajari!'],
          },
        }));
      }
    } catch (error) {
      console.error('Failed to get quiz:', error);
      setState(prev => ({
        ...prev,
        status: 'result',
        message: 'Belajar selesai! 📚',
        quizResult: {
          score: 100,
          recommendations: ['Materi sudah selesai dipelajari!'],
        },
      }));
    } finally {
      setLoading(false);
    }
  };

  const submitAnswer = (questionId: number, answer: string) => {
    setState(prev => ({
      ...prev,
      answers: { ...prev.answers, [questionId]: answer },
    }));
  };

  const submitQuiz = async () => {
    if (!state.sessionId) return;
    
    setLoading(true);
    try {
      const result = await learningAPI.submitQuiz(state.sessionId, state.answers);
      
      setState(prev => ({
        ...prev,
        status: 'result',
        quizResult: result,
      }));
    } catch (error) {
      console.error('Failed to submit quiz:', error);
      setState(prev => ({
        ...prev,
        status: 'error',
        message: 'Gagal submit kuis. Coba lagi.',
      }));
    } finally {
      setLoading(false);
    }
  };

  const reset = () => {
    setState(INITIAL_STATE);
    setTopicInput('');
    setLevel('pemula');
  };

  const renderIdle = () => (
    <div className="max-w-2xl mx-auto text-center py-12">
      <div className="mb-8">
        <div className="w-20 h-20 bg-green-100 rounded-full flex items-center justify-center mx-auto mb-4">
          <BookOpen className="w-10 h-10 text-green-600" />
        </div>
        <h2 className="text-2xl font-bold text-gray-900 mb-2">
          Learning Mode
        </h2>
        <p className="text-gray-600">
          Pilih topik yang ingin kamu pelajari dari dokumen yang sudah di-upload.
        </p>
      </div>

      <div className="space-y-4">
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 text-left">
            Topik Pembelajaran
          </label>
          <input
            type="text"
            value={topicInput}
            onChange={(e) => setTopicInput(e.target.value)}
            placeholder="Contoh: Machine Learning, Photosynthesis, dll..."
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
            onKeyDown={(e) => e.key === 'Enter' && startLesson()}
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2 text-left">
            Level Kamu
          </label>
          <div className="flex space-x-3 justify-center">
            {['pemula', 'menengah', 'mahir'].map((lvl) => (
              <button
                key={lvl}
                onClick={() => setLevel(lvl)}
                className={`px-4 py-2 rounded-lg border-2 transition-all ${
                  level === lvl
                    ? 'border-green-500 bg-green-50 text-green-700'
                    : 'border-gray-200 text-gray-600 hover:border-gray-300'
                }`}
              >
                {lvl.charAt(0).toUpperCase() + lvl.slice(1)}
              </button>
            ))}
          </div>
        </div>

        <button
          onClick={startLesson}
          disabled={!topicInput.trim() || loading}
          className="w-full btn-primary py-3 disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center space-x-2"
        >
          {loading ? (
            <>
              <Loader2 className="w-5 h-5 animate-spin" />
              <span>Memulai...</span>
            </>
          ) : (
            <>
              <BookOpen className="w-5 h-5" />
              <span>Mulai Belajar</span>
            </>
          )}
        </button>
      </div>

      {state.message && (
        <p className="mt-4 text-sm text-gray-600 bg-gray-50 p-3 rounded-lg">
          {state.message}
        </p>
      )}
    </div>
  );

  const renderLesson = () => {
    if (!state.sections || state.currentSection === undefined) return null;
    
    const currentSection = state.sections[state.currentSection];
    const isLast = state.currentSection === state.sections.length - 1;
    const progress = ((state.currentSection + 1) / state.sections.length) * 100;

    return (
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <div className="flex justify-between items-center mb-2">
            <span className="text-sm font-medium text-green-600">
              {state.topic}
            </span>
            <span className="text-sm text-gray-500">
              Bagian {state.currentSection + 1} dari {state.sections.length}
            </span>
          </div>
          <div className="w-full bg-gray-200 rounded-full h-2">
            <div 
              className="bg-green-500 h-2 rounded-full transition-all duration-300"
              style={{ width: `${progress}%` }}
            />
          </div>
        </div>

        <div 
          ref={contentRef}
          className="bg-white border border-gray-200 rounded-xl p-6 mb-6 shadow-sm"
        >
          <div className="prose max-w-none">
            <ReactMarkdown>{currentSection.content}</ReactMarkdown>
          </div>
        </div>

        <div className="flex justify-between items-center">
          <button
            onClick={reset}
            className="btn-secondary text-gray-600"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Mulai Ulang
          </button>

          <button
            onClick={continueLesson}
            disabled={loading}
            className="btn-primary flex items-center space-x-2"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              <>
                <span>{isLast ? 'Mulai Kuis' : 'Lanjut'}</span>
                <ArrowRight className="w-5 h-5" />
              </>
            )}
          </button>
        </div>
      </div>
    );
  };

  const renderQuiz = () => {
    if (!state.questions) return null;

    const allAnswered = state.questions.every(q => state.answers[q.id]);

    return (
      <div className="max-w-3xl mx-auto">
        <div className="mb-6">
          <h2 className="text-xl font-bold text-gray-900 mb-2">
            📝 Kuis: {state.topic}
          </h2>
          <p className="text-gray-600">
            Jawab pertanyaan berikut untuk menguji pemahamanmu.
          </p>
        </div>

        <div className="space-y-6">
          {state.questions.map((q, idx) => (
            <div key={q.id} className="bg-white border border-gray-200 rounded-xl p-6 shadow-sm">
              <div className="flex space-x-2 mb-3">
                <span className="bg-green-100 text-green-700 text-sm font-medium px-2 py-1 rounded">
                  {idx + 1}
                </span>
                <span className="bg-gray-100 text-gray-600 text-sm px-2 py-1 rounded">
                  {q.type === 'multiple_choice' ? 'Pilihan Ganda' : 
                   q.type === 'true_false' ? 'Benar/Salah' : 'Jawaban Singkat'}
                </span>
              </div>

              <p className="text-lg font-medium text-gray-900 mb-4">
                {q.question}
              </p>

              {q.type === 'multiple_choice' && q.options && (
                <div className="space-y-2">
                  {q.options.map((option, optIdx) => (
                    <label 
                      key={optIdx}
                      className={`flex items-center p-3 rounded-lg border-2 cursor-pointer transition-all ${
                        state.answers[q.id] === option
                          ? 'border-green-500 bg-green-50'
                          : 'border-gray-200 hover:border-gray-300'
                      }`}
                    >
                      <input
                        type="radio"
                        name={`question-${q.id}`}
                        value={option}
                        checked={state.answers[q.id] === option}
                        onChange={() => submitAnswer(q.id, option)}
                        className="sr-only"
                      />
                      <span className="text-gray-700">{option}</span>
                    </label>
                  ))}
                </div>
              )}

              {(q.type === 'true_false' || q.type === 'short_answer') && (
                <input
                  type="text"
                  value={state.answers[q.id] || ''}
                  onChange={(e) => submitAnswer(q.id, e.target.value)}
                  placeholder="Ketik jawabanmu..."
                  className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-green-500 focus:border-green-500"
                />
              )}
            </div>
          ))}
        </div>

        <div className="mt-6 flex justify-between items-center">
          <button
            onClick={reset}
            className="btn-secondary text-gray-600"
          >
            <RotateCcw className="w-4 h-4 mr-2" />
            Mulai Ulang
          </button>

          <button
            onClick={submitQuiz}
            disabled={!allAnswered || loading}
            className="btn-primary disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {loading ? (
              <Loader2 className="w-5 h-5 animate-spin" />
            ) : (
              'Submit Jawaban'
            )}
          </button>
        </div>
      </div>
    );
  };

  const renderResult = () => {
    if (!state.quizResult) return null;

    const scoreColor = state.quizResult.score >= 70 ? 'text-green-600' : 
                       state.quizResult.score >= 50 ? 'text-yellow-600' : 'text-red-600';

    return (
      <div className="max-w-2xl mx-auto text-center">
        <div className="bg-white border border-gray-200 rounded-xl p-8 shadow-sm">
          <div className={`text-6xl font-bold mb-4 ${scoreColor}`}>
            {Math.round(state.quizResult.score)}%
          </div>
          
          <p className="text-xl text-gray-700 mb-2">
            {state.quizResult.score >= 70 ? '🎉 Luar biasa!' :
             state.quizResult.score >= 50 ? '👍 Bagus! Terus belajar!' :
             '💪 Semangat! Coba lagi!'}
          </p>
          
          <p className="text-gray-600 mb-6">
            {state.quizResult.correct_count} dari {state.quizResult.total_questions} jawaban benar
          </p>

          {state.quizResult.recommendations && (
            <div className="text-left bg-gray-50 rounded-lg p-4 mb-6">
              <p className="font-medium text-gray-700 mb-2">💡 Saran:</p>
              <ul className="list-disc list-inside text-gray-600 space-y-1">
                {state.quizResult.recommendations.map((rec: string, idx: number) => (
                  <li key={idx}>{rec}</li>
                ))}
              </ul>
            </div>
          )}

          <div className="flex justify-center space-x-4">
            <button
              onClick={reset}
              className="btn-secondary"
            >
              <RotateCcw className="w-4 h-4 mr-2" />
              Belajar Topik Lain
            </button>
          </div>
        </div>
      </div>
    );
  };

  const renderLoading = () => (
    <div className="flex flex-col items-center justify-center py-12">
      <Loader2 className="w-12 h-12 text-green-600 animate-spin mb-4" />
      <p className="text-gray-600">Memuat pembelajaran...</p>
    </div>
  );

  const renderError = () => (
    <div className="max-w-2xl mx-auto text-center py-12">
      <div className="bg-red-50 border border-red-200 rounded-xl p-6">
        <p className="text-red-600 mb-4">{state.message}</p>
        <button onClick={reset} className="btn-primary">
          Coba Lagi
        </button>
      </div>
    </div>
  );

  return (
    <div className="py-6">
      {state.status === 'idle' && renderIdle()}
      {state.status === 'loading' && renderLoading()}
      {state.status === 'lesson' && renderLesson()}
      {state.status === 'quiz' && renderQuiz()}
      {state.status === 'result' && renderResult()}
      {state.status === 'error' && renderError()}
    </div>
  );
}
