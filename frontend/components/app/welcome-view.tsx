import { Button } from '@/components/livekit/button';
import { useState } from 'react';

function ImprovBattleIcon() {
  return (
    <svg
      width="70"
      height="70"
      viewBox="0 0 80 80"
      fill="none"
      xmlns="http://www.w3.org/2000/svg"
      className="mb-6 opacity-90"
    >
      <circle cx="28" cy="35" r="18" fill="#8B5CF6" />
      <circle cx="52" cy="35" r="18" fill="#EC4899" />
      <path d="M22 38 Q28 42 34 38" stroke="white" strokeWidth="2" fill="none" />
      <circle cx="24" cy="32" r="2" fill="white" />
      <circle cx="32" cy="32" r="2" fill="white" />
      <path d="M46 42 Q52 38 58 42" stroke="white" strokeWidth="2" fill="none" />
      <circle cx="48" cy="32" r="2" fill="white" />
      <circle cx="56" cy="32" r="2" fill="white" />
      <rect x="36" y="50" width="8" height="20" rx="4" fill="#8B5CF6" />
    </svg>
  );
}

interface WelcomeViewProps {
  startButtonText?: string;
  onStartCall: (playerName: string) => void;
}

export const WelcomeView = ({
  startButtonText = "Start Improv Battle",
  onStartCall,
  ref,
}: React.ComponentProps<'div'> & WelcomeViewProps) => {
  const [playerName, setPlayerName] = useState('');
  const [error, setError] = useState('');

  const handleStart = () => {
    if (!playerName.trim()) {
      setError('Please enter your name');
      return;
    }
    setError('');
    onStartCall(playerName.trim());
  };

  return (
    <div ref={ref} className="min-h-screen bg-gradient-to-b from-gray-900 to-gray-800 flex flex-col items-center justify-center px-4">
      {/* Card */}
      <div className="bg-gray-800/60 backdrop-blur-sm border border-gray-700 rounded-2xl p-10 max-w-lg w-full shadow-xl text-center">

        <ImprovBattleIcon />

        <h1 className="text-4xl font-extrabold text-white mb-3">
          Improv Battle
        </h1>

        <p className="text-gray-300 mb-2 text-base">
          A simple and fun 3-round improvisation challenge.
        </p>

        <p className="text-gray-400 mb-8 text-sm">
          Enter your stage name to begin.
        </p>

        {/* Input */}
        <input
          type="text"
          value={playerName}
          onChange={(e) => {
            setPlayerName(e.target.value);
            setError('');
          }}
          placeholder="Your stage name..."
          className="w-full px-4 py-3 rounded-xl bg-gray-900 border border-gray-700 text-white placeholder-gray-500 focus:outline-none focus:border-purple-500"
        />

        {error && (
          <p className="text-red-400 text-xs mt-2">
            {error}
          </p>
        )}

        {/* Button */}
        <Button
          variant="primary"
          size="lg"
          onClick={handleStart}
          disabled={!playerName.trim()}
          className="w-full mt-6 rounded-xl bg-purple-600 hover:bg-purple-500 transition font-semibold py-3 disabled:opacity-50"
        >
          🎭 {startButtonText}
        </Button>

        {/* Features */}
        <div className="grid grid-cols-3 gap-4 mt-10 text-sm">
          <div className="bg-gray-900 p-4 rounded-xl border border-gray-700">
            <div className="text-2xl">🎬</div>
            <p className="text-white mt-1 font-medium">3 Rounds</p>
          </div>
          <div className="bg-gray-900 p-4 rounded-xl border border-gray-700">
            <div className="text-2xl">🤖</div>
            <p className="text-white mt-1 font-medium">AI Host</p>
          </div>
          <div className="bg-gray-900 p-4 rounded-xl border border-gray-700">
            <div className="text-2xl">⭐</div>
            <p className="text-white mt-1 font-medium">Live Score</p>
          </div>
        </div>
      </div>

      {/* Footer */}
      <footer className="mt-6 text-gray-500 text-xs">
        Powered by Murf Falcon TTS • Voice Agent Challenge
      </footer>
    </div>
  );
};
