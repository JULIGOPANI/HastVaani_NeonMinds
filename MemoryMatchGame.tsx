import React, { useState, useEffect, useCallback } from 'react';
import { motion, AnimatePresence } from 'framer-motion';
import { Card } from '../types/game';
import { Trophy, Timer as TimerIcon, Move } from 'lucide-react';
import confetti from 'canvas-confetti';

interface MemoryMatchGameProps {
  category: 'numbers' | 'alphabets';
  difficulty: 'easy' | 'medium' | 'hard';
  onGameComplete: (moves: number, time: number) => void;
  onRestart: () => void;
}

export const MemoryMatchGame: React.FC<MemoryMatchGameProps> = ({
  category,
  difficulty,
  onGameComplete,
  onRestart,
}) => {
  const [cards, setCards] = useState<Card[]>([]);
  const [flippedIndexes, setFlippedIndexes] = useState<number[]>([]);
  const [moves, setMoves] = useState(0);
  const [timer, setTimer] = useState(0);
  const [isGameActive, setIsGameActive] = useState(true);
  const [isChecking, setIsChecking] = useState(false);
  const [matchedPairs, setMatchedPairs] = useState(0);
  const [gameData, setGameData] = useState<any>(null);
  const [showCelebration, setShowCelebration] = useState(false);
  // Removed unused gameStats state

  // Load game data
  useEffect(() => {
    fetch('/data/game.json')
      .then((response) => response.json())
      .then((data) => {
        setGameData(data);
      })
      .catch((error) => console.error('Error loading game data:', error));
  }, []);

  // Initialize game based on difficulty
  useEffect(() => {
    if (!gameData) return;

    const pairsCount = 
      difficulty === 'easy' ? 2 : 
      difficulty === 'medium' ? 8 : 
      18; // 6x6 grid needs 18 pairs (36 cards)
    
    const pairs = generatePairs(pairsCount, category);
    const shuffledCards = shuffleCards([...pairs, ...pairs]);
    setCards(shuffledCards);
    setTimer(0);
    setMoves(0);
    setMatchedPairs(0);
    setIsGameActive(true);
    setFlippedIndexes([]);
  }, [difficulty, category, gameData]);

  // Timer effect
  useEffect(() => {
    let interval: ReturnType<typeof setInterval>;
    if (isGameActive) {
      interval = setInterval(() => setTimer(prev => prev + 1), 1000);
    }
    return () => clearInterval(interval);
  }, [isGameActive]);

// Update the generatePairs function in MemoryMatchGame.tsx
const generatePairs = (count: number, category: 'numbers' | 'alphabets'): Card[] => {
  const categoryData = gameData[category].slice(0, count);
  return categoryData.map((item: Card) => ({
    ...item,
    isMatched: false,
    isFlipped: false
  }));
};

// Define the shuffleCards function
const shuffleCards = (cards: Card[]): Card[] => {
  return cards
    .map((card) => ({ ...card, sort: Math.random() }))
    .sort((a, b) => a.sort - b.sort)
    .map(({ sort, ...card }) => card);
};

// Update the initialization effect to handle 25 cards for hard difficulty
useEffect(() => {
  if (!gameData) return;

  const pairsCount = 
    difficulty === 'easy' ? 2 : // 2 pairs (4 cards) - 2x2 grid
    difficulty === 'medium' ? 8 : // 8 pairs (16 cards) - 4x4 grid
    12; // 12 pairs (24 cards) + 1 extra card (25 total) - 5x5 grid
  
  const pairs = generatePairs(pairsCount, category);
  
  // For hard difficulty (5x5 grid with 25 cards)
  let shuffledCards: Card[];
  if (difficulty === 'hard') {
    // Create pairs (24 cards) and add one extra card (25th card)
    const pairedCards = [...pairs, ...pairs];
    const extraCard = { ...pairs[0], id: pairs[0].id + 0.1 }; // Duplicate first card with a unique numeric id
    shuffledCards = shuffleCards([...pairedCards, extraCard]);
  } else {
    // Normal pairing for easy and medium
    shuffledCards = shuffleCards([...pairs, ...pairs]);
  }
  
  setCards(shuffledCards);
  setTimer(0);
  setMoves(0);
  setMatchedPairs(0);
  setIsGameActive(true);
  setFlippedIndexes([]);
}, [difficulty, category, gameData]);

// Update the getGridClass function
const getGridClass = () => {
  switch (difficulty) {
    case 'easy':
      return 'grid-cols-2 gap-4 justify-items-center'; // Reduced gap from 6 to 4
    case 'medium':
      return 'grid-cols-4 gap-3 justify-items-center'; // Reduced gap from 4 to 3
    case 'hard':
      return 'grid-cols-5 gap-2 justify-items-center'; // Reduced gap from 4 to 2
    default:
      return 'grid-cols-2 gap-4 justify-items-center';
  }
};

// Update the getContainerSize function
const getContainerSize = () => {
  switch (difficulty) {
    case 'easy':
      return 'max-w-sm mx-auto'; // Reduced from max-w-md
    case 'medium':
      return 'max-w-md mx-auto'; // Reduced from max-w-lg
    case 'hard':
      return 'max-w-2xl mx-auto px-4'; // Reduced from max-w-4xl
    default:
      return 'max-w-sm mx-auto';
  }
};

// Update the card size for Level 3
const getCardSizeClass = () => {
  switch (difficulty) {
    case 'easy':
      return 'w-32 h-32';
    case 'medium':
      return 'w-16 h-16 sm:w-20 sm:h-20';
    case 'hard':
      return 'w-24 h-24 sm:w-28 sm:h-28';
    default:
      return 'w-32 h-32';
  }
};

const handleCardClick = (index: number) => {
  if (isChecking || flippedIndexes.includes(index) || cards[index].isMatched) {
    return;
  }

  const newFlippedIndexes = [...flippedIndexes, index];
  setFlippedIndexes(newFlippedIndexes);

  if (newFlippedIndexes.length === 2) {
    setIsChecking(true);
    setTimeout(() => {
      const [firstIndex, secondIndex] = newFlippedIndexes;
      if (cards[firstIndex].id === cards[secondIndex].id) {
        const updatedCards = cards.map((card, i) =>
          i === firstIndex || i === secondIndex
            ? { ...card, isMatched: true }
            : card
        );
        setCards(updatedCards);
        setMatchedPairs((prev) => prev + 1);

        // Check if all cards are matched
        if (updatedCards.every((card) => card.isMatched)) {
          setIsGameActive(false);
          setShowCelebration(true); // Show the celebration modal
          onGameComplete(moves, timer); // Notify parent component about game completion

          // Trigger faster confetti celebration
          confetti({
            particleCount: 150,
            spread: 90,
            startVelocity: 80,
            decay: 0.7,
            origin: { y: 0.6 },
            colors: ['#00f2ff', '#ff00ff', '#ffff00', '#00ffff'],
            shapes: ['circle', 'square']
          });
        }
      }
      setFlippedIndexes([]);
      setIsChecking(false);
    }, 1000);
  }

  setMoves((prev) => prev + 1);
};

const handleGameComplete = useCallback((time: number) => {
  if (!showCelebration) {
    // Removed setGameStats call as gameStats is no longer used
    setShowCelebration(true);
  }
}, [showCelebration]);

  return (
    <div className="flex flex-col items-center justify-center min-h-screen pt-16 pb-4 px-2 space-y-2"> {/* Added pt-16 for top padding */}
      <div className={`holographic-panel p-3 rounded-xl w-full ${getContainerSize()} mt-8`}> {/* Added mt-8 for top margin */}
        <div className="text-center mb-4">
          <h1 className="text-2xl neon-text mb-2">
            {category === 'numbers' ? 'Match the Numbers!' : 'Match the Alphabets!'}
          </h1>
          
          <div className="flex justify-center gap-4 mb-4">
            <div className="flex items-center gap-1">
              <Move className="w-4 h-4 text-cyan-400" />
              <p className="text-sm neon-text">{moves}</p>
            </div>
            <div className="flex items-center gap-1">
              <TimerIcon className="w-4 h-4 text-purple-400" />
              <p className="text-sm neon-text">{timer}s</p>
            </div>
            <div className="flex items-center gap-1">
              <Trophy className="w-4 h-4 text-yellow-400" />
              <p className="text-sm neon-text">{matchedPairs}/{cards.length / 2}</p>
            </div>
          </div>

          <button 
            onClick={onRestart}
            className="btn-holographic px-3 py-1 text-xs"
          >
            Restart
          </button>
        </div>

        {/* Game Grid */}
        <div className={`grid ${getGridClass()} mx-auto`}>
          <AnimatePresence>
            {cards.map((card, index) => (
              <motion.div
                key={`${card.id}-${index}`}
                className={`${getCardSizeClass()} flex-shrink-0`}
                initial={{ scale: 0 }}
                animate={{ scale: 1 }}
                exit={{ scale: 0 }}
                whileHover={{ scale: 1.05 }}
                whileTap={{ scale: 0.95 }}
              >
                <div
                  className="relative w-full h-full cursor-pointer"
                  onClick={() => handleCardClick(index)}
                >
                  <motion.div
                    className="absolute w-full h-full"
                    initial={false}
                    animate={{
                      rotateY: flippedIndexes.includes(index) || card.isMatched ? 180 : 0
                    }}
                    transition={{ duration: 0.6 }}
                  >
                    {/* Card Back */}
                    <div 
                      className={`absolute w-full h-full backface-hidden rounded ${
                        flippedIndexes.includes(index) || card.isMatched
                          ? 'opacity-0'
                          : 'opacity-100'
                      }`}
                    >
                      <div className="w-full h-full holographic-panel flex items-center justify-center">
                        <span
                          className={`${
                            difficulty === 'easy'
                              ? 'text-4xl sm:text-5xl' // Larger size for easy level
                              : difficulty === 'medium'
                              ? 'text-3xl sm:text-4xl' // Slightly smaller size for medium level
                              : 'text-4xl sm:text-5xl' // Default size for hard level
                          } font-medium text-gray-300`} // Simpler styling for the question mark
                        >
                          ?
                        </span>
                      </div>
                    </div>

                    {/* Card Front */}
                    <div 
                      className={`absolute w-full h-full backface-hidden transform rotate-y-180 rounded ${
                        flippedIndexes.includes(index) || card.isMatched
                          ? 'opacity-100'
                          : 'opacity-0'
                      }`}
                    >
                      <img
                        src={card.imageUrl}
                        alt={`${category} Sign ${card.id}`}
                        className="w-full h-full object-cover rounded"
                      />
                    </div>
                  </motion.div>
                </div>
              </motion.div>
            ))}
          </AnimatePresence>
        </div>
      </div>

      {/* Celebration Modal */}
      <AnimatePresence>
        {showCelebration && (
          <motion.div
            key="celebration-modal"
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm"
          >
            <motion.div
              initial={{ scale: 0.8 }}
              animate={{ scale: 1 }}
              exit={{ scale: 0.8 }}
              className="relative holographic-panel rounded-2xl p-8 max-w-md w-full mx-4 border border-white/20"
            >
              <div className="absolute -top-12 left-1/2 transform -translate-x-1/2">
                <Trophy className="w-24 h-24 text-yellow-400 fill-yellow-400 drop-shadow-lg" />
              </div>
              
              <div className="text-center mt-8">
                <h2 className="text-3xl font-bold text-white mb-2 neon-text">
                  🎉 Level Complete!
                </h2>
                <p className="text-white/80 mb-6 neon-text">
                  You've mastered this challenge!
                </p>
                
                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="holographic-panel p-4 rounded-lg border border-white/10">
                    <p className="text-sm text-white/70">Moves</p>
                    <p className="text-2xl font-bold text-white neon-text">{moves}</p>
                  </div>
                  <div className="holographic-panel p-4 rounded-lg border border-white/10">
                    <p className="text-sm text-white/70">Time</p>
                    <p className="text-2xl font-bold text-white neon-text">{timer}s</p>
                  </div>
                </div>
                
                <button
                  onClick={() => {
                    confetti.reset(); // Stop the confetti animation
                    setShowCelebration(false); // Hide the celebration modal
                    onRestart(); // Call the parent restart handler
                  }}
                  className="btn-holographic w-full py-3 text-lg flex items-center justify-center gap-2 hover:bg-white/20"
                >
                  <Trophy className="w-5 h-5" />
                  Play Again
                </button>
              </div>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};