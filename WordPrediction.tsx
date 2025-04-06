import React, { useState, useRef, useEffect } from "react";
import Webcam from "react-webcam";
import { FaCamera, FaStop, FaHandPaper, FaTrash } from "react-icons/fa";
import { IoRefreshOutline, IoVolumeHigh } from "react-icons/io5";
import { motion } from "framer-motion";
import { correctTextWithGemini } from '../services/gemini';

const WordPrediction: React.FC = () => {
  const webcamRef = useRef<Webcam>(null);
  const lastPredictionRef = useRef<string>("");
  const [capturing, setCapturing] = useState(false);
  const [prediction, setPrediction] = useState<string>("No sign detected");
  const [confidence, setConfidence] = useState<number>(0);
  const [duration, setDuration] = useState(2);
  const [timer, setTimer] = useState(0);
  const [predictedWords, setPredictedWords] = useState<string[]>([]);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);

  const sendFrameToBackend = async (imageSrc: string) => {
    try {
      const response = await fetch("http://localhost:8000/predict/word", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({ image: imageSrc }),
      });

      const data = await response.json();
      if (data.prediction) {
        setPrediction(data.prediction);
        setConfidence(data.confidence || 0);

        if (data.prediction !== lastPredictionRef.current) {
          speakPrediction(data.prediction);
          lastPredictionRef.current = data.prediction;
          setPredictedWords(prev => [...prev, data.prediction]);
        }
      } else {
        setPrediction("No sign detected");
        setConfidence(0);
        lastPredictionRef.current = "";
      }
    } catch (error) {
      console.error("Error sending frame to backend:", error);
      setPrediction("Error detecting sign");
      setConfidence(0);
      lastPredictionRef.current = "";
    }
  };

  const speakPrediction = (text: string) => {
    const utterance = new SpeechSynthesisUtterance(`Predicted sign is ${text}`);
    utterance.lang = "en-US";
    speechSynthesis.speak(utterance);
  };

  const speakFullSentence = async () => {
    if (predictedWords.length === 0 || isSpeaking || isProcessing) return;
    
    setIsProcessing(true);
    setIsSpeaking(true);
    setPrediction("Processing...");

    try {
      const rawText = predictedWords.join(" ");
      const correctedText = await correctTextWithGemini(rawText);
      setPrediction(correctedText);
      
      const utterance = new SpeechSynthesisUtterance(correctedText);
      utterance.onend = () => {
        setIsSpeaking(false);
        setIsProcessing(false);
      };
      speechSynthesis.speak(utterance);
      
    } catch (error) {
      console.error("Error in speak process:", error);
      setIsProcessing(false);
      setIsSpeaking(false);
      setPrediction("Error processing text");
    }
  };

  const clearPredictedWords = () => {
    setPredictedWords([]);
  };

  const handleStartRecording = () => {
    setCapturing(true);
    setTimer(duration);

    const interval = setInterval(() => {
      setTimer((prev) => {
        if (prev === 1) {
          clearInterval(interval);
          setCapturing(false);
          captureFrame();
        }
        return prev - 1;
      });
    }, 1000);
  };

  const stopRecording = () => {
    setCapturing(false);
    setTimer(0);
  };

  const captureFrame = () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        sendFrameToBackend(imageSrc);
      }
    }
  };

  const resetPrediction = () => {
    setPrediction("No sign detected");
    setConfidence(0);
    lastPredictionRef.current = "";
  };

  return (
    <div className="bg-dark-lighter/50 rounded-xl border-2 border-accent p-6 shadow-lg w-full max-w-7xl mx-auto">
      {/* Header Section */}
      <motion.div
        className="text-center mb-6"
        initial={{ opacity: 0, y: -20 }}
        animate={{ opacity: 1, y: 0 }}
      >
        <motion.div
          className="inline-flex items-center gap-2 mb-2 px-4 py-1 bg-white/5 rounded-full"
          whileHover={{ scale: 1.05 }}
        >
          <span className="text-sm text-white/80 tracking-wide font-medium">
            Real-time Sign Language Recognition
          </span>
        </motion.div>

        <motion.h1 className="text-3xl font-bold mb-2">
          <span className="text-white/90">Words Just a Sign Away</span>
        </motion.h1>

        <motion.p className="text-white/60 text-sm">
          Communicate effortlessly with HastVaani
        </motion.p>
      </motion.div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* Webcam Section */}
        <div className="space-y-4">
          <motion.div
            className="relative aspect-square w-full max-w-2xl mx-auto rounded-xl overflow-hidden bg-gradient-to-br from-black/30 to-black/10 border border-white/10"
            whileHover={{ borderColor: "rgba(255,255,255,0.2)" }}
          >
            <Webcam
              audio={false}
              ref={webcamRef}
              screenshotFormat="image/jpeg"
              className="w-full h-full object-cover"
              style={{ transform: "scaleX(-1)" }}
              videoConstraints={{
                width: 640,
                height: 640,
                facingMode: "user"
              }}
            />
            {capturing && (
              <div className="absolute inset-0 flex items-center justify-center bg-black/20">
                <div className="absolute top-4 right-4 bg-black/60 text-white px-3 py-1.5 rounded-full">
                  Recording... {timer}s
                </div>
              </div>
            )}
          </motion.div>

          <div className="flex justify-center gap-3">
            {!capturing ? (
              <button
                onClick={handleStartRecording}
                className="btn-primary px-6 py-3 rounded-lg shadow-md hover:scale-105 transition-transform duration-300 flex items-center text-sm"
              >
                <FaCamera className="mr-2" /> Start Recording
              </button>
            ) : (
              <button
                onClick={stopRecording}
                className="bg-red-500 text-white px-6 py-3 rounded-lg flex items-center text-sm"
              >
                <FaStop className="mr-2" /> Stop Recording
              </button>
            )}
            <button
              onClick={resetPrediction}
              className="bg-gray-700 text-white px-6 py-3 rounded-lg flex items-center text-sm"
              disabled={capturing}
            >
              <IoRefreshOutline className="mr-2" /> Reset
            </button>
          </div>
        </div>

        {/* Results Section */}
        <div className="space-y-4">
          <div className="bg-white/[0.02] border border-white/10 rounded-xl p-4 h-full flex flex-col justify-between">
            <h3 className="text-xl font-bold text-center mb-4">Recognition Results</h3>

            <div className="flex-grow flex flex-col items-center justify-center space-y-6">
              <div className="flex justify-center">
                <button
                  className="bg-gray-800 text-white p-4 rounded-full shadow-lg border-2 border-accent hover:scale-110 transition-transform duration-300 relative"
                  style={{
                    boxShadow: "0 0 15px rgba(255, 255, 255, 0.5), 0 0 30px rgba(255, 255, 255, 0.3)",
                  }}
                >
                  <IoVolumeHigh className="text-3xl" />
                  <div
                    className="absolute inset-0 rounded-full border-2 border-accent animate-pulse"
                    style={{
                      boxShadow: "0 0 20px rgba(255, 255, 255, 0.5)",
                    }}
                  ></div>
                </button>
              </div>

              {prediction === "No sign detected" ? (
                <div className="flex flex-col items-center justify-center space-y-3">
                  <FaHandPaper className="text-4xl text-white/40" />
                  <p className="text-white/60 text-lg">Make a sign to begin</p>
                </div>
              ) : (
                <div className="w-full space-y-4 flex flex-col items-center">
                  <h4 className="text-2xl font-bold text-white text-center neon-text">
                    {prediction}
                  </h4>
                  <div className="w-3/4 bg-white/10 rounded-full h-2 relative">
                    <div
                      className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 rounded-full"
                      style={{ width: `${confidence}%` }}
                    />
                    <div
                      className="absolute top-0 right-0 h-full bg-white/10 rounded-full"
                      style={{ width: `${100 - confidence}%` }}
                    />
                  </div>
                  <p className="text-center text-white/60">
                    Confidence: {confidence.toFixed(2)}%
                  </p>
                </div>
              )}
            </div>
          </div>
        </div>
      </div>

      {/* Predicted Words Collection Section */}
      <div className="mt-6 bg-white/[0.02] border border-white/10 rounded-xl p-4 w-full">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 mb-4">
          <h3 className="text-xl font-bold">Predicted Sentence</h3>
          <div className="flex gap-2">
            <button
              onClick={speakFullSentence}
              disabled={predictedWords.length === 0 || isProcessing}
              className={`px-4 py-2 rounded-lg flex items-center ${
                predictedWords.length === 0 || isProcessing
                  ? 'bg-gray-700 text-gray-400'
                  : 'bg-accent text-white hover:bg-accent/80'
              }`}
            >
              <IoVolumeHigh className="mr-2" />
              Speak Sentence
            </button>
            <button
              onClick={clearPredictedWords}
              disabled={predictedWords.length === 0}
              className={`px-4 py-2 rounded-lg flex items-center ${
                predictedWords.length === 0
                  ? 'bg-gray-700 text-gray-400'
                  : 'bg-red-500 text-white hover:bg-red-600'
              }`}
            >
              <FaTrash className="mr-2" />
              Clear
            </button>
          </div>
        </div>
        
        <div className="bg-black/20 border border-white/10 rounded-lg p-4 min-h-16 w-full">
          {predictedWords.length > 0 ? (
            <p className="text-white text-lg">
              {predictedWords.join(" ")}
            </p>
          ) : (
            <p className="text-white/40 italic">Your predicted words will appear here as you make signs</p>
          )}
        </div>
        
        {isProcessing && (
          <div className="text-center text-accent mt-2 text-sm">
            Processing...
          </div>
        )}
      </div>
    </div>
  );
};

export default WordPrediction;