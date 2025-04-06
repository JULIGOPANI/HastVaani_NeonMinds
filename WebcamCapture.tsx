import React, { useState, useRef, useEffect } from 'react';
import Webcam from 'react-webcam';
import { FaCamera, FaStop, FaHandPaper, FaTrash } from 'react-icons/fa';
import { IoRefreshOutline, IoVolumeHigh } from 'react-icons/io5';
import { motion } from 'framer-motion';
import { correctTextWithGemini } from '../services/gemini';
import { Camera, CameraOff, RefreshCw, Clock, Play, X, Volume2 } from 'lucide-react';

interface WebcamCaptureProps {
  mode: 'word' | 'alphabet';
}

const WebcamCapture: React.FC<WebcamCaptureProps> = ({ mode }) => {
  const webcamRef = useRef<Webcam>(null);
  const [isEnabled, setIsEnabled] = useState(false);
  const [capturedImage, setCapturedImage] = useState<string | null>(null);
  const [result, setResult] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);
  const [detectedSign, setDetectedSign] = useState<string | null>(null);
  const [previousResult, setPreviousResult] = useState<string | null>('No hand detected');
  const [detectedSignsHistory, setDetectedSignsHistory] = useState<string[]>([]);
  const [processedSigns, setProcessedSigns] = useState<string[]>([]);
  const [isLive, setIsLive] = useState(false);
  const [isLiveLetter, setIsLiveLetter] = useState(false);
  const [intervalId, setIntervalId] = useState<NodeJS.Timeout | null>(null);
  const [isSpeaking, setIsSpeaking] = useState(false);
  const [confidence, setConfidence] = useState<number>(0);
  const [countdown, setCountdown] = useState<number | null>(null);
  const [captureTimeoutId, setCaptureTimeoutId] = useState<NodeJS.Timeout | null>(null);
  const [liveTimer, setLiveTimer] = useState(5);

  const predictSign = async (imageSrc: string) => {
    try {
      const endpoint = mode === 'alphabet' 
        ? 'http://localhost:8000/predict/alphanum' 
        : 'http://localhost:8000/predict/word';
      
      const response = await fetch(endpoint, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ image: imageSrc }),
      });

      if (!response.ok) {
        throw new Error('Prediction failed');
      }

      const data = await response.json();
      setConfidence(data.confidence || 0);
      return data.prediction || 'No prediction returned';
    } catch (error) {
      console.error('Error predicting sign:', error);
      return 'Error detecting sign';
    }
  };

  const toggleWebcam = () => {
    setIsEnabled(!isEnabled);
    setCapturedImage(null);
    setResult(null);
    stopLiveDetection();
    stopLiveLetterDetection();
  };

  const capture = async () => {
    if (webcamRef.current) {
      const imageSrc = webcamRef.current.getScreenshot();
      if (imageSrc) {
        setCapturedImage(imageSrc);
        setLoading(true);
        try {
          const prediction = await predictSign(imageSrc);
          setResult(prediction);
          if (prediction && prediction !== 'No hand detected') {
            if (previousResult === 'No hand detected') {
              setDetectedSign(prediction);
              setDetectedSignsHistory(prevSigns => [...prevSigns, prediction]);
              setPreviousResult(prediction);
              setTimeout(() => {
                setPreviousResult('No hand detected');
              }, 5000);
            }
          }
        } finally {
          setLoading(false);
        }
      }
    }
  };

  const reset = () => {
    setCapturedImage(null);
    setResult(null);
    setDetectedSign(null);
    setDetectedSignsHistory([]);
    setProcessedSigns([]);
    setPreviousResult('No hand detected');
    stopLiveDetection();
    stopLiveLetterDetection();
  };

  const toggleLiveDetection = () => {
    setIsLive(!isLive);
    setIsLiveLetter(false);
    
    if (!isLive) {
      setLiveTimer(5);
      const id = setInterval(() => {
        setLiveTimer(prev => {
          if (prev === 1) {
            if (webcamRef.current) {
              const imageSrc = webcamRef.current.getScreenshot();
              if (imageSrc) {
                predictSign(imageSrc)
                  .then(prediction => {
                    setResult(prediction);
                    if (prediction && prediction !== 'No hand detected') {
                      if (previousResult === 'No hand detected') {
                        setDetectedSign(prediction);
                        setDetectedSignsHistory(prevSigns => [...prevSigns, prediction]);
                        setPreviousResult(prediction);
                        setTimeout(() => {
                          setPreviousResult('No hand detected');
                        }, 5000);
                      }
                    }
                  })
                  .catch(error => {
                    console.error('Error predicting sign:', error);
                    setResult(null);
                  });
              }
            }
            return 5;
          }
          return prev - 1;
        });
      }, 1000);
      
      setIntervalId(id);
    } else {
      if (intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }
    }
  };

  const toggleLiveLetterDetection = () => {
    setIsLiveLetter(!isLiveLetter);
    setIsLive(false);
    
    if (!isLiveLetter) {
      const id = setInterval(() => {
        if (webcamRef.current) {
          const imageSrc = webcamRef.current.getScreenshot();
          if (imageSrc) {
            predictSign(imageSrc)
              .then(prediction => {
                setResult(prediction);
                if (prediction && prediction !== 'No hand detected') {
                  if (previousResult === 'No hand detected') {
                    setDetectedSign(prediction);
                    setDetectedSignsHistory(prevSigns => [...prevSigns, prediction]);
                    setPreviousResult(prediction);
                    setTimeout(() => {
                      setPreviousResult('No hand detected');
                    }, 5000);
                  }
                }
              })
              .catch(error => {
                console.error('Error predicting sign:', error);
                setResult(null);
              });
          }
        }
      }, 1000);
      
      setIntervalId(id);
    } else {
      if (intervalId) {
        clearInterval(intervalId);
        setIntervalId(null);
      }
    }
  };

  const stopLiveDetection = () => {
    setIsLive(false);
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  };

  const stopLiveLetterDetection = () => {
    setIsLiveLetter(false);
    if (intervalId) {
      clearInterval(intervalId);
      setIntervalId(null);
    }
  };

  const clearTextBoxes = () => {
    setDetectedSign(null);
    setProcessedSigns([]);
    setDetectedSignsHistory([]);
    setResult(null);
  };

  const speakText = async () => {
    if (isSpeaking) return;
  
    const textToProcess = processedSigns.join('');
    
    if (!textToProcess.trim()) {
      alert('No text to speak');
      return;
    }
  
    setIsSpeaking(true);
    setLoading(true);
  
    try {
      const correctedText = await correctTextWithGemini(textToProcess);
      setProcessedSigns(correctedText.split(''));
      
      const utterance = new SpeechSynthesisUtterance(correctedText);
      utterance.onend = () => {
        setIsSpeaking(false);
        setLoading(false);
      };
      window.speechSynthesis.speak(utterance);
    } catch (error) {
      console.error('Error:', error);
      setIsSpeaking(false);
      setLoading(false);
      const utterance = new SpeechSynthesisUtterance(textToProcess);
      window.speechSynthesis.speak(utterance);
    }
  };

  useEffect(() => {
    return () => {
      if (intervalId) {
        clearInterval(intervalId);
      }
      if (captureTimeoutId) {
        clearTimeout(captureTimeoutId);
      }
    };
  }, [intervalId, captureTimeoutId]);

  useEffect(() => {
    const processSigns = (signs: string[]) => {
      if (signs.length === 0) return [];
      const processed: string[] = [];
      let count = 1;
      for (let i = 1; i <= signs.length; i++) {
        if (signs[i] === signs[i - 1]) {
          count++;
        } else {
          if (count >= 4) {
            processed.push(signs[i - 1], signs[i - 1]);
          } else {
            processed.push(signs[i - 1]);
          }
          count = 1;
        }
      }
      return processed;
    };

    setProcessedSigns(processSigns(detectedSignsHistory));
  }, [detectedSignsHistory]);

  const videoConstraints = {
    width: 720,
    height: 480,
    facingMode: "user",
  };

  return (
    <div className="bg-dark-lighter/50 rounded-xl border-2 border-accent p-6 shadow-lg w-full max-w-7xl mx-auto">
      {/* Header Section */}
      <motion.div className="text-center mb-6">
        <motion.h1 className="text-3xl font-bold mb-2">
          <span className="text-white/90">{mode === 'alphabet' ? 'Alphabet' : 'Word'} Recognition</span>
        </motion.h1>
        <motion.p className="text-white/60 text-sm">
          Communicate effortlessly with sign language
        </motion.p>
      </motion.div>

      {/* Main Content */}
      <div className="grid md:grid-cols-2 gap-8">
        {/* Webcam Section */}
        <div className="space-y-4">
          <motion.div className="relative aspect-square w-full max-w-2xl mx-auto rounded-xl overflow-hidden bg-gradient-to-br from-black/30 to-black/10 border border-white/10">
            {isEnabled ? (
              <>
                <Webcam
                  audio={false}
                  ref={webcamRef}
                  screenshotFormat="image/jpeg"
                  videoConstraints={videoConstraints}
                  className="w-full h-full object-cover"
                  style={{ transform: "scaleX(-1)" }}
                />
                {(isLive || isLiveLetter) && (
                  <div className="absolute top-2 right-2 bg-black/50 text-white p-2 rounded-lg">
                    {liveTimer}
                  </div>
                )}
                {result && (
                  <motion.div className="absolute bottom-0 left-0 right-0 bg-accent/20 backdrop-blur-sm p-4 text-center">
                    <h3 className="text-xl font-semibold text-white">Detected Sign:</h3>
                    <p className="text-2xl font-bold text-accent neon-text">{result}</p>
                    <p className="text-white/60">Confidence: {confidence.toFixed(2)}%</p>
                  </motion.div>
                )}
              </>
            ) : (
              <div className="flex items-center justify-center h-full bg-primary-dark">
                <FaCamera className="w-16 h-16 text-dark-lightest opacity-50" />
              </div>
            )}
          </motion.div>

          <div className="flex flex-wrap justify-center gap-3">
            {!isEnabled ? (
              <button
                onClick={toggleWebcam}
                className="btn-primary px-6 py-3 rounded-lg shadow-md hover:scale-105 transition-transform duration-300 flex items-center text-sm"
              >
                <Camera size={20} className="mr-2" /> Turn On Camera
              </button>
            ) : (
              <>
                <button
                  onClick={toggleWebcam}
                  className="btn-primary px-6 py-3 rounded-lg flex items-center text-sm"
                >
                  <CameraOff size={20} className="mr-2" />  Camera Off
                </button>
                
                {/* {!isLive && !isLiveLetter && (
                  // <button
                  //   onClick={capture}
                  //   className="btn-primary px-6 py-3 rounded-lg flex items-center text-sm"
                  // >
                  //   <Camera size={20} className="mr-2" /> Capture {mode === 'alphabet' ? 'Letter' : 'Word'}
                  // </button>
                )} */}
                
                <button
                  onClick={toggleLiveDetection}
                  className={`px-6 py-3 rounded-lg flex items-center text-sm btn-primary  text-white`}
                >
                  <Play size={20} className={`mr-2 ${isLive ? "animate-pulse" : ""}`} />
                  {isLive ? 'Stop Live Word' : 'Live Word'}
                </button>
                
                {/* <button
                  onClick={toggleLiveLetterDetection}
                  className={`px-6 py-3 rounded-lg flex items-center text-sm ${
                    isLiveLetter ? 'bg-red-600' : 'bg-green-600'
                  } text-white`}
                >
                  <Play size={20} className={`mr-2 ${isLiveLetter ? "animate-pulse" : ""}`} />
                  {isLiveLetter ? 'Stop Live Letter' : 'Live Letter'}
                </button> */}
              </>
            )}
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

              {!result || result === 'No hand detected' ? (
                <div className="flex flex-col items-center justify-center space-y-3">
                  <FaHandPaper className="text-4xl text-white/40" />
                  <p className="text-white/60 text-lg">Make a sign to begin</p>
                </div>
              ) : (
                <div className="w-full space-y-4 flex flex-col items-center">
                  <h4 className="text-2xl font-bold text-white text-center neon-text">
                    {result}
                  </h4>
                  <div className="w-3/4 bg-white/10 rounded-full h-2 relative">
                    <div
                      className="absolute top-0 left-0 h-full bg-gradient-to-r from-green-400 via-yellow-400 to-red-500 rounded-full"
                      style={{ width: `${confidence}%` }}
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
          <h3 className="text-xl font-bold">Predicted {mode === 'alphabet' ? 'Letters' : 'Words'}</h3>
          <div className="flex gap-2">
            <button
              onClick={speakText}
              disabled={processedSigns.length === 0 || loading}
              className={`px-4 py-2 rounded-lg flex items-center ${
                processedSigns.length === 0 || loading
                  ? 'bg-gray-700 text-gray-400'
                  : 'bg-accent text-white hover:bg-accent/80'
              }`}
            >
              <Volume2 size={18} className="mr-2" />
              Speak
            </button>
            <button
              onClick={clearTextBoxes}
              disabled={processedSigns.length === 0}
              className={`px-4 py-2 rounded-lg flex items-center ${
                processedSigns.length === 0
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
          {processedSigns.length > 0 ? (
            <p className="text-white text-lg">
              {processedSigns.join(' ')}
            </p>
          ) : (
            <p className="text-white/40 italic">Your predicted {mode === 'alphabet' ? 'letters' : 'words'} will appear here</p>
          )}
        </div>
        
        {loading && (
          <div className="text-center text-accent mt-2 text-sm">
            Processing...
          </div>
        )}
      </div>
    </div>
  );
};

export default WebcamCapture;