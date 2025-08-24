"use client";

import { useState } from 'react';
import Image from 'next/image';

type PredictionResult = {
  prediction: string;
  confidence: number;
  heatmap: string; // This will be a base64 encoded image string
  filename: string;
};

export default function Home() {
  const [file, setFile] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResult | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  const handleFileChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const selectedFile = event.target.files?.[0];
    if (selectedFile) {
      setFile(selectedFile);
      setPrediction(null);
      setError(null);
      // Create a URL for image preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result as string);
      };
      reader.readAsDataURL(selectedFile);
    }
  };

  const handleSubmit = async () => {
    if (!file) {
      setError("Please select an image file first.");
      return;
    }

    setIsLoading(true);
    setError(null);
    setPrediction(null);

    const formData = new FormData();
    formData.append('file', file);

    try {
      const apiUrl = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:5001';
      const response = await fetch(`${apiUrl}/api/predict`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        const errData = await response.json();
        throw new Error(errData.error || `Server responded with status: ${response.status}`);
      }

      const data: PredictionResult = await response.json();
      setPrediction(data);
    } catch (e: any) {
      setError(e.message || "An unexpected error occurred during prediction.");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <main className="flex min-h-screen flex-col items-center justify-center p-4 sm:p-8 md:p-12 bg-[rgb(var(--background-rgb))] text-[rgb(var(--foreground-rgb))]">
      <div className="w-full max-w-6xl mx-auto">
        <header className="text-center mb-8">
          <h1 className="text-4xl sm:text-5xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-blue-500">
            Deepfake Detector
          </h1>
          <p className="text-lg text-gray-400 mt-2">
            Leveraging AI to distinguish real from synthetic media.
          </p>
        </header>

        <div className={`grid gap-8 ${prediction ? 'grid-cols-1 md:grid-cols-2' : 'grid-cols-1'}`}>
          {/* Left Column: Controls and Results */}
          <div className="bg-[rgb(var(--surface-rgb))] p-6 rounded-2xl border border-gray-700 shadow-lg">
            <h2 className="text-2xl font-semibold mb-4 border-b border-gray-600 pb-3">Upload & Analyze</h2>

            <div className="space-y-6">
              <div>
                <label htmlFor="file-upload" className="block text-sm font-medium text-gray-400 mb-2">
                  Choose an image to inspect
                </label>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileChange}
                  className="block w-full text-sm text-gray-400 file:mr-4 file:py-2 file:px-4 file:rounded-lg file:border file:border-gray-600 file:bg-gray-700 file:text-gray-300 hover:file:bg-gray-600 transition-colors"
                />
              </div>

              {previewUrl && !prediction && (
                <div className="text-center">
                  <Image src={previewUrl} alt="Image preview" width={200} height={200} className="rounded-lg mx-auto shadow-md" />
                </div>
              )}

              <button
                onClick={handleSubmit}
                disabled={!file || isLoading}
                className="w-full bg-[rgb(var(--accent-rgb))] text-white font-bold py-3 px-4 rounded-lg hover:bg-[rgb(var(--accent-hover-rgb))] disabled:bg-gray-600 disabled:cursor-not-allowed transition-all duration-300 transform hover:scale-105"
              >
                {isLoading ? 'Analyzing...' : 'Detect Deepfake'}
              </button>
            </div>

            {error && (
              <div className="mt-6 p-4 bg-red-900/50 text-red-300 border border-red-700 rounded-lg">
                <p className="font-bold">Analysis Failed</p>
                <p className="text-sm">{error}</p>
              </div>
            )}

            {prediction && (
              <div className="mt-6 p-4 bg-gray-900/50 border border-gray-700 rounded-lg space-y-3">
                <h3 className="text-xl font-semibold text-teal-400">Detection Result</h3>
                <p><strong>File:</strong> <span className="text-gray-400">{prediction.filename}</span></p>
                <div className="flex items-center space-x-2">
                  <strong>Prediction:</strong>
                  <span className={`font-bold text-lg px-3 py-1 rounded-full ${prediction.prediction === 'FAKE' ? 'bg-red-500/20 text-red-400' : 'bg-green-500/20 text-green-400'}`}>
                    {prediction.prediction}
                  </span>
                </div>
                <p><strong>Confidence Score:</strong> <span className="font-mono text-lg text-teal-400">{Math.round(prediction.confidence * 100)}%</span></p>
              </div>
            )}
          </div>

          {/* Right Column: Heatmap Visualization */}
          {prediction && (
            <div className="bg-[rgb(var(--surface-rgb))] p-6 rounded-2xl border border-gray-700 shadow-lg flex flex-col items-center justify-center">
              <h2 className="text-2xl font-semibold mb-4 text-center">Explainability Heatmap</h2>
              <p className="text-sm text-gray-400 mb-4 text-center">This heatmap highlights the areas the model focused on for its prediction.</p>
              <Image
                src={prediction.heatmap}
                alt="Prediction heatmap"
                width={400}
                height={400}
                className="rounded-lg shadow-2xl border-2 border-gray-600"
              />
            </div>
          )}
        </div>
      </div>
    </main>
  );
}
