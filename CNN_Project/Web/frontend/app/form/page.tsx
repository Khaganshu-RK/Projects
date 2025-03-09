"use client";

import { useRef, useState, useEffect } from "react";
import axios from "axios";

const url = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

type Prediction = {
  age: number;
  gender: string;
  ethnicity: string;
};

export default function Home() {
  const videoRef = useRef<HTMLVideoElement | null>(null);
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const [prediction, setPrediction] = useState<Prediction | null>(null);
  const [isLoading, setIsLoading] = useState(false);

  useEffect(() => {
    navigator.mediaDevices
      .getUserMedia({ video: true })
      .then((stream) => {
        if (videoRef.current) {
          videoRef.current.srcObject = stream;
        }
      })
      .catch((error) => console.error("Error accessing webcam:", error));
  }, []);

  const captureImage = () => {
    const canvas = canvasRef.current;
    const video = videoRef.current;

    if (!canvas || !video) {
      console.error("Canvas or Video element not found.");
      return;
    }

    const context = canvas.getContext("2d");
    if (!context) {
      console.error("Failed to get canvas context.");
      return;
    }

    context.drawImage(video, 0, 0, canvas.width, canvas.height);

    canvas.toBlob((blob) => {
      if (blob) {
        uploadImage(blob);
      } else {
        console.error("Failed to convert canvas to Blob.");
      }
    }, "image/jpeg");
  };

  const uploadImage = async (blob: Blob) => {
    setIsLoading(true);
    const formData = new FormData();
    formData.append("file", blob, "image.jpg");

    try {
      const response = await axios.post<Prediction>(
        `${url}/predict`,
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
          responseType: "json",
        }
      );
      console.log("Full Axios Response:", response);

      console.log("Response Data:", response.data);
      setPrediction(response.data);
    } catch (error) {
      console.error("Error predicting:", error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="flex h-screen flex-col items-center justify-center bg-gray-900 text-white">
      <video
        ref={videoRef}
        autoPlay
        playsInline
        width="320"
        height="240"
        className="rounded border-2 border-white"
      />

      <canvas ref={canvasRef} width="96" height="96" className="hidden" />

      <button
        onClick={captureImage}
        className="mt-4 rounded bg-blue-500 px-4 py-2">
        {isLoading ? "Processing..." : "Predict"}
      </button>

      {prediction && (
        <div className="mt-4 rounded bg-gray-800 p-4">
          <p>Age: {prediction.age}</p>
          <p>Gender: {prediction.gender}</p>
          <p>Ethnicity: {prediction.ethnicity}</p>
        </div>
      )}
    </div>
  );
}
