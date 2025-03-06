"use client";

import { useEffect, useState } from "react";

export default function FormPage() {
  const [categoricalData, setCategoricalData] = useState<
    Record<string, string[]>
  >({});
  const [numericalData, setNumericalData] = useState<
    Record<string, { min: number; max: number; type: "float" | "int" }>
  >({});
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [formData, setFormData] = useState<Record<string, string | number>>({});

  useEffect(() => {
    async function fetchData() {
      try {
        const response = await fetch("/api/getColumns");
        const result = await response.json();

        console.log("API Response:", result);

        if (!result.success) {
          throw new Error(result.error || "Unknown API error");
        }

        if (
          !result.schema ||
          !result.schema.categorical ||
          !result.schema.numerical
        ) {
          throw new Error(
            "Missing 'categorical' or 'numerical' data in API response"
          );
        }

        setCategoricalData(result.schema.categorical);
        setNumericalData(result.schema.numerical);
      } catch (err: unknown) {
        console.error("Fetch Error:", err);
        if (err instanceof Error) {
          setError(err.message);
        }
      } finally {
        setLoading(false);
      }
    }

    fetchData();
  }, []);

  const handleInputChange = (field: string, value: string | number) => {
    setFormData((prev) => ({ ...prev, [field]: value }));
  };

  const handleSubmit = async (event: React.FormEvent) => {
    event.preventDefault();
    const isFormValid = Object.values(formData).every(
      (value) => value !== "" && value !== null
    );
    if (!isFormValid) {
      alert("Please fill out all fields before submitting.");
      return;
    }

    console.log("Form Submitted:", formData);

    try {
      const response = await fetch("/api/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(formData),
      });

      const result = await response.json();

      if (!response.ok) {
        throw new Error(result.error || "Something went wrong!");
      }

      console.log("Prediction Result:", result);
      alert(`Predicted VehicleSales Price: ${Math.round(result.prediction)}`);
    } catch (error: unknown) {
      console.error("Submission Error:", error);
      alert("Failed to get prediction. Please try again.");
    }
  };

  if (loading) return <p className="text-blue-500">Loading...</p>;

  if (error) return <p className="text-red-500">Error: {error}</p>;

  return (
    <div className="mx-auto max-w-lg p-6">
      <h1 className="mb-4 text-center text-2xl font-bold">
        Predict Cars Sales Price
      </h1>
      <form onSubmit={handleSubmit} className="space-y-4">
        {Object.entries(categoricalData).map(([column, values]) => (
          <div key={column} className="flex flex-col">
            <label className="mb-1 font-semibold" htmlFor={column}>
              {column}
            </label>
            <select
              id={column}
              className="rounded-md border p-2 text-black"
              value={formData[column] || ""}
              onChange={(e) => handleInputChange(column, e.target.value)}>
              <option value="">Select {column}</option>
              {values.map((value) => (
                <option key={value} value={value}>
                  {value}
                </option>
              ))}
            </select>
          </div>
        ))}

        {Object.entries(numericalData)
          .filter(([column]) => column !== "sellingprice")
          .map(([column, { min, max, type }]) => (
            <div key={column} className="flex flex-col">
              <label className="mb-1 font-semibold" htmlFor={column}>
                {column} (Min: {min}, Max: {max})
              </label>
              <input
                type="number"
                step={type === "int" ? "1" : "0.1"}
                id={column}
                className="rounded-md border p-2 text-black"
                min={min}
                max={max}
                value={formData[column] || ""}
                onChange={(e) => {
                  const value =
                    type === "int"
                      ? parseInt(e.target.value, 10)
                      : parseFloat(e.target.value);
                  handleInputChange(column, isNaN(value) ? "" : value);
                }}
              />
            </div>
          ))}

        <button
          type="submit"
          className="mt-4 rounded-md bg-blue-500 px-4 py-2 text-white">
          Submit
        </button>
      </form>
    </div>
  );
}
