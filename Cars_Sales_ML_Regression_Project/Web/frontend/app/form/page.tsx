"use client";

import { useEffect, useState } from "react";

export default function FormPage() {
  const [carsData, setCarsData] = useState<
    Record<
      string,
      {
        models: Record<
          string,
          {
            bodies: Record<
              string,
              {
                trims: Record<
                  string,
                  {
                    transmissions: string[];
                  }
                >;
              }
            >;
          }
        >;
      }
    >
  >({});
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
          !result.schema.cars ||
          !result.schema.categorical ||
          !result.schema.numerical
        ) {
          throw new Error(
            "Missing 'cars', 'categorical', or 'numerical' data in API response"
          );
        }

        setCarsData(result.schema.cars);
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
      alert(`Predicted Vehicle Sales Price: ${Math.round(result.prediction)}`);
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
        {/* Make Dropdown */}
        <div className="flex flex-col">
          <label className="mb-1 font-semibold" htmlFor="make">
            Make
          </label>
          <select
            id="make"
            className="rounded-md border p-2 text-black"
            value={formData.make || ""}
            onChange={(e) => handleInputChange("make", e.target.value)}>
            <option value="">Select Make</option>
            {Object.keys(carsData).map((make) => (
              <option key={make} value={make}>
                {make}
              </option>
            ))}
          </select>
        </div>

        {/* Model Dropdown */}
        {formData.make && (
          <div className="flex flex-col">
            <label className="mb-1 font-semibold" htmlFor="model">
              Model
            </label>
            <select
              id="model"
              className="rounded-md border p-2 text-black"
              value={formData.model || ""}
              onChange={(e) => handleInputChange("model", e.target.value)}>
              <option value="">Select Model</option>
              {Object.keys(carsData[formData.make]?.models || {}).map(
                (model) => (
                  <option key={model} value={model}>
                    {model}
                  </option>
                )
              )}
            </select>
          </div>
        )}

        {/* Body Dropdown */}
        {formData.model && (
          <div className="flex flex-col">
            <label className="mb-1 font-semibold" htmlFor="body">
              Body
            </label>
            <select
              id="body"
              className="rounded-md border p-2 text-black"
              value={formData.body || ""}
              onChange={(e) => handleInputChange("body", e.target.value)}>
              <option value="">Select Body</option>
              {Object.keys(
                carsData[formData.make]?.models[formData.model]?.bodies || {}
              ).map((body) => (
                <option key={body} value={body}>
                  {body}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Trim Dropdown */}
        {formData.body && (
          <div className="flex flex-col">
            <label className="mb-1 font-semibold" htmlFor="trim">
              Trim
            </label>
            <select
              id="trim"
              className="rounded-md border p-2 text-black"
              value={formData.trim || ""}
              onChange={(e) => handleInputChange("trim", e.target.value)}>
              <option value="">Select Trim</option>
              {Object.keys(
                carsData[formData.make]?.models[formData.model]?.bodies[
                  formData.body
                ]?.trims || {}
              ).map((trim) => (
                <option key={trim} value={trim}>
                  {trim}
                </option>
              ))}
            </select>
          </div>
        )}
        {/* Rest of the categorical fields */}
        {Object.entries(categoricalData)
          .filter(
            ([column]) =>
              !["make", "model", "body", "trim", "transmission"].includes(
                column
              )
          )
          .map(([column, options]) => (
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
                {options.map((option) => (
                  <option key={option} value={option}>
                    {option}
                  </option>
                ))}
              </select>
            </div>
          ))}

        {/* Transmission Dropdown (Fixed Duplicates) */}
        {formData.trim && (
          <div className="flex flex-col">
            <label className="mb-1 font-semibold" htmlFor="transmission">
              Transmission
            </label>
            <select
              id="transmission"
              className="rounded-md border p-2 text-black"
              value={formData.transmission || ""}
              onChange={(e) =>
                handleInputChange("transmission", e.target.value)
              }>
              <option value="">Select Transmission</option>
              {[
                ...new Set(
                  carsData[formData.make]?.models[formData.model]?.bodies[
                    formData.body
                  ]?.trims[formData.trim]?.transmissions || []
                ),
              ].map((transmission) => (
                <option key={transmission} value={transmission}>
                  {transmission}
                </option>
              ))}
            </select>
          </div>
        )}

        {/* Numerical Fields */}
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
                onChange={(e) => handleInputChange(column, e.target.value)}
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
