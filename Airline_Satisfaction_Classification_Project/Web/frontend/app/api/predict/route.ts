import { NextResponse } from "next/server";

const url = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

export async function POST(req: Request) {
  const data = await req.json();

  const response = await fetch(`${url}/predict`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ data }),
  });

  const result = await response.json();
  return NextResponse.json({ prediction: result.prediction });
}
